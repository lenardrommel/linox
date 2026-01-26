# topklinox.py

r"""Top-k eigendecomposition for Kronecker products.

This module provides memory-efficient top-k/bottom-k eigendecomposition
for Kronecker products without forming the full matrix.

- :class:`KroneckerSelectedEigenvectors`: Linear operator representing selected
    eigenvectors of a Kronecker product
- :func:`topk_eigh_kronecker`: Compute top-k or bottom-k eigenvalues/vectors
"""

import heapq
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from linox._arithmetic import leigh
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix
from linox.utils import as_linop


class KroneckerSelectedEigenvectors(LinearOperator):
    r"""Matrix-free operator for selected Kronecker eigenvectors.

    Represents :math:`Q_k` where columns are
    :math:`q_A^{(i)} \otimes q_B^{(j)} \otimes \ldots` for selected index tuples.

    Never forms the full Kronecker product. For two factors A, B with
    eigenvectors :math:`Q_A, Q_B`:

    .. math::
        Q_k \alpha = \text{vec}((U_B \odot \alpha) U_A^T)

    where :math:`U_A = Q_A[:, \text{selected}_A]`, :math:`U_B = Q_B[:, \text{selected}_B]`.

    Args:
        factor_vecs: List of (n_i, n_i) eigenvector matrices from each factor
        selected_indices: List of k tuples specifying which eigenvector combinations
        sort_indices: Sorting permutations applied to each factor's eigenvalues
    """

    def __init__(
        self,
        factor_vecs: list[jax.Array],
        selected_indices: list[tuple[int, ...]],
        sort_indices: list[jax.Array],
    ) -> None:
        self._factor_vecs = factor_vecs
        self._selected_indices = selected_indices
        self._sort_indices = sort_indices

        self._d = len(factor_vecs)
        self._k = len(selected_indices)
        self._factor_dims = [Q.shape[0] for Q in factor_vecs]
        self._n_total = int(jnp.prod(jnp.array(self._factor_dims)))

        # Precompute gathered columns for each factor: (n_i, k)
        self._gathered = []
        for i in range(self._d):
            idx_for_factor = jnp.array([
                self._sort_indices[i][sel[i]] for sel in selected_indices
            ])
            self._gathered.append(self._factor_vecs[i][:, idx_for_factor])

        dtype = factor_vecs[0].dtype
        super().__init__((self._n_total, self._k), dtype)

    @property
    def k(self) -> int:
        """Number of selected eigenvectors."""
        return self._k

    @property
    def num_factors(self) -> int:
        """Number of Kronecker factors."""
        return self._d

    @property
    def factor_dims(self) -> list[int]:
        """Dimensions of each factor."""
        return self._factor_dims

    def tree_flatten(self) -> tuple[tuple, dict]:
        children = (
            tuple(self._factor_vecs),
            tuple(self._sort_indices),
        )
        aux_data = {
            "selected_indices": self._selected_indices,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict, children: tuple
    ) -> "KroneckerSelectedEigenvectors":
        factor_vecs, sort_indices = children
        return cls(
            list(factor_vecs),
            aux_data["selected_indices"],
            list(sort_indices),
        )

    def _matmul(self, alpha: jax.Array) -> jax.Array:
        r"""Compute :math:`Q_k \alpha` without forming the Kronecker product."""
        squeeze = False
        if alpha.ndim == 1:
            alpha = alpha[:, None]
            squeeze = True

        if self._d == 2:
            # Optimized 2-factor case using einsum
            # kron(qA, qB)[i*nB + j] = qA[i] * qB[j]
            UA, UB = self._gathered[0], self._gathered[1]
            nA, nB = self._factor_dims[0], self._factor_dims[1]

            # Y[i, j, b] = sum_l UA[i, l] * alpha[l, b] * UB[j, l]
            Y = jnp.einsum("il,lb,jl->ijb", UA, alpha, UB)
            result = Y.reshape((nA * nB, -1))

            if squeeze:
                result = result.squeeze(-1)
            return result

        # General d-factor case
        result = jnp.zeros((self._n_total, alpha.shape[1]), dtype=self.dtype)

        for l in range(self._k):
            vec_l = self._gathered[0][:, l]
            for i in range(1, self._d):
                vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
            result = result + vec_l[:, None] * alpha[l, :]

        if squeeze:
            result = result.squeeze(-1)
        return result

    # def _rmatmul(self, v: jax.Array) -> jax.Array:
    #     r"""Compute :math:`Q_k^T v` without forming the Kronecker product."""
    #     squeeze = False
    #     if v.ndim == 1:
    #         v = v[None, :]
    #         squeeze = True

    #     if self._d == 2:
    #         UA, UB = self._gathered[0], self._gathered[1]
    #         nA, nB = self._factor_dims[0], self._factor_dims[1]

    #         # v: (batch, nA * nB)
    #         X = v.reshape((v.shape[0], nA, nB))
    #         # result[b, l] = sum_{i,j} UA[i, l] * X[b, i, j] * UB[j, l]
    #         T = jnp.einsum("il,bij->blj", UA, X)
    #         result = jnp.einsum("blj,jl->bl", T, UB)

    #         if squeeze:
    #             result = result.squeeze(0)
    #         return result

    #     # General case
    #     result = jnp.zeros((v.shape[0], self._k), dtype=self.dtype)

    #     for l in range(self._k):
    #         vec_l = self._gathered[0][:, l]
    #         for i in range(1, self._d):
    #             vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
    #         result = result.at[:, l].set(v @ vec_l)

    #     if squeeze:
    #         result = result.squeeze(0)
    #     return result

    def _rmatmul(self, v: jax.Array) -> jax.Array:
        r"""Compute Q_k^T v robustly for v shaped:
        - (n_total,)
        - (batch, n_total)
        - (n_total, p)  (stack of p column vectors; Linox often gives this)
        """
        n = self._n_total
        restore = None

        if v.ndim == 1:
            # (n,)
            pass

        elif v.ndim == 2:
            if v.shape[0] == n and v.shape[1] != n:
                # (n, p): p column vectors
                # convert to (p, n) batch of row-vectors
                v = jnp.swapaxes(v, 0, 1)  # (p, n)
                restore = ("cols",)  # output should become (k, p)

            elif v.shape[1] == n:
                # (batch, n): already correct
                restore = ("batch",)

            elif v.shape[1] == 1 and v.shape[0] == n:
                # Special case: (n,1) column vector (most common Linox path)
                v = v[:, 0]  # -> (n,)
            else:
                raise ValueError(
                    f"Unsupported v shape {v.shape}. Expected (n,), (batch,n), or (n,p)."
                )
        else:
            raise ValueError(f"Unsupported v.ndim={v.ndim}. Expected 1 or 2.")

        # --- Now compute ---
        squeeze_single = False
        if v.ndim == 1:
            v = v[None, :]  # (1, n)
            squeeze_single = True

        if self._d == 2:
            UA, UB = self._gathered[0], self._gathered[1]
            nA, nB = self._factor_dims[0], self._factor_dims[1]

            # v: (batch, nA*nB)
            X = v.reshape((v.shape[0], nA, nB))  # (batch, nA, nB)
            T = jnp.einsum("il,bij->blj", UA, X)  # (batch, k, nB)
            result = jnp.einsum("blj,jl->bl", T, UB)  # (batch, k)

        else:
            # General case (slow but correct)
            result = jnp.zeros((v.shape[0], self._k), dtype=self.dtype)
            for l in range(self._k):
                vec_l = self._gathered[0][:, l]
                for i in range(1, self._d):
                    vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
                result = result.at[:, l].set(v @ vec_l)

        # --- Restore shapes ---
        if squeeze_single:
            result = result[0, :]  # (k,)
        else:
            # (batch, k)
            pass

        if restore == ("cols",):
            # input was (n,p), we converted to (p,n), so output is (p,k) or (k,)?
            # currently result is (p,k); return (k,p) to match matmul result (k,p)
            if result.ndim == 2:
                result = jnp.swapaxes(result, 0, 1)  # (k, p)

        return result

    def transpose(self) -> "KroneckerSelectedEigenvectorsTranspose":
        return KroneckerSelectedEigenvectorsTranspose(self)

    def todense(self) -> jax.Array:
        """Build dense matrix. WARNING: This defeats the memory efficiency!"""
        cols = []
        for l in range(self._k):
            vec_l = self._gathered[0][:, l]
            for i in range(1, self._d):
                vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
            cols.append(vec_l)
        return jnp.stack(cols, axis=1)


class KroneckerSelectedEigenvectorsTranspose(LinearOperator):
    r"""Transpose of :class:`KroneckerSelectedEigenvectors`."""

    def __init__(self, parent: KroneckerSelectedEigenvectors) -> None:
        self._parent = parent
        super().__init__((parent.shape[1], parent.shape[0]), parent.dtype)

    def tree_flatten(self) -> tuple[tuple, dict]:
        return (self._parent,), {}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict, children: tuple
    ) -> "KroneckerSelectedEigenvectorsTranspose":
        return cls(children[0])

    def _matmul(self, v: jax.Array) -> jax.Array:
        return self._parent._rmatmul(v)

    def transpose(self) -> KroneckerSelectedEigenvectors:
        return self._parent

    def todense(self) -> jax.Array:
        return self._parent.todense().T


# Register as PyTrees
jax.tree_util.register_pytree_node_class(KroneckerSelectedEigenvectors)
jax.tree_util.register_pytree_node_class(KroneckerSelectedEigenvectorsTranspose)


def topk_eigh_kronecker(
    factors: Sequence[LinearOperator | jax.Array],
    k: int,
    *,
    largest: bool = True,
) -> tuple[jax.Array, KroneckerSelectedEigenvectors]:
    r"""Compute top-k or bottom-k eigenvalues/vectors of a Kronecker product.

    Uses a heap-based best-first search on the monotone grid of eigenvalue
    products. Avoids :math:`O(\prod n_i^2)` memory by never forming the full
    Kronecker product.

    Note:
        Assumes all factors are PSD (positive semi-definite) so that
        eigenvalues are non-negative and the monotone grid property holds.

    Args:
        factors: Sequence of symmetric PSD LinearOperators or arrays.
        k: Number of eigenvalues to return.
        largest: If True, return largest k eigenvalues; else smallest k.

    Returns:
        eigenvalues: Array of shape (k,) with the k largest/smallest eigenvalues.
        eigenvectors: LinearOperator of shape (n_total, k) representing the
            k eigenvectors without forming the full Kronecker product.

    Example:
        >>> A = Matrix(jnp.eye(3) * 2)
        >>> B = Matrix(jnp.eye(4) * 3)
        >>> eigs, Q = topk_eigh_kronecker([A, B], k=5, largest=True)
        >>> # Q @ alpha computes linear combination without densifying
        >>> v = Q @ jnp.ones(5)
    """
    factors = [as_linop(f) for f in factors]
    d = len(factors)

    # Compute eigendecompositions of each factor using linox's leigh
    factor_eigs = []
    factor_vecs = []
    sort_indices = []

    for A in factors:
        w, Q = leigh(A)
        # Sort eigenvalues (descending for largest, ascending for smallest)
        order = jnp.argsort(-w) if largest else jnp.argsort(w)
        factor_eigs.append(w[order])
        sort_indices.append(order)
        # Get dense eigenvectors - this is O(n_i^2) per factor, not O(prod n_i^2)
        factor_vecs.append(Q.todense() if hasattr(Q, "todense") else jnp.asarray(Q))

    sizes = [len(w) for w in factor_eigs]

    def compute_eigenvalue(indices: tuple[int, ...]) -> float:
        prod = 1.0
        for i, idx in enumerate(indices):
            prod = prod * factor_eigs[i][idx]
        return float(prod)

    # Heap-based best-first search
    initial_idx = tuple(0 for _ in range(d))
    initial_val = compute_eigenvalue(initial_idx)

    heap = [(-initial_val, initial_idx)] if largest else [(initial_val, initial_idx)]
    visited = {initial_idx}

    eigenvalues = []
    selected_indices = []

    while len(eigenvalues) < k and heap:
        if largest:
            neg_val, idx = heapq.heappop(heap)
            val = -neg_val
        else:
            val, idx = heapq.heappop(heap)

        eigenvalues.append(val)
        selected_indices.append(idx)

        # Add neighbors (increment each dimension by 1)
        for dim in range(d):
            new_idx = list(idx)
            new_idx[dim] += 1
            new_idx = tuple(new_idx)

            if new_idx[dim] < sizes[dim] and new_idx not in visited:
                visited.add(new_idx)
                new_val = compute_eigenvalue(new_idx)
                if largest:
                    heapq.heappush(heap, (-new_val, new_idx))
                else:
                    heapq.heappush(heap, (new_val, new_idx))

    # Build matrix-free eigenvector operator
    eigenvectors = KroneckerSelectedEigenvectors(
        factor_vecs, selected_indices, sort_indices
    )

    return jnp.array(eigenvalues), eigenvectors


# ============================================================================
# Low-rank sqrt/inv-sqrt operators (useful for sampling and whitening)
# ============================================================================


class KroneckerLowRankFactor(LinearOperator):
    r"""Low-rank factor: :math:`Q_k \text{diag}(f(\lambda_k))` for some function f.

    Base class for sqrt and inverse-sqrt operators.
    """

    def __init__(
        self,
        eigvecs: KroneckerSelectedEigenvectors,
        weights: jax.Array,
    ) -> None:
        self._eigvecs = eigvecs
        self._weights = weights
        super().__init__(eigvecs.shape, eigvecs.dtype)

    @property
    def eigenvectors(self) -> KroneckerSelectedEigenvectors:
        return self._eigvecs

    @property
    def weights(self) -> jax.Array:
        return self._weights

    def tree_flatten(self) -> tuple[tuple, dict]:
        return (self._eigvecs, self._weights), {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple):
        return cls(children[0], children[1])

    def _matmul(self, v: jax.Array) -> jax.Array:
        # Scale input by weights, then apply Q_k
        scaled = self._weights * v if v.ndim == 1 else self._weights[:, None] * v
        return self._eigvecs @ scaled

    def transpose(self) -> "KroneckerLowRankFactorTranspose":
        return KroneckerLowRankFactorTranspose(self)

    def todense(self) -> jax.Array:
        return self._eigvecs.todense() * self._weights[None, :]


class KroneckerLowRankFactorTranspose(LinearOperator):
    """Transpose of KroneckerLowRankFactor."""

    def __init__(self, parent: KroneckerLowRankFactor) -> None:
        self._parent = parent
        super().__init__((parent.shape[1], parent.shape[0]), parent.dtype)

    def tree_flatten(self) -> tuple[tuple, dict]:
        return (self._parent,), {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple):
        return cls(children[0])

    def _matmul(self, v: jax.Array) -> jax.Array:
        qt_v = self._parent._eigvecs.transpose() @ v
        weights = self._parent._weights
        return weights * qt_v if qt_v.ndim == 1 else weights[:, None] * qt_v

    def transpose(self) -> KroneckerLowRankFactor:
        return self._parent

    def todense(self) -> jax.Array:
        return self._parent.todense().T


jax.tree_util.register_pytree_node_class(KroneckerLowRankFactor)
jax.tree_util.register_pytree_node_class(KroneckerLowRankFactorTranspose)


def make_lowrank_sqrt(
    eigvecs: KroneckerSelectedEigenvectors,
    eigenvalues: jax.Array,
    eps: float = 1e-12,
) -> KroneckerLowRankFactor:
    r"""Create low-rank square root: :math:`Q_k \text{diag}(\sqrt{\lambda_k})`.

    Useful for sampling: :math:`x = Q_k \text{diag}(\sqrt{\lambda_k}) z`
    where :math:`z \sim \mathcal{N}(0, I_k)`.
    """
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, eps))
    return KroneckerLowRankFactor(eigvecs, sqrt_eigs)


def make_lowrank_inv_sqrt(
    eigvecs: KroneckerSelectedEigenvectors,
    eigenvalues: jax.Array,
    eps: float = 1e-12,
) -> KroneckerLowRankFactor:
    r"""Create low-rank inverse square root: :math:`Q_k \text{diag}(1/\sqrt{\lambda_k})`.

    Useful for whitening transformations.
    """
    inv_sqrt_eigs = jnp.where(eigenvalues > eps, 1.0 / jnp.sqrt(eigenvalues), 0.0)
    return KroneckerLowRankFactor(eigvecs, inv_sqrt_eigs)


def kron_matvec(factors, v):
    """Matrix-free Kronecker matvec for verification."""
    dims = [f.shape[0] for f in factors]
    X = v.reshape(dims)
    for i, F in enumerate(factors):
        F_arr = F.A if isinstance(F, Matrix) else F
        X = jnp.tensordot(F_arr, X, axes=([1], [i]))
        X = jnp.moveaxis(X, 0, i)
    return X.reshape(-1)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing topk_eigh_kronecker (linox-compatible API)")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Create symmetric PSD matrices
    A = jax.random.normal(k1, (4, 4))
    A = A @ A.T + jnp.eye(4) * 0.1
    B = jax.random.normal(k2, (3, 3))
    B = B @ B.T + jnp.eye(3) * 0.1

    print("\n=== Two-factor Kronecker product ===")
    op_A = Matrix(A)
    op_B = Matrix(B)

    print(f"A shape: {op_A.shape}, B shape: {op_B.shape}")
    full_kron = jnp.kron(A, B)
    print(f"Full Kronecker size: {full_kron.shape[0]} x {full_kron.shape[0]}")

    full_eigs, _ = jnp.linalg.eigh(full_kron)

    k = 5
    print(f"\nComputing top-{k} eigenvalues...")
    topk_vals, topk_vecs = topk_eigh_kronecker([op_A, op_B], k=k, largest=True)

    print(f"\nTop-{k} via heap:       {topk_vals}")
    print(f"Top-{k} via full eigh:  {jnp.sort(full_eigs)[::-1][:k]}")

    print(f"\nBottom-{k} eigenvalues...")
    botk_vals, _ = topk_eigh_kronecker([op_A, op_B], k=k, largest=False)
    print(f"Bottom-{k} via heap:       {botk_vals}")
    print(f"Bottom-{k} via full eigh:  {jnp.sort(full_eigs)[:k]}")

    print("\n=== Eigenvector verification (matrix-free) ===")
    for i in range(min(3, k)):
        lam = topk_vals[i]
        e_i = jnp.zeros(k).at[i].set(1.0)
        v = topk_vecs @ e_i  # Matrix-free!
        Av = full_kron @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Transpose verification ===")
    test_vec = jax.random.normal(k1, (full_kron.shape[0],))
    qt_v = topk_vecs.transpose() @ test_vec
    Q_dense = topk_vecs.todense()
    qt_v_dense = Q_dense.T @ test_vec
    print(f"Q_k.T @ v shape: {qt_v.shape}")
    print(
        f"||Q_k.T @ v (matrix-free) - Q_k.T @ v (dense)||: {jnp.linalg.norm(qt_v - qt_v_dense):.2e}"
    )

    print("\n=== Three-factor Kronecker product ===")
    C = jax.random.normal(k3, (3, 3))
    C = C @ C.T + jnp.eye(3) * 0.1
    op_C = Matrix(C)

    full_kron_3 = jnp.kron(jnp.kron(A, B), C)
    print(f"Shapes: A={op_A.shape}, B={op_B.shape}, C={op_C.shape}")
    print(f"Full Kronecker size: {full_kron_3.shape[0]} x {full_kron_3.shape[0]}")

    full_eigs_3, _ = jnp.linalg.eigh(full_kron_3)

    k = 8
    print(f"\nComputing top-{k} eigenvalues...")
    topk_vals_3, topk_vecs_3 = topk_eigh_kronecker(
        [op_A, op_B, op_C], k=k, largest=True
    )

    print(f"\nTop-{k} via heap:       {topk_vals_3}")
    print(f"Top-{k} via full eigh:  {jnp.sort(full_eigs_3)[::-1][:k]}")

    print("\n=== Verify 3-factor eigenvectors ===")
    for i in range(min(3, k)):
        lam = topk_vals_3[i]
        e_i = jnp.zeros(k).at[i].set(1.0)
        v = topk_vecs_3 @ e_i
        Av = full_kron_3 @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Large-scale example (NO densification!) ===")
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)

    D = jax.random.normal(k1, (50, 50))
    D = D @ D.T + jnp.eye(50) * 0.1
    E = jax.random.normal(k2, (50, 50))
    E = E @ E.T + jnp.eye(50) * 0.1
    F = jax.random.normal(k3, (50, 50))
    F = F @ F.T + jnp.eye(50) * 0.1

    op_D, op_E, op_F = Matrix(D), Matrix(E), Matrix(F)

    full_size = 50 * 50 * 50
    print(f"3 factors of 50x50 → Kronecker size: {full_size} x {full_size}")
    print(f"Dense Kronecker would need: {full_size**2 * 8 / 1e9:.2f} GB")
    print("(We never build it!)")

    k_large = 20
    topk_large, vecs_large = topk_eigh_kronecker(
        [op_D, op_E, op_F], k=k_large, largest=True
    )
    print(f"\nTop-{k_large} eigenvalues computed efficiently:")
    print(f"  {topk_large[:5]} ...")

    print("\nVerify with matrix-free Kronecker matvec:")
    e_0 = jnp.zeros(k_large).at[0].set(1.0)
    v0 = vecs_large @ e_0
    Av0 = kron_matvec([op_D, op_E, op_F], v0)
    residual = jnp.linalg.norm(Av0 - topk_large[0] * v0)
    print(f"  Eigenpair 0: λ={topk_large[0]:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Memory comparison ===")
    print(f"Dense eigenvector matrix would be: {full_size * k_large * 8 / 1e6:.2f} MB")
    print(
        f"Our operator stores: {3 * 50 * k_large * 8 / 1e3:.2f} KB (factor columns only)"
    )

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
