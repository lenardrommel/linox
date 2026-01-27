# _kronecker.py

r"""Kronecker product operations for linear operators.

This module includes:

- :class:`Kronecker`: Represents the Kronecker product :math:`A \otimes B` of two
    linear operators :math:`A` and :math:`B`
- :class:`KroneckerSelectedEigenvectors`: Matrix-free operator for selected
    eigenvectors of a Kronecker product
- :func:`topk_eigh`: Compute top-k or bottom-k eigenvalues/vectors of a Kronecker
    product without forming the full matrix
"""

import heapq
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from linox import utils
from linox._arithmetic import (
    ProductLinearOperator,
    diagonal,
    lcholesky,
    ldet,
    leigh,
    linverse,
    lpinverse,
    lqr,
    lsolve,
    lsqrt,
    psolve,
    slogdet,
    svd,
)
from linox._linear_operator import LinearOperator
from linox._registry import get, register


class Kronecker(LinearOperator):
    """A Kronecker product of two linear operators.

    Example usage:

    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    B = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)
    op = Kronecker(A, B)
    vec = jnp.ones((4,))
    result = op @ vec
    result_true = jnp.kron(A, B) @ vec
    jnp.allclose(result, result_true)
    """

    def __init__(
        self, A: LinearOperator | jax.Array, B: LinearOperator | jax.Array
    ) -> None:
        self._A = utils.as_linop(A)
        self._B = utils.as_linop(B)
        A_shape = self._A.shape if len(self._A.shape) == 2 else (self._A.shape[0], 1)
        B_shape = self._B.shape if len(self._B.shape) == 2 else (self._B.shape[0], 1)

        self._shape = (
            A_shape[0] * B_shape[0],
            A_shape[1] * B_shape[1],
        )

        dtype = jnp.result_type(self._A.dtype, self._B.dtype)
        super().__init__(self._shape, dtype)

    @property
    def A(self) -> LinearOperator:
        return self._A

    @property
    def B(self) -> LinearOperator:
        return self._B

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def tree_flatten(self) -> tuple[tuple, dict]:
        children = (self.A, self.B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict,
        children: tuple,
    ) -> "Kronecker":
        return cls(*children)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        if len(vec.shape) == 1:
            vec = vec[:, None]

        _, mA = self.A.shape
        _, mB = self.B.shape

        y = jnp.swapaxes(vec, -2, -1)
        y = y.reshape((*y.shape[:-1], mA, mB))
        y = self.B @ jnp.swapaxes(y, -1, -2)
        y = self.A @ jnp.swapaxes(y, -1, -2)
        y = y.reshape((*y.shape[:-2], -1))
        y = jnp.swapaxes(y, -1, -2)

        return y

    def _todense(self) -> jax.Array:
        return jnp.kron(self.A._todense(), self.B._todense())

    def transpose(self) -> "Kronecker":
        return Kronecker(self.A.transpose(), self.B.transpose())

    def trace(self) -> jax.Array:
        return jnp.trace(self.A) * jnp.trace(self.B)


@linverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(linverse(op.A), linverse(op.B))


def _solve_left(A, M):
    """
    Solve A X = M for X, where A is (n,n) linop and
    M is (..., n, k). Returns (..., n, k).
    """
    n = A.shape[0]
    k = M.shape[-1]
    batch = M.shape[:-2]
    M2 = M.reshape((-1, n, k))  # (B, n, k)
    X2 = jax.vmap(lambda rhs: lsolve(A, rhs))(M2)
    return X2.reshape(batch + (n, k))


@lsolve.dispatch
def _(op: Kronecker, b: jax.Array) -> jax.Array:
    squeeze_vec = False
    if b.ndim == 1:
        b = b[:, None]
        squeeze_vec = True

    mA, nA = op.A.shape
    mB, nB = op.B.shape
    if mA != nA or mB != nB:
        msg = f"Square factors required, got {op.A.shape}, {op.B.shape}"
        raise ValueError(msg)
    if b.shape[-2] != mA * mB:
        msg = f"Shape mismatch: op.shape={op.shape}, b.shape={b.shape}"
        raise ValueError(msg)

    # reshape b into (..., mA, mB, r) in a way consistent with your matmul
    r = b.shape[-1]
    y = jnp.swapaxes(b, -2, -1)  # (..., r, m)
    y = y.reshape((*y.shape[:-1], mA, mB))  # (..., r, mA, mB)

    # bring r to the end: (..., mA, mB, r)
    y = jnp.swapaxes(y, -3, -1)  # (..., mB, mA, r)
    y = jnp.swapaxes(y, -3, -2)  # (..., mA, mB, r)

    # Undo A-step (A acted on the mA dimension in your matmul pipeline)
    # We need solve A X = Y for each (mB, r) slice
    Y = y.reshape(y.shape[:-3] + (mA, mB * r))  # (..., mA, mB*r)
    X = _solve_left(op.A, Y)  # (..., mA, mB*r)
    X = X.reshape(y.shape[:-3] + (mA, mB, r))  # (..., mA, mB, r)

    # Undo B-step: solve B Z = (X swapped appropriately)
    Xs = jnp.swapaxes(X, -3, -2)  # (..., mB, mA, r)
    Y2 = Xs.reshape(Xs.shape[:-3] + (mB, mA * r))  # (..., mB, mA*r)
    Z2 = _solve_left(op.B, Y2)  # (..., mB, mA*r)
    Z = Z2.reshape(Xs.shape[:-3] + (mB, mA, r))  # (..., mB, mA, r)

    # swap back and vectorize to (mA*mB, r)
    Z = jnp.swapaxes(Z, -3, -2)  # (..., mA, mB, r)
    out = Z.reshape(Z.shape[:-3] + (mA * mB, r))

    if squeeze_vec:
        out = out[:, 0]
    return out


@lpinverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(lpinverse(op.A), lpinverse(op.B))


def _psolve_left(A, M):
    """
    Solve A X = M for X, where A is (n,n) linop and
    M is (..., n, k). Returns (..., n, k).
    """
    n = A.shape[0]
    k = M.shape[-1]
    batch = M.shape[:-2]
    M2 = M.reshape((-1, n, k))  # (B, n, k)
    X2 = jax.vmap(lambda rhs: psolve(A, rhs))(M2)
    return X2.reshape(batch + (n, k))


@psolve.dispatch
def _(op: Kronecker, b: jax.Array) -> jax.Array:
    squeeze_vec = False
    if b.ndim == 1:
        b = b[:, None]
        squeeze_vec = True

    mA, nA = op.A.shape
    mB, nB = op.B.shape
    if mA != nA or mB != nB:
        msg = f"Square factors required, got {op.A.shape}, {op.B.shape}"
        raise ValueError(msg)
    if b.shape[-2] != mA * mB:
        msg = f"Shape mismatch: op.shape={op.shape}, b.shape={b.shape}"
        raise ValueError(msg)

    # reshape b into (..., mA, mB, r) in a way consistent with your matmul
    r = b.shape[-1]
    y = jnp.swapaxes(b, -2, -1)  # (..., r, m)
    y = y.reshape((*y.shape[:-1], mA, mB))  # (..., r, mA, mB)

    # bring r to the end: (..., mA, mB, r)
    y = jnp.swapaxes(y, -3, -1)  # (..., mB, mA, r)
    y = jnp.swapaxes(y, -3, -2)  # (..., mA, mB, r)

    # Undo A-step (A acted on the mA dimension in your matmul pipeline)
    # We need solve A X = Y for each (mB, r) slice
    Y = y.reshape(y.shape[:-3] + (mA, mB * r))  # (..., mA, mB*r)
    X = _psolve_left(op.A, Y)  # (..., mA, mB*r)
    X = X.reshape(y.shape[:-3] + (mA, mB, r))  # (..., mA, mB, r)

    # Undo B-step: solve B Z = (X swapped appropriately)
    Xs = jnp.swapaxes(X, -3, -2)  # (..., mB, mA, r)
    Y2 = Xs.reshape(Xs.shape[:-3] + (mB, mA * r))  # (..., mB, mA*r)
    Z2 = _psolve_left(op.B, Y2)  # (..., mB, mA*r)
    Z = Z2.reshape(Xs.shape[:-3] + (mB, mA, r))  # (..., mB, mA, r)

    # swap back and vectorize to (mA*mB, r)
    Z = jnp.swapaxes(Z, -3, -2)  # (..., mA, mB, r)
    out = Z.reshape(Z.shape[:-3] + (mA * mB, r))

    if squeeze_vec:
        out = out[:, 0]
    return out


@lsqrt.dispatch
def _(op: Kronecker) -> Kronecker:
    r"""Square root of a Kronecker product.

    For a Kronecker product :math:`A \otimes B`, this represents
    :math:`\sqrt{A \otimes B} = \sqrt{A} \otimes \sqrt{B}`
    """
    return Kronecker(lsqrt(op.A), lsqrt(op.B))


@leigh.dispatch
def _(op: Kronecker) -> tuple[Kronecker, Kronecker]:
    r"""Eigendecomposition of a Kronecker product.

    For :math:`A \otimes B` with :math:`A = Q_A \Lambda_A Q_A^T` and
    :math:`B = Q_B \Lambda_B Q_B^T`, returns:

    - Eigenvalues as :math:`\Lambda_A \otimes \Lambda_B` (Kronecker of Diagonals)
    - Eigenvectors as :math:`Q_A \otimes Q_B` (Kronecker of orthogonal matrices)

    Both are LinearOperators, avoiding dense eigenvalue arrays for large products.
    Handles nested Kronecker structures by checking if eigenvalues are already
    LinearOperators.
    """
    from linox._matrix import Diagonal

    wA, QA = leigh(op.A)
    wB, QB = leigh(op.B)

    # Handle nested Kronecker: eigenvalues may already be LinearOperators
    if isinstance(wA, LinearOperator):
        LamA = wA
    else:
        LamA = Diagonal(wA)

    if isinstance(wB, LinearOperator):
        LamB = wB
    else:
        LamB = Diagonal(wB)

    Lambda = Kronecker(LamA, LamB)
    Q = Kronecker(QA, QB)

    return Lambda, Q


@lqr.dispatch
def _(op: Kronecker) -> tuple[Kronecker, Kronecker]:
    """QR decomposition of a kronecker product.

    Returns:
        Q(Q_A, Q_B): Orthogonal matrix
        R(R_A, R_B): Upper triangular matrix.
    """
    Q_A, R_A = lqr(op.A)
    Q_B, R_B = lqr(op.B)
    return Kronecker(Q_A, Q_B), Kronecker(R_A, R_B)


@svd.dispatch
def _(op: Kronecker) -> tuple[Kronecker, jax.Array, Kronecker]:
    """SVD decomposition of a kronecker product.

    Returns:
        U(U_A, U_B): Left singular vectors
        S: Singular values
        Vh(Vh_A, Vh_B): Right singular vectors (Hermitian transposed).
    """
    U_A, S_A, Vh_A = svd(op.A)
    U_B, S_B, Vh_B = svd(op.B)

    return (
        Kronecker(U_A, U_B),
        jnp.outer(S_A, S_B).flatten(),
        Kronecker(Vh_A, Vh_B),
    )


@lcholesky.dispatch
def _(op: Kronecker) -> Kronecker:
    L_A = lcholesky(op.A)
    L_B = lcholesky(op.B)
    return Kronecker(L_A, L_B)


@ldet.dispatch
def _(op: Kronecker) -> ProductLinearOperator:
    return ProductLinearOperator(
        [
            ldet(op.A) ** op.B.shape[0],
            ldet(op.B) ** op.A.shape[0],
        ]
    )


@slogdet.dispatch
def _(op: Kronecker) -> tuple[jax.Array, jax.Array]:
    sign_A, logdet_A = slogdet(op.A)
    sign_B, logdet_B = slogdet(op.B)

    dim_A = op.A.shape[0]
    dim_B = op.B.shape[0]

    final_sign = sign_A**dim_B * sign_B**dim_A
    final_logdet = dim_B * logdet_A + dim_A * logdet_B

    return final_sign, final_logdet


@diagonal.dispatch
def _(op: Kronecker) -> jax.Array:
    diag_A = jnp.asarray(diagonal(op.A))
    diag_B = jnp.asarray(diagonal(op.B))
    batch_shape = jnp.broadcast_shapes(diag_A.shape[:-1], diag_B.shape[:-1])
    diag_A = jnp.broadcast_to(diag_A, batch_shape + (diag_A.shape[-1],))
    diag_B = jnp.broadcast_to(diag_B, batch_shape + (diag_B.shape[-1],))
    diag = jnp.einsum("...i,...j->...ij", diag_A, diag_B)
    return diag.reshape(batch_shape + (diag_A.shape[-1] * diag_B.shape[-1],))


jax.tree_util.register_pytree_node_class(Kronecker)


def _factor_pair(total: int) -> tuple[int, int]:
    for k in range(2, int(total**0.5) + 1):
        if total % k == 0:
            return k, total // k
    return total, 1


@register("kronecker", tags=("rectangular",))
def make_kronecker(
    key: jax.random.PRNGKey,
    shape: tuple[int, int],
    dtype: jnp.dtype = jnp.float32,
    require: str | None = None,
    *,
    maker_A: str = "matrix",
    maker_B: str = "matrix",
) -> Kronecker:
    m, n = shape
    mA, mB = _factor_pair(m)
    nA, nB = _factor_pair(n)

    keyA, keyB = jax.random.split(key)
    A = get(maker_A).maker(keyA, (mA, nA), dtype, require=require)
    B = get(maker_B).maker(keyB, (mB, nB), dtype, require=require)

    return Kronecker(A, B)


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

        # TRIAL
        sel_np = np.asarray(selected_indices, dtype=np.int32)  # (k, d)

        # self._gathered = []
        # for i in range(self._d):
        #     idx_for_factor = jnp.array(
        #         [self._sort_indices[i][sel[i]] for sel in selected_indices]
        #     )
        #     self._gathered.append(self._factor_vecs[i][:, idx_for_factor])

        # dtype = factor_vecs[0].dtype
        # super().__init__((self._n_total, self._k), dtype)
        self._gathered = []
        for i in range(self._d):
            # idx_sorted: (k,)
            idx_sorted = jnp.asarray(sel_np[:, i])

            # idx_orig = order[idx_sorted] in one shot (no Python loop!)
            idx_orig = jnp.take(self._sort_indices[i], idx_sorted, mode="clip")

            # gather columns
            self._gathered.append(self._factor_vecs[i][:, idx_orig])

        dtype = factor_vecs[0].dtype
        super().__init__((self._n_total, self._k), dtype)

    @property
    def k(self) -> int:
        return self._k

    @property
    def num_factors(self) -> int:
        return self._d

    @property
    def factor_dims(self) -> list[int]:
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
            UA, UB = self._gathered[0], self._gathered[1]
            nA, nB = self._factor_dims[0], self._factor_dims[1]
            Y = jnp.einsum("il,lb,jl->ijb", UA, alpha, UB)
            result = Y.reshape((nA * nB, -1))

            if squeeze:
                result = result.squeeze(-1)
            return result

        result = jnp.zeros((self._n_total, alpha.shape[1]), dtype=self.dtype)

        for l in range(self._k):
            vec_l = self._gathered[0][:, l]
            for i in range(1, self._d):
                vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
            result += vec_l[:, None] * alpha[l, :]

        if squeeze:
            result = result.squeeze(-1)
        return result

    def _rmatmul(self, v: jax.Array) -> jax.Array:
        r"""Compute Q_k^T v."""
        n = self._n_total
        restore = None

        if v.ndim == 1:
            pass

        elif v.ndim == 2:
            if v.shape[0] == n and v.shape[1] != n:
                v = jnp.swapaxes(v, 0, 1)
                restore = ("cols",)

            elif v.shape[1] == n:
                restore = ("batch",)

            elif v.shape[1] == 1 and v.shape[0] == n:
                v = v[:, 0]
            else:
                msg = (
                    f"Unsupported v shape {v.shape}. Expected (n,), (batch,n), or (n,p)"
                )
                raise ValueError(msg)
        else:
            msg = f"Unsupported v.ndim={v.ndim}. Expected 1 or 2."
            raise ValueError(msg)

        squeeze_single = False
        if v.ndim == 1:
            v = v[None, :]
            squeeze_single = True

        if self._d == 2:
            UA, UB = self._gathered[0], self._gathered[1]
            nA, nB = self._factor_dims[0], self._factor_dims[1]

            X = v.reshape((v.shape[0], nA, nB))
            T = jnp.einsum("il,bij->blj", UA, X)
            result = jnp.einsum("blj,jl->bl", T, UB)

        else:
            result = jnp.zeros((v.shape[0], self._k), dtype=self.dtype)
            for l in range(self._k):
                vec_l = self._gathered[0][:, l]
                for i in range(1, self._d):
                    vec_l = jnp.kron(vec_l, self._gathered[i][:, l])
                result = result.at[:, l].set(v @ vec_l)

        if squeeze_single:
            result = result[0, :]

        if restore == ("cols",) and result.ndim == 2:
            result = jnp.swapaxes(result, 0, 1)

        return result

    def transpose(self) -> "KroneckerSelectedEigenvectorsTranspose":
        return KroneckerSelectedEigenvectorsTranspose(self)

    def _todense(self) -> jax.Array:
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
        return self._parent._rmatmul(v)  # noqa: SLF001

    def transpose(self) -> KroneckerSelectedEigenvectors:
        return self._parent

    def _todense(self) -> jax.Array:
        return self._parent._todense().T


jax.tree_util.register_pytree_node_class(KroneckerSelectedEigenvectors)
jax.tree_util.register_pytree_node_class(KroneckerSelectedEigenvectorsTranspose)


def extract_kronecker_factors(
    op: LinearOperator,
) -> tuple[list[LinearOperator], jax.Array | None]:
    r"""Extract leaf factors from a (possibly nested) Kronecker structure.

    Handles complex nested structures like:
    - ``Kronecker(A, Kronecker(B, C))`` → ``[A, B, C]``
    - ``ScaledLinearOperator(Kronecker(...), scalar)`` → ``([factors], scalar)``

    Args:
        op: A LinearOperator that may be a Kronecker product, possibly nested
            or wrapped in a ScaledLinearOperator.

    Returns:
        factors: List of leaf LinearOperators (non-Kronecker factors).
        scalar: The scalar multiplier if op was wrapped in ScaledLinearOperator,
            otherwise None.

    Example:
        >>> A = Matrix(jnp.eye(3))
        >>> B = Matrix(jnp.eye(4))
        >>> C = Matrix(jnp.eye(5))
        >>> kron = Kronecker(A, Kronecker(B, C))
        >>> factors, scalar = extract_kronecker_factors(kron)
        >>> len(factors)  # 3
    """
    from linox._arithmetic import ScaledLinearOperator

    scalar = None

    # Unwrap ScaledLinearOperator if present
    if isinstance(op, ScaledLinearOperator):
        scalar = op.scalar
        op = op.operator

    def _collect_factors(node: LinearOperator) -> list[LinearOperator]:
        """Recursively collect leaf factors from Kronecker tree."""
        if isinstance(node, Kronecker):
            return _collect_factors(node.A) + _collect_factors(node.B)
        return [node]

    factors = _collect_factors(op)
    return factors, scalar


def topk_eigh(
    op_or_factors: LinearOperator | Sequence[LinearOperator | jax.Array],
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
        op_or_factors: Either a single LinearOperator (which may be a nested
            Kronecker structure, optionally wrapped in ScaledLinearOperator),
            or a sequence of symmetric PSD LinearOperators/arrays as factors.
        k: Number of eigenvalues to return.
        largest: If True, return largest k eigenvalues; else smallest k.

    Returns:
        eigenvalues: Array of shape (k,) with the k largest/smallest eigenvalues.
        eigenvectors: LinearOperator of shape (n_total, k) representing the
            k eigenvectors without forming the full Kronecker product.

    Example:
        >>> from linox import Matrix, topk_eigh, Kronecker
        >>> A = Matrix(jnp.eye(3) * 2)
        >>> B = Matrix(jnp.eye(4) * 3)
        >>> # Pass factors directly
        >>> eigs, Q = topk_eigh([A, B], k=5, largest=True)
        >>> # Or pass a Kronecker operator
        >>> kron = Kronecker(A, B)
        >>> eigs2, Q2 = topk_eigh(kron, k=5, largest=True)
    """
    # Handle single LinearOperator input (possibly nested Kronecker)
    scalar = None
    if isinstance(op_or_factors, LinearOperator):
        factors, scalar = extract_kronecker_factors(op_or_factors)
    else:
        factors = [utils.as_linop(f) for f in op_or_factors]
    d = len(factors)

    factor_eigs = []
    factor_vecs = []
    sort_indices = []

    for A in factors:
        w, Q = leigh(A)
        if isinstance(w, LinearOperator):
            w = diagonal(w)
        order = jnp.argsort(-w) if largest else jnp.argsort(w)
        factor_eigs.append(w[order])
        sort_indices.append(order)
        factor_vecs.append(Q._todense() if hasattr(Q, "_todense") else jnp.asarray(Q))

    sizes = [len(w) for w in factor_eigs]

    def compute_eigenvalue(indices: tuple[int, ...]) -> float:
        prod = 1.0
        for i, idx in enumerate(indices):
            prod *= factor_eigs[i][idx]
        return float(prod)

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

    eigenvectors = KroneckerSelectedEigenvectors(
        factor_vecs, selected_indices, sort_indices
    )

    eig_array = jnp.array(eigenvalues)

    # If input was ScaledLinearOperator, multiply eigenvalues by scalar
    if scalar is not None:
        eig_array = scalar * eig_array

    return eig_array, eigenvectors
