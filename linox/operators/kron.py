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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from linox import utils
from linox.operators.arithmetic import (
    AddLinearOperator,
    diagonal,
    lcholesky,
    ldet,
    leigh,
    lexp,
    linverse,
    llog,
    lpinverse,
    lpow,
    lqr,
    lsolve,
    lsqrt,
    ltrace,
    psolve,
    slogdet,
    svd,
)
from linox.operators.base import LinearOperator
from linox.operators.special import Identity


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

    def __init__(self, A: LinearOperator | jax.Array, B: LinearOperator | jax.Array) -> None:
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
        """First factor of the Kronecker product."""
        return self._A

    @property
    def B(self) -> LinearOperator:
        """Second factor of the Kronecker product."""
        return self._B

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Kronecker product."""
        return self._shape

    @property
    def is_symmetric(self) -> bool:
        """Check if Kronecker product is symmetric."""
        return self.A.is_symmetric and self.B.is_symmetric

    @property
    def is_psd(self) -> bool:
        """Check if Kronecker product is positive semi-definite."""
        return self.A.is_psd and self.B.is_psd

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten for JAX pytree registration."""
        children = (self.A, self.B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict,
        children: tuple,
    ) -> "Kronecker":
        """Unflatten for JAX pytree registration."""
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
        """Return transposed Kronecker product."""
        return Kronecker(self.A.transpose(), self.B.transpose())

    def trace(self) -> jax.Array:
        """Compute trace of Kronecker product: tr(A (x) B) = tr(A) tr(B)."""
        # `self.A` / `self.B` are LinearOperators, so take their diagonals
        # through the dispatch rather than calling `jnp.trace` on them.
        return jnp.sum(jnp.asarray(diagonal(self.A)), axis=-1) * jnp.sum(jnp.asarray(diagonal(self.B)), axis=-1)


@linverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(linverse(op.A), linverse(op.B))


def _solve_left(A, M):
    """Solve A X = M for X.

    Where A is (n,n) linop and M is (..., n, k). Returns (..., n, k).
    """
    n = A.shape[0]
    k = M.shape[-1]
    batch = M.shape[:-2]
    M2 = M.reshape((-1, n, k))  # (B, n, k)
    X2 = jax.vmap(lambda rhs: lsolve(A, rhs))(M2)
    return X2.reshape((*batch, n, k))


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
    Y = y.reshape((*y.shape[:-3], mA, mB * r))  # (..., mA, mB*r)
    X = _solve_left(op.A, Y)  # (..., mA, mB*r)
    X = X.reshape((*y.shape[:-3], mA, mB, r))  # (..., mA, mB, r)

    # Undo B-step: solve B Z = (X swapped appropriately)
    Xs = jnp.swapaxes(X, -3, -2)  # (..., mB, mA, r)
    Y2 = Xs.reshape((*Xs.shape[:-3], mB, mA * r))  # (..., mB, mA*r)
    Z2 = _solve_left(op.B, Y2)  # (..., mB, mA*r)
    Z = Z2.reshape((*Xs.shape[:-3], mB, mA, r))  # (..., mB, mA, r)

    # swap back and vectorize to (mA*mB, r)
    Z = jnp.swapaxes(Z, -3, -2)  # (..., mA, mB, r)
    out = Z.reshape((*Z.shape[:-3], mA * mB, r))

    if squeeze_vec:
        out = out[:, 0]
    return out


@lpinverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(lpinverse(op.A), lpinverse(op.B))


def _psolve_left(A, M):
    """Solve A X = M for X.

    Where A is (n,n) linop and M is (..., n, k). Returns (..., n, k).
    """
    n = A.shape[0]
    k = M.shape[-1]
    batch = M.shape[:-2]
    M2 = M.reshape((-1, n, k))  # (B, n, k)
    X2 = jax.vmap(lambda rhs: psolve(A, rhs))(M2)
    return X2.reshape((*batch, n, k))


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
    Y = y.reshape((*y.shape[:-3], mA, mB * r))  # (..., mA, mB*r)
    X = _psolve_left(op.A, Y)  # (..., mA, mB*r)
    X = X.reshape((*y.shape[:-3], mA, mB, r))  # (..., mA, mB, r)

    # Undo B-step: solve B Z = (X swapped appropriately)
    Xs = jnp.swapaxes(X, -3, -2)  # (..., mB, mA, r)
    Y2 = Xs.reshape((*Xs.shape[:-3], mB, mA * r))  # (..., mB, mA*r)
    Z2 = _psolve_left(op.B, Y2)  # (..., mB, mA*r)
    Z = Z2.reshape((*Xs.shape[:-3], mB, mA, r))  # (..., mB, mA, r)

    # swap back and vectorize to (mA*mB, r)
    Z = jnp.swapaxes(Z, -3, -2)  # (..., mA, mB, r)
    out = Z.reshape((*Z.shape[:-3], mA * mB, r))

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
    from linox.operators.diagonal import Diagonal

    wA, QA = leigh(op.A)
    wB, QB = leigh(op.B)

    # Handle nested Kronecker: eigenvalues may already be LinearOperators
    LamA = wA if isinstance(wA, LinearOperator) else Diagonal(wA)

    LamB = wB if isinstance(wB, LinearOperator) else Diagonal(wB)

    Lambda = Kronecker(LamA, LamB)
    Q = Kronecker(QA, QB)

    return Lambda, Q


@lqr.dispatch
def _(op: Kronecker) -> tuple[Kronecker, Kronecker]:
    """QR decomposition of a kronecker product.

    Returns
    -------
        Q(Q_A, Q_B): Orthogonal matrix
        R(R_A, R_B): Upper triangular matrix.
    """
    Q_A, R_A = lqr(op.A)
    Q_B, R_B = lqr(op.B)
    return Kronecker(Q_A, Q_B), Kronecker(R_A, R_B)


@svd.dispatch
def _(op: Kronecker, **kwargs) -> tuple[Kronecker, jax.Array, Kronecker]:
    """SVD decomposition of a kronecker product.

    Exploits the structure: SVD(A ⊗ B) = (U_A ⊗ U_B) (S_A ⊗ S_B) (V_A^H ⊗ V_B^H)

    Returns
    -------
        U(U_A, U_B): Left singular vectors as Kronecker product
        S: Singular values (outer product of S_A and S_B, flattened)
        Vh(Vh_A, Vh_B): Right singular vectors (Hermitian) as Kronecker product

    Notes
    -----
        Passes through all kwargs (k, num_iters, u0, etc.) to constituent SVDs.
    """
    U_A, S_A, Vh_A = svd(op.A, **kwargs)
    U_B, S_B, Vh_B = svd(op.B, **kwargs)

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
def _(op: Kronecker) -> jax.Array:
    # det(A (x) B) = det(A)^nB * det(B)^nA -- a product of two *scalars*, not
    # a composition of operators.
    return ldet(op.A) ** op.B.shape[-1] * ldet(op.B) ** op.A.shape[-1]


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
    diag_A = jnp.broadcast_to(diag_A, (*batch_shape, diag_A.shape[-1]))
    diag_B = jnp.broadcast_to(diag_B, (*batch_shape, diag_B.shape[-1]))
    diag = jnp.einsum("...i,...j->...ij", diag_A, diag_B)
    return diag.reshape((*batch_shape, diag_A.shape[-1] * diag_B.shape[-1]))


# New matrix-free function dispatches for Kronecker
@ltrace.dispatch
def _(
    op: Kronecker,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Trace of Kronecker product: trace(A ⊗ B) = trace(A) * trace(B)."""
    from linox.operators.arithmetic import ltrace

    trace_A, std_A = ltrace(op.A, key=key, num_samples=num_samples, distribution=distribution)
    trace_B, std_B = ltrace(op.B, key=key, num_samples=num_samples, distribution=distribution)

    # trace(A ⊗ B) = trace(A) * trace(B)
    trace_value = trace_A * trace_B

    # Error propagation for product: σ(xy) ≈ |y|σ(x) + |x|σ(y)
    trace_std = jnp.abs(trace_B) * std_A + jnp.abs(trace_A) * std_B

    return trace_value, trace_std


@lexp.dispatch
def _(
    op: Kronecker,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix exponential of Kronecker: exp(A ⊗ B) = exp(A) ⊗ exp(B)."""
    if v is None:
        # Return lazy operator: exp(A) ⊗ exp(B)
        exp_A = lexp(op.A, v=None, num_iters=num_iters, method=method)
        exp_B = lexp(op.B, v=None, num_iters=num_iters, method=method)
        return Kronecker(exp_A, exp_B)
    # For Kronecker product, we can use the vec-trick
    # But for simplicity, fall back to general algorithm
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    return lanczos_matrix_function(op, v, jnp.exp, num_iters, reortho=True)


@llog.dispatch
def _(
    op: Kronecker,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix logarithm of Kronecker product.

    Note: log(A ⊗ B) ≠ log(A) ⊗ log(B) in general.
    Falls back to general Lanczos method.
    """
    if v is None:
        # Fall back to general algorithm
        from linox.config import warn as _warn

        _warn("Computing log(A ⊗ B) using dense method - no efficient structured formula available")
        eigvals, eigvecs = jnp.linalg.eigh(op.todense())
        return utils.as_linop(eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T)
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    return lanczos_matrix_function(op, v, jnp.log, num_iters, reortho=True)


@lpow.dispatch
def _(
    op: Kronecker,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix power of Kronecker: (A ⊗ B)^p = A^p ⊗ B^p."""
    if v is None:
        # Return lazy operator: A^p ⊗ B^p
        pow_A = lpow(op.A, power=power, v=None, num_iters=num_iters, method=method)
        pow_B = lpow(op.B, power=power, v=None, num_iters=num_iters, method=method)
        return Kronecker(pow_A, pow_B)
    # Can use structure, but for simplicity use general algorithm
    from linox.linalg.approx.lanczos import lanczos_matrix_function

    def power_func(eigvals):
        return eigvals**power

    return lanczos_matrix_function(op, v, power_func, num_iters, reortho=True)


jax.tree_util.register_pytree_node_class(Kronecker)


def _factor_pair(total: int) -> tuple[int, int]:
    for k in range(2, int(total**0.5) + 1):
        if total % k == 0:
            return k, total // k
    return total, 1


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

        sel_np = np.asarray(selected_indices, dtype=np.int32)  # (k, d)

        # `factor_vecs` arrive with their columns ALREADY permuted into
        # eigenvalue-sorted order (see `topk_eigh`), and `selected_indices`
        # index into that same sorted order. Re-applying `sort_indices` here
        # would permute a second time and select the wrong columns entirely --
        # in practice the bottom of the spectrum instead of the top.
        # `sort_indices` is retained only so callers can recover the mapping
        # back to each factor's original eigenvector ordering.
        self._gathered = []
        for i in range(self._d):
            # idx_sorted: (k,) -- indices into the already-sorted columns
            idx_sorted = jnp.asarray(sel_np[:, i])
            self._gathered.append(self._factor_vecs[i][:, idx_sorted])

        dtype = factor_vecs[0].dtype
        super().__init__((self._n_total, self._k), dtype)

    @property
    def k(self) -> int:
        """Number of selected eigenvector combinations."""
        return self._k

    @property
    def num_factors(self) -> int:
        """Number of Kronecker factors."""
        return self._d

    @property
    def factor_dims(self) -> list[int]:
        """Dimensions of each Kronecker factor."""
        return self._factor_dims

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten for JAX pytree registration."""
        children = (
            tuple(self._factor_vecs),
            tuple(self._sort_indices),
        )
        aux_data = {
            "selected_indices": self._selected_indices,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "KroneckerSelectedEigenvectors":
        """Unflatten for JAX pytree registration."""
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
                msg = f"Unsupported v shape {v.shape}. Expected (n,), (batch,n), or (n,p)"
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
        """Return transpose operator."""
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
        """Flatten for JAX pytree registration."""
        return (self._parent,), {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "KroneckerSelectedEigenvectorsTranspose":
        """Unflatten for JAX pytree registration."""
        return cls(children[0])

    def _matmul(self, v: jax.Array) -> jax.Array:
        return self._parent._rmatmul(v)

    def transpose(self) -> KroneckerSelectedEigenvectors:
        """Return transpose (the parent operator)."""
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

    Returns
    -------
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
    from linox.operators.arithmetic import ScaledLinearOperator

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


class KronTopkEighInfo(NamedTuple):
    """Extra info to keep top-k eigenpairs factorized.

    Notes
    -----
    - factor_vecs[i] are the eigenvectors Q_i in *original* column order (as returned by leigh)
    - factor_eigs[i] are the eigenvalues w_i in *sorted* order (descending if largest=True)
    - sort_indices[i] is the permutation 'order' such that w_sorted = w[order]
    - selected_indices are tuples in *sorted coordinates* (i.e. indices into w_sorted)
    """

    factor_vecs: list[jax.Array]  # [Q_0, Q_1, ...], each (n_i, n_i)
    factor_eigs: list[jax.Array]  # [w_0_sorted, ...], each (n_i,)
    sort_indices: list[jax.Array]  # [order_0, ...], each (n_i,)
    selected_indices: list[tuple[int, ...]]  # [(i0,i1,...), ...], length k
    scalar: jax.Array | None  # scalar if ScaledLinearOperator was unwrapped


def whitened_selected_columns(
    info: KronTopkEighInfo,
    *,
    eps: float = 1e-12,
    sigma2: float | jax.Array | None = None,
    scale_on_factor: int = 0,  # 0 = u-factor
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array], jax.Array]:
    """Return factorwise columns for selected Kronecker eigenvectors with correct per-column whitening.

    We build columns for each factor r: Q_r[:, idx_orig_r(i)].
    Then compute full eigenvalue λ_i = scalar * Π_r w_r_sorted[idx_sorted_r(i)].
    Whitening scale is s_i = (max(λ_i, floor))^{-1/2}, and we absorb s_i into one factor (default u-factor).

    Returns
    -------
    factor_cols : list[jax.Array]
        factor_cols[r] has shape (n_r, k), with column i the eigenvector column for factor r.
        The whitening scale s_i is applied ONLY on factor `scale_on_factor`.
    idx_sorted_list, idx_orig_list : list[jax.Array]
        indices per factor
    full_eigs : jax.Array
        shape (k,), the full Kronecker eigenvalues (incl. scalar)
    """
    d = len(info.factor_vecs)
    k = len(info.selected_indices)

    idx_sorted_list: list[jax.Array] = []
    idx_orig_list: list[jax.Array] = []
    factor_cols: list[jax.Array] = []

    # gather columns
    for r in range(d):
        Qr = info.factor_vecs[r]  # (n_r, n_r) in original order
        order = info.sort_indices[r]  # sorted -> original
        info.factor_eigs[r]  # sorted eigenvalues (n_r,)

        idx_sorted = jnp.array([info.selected_indices[i][r] for i in range(k)])  # (k,)
        idx_orig = order[idx_sorted]  # (k,)
        cols = Qr[:, idx_orig]  # (n_r, k)

        idx_sorted_list.append(idx_sorted)
        idx_orig_list.append(idx_orig)
        factor_cols.append(cols)

    # full eigenvalues λ_i = scalar * Π_r wr_sorted[idx_sorted_r(i)]
    full_eigs = jnp.ones((k,), dtype=info.factor_eigs[0].dtype)
    for r in range(d):
        full_eigs = full_eigs * info.factor_eigs[r][idx_sorted_list[r]]

    if info.scalar is not None:
        full_eigs = full_eigs * jnp.asarray(info.scalar, dtype=full_eigs.dtype)

    # floor for pseudo-inverse sqrt
    floor = jnp.asarray(eps, dtype=full_eigs.dtype)
    if sigma2 is not None:
        floor = jnp.maximum(floor, jnp.asarray(sigma2, dtype=full_eigs.dtype))

    inv_sqrt = jnp.exp(-0.5 * jnp.log(jnp.maximum(full_eigs, floor)))  # (k,)

    # absorb scale into one factor to keep factorization
    factor_cols[scale_on_factor] = factor_cols[scale_on_factor] * inv_sqrt[None, :]

    return factor_cols, idx_sorted_list, idx_orig_list, full_eigs


def build_kron_columns_from_factors(
    factor_cols: list[jax.Array],
) -> jax.Array:
    """Build explicit kron columns from factor columns.

    Parameters
    ----------
    factor_cols : list[jax.Array]
        List [C0, C1, ...] where each Cr has shape (n_r, k)

    Returns
    -------
    X : jax.Array
        Explicit columns X[:, l] = kron_r Cr[:, l], shape (prod n_r, k)

    Notes
    -----
    This is O(prod n_r * k) memory/time; fine for testing and for cases where
    prod n_r is not enormous.
    """
    k = factor_cols[0].shape[1]

    cols_out = []
    for l_idx in range(k):
        v = factor_cols[0][:, l_idx]
        for r in range(1, len(factor_cols)):
            v = jnp.kron(v, factor_cols[r][:, l_idx])
        cols_out.append(v)
    return jnp.stack(cols_out, axis=1)  # (prod n_r, k)


def _topk_product_grid_indices_jax(
    w_list_sorted: list[jax.Array],
    k: int,
    *,
    largest: bool,
    scalar: jax.Array,
    add_shift: jax.Array,
    oversample: int = 4,
):
    d = len(w_list_sorted)
    keep = int(k * oversample)

    w0 = w_list_sorted[0]
    m0 = min(int(w0.shape[0]), keep)
    vals = w0[:m0]
    idx = jnp.arange(m0, dtype=jnp.int32)[:, None]

    for r in range(1, d):
        wr = w_list_sorted[r]
        mr = min(int(wr.shape[0]), keep)
        wr = wr[:mr]

        prod = vals[:, None] * wr[None, :]
        flat = prod.reshape(-1)

        score = flat if largest else -flat
        k2 = min(int(flat.shape[0]), keep)

        top_score, top_flat_idx = jax.lax.top_k(score, k2)
        top_vals = top_score if largest else -top_score

        i_prev = top_flat_idx // mr
        i_r = top_flat_idx % mr

        idx = jnp.concatenate([idx[i_prev], i_r[:, None].astype(jnp.int32)], axis=1)
        vals = top_vals

    vals = vals[:k]
    idx = idx[:k, :]
    eigvals = scalar * vals + add_shift
    return eigvals, idx


def _topk_product_grid_indices_host(
    w_list_sorted: list[np.ndarray],  # each (m_r,) sorted asc if smallest else desc
    k: int,
    *,
    largest: bool,
    eps_cutoff: float,
    scalar: float = 1.0,
    add_shift: float = 0.0,
):
    d = len(w_list_sorted)
    sizes = [w.shape[0] for w in w_list_sorted]
    if any(s == 0 for s in sizes):
        return [], []

    idx0 = tuple(0 for _ in range(d))

    def base_prod(idx):
        p = float(scalar)
        for r, ir in enumerate(idx):
            p *= float(w_list_sorted[r][ir])
        return p

    p0 = base_prod(idx0)
    v0 = p0 + add_shift

    heap = [((-v0 if largest else v0), p0, idx0)]
    visited = {idx0}

    eigvals: list[float] = []
    selected: list[tuple[int, ...]] = []

    while heap and len(eigvals) < k:
        prio, p, idx = heapq.heappop(heap)
        val = -prio if largest else prio  # == p + add_shift

        # expand neighbors first (so skipping doesn't block exploration)
        for dim in range(d):
            nxt = list(idx)
            nxt[dim] += 1
            nxt = tuple(nxt)
            if nxt[dim] >= sizes[dim] or nxt in visited:
                continue
            visited.add(nxt)

            w_old = w_list_sorted[dim][idx[dim]]
            w_new = w_list_sorted[dim][nxt[dim]]
            p_nxt = p * (w_new / w_old) if w_old != 0.0 else base_prod(nxt)

            v_nxt = p_nxt + add_shift
            heapq.heappush(heap, ((-v_nxt if largest else v_nxt), p_nxt, nxt))

        # "numerical zero" filter
        if val <= eps_cutoff:
            if largest:
                # for PSD + largest, once you hit ~0, the rest are <= 0
                break
            else:
                # for smallest, keep going until positive entries appear
                continue

        eigvals.append(val)
        selected.append(idx)

    return eigvals, selected


def topk_eigh(
    op_or_factors,
    k: int,
    *,
    largest: bool = True,
    sigma2: float | jax.Array | None = None,
    include_noise_shift: bool = False,
    return_full_eigs: bool = False,
    mode: str = "jax",
):
    """Compute top-k eigenvalues/vectors of a Kronecker product.

    Parameters
    ----------
    op_or_factors : LinearOperator or list[LinearOperator]
        Either a Kronecker operator or a list of factor operators
    k : int
        Number of eigenvalues/vectors to compute
    largest : bool, default=True
        If True, compute largest eigenvalues; if False, compute smallest
    sigma2 : float or jax.Array, optional
        Noise variance for whitening
    include_noise_shift : bool, default=False
        Whether to include noise shift in eigenvalues
    return_full_eigs : bool, default=False
        If True, return full eigenvalue arrays for each factor
    mode : str, default="jax"
        Computation mode

    Returns
    -------
    eigenvalues : jax.Array
        Top-k eigenvalues
    eigenvectors : LinearOperator
        Top-k eigenvectors as a linear operator
    info : KronTopkEighInfo
        Factorized representation with additional information
    """
    scalar = None
    if isinstance(op_or_factors, LinearOperator):
        factors, scalar = extract_kronecker_factors(op_or_factors)
    else:
        factors = [utils.as_linop(f) for f in op_or_factors]

    factor_eigs: list[jax.Array] = []
    factor_vecs: list[jax.Array] = []
    sort_indices: list[jax.Array] = []
    full_factor_eigs: list[jax.Array] = []

    # compute eigendecomp of each factor
    for A in factors:
        w, Q = leigh(A)
        if isinstance(w, LinearOperator):
            w = diagonal(w)

        eps = jnp.finfo(w.dtype).eps
        w_safe = jnp.maximum(w, eps)  # (n,)

        Q_dense = Q._todense() if hasattr(Q, "_todense") else jnp.asarray(Q)  # (n,n)

        order = jnp.argsort(-w_safe) if largest else jnp.argsort(w_safe)  # (n,)
        w_sorted = w_safe[order]
        Q_sorted = Q_dense[:, order]

        sort_indices.append(order)
        factor_eigs.append(w_sorted)
        factor_vecs.append(Q_sorted)

        if return_full_eigs:
            full_factor_eigs.append(w_sorted)

    # numerical eps cutoff (host float)
    dtype = factor_eigs[0].dtype

    # time_start = time.time()  # Timing removed

    if mode == "jax":
        # JAX branch: NO device_get, NO python if on traced values
        scalar_j = jnp.asarray(1.0, dtype=dtype) if scalar is None else jnp.asarray(scalar, dtype=dtype)
        add_shift_j = jnp.asarray(0.0, dtype=dtype)
        if include_noise_shift and sigma2 is not None:
            add_shift_j = jnp.asarray(sigma2, dtype=dtype)

        eig_array, selected_indices = _topk_product_grid_indices_jax(
            factor_eigs,
            k,
            largest=largest,
            scalar=scalar_j,
            add_shift=add_shift_j,
        )

    else:
        # Host/debug branch only
        eps_cutoff = float(np.finfo(np.dtype(dtype)).eps)

        scalar_f = 1.0
        if scalar is not None:
            scalar_f = float(jax.device_get(jnp.asarray(scalar)))
            if scalar_f < 0.0:
                raise ValueError("Negative scalar breaks PSD monotone-grid assumptions.")

        add_shift = 0.0
        if include_noise_shift and sigma2 is not None:
            add_shift = float(jax.device_get(jnp.asarray(sigma2)))

        w_host_list = [np.asarray(jax.device_get(w)) for w in factor_eigs]
        eigvals_list, selected_indices = _topk_product_grid_indices_host(
            w_host_list,
            k,
            largest=largest,
            eps_cutoff=eps_cutoff,
            scalar=scalar_f,
            add_shift=add_shift,
        )
        eig_array = jnp.asarray(eigvals_list, dtype=dtype)

    # time_end = time.time() # Timing removed
    # logger.info(...) # Removed

    Qk = KroneckerSelectedEigenvectors(factor_vecs, selected_indices, sort_indices)

    info = KronTopkEighInfo(
        factor_vecs=factor_vecs,
        factor_eigs=factor_eigs,
        sort_indices=sort_indices,
        selected_indices=selected_indices,
        scalar=scalar,
    )
    if return_full_eigs:
        # return eig_array, Qk, info, full_factor_eigs
        # Compatibility wrapper: existing calls might expect (vals, vecs).
        # But this function is new (renamed from topk_eigh which was slightly different).
        # We'll return (vals, vecs, info, ...) and update users.
        return eig_array, Qk, info, full_factor_eigs
    return eig_array, Qk, info


class KroneckerAdditiveIsotropicAdditiveLinearOperator(AddLinearOperator):
    """Kronecker product with isotropic additive term (Kron + sI)."""

    def __init__(self, kronecker: Kronecker, s: jax.Array) -> None:
        self._kronecker = kronecker
        self._s = s
        super().__init__(kronecker.shape, kronecker.dtype)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self._kronecker._matmul(vec) + self._s * vec

    def _todense(self) -> jax.Array:
        return self._kronecker._todense() + self._s * Identity(self._kronecker.shape[0])._todense()

    def transpose(self) -> "KroneckerAdditiveIsotropicAdditiveLinearOperator":
        """Return transposed operator."""
        return KroneckerAdditiveIsotropicAdditiveLinearOperator(self._kronecker.transpose(), self._s)

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten for JAX pytree registration."""
        return (self._kronecker, self._s), {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "KroneckerAdditiveIsotropicAdditiveLinearOperator":
        """Unflatten for JAX pytree registration."""
        return cls(children[0], children[1])


# @linverse.dispatch
