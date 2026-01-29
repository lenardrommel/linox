# _kronecker.py

r"""Kronecker product operations for linear operators.

This module includes a linear operator that represents the Kronecker product
of two linear operators.

- :class:`Kronecker`: Represents the Kronecker product :math:`A \otimes B` of two
    linear operators :math:`A` and :math:`B`
"""

import jax
import jax.numpy as jnp

from linox import utils
from linox._arithmetic import (
    ProductLinearOperator,
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
    lsqrt,
    ltrace,
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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.A, self.B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],  # noqa: ARG003
        children: tuple[any, ...],
    ) -> "Kronecker":
        return cls(*children)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        if len(vec.shape) == 1:
            vec = vec[:, None]

        _, mA = self.A.shape
        _, mB = self.B.shape

        # vec(X) -> X, i.e., reshape into stack of matrices
        y = jnp.swapaxes(vec, -2, -1)
        y = y.reshape((*y.shape[:-1], mA, mB))

        # (X @ B.T).T = B @ X.T
        y = self.B @ jnp.swapaxes(y, -1, -2)

        # A @ X @ B.T = A @ (B @ X.T).T
        y = self.A @ jnp.swapaxes(y, -1, -2)

        # vec(A @ X @ B.T), i.e., revert to stack of vectorized matrices
        y = y.reshape((*y.shape[:-2], -1))
        y = jnp.swapaxes(y, -1, -2)

        return y

    def todense(self) -> jax.Array:
        return jnp.kron(self.A.todense(), self.B.todense())

    def transpose(self) -> "Kronecker":
        return Kronecker(self.A.transpose(), self.B.transpose())

    def trace(self) -> jax.Array:
        return jnp.trace(self.A) * jnp.trace(self.B)


@linverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(linverse(op.A), linverse(op.B))


@lpinverse.dispatch
def _(op: Kronecker) -> Kronecker:
    return Kronecker(lpinverse(op.A), lpinverse(op.B))


@lsqrt.dispatch
def _(op: Kronecker) -> Kronecker:
    r"""Square root of a Kronecker product.

    For a Kronecker product :math:`A \otimes B`, this represents
    :math:`\sqrt{A \otimes B} = \sqrt{A} \otimes \sqrt{B}`
    """
    return Kronecker(lsqrt(op.A), lsqrt(op.B))


@leigh.dispatch
def _(op: Kronecker) -> tuple[jax.Array, Kronecker]:
    wA, QA = leigh(op.A)
    wB, QB = leigh(op.B)

    return jnp.kron(wA, wB), Kronecker(QA, QB)


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
def _(op: Kronecker, **kwargs) -> tuple[Kronecker, jax.Array, Kronecker]:
    """SVD decomposition of a kronecker product.

    Exploits the structure: SVD(A ⊗ B) = (U_A ⊗ U_B) (S_A ⊗ S_B) (V_A^H ⊗ V_B^H)

    Returns:
        U(U_A, U_B): Left singular vectors as Kronecker product
        S: Singular values (outer product of S_A and S_B, flattened)
        Vh(Vh_A, Vh_B): Right singular vectors (Hermitian) as Kronecker product

    Notes:
        Passes through all kwargs (k, num_iters, u0, etc.) to constituent SVDs.
    """
    U_A, S_A, Vh_A = svd(op.A, **kwargs)
    U_B, S_B, Vh_B = svd(op.B, **kwargs)

    return (
        Kronecker(U_A, U_B),
        jnp.outer(S_A, S_B).flatten(),
        Kronecker(Vh_A, Vh_B),
    )


# Not properly tested yet.
# @lsolve.dispatch
# def _(op: Kronecker, v: jax.Array) -> jax.Array:
#     m_A, _ = op.A.shape
#     m_B, _ = op.B.shape


#     V = v.reshape((m_A, m_B))
#     return jnp.ravel(lsolve(op.A, lsolve(op.B, V.T).T))  # op.A.solve(op.B.solve(V.T)
# .T)


@lcholesky.dispatch
def _(op: Kronecker) -> Kronecker:
    L_A = lcholesky(op.A)
    L_B = lcholesky(op.B)
    return Kronecker(L_A, L_B)


# Not properly tested yet.
@ldet.dispatch
def _(op: Kronecker) -> ProductLinearOperator:
    return ProductLinearOperator([
        ldet(op.A) ** op.B.shape[0],
        ldet(op.B) ** op.A.shape[0],
    ])


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
    from linox._arithmetic import ltrace  # noqa: PLC0415

    trace_A, std_A = ltrace(
        op.A, key=key, num_samples=num_samples, distribution=distribution
    )
    trace_B, std_B = ltrace(
        op.B, key=key, num_samples=num_samples, distribution=distribution
    )

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
    from linox._algorithms._matrix_functions import (
        lanczos_matrix_function,
    )

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
        from linox._algorithms._matrix_functions import (
            lanczos_matrix_function,
        )
        from linox.config import warn as _warn  # noqa: PLC0415

        _warn(
            "Computing log(A ⊗ B) using dense method - no efficient structured formula available"
        )
        return utils.as_linop(jnp.linalg.matrix_exp(op.todense()))
    from linox._algorithms._matrix_functions import (
        lanczos_matrix_function,
    )

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
    from linox._algorithms._matrix_functions import (
        lanczos_matrix_function,
    )

    def power_func(eigvals):
        return eigvals**power

    return lanczos_matrix_function(op, v, power_func, num_iters, reortho=True)


# Register Kronecker as a PyTree
jax.tree_util.register_pytree_node_class(Kronecker)


# Experimental registry integration
def _factor_pair(total: int) -> tuple[int, int]:
    for k in range(2, int(total**0.5) + 1):
        if total % k == 0:
            return k, total // k
    return total, 1


@register("kronecker", tags=("rectangular",))
def make_kronecker(
    key: jax.random.PRNGKey,
    shape: tuple[int, int],
    dtype=jnp.float32,
    require=None,
    *,
    maker_A: str = "matrix",
    maker_B: str = "matrix",
    **kwargs,
) -> Kronecker:
    m, n = shape
    mA, mB = _factor_pair(m)
    nA, nB = _factor_pair(n)

    keyA, keyB = jax.random.split(key)
    A = get(maker_A).maker(keyA, (mA, nA), dtype, require=require)
    B = get(maker_B).maker(keyB, (mB, nB), dtype, require=require)

    return Kronecker(A, B)
