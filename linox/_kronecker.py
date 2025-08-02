r"""Kronecker product operations for linear operators.

This module includes a linear operator that represents the Kronecker product
of two linear operators.

- :class:`Kronecker`: Represents the Kronecker product :math:`A \otimes B` of two
    linear operators :math:`A` and :math:`B`
"""

import jax
import jax.numpy as jnp

from linox._arithmetic import (
    ProductLinearOperator,
    lcholesky,
    ldet,
    leigh,
    linverse,
    lpsolve,
    lqr,
    lsolve,
    lsqrt,
    slogdet,
)
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix
from linox.utils import as_linop


class Kronecker(LinearOperator):
    r"""Kronecker product of two linear operators.

    For linear operators :math:`A` and :math:`B`, this represents their Kronecker
    product :math:`A \otimes B`. The action on a vector :math:`x` is given by
    :math:`(A \otimes B)x = \text{vec}(A \cdot \text{unvec}(x) \cdot B^T)`
    where :math:`\text{vec}` and :math:`\text{unvec}` are vectorization operations.

    Args:
        A: First linear operator
        B: Second linear operator
    """

    def __init__(
        self, A: LinearOperator | jax.Array, B: LinearOperator | jax.Array
    ) -> None:
        self.A = as_linop(A)
        self.B = as_linop(B)
        shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        dtype = A.dtype
        super().__init__(shape, dtype)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        mA, nA = self.A.shape
        mB, nB = self.B.shape
        assert mA == nA and mB == nB, (  # noqa: PT018
            f"Kronecker product requires square matrices, got {self.A.shape} and {self.B.shape}"
        )

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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.A, self.B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Kronecker":
        del aux_data
        A, B = children
        return cls(A=A, B=B)


@linverse.dispatch
def _(op: Kronecker) -> Kronecker:
    r"""Inverse of a Kronecker product.

    For a Kronecker product :math:`A \otimes B`, this represents
    :math:`(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}`
    """
    return Kronecker(linverse(op.A), linverse(op.B))


@lsqrt.dispatch
def _(op: Kronecker) -> Kronecker:
    r"""Square root of a Kronecker product.

    For a Kronecker product :math:`A \otimes B`, this represents
    :math:`\sqrt{A \otimes B} = \sqrt{A} \otimes \sqrt{B}`
    """
    return Kronecker(lsqrt(op.A), lsqrt(op.B))


# Not properly tested yet.
@leigh.dispatch
def _(op: Kronecker) -> Kronecker:
    wA, QA = leigh(op.A)
    wB, QB = leigh(op.B)
    eigvals = jnp.outer(wA, wB).flatten()
    return Matrix(eigvals), Kronecker(QA, QB)


# Not properly tested yet.
@lqr.dispatch
def _(op: Kronecker) -> tuple[Kronecker, Kronecker]:
    """QR decomposition of a kronecker product.
    Returns:
        Q(Q_A, Q_B): Orthogonal matrix
        R(R_A, R_B): Upper triangular matrix.
    """  # noqa: D205
    Q_A, R_A = lqr(op.A)
    Q_B, R_B = lqr(op.B)
    return Kronecker(Q_A, Q_B), Kronecker(R_A, R_B)


# Not properly tested yet.
# @lsolve.dispatch
# def _(op: Kronecker, v: jax.Array) -> jax.Array:
#     m_A, _ = op.A.shape
#     m_B, _ = op.B.shape


#     V = v.reshape((m_A, m_B))
#     return jnp.ravel(lsolve(op.A, lsolve(op.B, V.T).T))  # op.A.solve(op.B.solve(V.T).T)
@lsolve.dispatch
def _(op: Kronecker, v: jax.Array) -> jax.Array:
    mA, nA = op.A.shape
    mB, nB = op.B.shape
    if mA == nA and mB == nB:
        return linverse(op) @ v.reshape(-1)  # noqa: SLF001
    raise ValueError(  # noqa: TRY003
        f"Cannot solve Kronecker({mA}×{nA}, {mB}×{nB}) unless both factors are square"  # noqa: EM102
    )


# Not properly tested yet.
@lpsolve.dispatch
def _(op: Kronecker, v: jax.Array) -> jax.Array:
    m_A, _ = op.A.shape
    m_B, _ = op.B.shape
    V = v.reshape((m_A, m_B))
    return jnp.ravel(lpsolve(op.A, lpsolve(op.B, V.T).T))


# Not properly tested yet.
@lcholesky.dispatch
def _(op: Kronecker) -> tuple[jax.Array, jax.Array]:
    L_A, U_A = lcholesky(op.A)
    L_B, U_B = lcholesky(op.B)
    return Kronecker(L_A, L_B), Kronecker(U_A, U_B)


# Not properly tested yet.
@ldet.dispatch
def _(op: Kronecker) -> ProductLinearOperator:
    return ProductLinearOperator([
        ldet(op.A) ** op.B.shape[0],
        ldet(op.B) ** op.A.shape[0],
    ])


# Not properly tested yet.
@slogdet.dispatch
def _(op: Kronecker) -> tuple[jax.Array, jax.Array]:
    sign_A, logdet_A = slogdet(op.A)
    sign_B, logdet_B = slogdet(op.B)

    dim_A = op.A.shape[0]
    dim_B = op.B.shape[0]

    final_sign = sign_A**dim_B * sign_B**dim_A
    final_logdet = dim_B * logdet_A + dim_A * logdet_B

    return final_sign, final_logdet


# Register Kronecker as a PyTree
jax.tree_util.register_pytree_node_class(Kronecker)
