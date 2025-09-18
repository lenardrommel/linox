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
    lcholesky,
    ldet,
    leigh,
    linverse,
    lpinverse,
    lqr,
    lsqrt,
    slogdet,
    svd,
)
from linox._linear_operator import LinearOperator


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
        self._shape = (
            self._A.shape[0] * self._B.shape[0],
            self._A.shape[1] * self._B.shape[1],
        )
        dtype = A.dtype
        super().__init__(self._shape, dtype)

    @property
    def A(self) -> LinearOperator:
        return self._A

    @property
    def B(self) -> LinearOperator:
        return self._B

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self._A.shape[0] * self._B.shape[0],
            self._A.shape[1] * self._B.shape[1],
        )

    @classmethod
    def tree_flatten(cls) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (cls.A, cls.B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
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
        y = y.reshape(y.shape[:-1] + (mA, mB))

        # (X @ B.T).T = B @ X.T
        y = self.B @ jnp.swapaxes(y, -1, -2)

        # A @ X @ B.T = A @ (B @ X.T).T
        y = self.A @ jnp.swapaxes(y, -1, -2)

        # vec(A @ X @ B.T), i.e., revert to stack of vectorized matrices
        y = y.reshape(y.shape[:-2] + (-1,))
        y = jnp.swapaxes(y, -1, -2)

        return y

    def todense(self) -> jax.Array:
        return jnp.kron(self.A.todense(), self.B.todense())

    def transpose(self) -> "Kronecker":
        return Kronecker(self.A.transpose(), self.B.transpose())

    def diag(self) -> "Kronecker":
        return Kronecker(self.A.diagonal(), self.B.diagonal())

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
    eigvals = jnp.outer(wA, wB).flatten()
    return eigvals, Kronecker(QA, QB)


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


# Not properly tested yet.
# @lsolve.dispatch
# def _(op: Kronecker, v: jax.Array) -> jax.Array:
#     m_A, _ = op.A.shape
#     m_B, _ = op.B.shape


#     V = v.reshape((m_A, m_B))
#     return jnp.ravel(lsolve(op.A, lsolve(op.B, V.T).T))  # op.A.solve(op.B.solve(V.T).T)


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


# Register Kronecker as a PyTree
jax.tree_util.register_pytree_node_class(Kronecker)
