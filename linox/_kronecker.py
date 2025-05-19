r"""Kronecker product operations for linear operators.

This module includes a linear operator that represents the Kronecker product
of two linear operators.

- :class:`Kronecker`: Represents the Kronecker product :math:`A \otimes B` of two
    linear operators :math:`A` and :math:`B`
"""

import jax
import jax.numpy as jnp

from linox._arithmetic import linverse, lsqrt
from linox._linear_operator import LinearOperator
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


# Register Kronecker as a PyTree
jax.tree_util.register_pytree_node_class(Kronecker)
