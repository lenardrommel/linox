# _eigen.py

r"""Linear operator for representing an eigenvalue decomposition.

This module includes a linear operator that represents an eigenvalue decomposition
of an array.

- :class:`EigenD`: Represents a linear operator :math:`A` in its eigenvalue
    decomposition form :math:`A = U \Lambda U^T` where :math:`U` is orthogonal
    and :math:`\Lambda` is diagonal
"""

import jax
from jax import numpy as jnp

from linox._arithmetic import leigh
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal
from linox.typing import ArrayLike
from linox.utils import as_linop


class EigenD(LinearOperator):
    r"""A linear operator representing an eigenvalue decomposition.
    Represents a linear operator :math:`A` in its eigenvalue decomposition form
    :math:`A = U \Lambda U^T` where :math:`U` is orthogonal and :math:`\Lambda`
    is diagonal.

    Args:
        A: The square matrix to decompose.
    """  # noqa: D205

    def __init__(self, A: ArrayLike) -> None:
        self.A = as_linop(A)
        super().__init__(shape=(A.shape[0], A.shape[0]), dtype=A.dtype)

    def _ensure_eigh(self) -> None:
        if (self._S is None) or (self._Q is None):
            self._S, self._Q = leigh(self.A)

    @property
    def U(self) -> LinearOperator:
        self._ensure_eigh()
        return self._Q

    @property
    def S(self) -> Diagonal:
        self._ensure_eigh()
        return self._S

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self.U @ (self.S[:, None] * (self.U.T @ vec))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.U, self.S)
        aux_data = {}
        return children, aux_data

    def todense(self) -> jax.Array:
        """Convert the linear operator to a dense matrix."""
        U = self.U.todense()
        S = self.S.todense()
        eigvals, eigvecs = jnp.linalg.eigh(U @ jnp.diag(S) @ U.T)
        return eigvals, eigvecs

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "EigenD":
        del aux_data
        U, S = children
        return cls(U=U, S=S)


@leigh.dispatch
def _(a: EigenD) -> tuple[LinearOperator, LinearOperator]:
    return a.eigvals, a.eigvecs


# Register EigenD as a PyTree
jax.tree_util.register_pytree_node_class(EigenD)


# A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
# A = A @ A.T + jnp.eye(3) * 1e-6
# print("Original matrix:\n", A)
# op = EigenD(A)
# print("Dense matrix:\n", op.todense())
# print("Shape:", op.shape)
