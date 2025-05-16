r"""Linear operator for representing an eigenvalue decomposition.

This module includes a linear operator that represents an eigenvalue decomposition
of an array.

- :class:`EigenD`: Represents a linear operator :math:`A` in its eigenvalue
    decomposition form :math:`A = U \Lambda U^T` where :math:`U` is orthogonal
    and :math:`\Lambda` is diagonal
"""

import jax

from linox._linear_operator import LinearOperator


class EigenD(LinearOperator):
    r"""Eigenvalue-representation of an array as an linear operator.

    For a eigenvectors :math:`U` and eigenvalues :math:`S`, this represents the
    linear operator :math:`A = U \Lambda U^T` where :math:`\Lambda` is diagonal.
    The action on a vector :math:`x` is given by :math:`Ax = U(\Lambda(U^T x))`

    Args:
        U: Orthogonal matrix of eigenvectors
        S: Vector of eigenvalues (diagonal elements of :math:`\Lambda`)
    """

    def __init__(self, U: jax.Array, S: jax.Array) -> None:
        self.U = U
        self.S = S
        super().__init__(shape=(U.shape[0], U.shape[0]), dtype=S.dtype)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self.U @ (self.S[:, None] * (self.U.T @ vec))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.U, self.S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "EigenD":
        del aux_data
        U, S = children
        return cls(U=U, S=S)


# Register EigenD as a PyTree
jax.tree_util.register_pytree_node_class(EigenD)
