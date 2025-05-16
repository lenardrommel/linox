r"""Permutation operations for linear operators.

This module implements permutation operations for linear operators, including:

- :class:`Permutation`: Represents a permutation matrix :math:`P` that permutes the rows
    of a vector according to a given permutation
"""

from functools import partial

import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator
from linox.typing import ArrayLike

DTYPE = jnp.float32


@partial(jnp.vectorize, signature="(n,k),(n)->(n,k)")
def _perm_op(_arr: jnp.ndarray, _indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(_arr, _indices, axis=-2)


class Permutation(LinearOperator):
    r"""A linear operator defined via a permutation matrix.

    For a permutation vector :math:`p`, this represents the permutation matrix :math:`P`
    where :math:`P_{ij} = 1` if :math:`j = p_i` and :math:`0` otherwise. The action on a
    vector :math:`x` is given by :math:`(Px)_i = x_{p_i}`, i.e., it permutes the
    elements of :math:`x` according to the permutation :math:`p`.

    Args:
        perm: The permutation vector defining the operator
        perm_inv: The inverse permutation vector (optional, computed if not provided)
    """

    def __init__(self, perm: ArrayLike, perm_inv: ArrayLike | None = None) -> None:
        self._perm = jnp.asarray(perm, dtype=jnp.int32)
        self._perm_inv = (
            jnp.asarray(perm_inv, dtype=jnp.int32)
            if perm_inv is not None
            else jnp.argsort(self._perm, axis=-1)
        )
        _perm_size = self._perm.shape[-1]
        super().__init__(
            shape=(*self._perm.shape[:-1], _perm_size, _perm_size),
            dtype=DTYPE,  # Otherwise operation not allowed
        )

    def _matmul(self, x: jnp.ndarray) -> jnp.ndarray:
        return _perm_op(x, self._perm)

    def transpose(self) -> "Permutation":
        return Permutation(self._perm_inv, self._perm)

    def inverse(self) -> "Permutation":
        return self.transpose()

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self._perm, self._perm_inv)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Permutation":
        del aux_data
        perm, perm_inv = children
        return cls(perm=perm, perm_inv=perm_inv)


# Register Permutation as a PyTree
jax.tree_util.register_pytree_node_class(Permutation)
