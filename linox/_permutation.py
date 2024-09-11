from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from linox._linear_operator import LinearOperator
from linox._typing import ArrayLike

DTYPE = jnp.float32


@partial(jnp.vectorize, signature="(n,k),(n)->(n,k)")
def _perm_op(_arr, _indices):
    return jnp.take(_arr, _indices, axis=-2)


# TODO(2bys): Write tests and docstrings for class.
class Permutation(LinearOperator):
    def __init__(self, perm: ArrayLike, perm_inv: ArrayLike | None = None) -> None:
        self._perm = jnp.asarray(perm, dtype=jnp.int_)
        self._perm_inv = (
            jnp.asarray(perm_inv, dtype=jnp.int_)
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

    def transpose(self) -> Permutation:
        return Permutation(self._perm_inv, self._perm)

    def inverse(self) -> Permutation:
        return self.transpose()
