from __future__ import annotations

import jax.numpy as jnp

# import probnum as pn
from linox._linear_operator import LinearOperator
from linox._typing import ArrayLike

DTYPE = jnp.float32


class Permutation(LinearOperator):
    def __init__(self, perm: ArrayLike, perm_inv: ArrayLike | None = None) -> None:
        self._perm = jnp.asarray(perm, dtype=jnp.int_)
        self._perm_inv = (
            jnp.asarray(perm_inv, dtype=jnp.int_)
            if perm_inv is not None
            else jnp.argsort(self._perm)
        )

        super().__init__(
            shape=(self._perm.size, self._perm.size),
            dtype=DTYPE,  # Otherwise operation not allowed
        )

    # def mv(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.take(x, self._perm, axis=-2)

    def mv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(x, self._perm)

    def transpose(self) -> Permutation:
        return Permutation(self._perm_inv, self._perm)

    def inverse(self) -> Permutation:
        return self.transpose()
