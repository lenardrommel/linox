# _toeplity.py

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from linox._algorithms._toeplitz import solve_toeplitz_jax
from linox._arithmetic import diagonal, lsolve
from linox._linear_operator import LinearOperator
from linox._matrix import Identity
from linox.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# Toeplitz Linear Operator
# --------------------------------------------------------------------------- #


class Toeplitz(LinearOperator):
    """A Toeplitz matrix which is constructed from a 1D array."""

    def __init__(self, v: ArrayLike) -> None:
        self.v = jnp.asarray(v)
        super().__init__(self.v.shape, self.v.dtype)

    @property
    def shape(self) -> jax.Array:
        return *self.v.shape, *self.v.shape

    def _matmul(self, vector: jax.Array) -> jax.Array:
        n = self.v.shape[0]

        if vector.ndim == 1:
            vector = vector.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False

        embedded_col = jnp.concatenate([self.v, self.v[-1:0:-1]])

        p = len(embedded_col)

        fft_col = jnp.fft.fft(embedded_col)

        vector_padded = jnp.concatenate(
            [vector, jnp.zeros((p - n, vector.shape[1]))], axis=0
        )

        fft_vector = jnp.fft.fft(vector_padded, axis=0)

        fft_result = fft_col.reshape(-1, 1) * fft_vector

        result = jnp.fft.ifft(fft_result, axis=0).real[:n]

        if squeeze_output:
            result = result.squeeze(axis=1)

        return result

    def todense(self) -> jax.Array:
        return jsp.linalg.toeplitz(self.v)

    def from_matrix(self, matrix: jax.Array) -> "Toeplitz":
        self.v = matrix[0, :]
        return Toeplitz(self.v)

    def transpose(self) -> "Toeplitz":
        return Toeplitz(self.v)


@diagonal.dispatch
def _(A: Toeplitz) -> LinearOperator:
    return A.v[0] * Identity(A.shape[0])


@lsolve.dispatch
def _(A: Toeplitz, b: jax.Array) -> jax.Array:
    """Solve a Toeplitz system."""
    # Use the algorithm module implementation (hybrid SciPy+JAX with custom VJP)
    return solve_toeplitz_jax(A.v, b, check_finite=False)


# Solver and related algorithms moved to linox/_algorithms/_toeplitz.py
