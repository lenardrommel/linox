import jax
import jax.numpy as jnp
from scipy import linalg as jsp

from linox._arithmetic import lsolve
from linox._linear_operator import LinearOperator
from linox._matrix import solve_toeplitz
from linox.utils import ArrayLike

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
        return self.A.shape

    def _matmul(self, vector: jax.Array) -> jax.Array:
        n = self.v.shape[0]

        def compute_element(i):  # noqa: ANN202
            indices = jnp.abs(i - jnp.arange(n))
            return jnp.sum(self.v[indices] * vector)

        result = jax.vmap(compute_element)(jnp.arange(n))
        return result

    def todense(self) -> jax.Array:
        return jsp.linalg.toeplitz(self.v)

    def from_matrix(self, matrix: jax.Array) -> "Toeplitz":
        self.v = matrix[0, :]
        return Toeplitz(self.v)

    def transpose(self) -> "Toeplitz":
        return Toeplitz(self.v)


@lsolve.dispatch
def _(A: Toeplitz, b: jax.Array) -> jax.Array:
    """Solve a Toeplitz system."""
    return solve_toeplitz(
        c=A.v,
        b=b,
        check_finite=False,
    )
