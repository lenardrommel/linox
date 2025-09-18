import jax
import jax.numpy as jnp

from linox._arithmetic import AddLinearOperator, ScaledLinearOperator, lcholesky
from linox._linear_operator import LinearOperator
from linox._matrix import Identity


class IsotropicAdditiveLinearOperator(AddLinearOperator):
    """Isotropic additive linear operator.

    Represents a linear operator of the form A = s * I + A where A is a
    linear operator and I is the identity matrix.

    Args:
        s: Scalar value to be added to the diagonal.
        A: Linear operator to be added.
    """

    def __init__(self, s: jax.Array, A: LinearOperator) -> None:
        self._A = A
        self._s = ScaledLinearOperator(
            Identity(self._A.shape[0], dtype=self._A.dtype), s
        )

        super().__init__(self._s, self._A)

    @property
    def s(self) -> jax.Array:
        return self._s

    @property
    def shape(self) -> tuple[int, int]:
        return self._A.shape

    def _matmul(self, arr):  # noqa: ANN202
        return self._s @ arr + self._A @ arr

    def todense(self) -> jax.Array:
        return self._s.todense() + self._A.todense()


@lcholesky.dispatch
def _(a: IsotropicAdditiveLinearOperator, jitter: float = 1e-10) -> jax.Array:
    return jnp.linalg.cholesky(a.todense() + jitter * jnp.eye(a.shape[0]))


# we need a log-determinant function here
# TODO: Implement lsolve for IsotropicAdditiveLinearOperator via eigendecomposition
# A \kron B + s I = (Q_A \kron Q_B) (Lambda_A \kron Lambda_B + s I) (Q_A \kron Q_B)^T
