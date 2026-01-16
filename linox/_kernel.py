# _kernel.py

from collections.abc import Callable

import jax
import jax.numpy as jnp

from linox._arithmetic import lsqrt
from linox._linear_operator import LinearOperator
from linox._toeplitz import Toeplitz


class KernelOperator(LinearOperator):
    def __init__(
        self,
        kernel: Callable[[jax.Array], jax.Array],
        x0: jax.Array,
        x1: jax.Array | None = None,
    ) -> None:
        self.kernel = kernel
        self.x0 = x0
        if x1 is None:
            self.x1 = x0
        else:
            self.x1 = x1

        super().__init__(shape=(self.x0.shape[0], self.x1.shape[0]), dtype=x0.dtype)

    # @property
    # def kernel(self) -> Callable[[jax.Array], jax.Array]:
    #     return self.kernel


class ArrayKernel(KernelOperator):
    def __init__(
        self,
        kernel,
        x0: jax.Array,
        x1: jax.Array | None = None,
    ) -> None:
        super().__init__(kernel, x0, x1)
        self._kernel_matrix = self._compute_kernel_matrix()

    def _compute_kernel_matrix(self) -> LinearOperator:
        """Compute kernel matrix appropriately based on kernel type.

        Args:
            x_batch: Optional batch of points/domains

        Returns:
            K: Kernel matrix (batch_size, batch_size)
        """
        # self.context_points shape: (N_batch,) + input_shape
        x0 = self.x0
        x1 = self.x1

        def row_kernel(xi):
            return jax.lax.map(
                lambda xj: self.kernel(xi, xj),
                x1,
                batch_size=1,
            )

        return jax.lax.map(row_kernel, x0)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        """Compute matrix-vector product: K @ v.

        Args:
            v: Vector to multiply with (batch_size,)
            x_batch: Optional batch of points/domains

        Returns:
            K @ v: Result (batch_size,)
        """
        # Use either precomputed or newly computed kernel matrix
        if self._kernel_matrix is None:
            self._kernel_matrix = self._compute_kernel_matrix()

        return self._kernel_matrix @ vec

    def transpose(self):
        return ArrayKernel(
            kernel=lambda x, y: self.kernel(y, x), x0=self.x1, x1=self.x0
        )

    def todense(self):
        """Convert the kernel matrix to a dense format.

        Returns:
            Dense kernel matrix.
        """
        return jnp.asarray(self._kernel_matrix)


@lsqrt.dispatch
def _(a: ArrayKernel) -> jax.Array:
    _jitter = 1e-6 if a.dtype == jnp.float32 else 1e-10
    return jnp.linalg.cholesky(a.todense() + _jitter * jnp.eye(a.shape[0]))


class ToeplitzKernel(KernelOperator):
    def __init__(
        self,
        kernel,
        x0: jax.Array,
        x1: jax.Array | None = None,
    ) -> None:
        super().__init__(kernel, x0, x1)

        if x1 is not None and not jnp.allclose(x0, x1):
            msg = (
                "ToeplitzKernel requires x0 == x1 (symmetric case). "
                "For non-symmetric cases, use ArrayKernel instead."
            )
            raise ValueError(msg)

        self._toeplitz_vector = self._compute_toeplitz_vector()
        self._toeplitz_operator = Toeplitz(self._toeplitz_vector)

    def _compute_toeplitz_vector(self) -> jax.Array:
        return jax.vmap(lambda x: self.kernel(self.x0[0], x))(self.x0)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self._toeplitz_operator @ vec

    def transpose(self) -> "ToeplitzKernel":
        return ToeplitzKernel(
            kernel=lambda x, y: self.kernel(y, x), x0=self.x0, x1=None
        )

    def todense(self) -> jax.Array:
        return self._toeplitz_operator.todense()
