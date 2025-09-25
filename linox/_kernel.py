# _kernel.py

from collections.abc import Callable

import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator


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

        return jax.vmap(
            jax.vmap(
                self.kernel,
                in_axes=(None, 0),
                out_axes=0,
            ),
            in_axes=(0, None),
            out_axes=0,
        )(self.x0, self.x1)

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
