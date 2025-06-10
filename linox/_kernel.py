import jax
import jax.numpy as jnp
import numpy as np

from linox._linear_operator import LinearOperator
from linox._matrix import Matrix, Toeplitz
from linox.kernels.kernel import Kernel as BaseKernel


class Kernel(LinearOperator):
    def __init__(
        self, kernel: BaseKernel, x0: jax.Array, x1: jax.Array | None = None
    ) -> None:
        self._kernel = kernel
        self.toeplitz = False
        if x1 is None:
            self.toeplitz = False
            self.x1 = x0
        self.x0 = x0
        super().__init__(shape=(self.x0.shape[0], self.x1.shape[0]), dtype=x0.dtype)


class ArrayKernel(Kernel):
    def __init__(self, kernel: Kernel, x0: jax.Array, x1: jax.Array | None = None):
        super().__init__(kernel, x0, x1)
        if self.toeplitz:
            self._kernel_matrix = Toeplitz(v=jnp.array(kernel(x0[0:1], x0).reshape(-1)))
        else:
            self._kernel_matrix = self._compute_kernel_matrix()
        self._dtype = self.x0.dtype

    @property
    def kernel(self) -> BaseKernel:
        return self._kernel

    @property
    def shape(self):
        return (self.x0.shape[0], self.x1.shape[0])

    def _compute_kernel_matrix(self) -> Matrix:
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

    def __matmul__(self, v):  # noqa: ANN204
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

        return self._kernel_matrix @ v

    def transpose(self):
        return ArrayKernel(self.kernel, self.x1, self.x0)

    def todense(self):  # noqa: ANN202
        """Convert the kernel matrix to a dense format.

        Returns:
            Dense kernel matrix.
        """
        return jnp.asarray(self._kernel_matrix)

    def _matmul(self, other):  # noqa: ANN202
        return super()._matmul(other)

    def diagonal(self):  # noqa: ANN202
        D = np.min(self.shape)
        diag = jnp.zeros(D, dtype=self.dtype)

        def body_function(i, diag):  # noqa: ANN202
            vec = jnp.zeros(self.shape[1], dtype=self.dtype)
            vec = vec.at[i].set(1.0)
            diag_val = (self._kernel_matrix @ vec)[i]
            return diag.at[i].set(diag_val)

        diagonal = jax.lax.fori_loop(0, D, body_function, diag)

        return diagonal
