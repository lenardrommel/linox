# _kernel.py

from collections.abc import Callable

import jax
import jax.numpy as jnp

from linox._arithmetic import leigh, lsqrt
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix
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

    def _compute_kernel_matrix(self) -> jax.Array:
        """Compute kernel matrix using chunked vmap for efficiency.

        Uses vmap over both dimensions with row chunking to balance memory
        usage and speed. This is much faster than nested lax.map with batch_size=1.

        Returns
        -------
        K : jax.Array
            Kernel matrix of shape (n0, n1)
        """
        x0 = self.x0
        x1 = self.x1

        kernel_fn = jax.vmap(
            jax.vmap(self.kernel, in_axes=(None, 0)),
            in_axes=(0, None),
        )

        n0 = x0.shape[0]
        chunk_size = min(256, n0)

        if n0 <= chunk_size:
            return kernel_fn(x0, x1)

        rows = []
        for i in range(0, n0, chunk_size):
            end_i = min(i + chunk_size, n0)
            row_chunk = kernel_fn(x0[i:end_i], x1)
            rows.append(row_chunk)
        return jnp.concatenate(rows, axis=0)

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

    def _todense(self):
        """Convert the kernel matrix to a dense format.

        Returns:
            Dense kernel matrix.
        """
        return jnp.asarray(self._kernel_matrix)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.kernel, self.x0, self.x1)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ArrayKernel":
        del aux_data
        kernel, x0, x1 = children
        return cls(kernel=kernel, x0=x0, x1=x1)


@lsqrt.dispatch
def _(a: ArrayKernel) -> jax.Array:
    _jitter = 1e-6 if a.dtype == jnp.float32 else 1e-10
    return jnp.linalg.cholesky(a._todense() + _jitter * jnp.eye(a.shape[0]))


# Register ArrayKernel as a PyTree
jax.tree_util.register_pytree_node_class(ArrayKernel)


class LazyArrayKernel(KernelOperator):
    """Lazy kernel operator that doesn't materialize the full matrix.

    Unlike ArrayKernel, this class computes kernel evaluations on-the-fly
    during matrix-vector products, enabling operations on matrices that
    would be too large to fit in memory.

    Use this for large-scale problems (e.g., 80k+ points) where the full
    kernel matrix would exceed GPU memory.
    """

    def __init__(
        self,
        kernel,
        x0: jax.Array,
        x1: jax.Array | None = None,
        chunk_size: int = 256,
    ) -> None:
        super().__init__(kernel, x0, x1)
        self.chunk_size = chunk_size

    def _matmul(self, vec: jax.Array) -> jax.Array:
        """Compute K @ v without materializing K.

        Uses chunked computation to control memory usage.
        """
        x0 = self.x0
        x1 = self.x1
        n0 = x0.shape[0]
        chunk_size = min(self.chunk_size, n0)

        kernel_row_fn = jax.vmap(self.kernel, in_axes=(None, 0))

        def compute_row_dot(xi):
            row = kernel_row_fn(xi, x1)
            return jnp.dot(row, vec)

        if n0 <= chunk_size:
            return jax.vmap(compute_row_dot)(x0)

        results = []
        for i in range(0, n0, chunk_size):
            end_i = min(i + chunk_size, n0)
            chunk_result = jax.vmap(compute_row_dot)(x0[i:end_i])
            results.append(chunk_result)
        return jnp.concatenate(results, axis=0)

    def transpose(self) -> "LazyArrayKernel":
        return LazyArrayKernel(
            kernel=lambda x, y: self.kernel(y, x),
            x0=self.x1,
            x1=self.x0,
            chunk_size=self.chunk_size,
        )

    def _todense(self) -> jax.Array:
        """Materialize the full kernel matrix (caution: may OOM for large n)."""
        x0 = self.x0
        x1 = self.x1
        n0 = x0.shape[0]
        chunk_size = min(self.chunk_size, n0)

        kernel_fn = jax.vmap(
            jax.vmap(self.kernel, in_axes=(None, 0)),
            in_axes=(0, None),
        )

        if n0 <= chunk_size:
            return kernel_fn(x0, x1)

        rows = []
        for i in range(0, n0, chunk_size):
            end_i = min(i + chunk_size, n0)
            row_chunk = kernel_fn(x0[i:end_i], x1)
            rows.append(row_chunk)
        return jnp.concatenate(rows, axis=0)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.kernel, self.x0, self.x1)
        aux_data = {"chunk_size": self.chunk_size}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "LazyArrayKernel":
        kernel, x0, x1 = children
        return cls(kernel=kernel, x0=x0, x1=x1, chunk_size=aux_data["chunk_size"])


jax.tree_util.register_pytree_node_class(LazyArrayKernel)


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

    def _todense(self) -> jax.Array:
        return self._toeplitz_operator._todense()

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.kernel, self.x0, self.x1)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ToeplitzKernel":
        del aux_data
        kernel, x0, x1 = children
        return cls(kernel=kernel, x0=x0, x1=x1)


# Register ToeplitzKernel as a PyTree
jax.tree_util.register_pytree_node_class(ToeplitzKernel)


@leigh.dispatch
def _(a: LazyArrayKernel, max_iter: int = 100) -> tuple[jax.Array, "Matrix"]:
    """Eigendecomposition of LazyArrayKernel using randomized methods.

    For large kernels where full eigendecomposition is infeasible, we use
    randomized SVD which only requires matrix-vector products.
    """
    n = a.shape[0]
    dtype = a.dtype

    if n <= 4096:
        K_dense = a._todense()
        jitter = 1e-6 if dtype == jnp.float32 else 1e-10
        K_dense = K_dense + jitter * jnp.eye(n, dtype=dtype)
        w, Q = jnp.linalg.eigh(K_dense)
        return w, Matrix(Q)

    key = jax.random.key(0)
    k = min(n - 1, max(1000, n // 10))
    oversampling = min(10, n - k)

    Omega = jax.random.normal(key, (n, k + oversampling), dtype=dtype)
    Y = a @ Omega
    Q, _ = jnp.linalg.qr(Y)

    for _ in range(2):
        Z = a @ Q
        Q, _ = jnp.linalg.qr(Z)

    B = Q.T @ (a @ Q)
    jitter = 1e-6 if dtype == jnp.float32 else 1e-10
    B = B + jitter * jnp.eye(B.shape[0], dtype=dtype)
    w_small, V_small = jnp.linalg.eigh(B)

    w = w_small
    Q_full = Q @ V_small

    return w, Matrix(Q_full)
