"""Kernel-based linear operators with automatic structure detection.

Provides lazy kernel operators that avoid materializing full kernel matrices,
with automatic selection of Toeplitz structure for stationary kernels on uniform grids.
"""
# _kernel.py

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

from linox import config
from linox.operators.arithmetic import lsqrt
from linox.operators.base import LinearOperator
from linox.operators.toeplitz import Toeplitz

DENSE_THRESHOLD = 512


def _is_self_covariance_cheap(x0: jax.Array, x1: jax.Array | None) -> bool:
    """Cheap check for self-covariance (no device sync).

    Only returns True for obvious cases: x1 is None or x1 is x0 (identity).
    Does NOT do value comparison to avoid device->host sync.
    """
    return x1 is None or x1 is x0


def _is_uniform_1d_host(x: jax.Array, rtol: float = 1e-5) -> bool:
    """Check if points form a uniform 1D grid (host-side check).

    Uses device_get for small arrays only. For large arrays or JIT contexts,
    use assume_uniform=True instead.
    """
    import numpy as np

    if x.ndim == 2 and x.shape[1] == 1:
        x = x.ravel()
    if x.ndim != 1:
        return False
    n = x.shape[0]
    if n < 2:
        return True
    if n > 10000:
        return False
    x_np = np.asarray(x)
    diffs = np.diff(x_np)
    return bool(np.allclose(diffs, diffs[0], rtol=rtol))


def kernel_operator(
    kernel: Callable[[jax.Array, jax.Array], jax.Array],
    x0: jax.Array,
    x1: jax.Array | None = None,
    is_stationary: bool = False,
    assume_uniform: bool = False,
    chunk_size: int = 256,
) -> "KernelOperator":
    """Create the optimal kernel operator based on structure detection.

    Automatically selects ToeplitzKernel when:
    - Self-covariance (x1 is None or x1 is x0 - identity check only, no value comparison)
    - Points are uniform 1D grid (assume_uniform=True or host-side check for small n)
    - Kernel is stationary (is_stationary=True)

    Otherwise creates a lazy ArrayKernel that never materializes the full matrix.

    Parameters
    ----------
    kernel : Callable[[jax.Array, jax.Array], jax.Array]
        Kernel function k(x, y) -> scalar
    x0 : jax.Array
        First set of points
    x1 : jax.Array, optional
        Second set of points (None for self-covariance)
    is_stationary : bool, default=False
        True if kernel is stationary k(x,y) = f(x-y)
    assume_uniform : bool, default=False
        True to skip uniformity check (use when creating grid with arange)
    chunk_size : int, default=256
        Chunk size for lazy matmul computation

    Returns
    -------
    KernelOperator
        Either ToeplitzKernel or ArrayKernel depending on structure
    """
    is_self_cov = _is_self_covariance_cheap(x0, x1)
    is_uniform = assume_uniform or _is_uniform_1d_host(x0)

    if is_self_cov and is_uniform and is_stationary:
        return ToeplitzKernel(kernel, x0, chunk_size=chunk_size)

    return ArrayKernel(kernel, x0, x1, chunk_size=chunk_size)


class KernelOperator(LinearOperator):
    """Base class for kernel-based linear operators."""

    def __init__(
        self,
        kernel: Callable[[jax.Array, jax.Array], jax.Array],
        x0: jax.Array,
        x1: jax.Array | None = None,
    ) -> None:
        self.kernel = kernel
        self.x0 = x0
        self.x1 = x0 if x1 is None else x1
        super().__init__(shape=(self.x0.shape[0], self.x1.shape[0]), dtype=x0.dtype)


# Legacy aliases
KernelLinearOperator = KernelOperator
Kernel = KernelOperator


class ArrayKernel(KernelOperator):
    """Lazy kernel operator that NEVER materializes the full matrix.

    All matrix-vector products are computed on-the-fly using lax.map
    for JIT compatibility. This operator is designed for large-scale
    problems where the full kernel matrix would exceed memory.

    WARNING: Calling _todense() on large operators will cause OOM.

    Args:
        kernel: Kernel function k(x, y) -> scalar
        x0: First set of points (n0, d)
        x1: Second set of points (n1, d), defaults to x0 if None
        chunk_size: Chunk size for chunked computation
    """

    def __init__(
        self,
        kernel: Callable[[jax.Array, jax.Array], jax.Array],
        x0: jax.Array,
        x1: jax.Array | None = None,
        chunk_size: int = 256,
    ) -> None:
        super().__init__(kernel, x0, x1)
        self.chunk_size = chunk_size

    def _matmul(self, vec: jax.Array) -> jax.Array:
        """Matrix-free matmul using lax.map (JIT-compatible).

        Computes K @ v row-by-row without ever building K.
        Uses lax.map for JIT compatibility.
        """
        x0 = self.x0
        x1 = self.x1

        kernel_row_fn = jax.vmap(self.kernel, in_axes=(None, 0))

        def compute_row_dot(xi):
            row = kernel_row_fn(xi, x1)
            return jnp.dot(row, vec)

        return lax.map(compute_row_dot, x0)

    def transpose(self) -> "ArrayKernel":
        """Return transposed kernel operator."""
        return ArrayKernel(
            kernel=lambda x, y: self.kernel(y, x),
            x0=self.x1,
            x1=self.x0,
            chunk_size=self.chunk_size,
        )

    def _todense(self) -> jax.Array:
        """Materialize the full kernel matrix.

        WARNING: This will cause OOM for large operators.
        Only use for debugging with small n or when explicitly needed.
        """
        n0, n1 = self.shape
        if n0 > DENSE_THRESHOLD or n1 > DENSE_THRESHOLD:
            config.warn(
                f"Densifying large kernel ({n0}x{n1}). This may cause OOM. "
                "Consider using matrix-free operations instead."
            )

        x0 = self.x0
        x1 = self.x1

        kernel_fn = jax.vmap(
            jax.vmap(self.kernel, in_axes=(None, 0)),
            in_axes=(0, None),
        )

        return kernel_fn(x0, x1)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = (self.kernel, self.x0, self.x1)
        aux_data = {"chunk_size": self.chunk_size}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ArrayKernel":
        """Unflatten for JAX pytree registration."""
        kernel, x0, x1 = children
        return cls(kernel=kernel, x0=x0, x1=x1, chunk_size=aux_data["chunk_size"])


jax.tree_util.register_pytree_node_class(ArrayKernel)


@lsqrt.dispatch
def _(a: ArrayKernel) -> jax.Array:
    jitter = 1e-6 if a.dtype == jnp.float32 else 1e-10
    return jnp.linalg.cholesky(a.todense() + jitter * jnp.eye(a.shape[0]))


class ToeplitzKernel(KernelOperator):
    """Kernel operator using Toeplitz structure for O(n log n) matmul via FFT.

    This operator is completely matrix-free - it only stores the Toeplitz vector
    and uses FFT-based multiplication. Suitable for:
    - Self-covariance (x1 == x0)
    - Uniform 1D grid
    - Stationary kernel k(x,y) = f(x-y)

    Note: Since we use symmetric Toeplitz matrices, transpose() returns self.

    Args:
        kernel: Stationary kernel function
        x0: Points array (n,) or (n, 1) - must be uniform 1D grid
        chunk_size: Chunk size (for compatibility, not used in Toeplitz matmul)
    """

    def __init__(
        self,
        kernel: Callable[[jax.Array, jax.Array], jax.Array],
        x0: jax.Array,
        x1: jax.Array | None = None,
        chunk_size: int = 256,
    ) -> None:
        if x1 is not None and not _is_self_covariance_cheap(x0, x1):
            msg = (
                "ToeplitzKernel requires self-covariance (x1 must be None or x1 is x0)"
            )
            raise ValueError(msg)

        super().__init__(kernel, x0, None)
        self.chunk_size = chunk_size
        self._toeplitz_vector = self._compute_toeplitz_vector()
        self._toeplitz_op = Toeplitz(self._toeplitz_vector)

    def _compute_toeplitz_vector(self) -> jax.Array:
        """Compute the first row of the Toeplitz matrix."""
        x0 = self.x0.ravel() if self.x0.ndim == 2 else self.x0
        x0_reshaped = x0.reshape(-1, 1) if x0.ndim == 1 else x0
        return jax.vmap(lambda xi: self.kernel(x0_reshaped[0], xi))(x0_reshaped)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        """FFT-based O(n log n) matmul - completely matrix-free."""
        return self._toeplitz_op @ vec

    def transpose(self) -> "ToeplitzKernel":
        """Symmetric Toeplitz: transpose is self."""
        return self

    def _todense(self) -> jax.Array:
        """Materialize via Toeplitz operator.

        WARNING: This defeats the purpose of using ToeplitzKernel.
        """
        n = self.shape[0]
        if n > DENSE_THRESHOLD:
            config.warn(
                f"Densifying large ToeplitzKernel ({n}x{n}). This may cause OOM."
            )
        return self._toeplitz_op._todense()

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = (self.kernel, self.x0)
        aux_data = {"chunk_size": self.chunk_size}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ToeplitzKernel":
        """Unflatten for JAX pytree registration."""
        kernel, x0 = children
        return cls(kernel=kernel, x0=x0, chunk_size=aux_data["chunk_size"])


jax.tree_util.register_pytree_node_class(ToeplitzKernel)
