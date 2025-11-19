import abc
import math

import jax
import jax.numpy as jnp


class Kernel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, learnable: bool) -> None:
        """Initialize kernel parameters."""
        self.learnable = learnable
        self.__kernel_name = self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, x, y):
        """Compute the kernel function between two sets of inputs.

        Args:
            x1: First input array of shape (n, d).
            x2: Second input array of shape (m, d).

        Returns:
            Kernel matrix of shape (n, m).
        """

    @property
    def params(self):
        """Get a dictionary of the kernel's parameters."""
        if not self.learnable:
            return {}

        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and k != "learnable"
        }


class L2InnerProductKernel(Kernel):
    def __init__(self, learnable: bool = False, bias=1e-4) -> None:
        super().__init__(learnable=learnable)
        self.bias = bias

    def __call__(self, x1: jax.Array, x2: jax.Array | None = None) -> jax.Array:
        """Compute L² inner product kernel between x1 and x2."""
        if x2 is None:
            x2 = x1

        return jnp.sum(x1 * x2) + self.bias


class RBFKernel(Kernel):
    def __init__(self, learnable: bool = True, lengthscale=1.0) -> None:
        super().__init__(learnable=learnable)
        self.lengthscale = lengthscale

    def __call__(self, x, y: jax.Array | None = None) -> jax.Array:
        """Compute RBF kernel between individual points."""
        if y is None:
            y = x

        sq_dist = jnp.sum((x - y) ** 2)

        return jnp.exp(-0.5 * sq_dist / self.lengthscale**2)


class Matern32Kernel(Kernel):
    def __init__(self, learnable: bool = True, lengthscale=1.0) -> None:
        super().__init__(learnable=learnable)
        self.lengthscale = lengthscale
        self.nu = 1.5
        self._sqrt3 = math.sqrt(3)

    def __call__(self, x, y):
        """Compute Matern 3/2 kernel between individual points."""
        if y is None:
            y = x

        d = jnp.linalg.norm(x - y)
        k = (1 + self._sqrt3 * d / self.lengthscale) * jnp.exp(
            -self._sqrt3 * d / self.lengthscale
        )

        return k
