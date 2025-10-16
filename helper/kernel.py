# kernel.py

import abc

import jax
from jax import numpy as jnp


class Kernel(abc.ABC):  # noqa: D101
    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize kernel parameters."""
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
        pass

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


class L2InnerProductKernel(Kernel):  # noqa: D101
    def __init__(self, bias=1e-4):
        super().__init__()
        self.bias = bias

    def __call__(self, x1: jax.Array, x2: jax.Array | None = None) -> jax.Array:
        """Compute L² inner product kernel between x1 and x2."""
        if x2 is None:
            x2 = x1

        return jnp.sum(x1 * x2) + self.bias


class RBFKernel(Kernel):
    def __init__(self, lengthscale=1.0):
        super().__init__()
        self.lengthscale = lengthscale

    def __call__(self, x, y: jax.Array | None = None) -> jax.Array:
        """Compute RBF kernel between individual points"""
        if y is None:
            y = x

        sq_dist = jnp.sum((x - y) ** 2)

        return jnp.exp(-0.5 * sq_dist / self.lengthscale**2)
