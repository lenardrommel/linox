"""Tests for kernel operators with Kronecker products."""

import jax
import jax.numpy as jnp
import pytest

from linox import Kronecker
from linox._kernel import ArrayKernel, ToeplitzKernel, kernel_operator

jax.config.update("jax_enable_x64", True)


def rbf_kernel(x, y):
    return jnp.exp(-jnp.sum((x - y) ** 2))


class TestArrayKernelMatmul:
    def test_1d_vector(self):
        x0 = jnp.array([[0.0], [1.0], [2.0]])
        x1 = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        K = ArrayKernel(rbf_kernel, x0, x1)

        vec = jnp.ones((4,))
        result = K @ vec
        expected = K._todense() @ vec

        assert result.shape == (3,)
        assert jnp.allclose(result, expected)

    def test_2d_matrix(self):
        x0 = jnp.array([[0.0], [1.0], [2.0]])
        x1 = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        K = ArrayKernel(rbf_kernel, x0, x1)

        vec = jnp.ones((4, 2))
        result = K @ vec
        expected = K._todense() @ vec

        assert result.shape == (3, 2)
        assert jnp.allclose(result, expected)

    def test_with_kronecker(self):
        x_a = jnp.array([[0.0], [1.0]])
        x_b = jnp.array([[0.0], [1.0], [2.0], [3.0]])

        K_a = ArrayKernel(rbf_kernel, x_a, x_a)
        K_b = ArrayKernel(rbf_kernel, x_b, x_b)
        K_kron = Kronecker(K_a, K_b)

        vec = jnp.ones((K_kron.shape[1], 1))
        result = K_kron @ vec
        expected = K_kron._todense() @ vec

        assert result.shape == expected.shape
        assert jnp.allclose(result, expected, rtol=1e-5)

    def test_nested_kronecker(self):
        x0 = jnp.array([[0.0], [1.0]])
        x1 = jnp.array([[0.0], [1.0], [2.0]])
        x2 = jnp.array([[0.0], [1.0], [2.0], [3.0]])

        K0 = ArrayKernel(rbf_kernel, x0, x0)
        K1 = ArrayKernel(rbf_kernel, x1, x1)
        K2 = ArrayKernel(rbf_kernel, x2, x2)

        K_kron = Kronecker(Kronecker(K0, K1), K2)

        vec = jnp.ones((K_kron.shape[1], 1))
        result = K_kron @ vec
        expected = K_kron._todense() @ vec

        assert result.shape == expected.shape
        assert jnp.allclose(result, expected, rtol=1e-4)


class TestToeplitzKernelMatmul:
    def test_1d_vector(self):
        x = jnp.linspace(0, 1, 8).reshape(-1, 1)
        K = ToeplitzKernel(rbf_kernel, x)

        vec = jnp.ones((8,))
        result = K @ vec
        expected = K._todense() @ vec

        assert result.shape == (8,)
        assert jnp.allclose(result, expected)

    def test_2d_matrix(self):
        x = jnp.linspace(0, 1, 8).reshape(-1, 1)
        K = ToeplitzKernel(rbf_kernel, x)

        vec = jnp.ones((8, 3))
        result = K @ vec
        expected = K._todense() @ vec

        assert result.shape == (8, 3)
        assert jnp.allclose(result, expected)

    def test_with_kronecker(self):
        x_a = jnp.linspace(0, 1, 4).reshape(-1, 1)
        x_b = jnp.linspace(0, 1, 8).reshape(-1, 1)

        K_a = ToeplitzKernel(rbf_kernel, x_a)
        K_b = ToeplitzKernel(rbf_kernel, x_b)
        K_kron = Kronecker(K_a, K_b)

        vec = jnp.ones((K_kron.shape[1], 1))
        result = K_kron @ vec
        expected = K_kron._todense() @ vec

        assert result.shape == expected.shape
        assert jnp.allclose(result, expected, rtol=1e-5)


class TestMixedKernelKronecker:
    def test_array_and_toeplitz(self):
        x_arr = jnp.array([[0.0], [1.0], [2.0]])
        x_toe = jnp.linspace(0, 1, 4).reshape(-1, 1)

        K_arr = ArrayKernel(rbf_kernel, x_arr, x_arr)
        K_toe = ToeplitzKernel(rbf_kernel, x_toe)
        K_kron = Kronecker(K_arr, K_toe)

        vec = jnp.ones((K_kron.shape[1], 1))
        result = K_kron @ vec
        expected = K_kron._todense() @ vec

        assert result.shape == expected.shape
        assert jnp.allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
