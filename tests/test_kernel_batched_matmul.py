"""Regression tests: ArrayKernel must follow the batched matmul convention.

``ArrayKernel._matmul`` used to return ``(n0, *batch, k)`` for a batched operand
``(*batch, n1, k)``. ``Kronecker._matmul`` passes such batched operands to its
factors, so any Kronecker product with an ArrayKernel factor returned wrong
shapes, or wrong values when the shapes happened to line up.
"""

import jax
import jax.numpy as jnp
import pytest

import linox as lo

jax.config.update("jax_enable_x64", True)


def _rbf_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
    diff = x1 - x2
    return jnp.exp(-0.5 * jnp.dot(diff, diff))


def _array_kernel(n: int, seed: int = 0) -> lo.ArrayKernel:
    x = jax.random.normal(jax.random.PRNGKey(seed), (n, 3))
    return lo.ArrayKernel(_rbf_kernel, x)


def _dense_matrix(n: int, seed: int) -> lo.Matrix:
    a = jax.random.normal(jax.random.PRNGKey(seed), (n, n))
    return lo.Matrix(a @ a.T + n * jnp.eye(n))


def test_array_kernel_batched_operand_matches_dense():
    A = _array_kernel(6)
    V = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 3))
    out = A @ V
    assert out.shape == (4, 6, 3)
    assert jnp.allclose(out, A.todense() @ V)


@pytest.mark.parametrize("shape", [(6,), (6, 1), (6, 5)])
def test_array_kernel_unbatched_operands_unchanged(shape):
    A = _array_kernel(6)
    v = jax.random.normal(jax.random.PRNGKey(2), shape)
    out = A @ v
    assert out.shape == shape
    assert jnp.allclose(out, A.todense() @ v)


@pytest.mark.parametrize("array_kernel_first", [True, False])
@pytest.mark.parametrize("ncols", [1, 3, 8, 16])
def test_kronecker_with_array_kernel_factor_matches_dense(array_kernel_first, ncols):
    # ncols == 8 equals the ArrayKernel size: shapes used to line up there, but
    # the values were wrong.
    K_arr, M = _array_kernel(8), _dense_matrix(5, seed=3)
    K = lo.Kronecker(K_arr, M) if array_kernel_first else lo.Kronecker(M, K_arr)
    dense = K.A.todense(), K.B.todense()
    dense = jnp.kron(*dense)
    R = jax.random.normal(jax.random.PRNGKey(ncols), (40, ncols))
    out = K @ R
    assert out.shape == (40, ncols)
    assert jnp.allclose(out, dense @ R)


def test_kronecker_with_array_kernel_factor_vector_and_jit():
    K = lo.Kronecker(_array_kernel(8), _dense_matrix(5, seed=4))
    dense = jnp.kron(K.A.todense(), K.B.todense())
    v = jax.random.normal(jax.random.PRNGKey(5), (40,))
    assert (K @ v).shape == (40,)
    assert jnp.allclose(K @ v, dense @ v)
    out = jax.jit(lambda op, r: op @ r)(K, v)
    assert jnp.allclose(out, dense @ v)


def test_isotropic_additive_kronecker_todense():
    # The original failure: densifying noise * I + Kronecker(ArrayKernel, ...).
    K = lo.Kronecker(_array_kernel(8), _dense_matrix(5, seed=6))
    op = lo.IsotropicAdditiveLinearOperator(jnp.array(1e-3), K)
    expected = jnp.kron(K.A.todense(), K.B.todense()) + 1e-3 * jnp.eye(40)
    assert jnp.allclose(op.todense(), expected)
