# test_kernel.py

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox as lo
from linox.typing import ShapeType

CaseType = tuple[lo.LinearOperator, jax.Array]

basic_shapes = [
    1,
    2,
    10,
    100,
]


def _inner_product_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
    return jnp.dot(x1, x2) + 1e-8


def sample_kernel(shape: ShapeType) -> CaseType:
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, shape)
    y = jax.random.normal(key, shape)
    linop = lo.ArrayKernel(_inner_product_kernel, x, y)
    matrix = jnp.dot(x, y.T)
    return linop, matrix


@pytest.mark.parametrize("shape", basic_shapes)
def case_kernel(shape: ShapeType) -> CaseType:
    linop, matrix = sample_kernel((shape, shape))
    return linop, matrix


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 22, 278]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[pytest.param(ncols, id=f"ncols{ncols}") for ncols in [1, 3, 5]],
)
def ncols(request: pytest.FixtureRequest) -> int:
    return request.param


# ============================================================================
# Basic Arithmetic Operations Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_to_dense(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix), "Dense matrix does not match"


@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_mv(linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey) -> None:
    vector = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop @ vector, matrix @ vector), (
        "MatVec does not match dense matmul"
    )


@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_shape(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    assert linop.shape == matrix.shape, "Shape does not match"
    assert linop.todense().shape == matrix.shape, "Dense shape does not match"
    assert linop.todense().shape == linop.shape, "Dense shape does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_matmat(
    linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop @ vec, matrix @ vec, atol=1e-6), (
        "MatVec does not match dense matmul"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_scalar_mul(
    linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix)


# ============================================================================
# Transpose Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_transpose(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    result_linop = linop.T
    expected_transposed = matrix.swapaxes(-1, -2)
    assert jnp.allclose(result_linop.todense(), expected_transposed)

    result_t = linop.T
    assert jnp.allclose(result_t.todense(), expected_transposed)

    result_transpose = linop.transpose()
    assert jnp.allclose(result_transpose.todense(), expected_transposed)


# ============================================================================
# Test Positive Definiteness
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_positive_definiteness(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    assert jnp.all(jnp.linalg.eigvals(matrix) >= 0), "Matrix is not positive definite"
    assert jnp.all(jnp.linalg.eigvals(linop.todense()) >= 0), (
        "Linop is not positive definite"
    )
