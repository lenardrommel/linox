# test_eigen.py

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox._eigen import EigenD, leigh
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix
from tests.test_linox_cases._matrix_cases import (
    case_add_operator,
    case_diagonal,
    case_identity,
    case_matrix,
    case_ones,
    case_product_operator,
    case_scaled_operator,
    case_transposed_operator,
    case_zero,
)

special_linops = [
    case_matrix,
    case_identity,
    case_zero,
    case_ones,
    case_diagonal,
    case_add_operator,
    case_scaled_operator,
    case_product_operator,
    case_transposed_operator,
]

DType = jnp.float32
CaseType = tuple[linox.LinearOperator, jax.Array]
KeyType = jax.random.PRNGKey


jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures
# ============================================================================


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


@pytest.fixture
def square_spd_matrix(key: jax.random.PRNGKey) -> tuple[LinearOperator, jax.Array]:
    """Generate a square symmetric positive definite matrix for testing."""
    size = 5
    A = jax.random.normal(key, (size, size))
    A = A @ A.T + jnp.eye(size) * 1e-6
    op = leigh(Matrix(A))
    matrix = A
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


# ============================================================================
# Basic Arithmetic Operations Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=square_spd_matrix)
def test_to_dense(linop: EigenD, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix), "Dense matrix does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=square_spd_matrix)
def test_shape(linop: EigenD, matrix: jax.Array) -> None:
    assert linop.shape == matrix.shape, "Shape does not match"
    assert linop.todense().shape == matrix.shape, "Dense shape does not match"
    assert linop.todense().shape == linop.shape, "Dense shape does not match"
