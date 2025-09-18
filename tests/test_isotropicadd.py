import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox._isotropicadd import IsotropicAdditiveLinearOperator

CaseType = tuple[linox.Kronecker, jax.Array]
jax.config.update("jax_enable_x64", True)
from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import pytest

from linox.types import ShapeType

DType = jnp.float32
CaseType = tuple[linox.LinearOperator, jax.Array]
KeyType = jax.random.PRNGKey


basic_shapes = [
    (2, 2),
    (3, 3),
    (4, 4),
]


def sample_isotropic_additive(shape: ShapeType) -> CaseType:
    key = jax.random.PRNGKey(1)
    arr = jax.random.normal(key, shape)
    M = linox.Matrix(arr)
    scalar = jax.random.normal(jax.random.PRNGKey(2), ())
    isotropicadd = IsotropicAdditiveLinearOperator(scalar, M)
    return isotropicadd, scalar * jnp.eye(shape[-1]) + arr


@pytest.mark.parametrize("shape", basic_shapes)
def case_isotropic_additive(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_isotropic_additive(shape)
    return linop, matrix


@pytest.mark.parametrize("shape", basic_shapes)
def case_add(
    shape: ShapeType,
) -> CaseType:
    linop1 = sample_isotropic_additive(shape)
    linop2 = sample_isotropic_additive(shape)
    return linop1, linop2


@pytest.mark.parametrize("shape", basic_shapes)
def case_matmul(
    shape: ShapeType,
) -> CaseType:
    linop1 = sample_isotropic_additive(shape)
    linop2 = sample_isotropic_additive(shape)
    return linop1, linop2


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
def square_spd_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a square symmetric positive definite matrix for testing."""
    size = 4
    A = jax.random.normal(key, (size, size))
    return A @ A.T + jnp.eye(size) * 1e-6  # Ensure positive definiteness


@pytest.fixture
def square_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a square matrix for testing."""
    size = 4
    return jax.random.normal(key, (size, size))


# ============================================================================
# Basic Arithmetic Operations Tests
# ============================================================================


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_isotropic_additive)
def test_todense(
    linop: linox.LinearOperator,
    matrix: jax.Array,
) -> None:
    assert jnp.allclose(linop.todense(), matrix, atol=1e-6), (
        f"Linop:\n{linop.todense()}\nMatrix:\n{matrix}"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_isotropic_additive)
def test_mv(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    vector = jax.random.normal(key, matrix.shape[-1])
    assert jnp.allclose(linop @ vector, matrix @ vector, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_isotropic_additive)
def test_matmat(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_isotropic_additive)
def test_scalar_mul(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=case_add)
def test_add(
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 + linop2).todense(), matrix1 + matrix2, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=case_matmul)
def test_lmatmul(
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 @ linop2).todense(), matrix1 @ matrix2, atol=1e-6)


# ============================================================================
# Function Tests
# ============================================================================
