# test_arithmetics.py

# test_arithmetics.py

# test_arithmetics.py

# test_arithmetics.py

"""Test for linear operator arithmetics."""

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from tests.test_linox_cases._matrix_cases import (
    case_add,
    case_add_operator,
    case_diagonal,
    case_identity,
    case_matmul,
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


CaseType = tuple[linox.LinearOperator, jax.Array]


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


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_to_dense(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_shape(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    assert linop.shape == matrix.shape, "Shape does not match"
    assert linop.todense().shape == matrix.shape, "Dense shape does not match"
    assert linop.todense().shape == linop.shape, "Dense shape does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_mv(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    vector = jax.random.normal(key, matrix.shape[-1])
    assert jnp.allclose(linop @ vector, matrix @ vector, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_matmat(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_scalar_mul(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=[case_add])
def test_add(
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 + linop2).todense(), matrix1 + matrix2, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=[case_matmul])
def test_lmatmul(
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 @ linop2).todense(), matrix1 @ matrix2, atol=1e-6)


@pytest_cases.parametrize_with_cases(
    "linop, matrix",
    cases=[case_product_operator],
)
def test_diagonal_product_matches_dense(
    linop: linox.LinearOperator,
    matrix: jax.Array,
) -> None:
    diag_linop = linox.diagonal(linop)
    expected = jnp.diag(matrix)
    assert jnp.allclose(diag_linop.todense(), expected, atol=1e-6)


def test_diagonal_isotropic_add_scaled_product_in_kronecker() -> None:
    mat_a = jnp.arange(4.0, dtype=jnp.float64).reshape(2, 2)
    mat_b = jnp.linspace(1.0, 4.0, num=4).reshape(2, 2)

    factor_left = linox.Matrix(mat_a)
    factor_right = linox.Matrix(mat_b)
    product = factor_left @ factor_right
    scaled_product = linox.ScaledLinearOperator(product, jnp.array(1.3))
    kron_wrapper = linox.Kronecker(scaled_product, linox.Matrix(jnp.ones((1, 1))))

    additive = linox.AddLinearOperator(kron_wrapper, linox.Matrix(jnp.eye(2)))
    iso = linox.IsotropicAdditiveLinearOperator(jnp.array(0.5), additive)

    result = linox.diagonal(iso)
    expected = jnp.diag(
        jnp.kron(1.3 * (mat_a @ mat_b), jnp.ones((1, 1)))
        + jnp.eye(2)
        + jnp.eye(2) * 0.5
    )
    assert jnp.allclose(result.todense(), expected, atol=1e-6), (
        "Diagonal does not match"
    )


# TODO(2bys): Add test for transpose.
def test_diagonal():
    mat = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    linop = linox.Matrix(mat)
    diag_linop = linox.diagonal(linop)
    expected = jnp.diag(mat)
    assert jnp.allclose(diag_linop.todense(), expected, atol=1e-6)
