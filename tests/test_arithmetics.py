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
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [2, 43, 257]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:  # noqa: D103
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[pytest.param(ncols, id=f"ncols{ncols}") for ncols in [1, 3, 5]],
)
def ncols(request: pytest.FixtureRequest) -> int:  # noqa: D103
    return request.param


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_to_dense(linop: linox.LinearOperator, matrix: jax.Array) -> None:  # noqa: D103
    assert jnp.allclose(linop.todense(), matrix)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_mv(  # noqa: D103
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    vector = jax.random.normal(key, matrix.shape[-1])
    assert jnp.allclose(linop @ vector, matrix @ vector, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_matmat(  # noqa: D103
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_scalar_mul(  # noqa: D103
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=[case_add])
def test_add(  # noqa: D103
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 + linop2).todense(), matrix1 + matrix2, atol=1e-7)


@pytest_cases.parametrize_with_cases("linop1, linop2", cases=[case_matmul])
def test_lmatmul(  # noqa: D103
    linop1: CaseType,
    linop2: CaseType,
) -> None:
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 @ linop2).todense(), matrix1 @ matrix2, atol=1e-6)


# TODO(2bys): Add test for transpose.
