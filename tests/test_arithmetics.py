"""Test for linear operator arithmetics"""

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from tests.test_linox_cases._linops_cases import (
    case_add_combinations,
    case_identity,
    case_matrix,
    case_mul_combinations,
    case_symmetric_combinations,
    case_zero,
)

# # create list of possible test_cases and iterate over them wit
# case_modules = [
#     ".test_linox_cases." + path.stem
#     for path in (pathlib.Path(__file__).parent / "test_linox_cases").glob("*_cases.py")
# ]

special_linops = [
    case_matrix,
    case_identity,
    case_zero,
]


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def key(request) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[pytest.param(ncols, id=f"ncols{ncols}") for ncols in [1, 2, 3, 4, 5]],
)
def ncols(request) -> int:
    return request.param


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_to_dense(linop, matrix):
    assert jnp.allclose(linop.todense(), matrix)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_mv(linop, matrix, key):
    vector = jax.random.normal(key, matrix.shape[1])
    assert jnp.allclose(linop.mv(vector), linop @ vector)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_matmat(linop, matrix, key, ncols):
    vector = jax.random.normal(key, (matrix.shape[1], ncols))
    assert jnp.allclose(linop @ vector, linop @ vector)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_scalar_mul(linop, matrix, key):
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix)


@pytest_cases.parametrize_with_cases(
    "linop1, linop2", cases=[case_add_combinations, case_symmetric_combinations]
)
def test_add(
    linop1,
    linop2,
):
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 + linop2).todense(), matrix1 + matrix2)


@pytest_cases.parametrize_with_cases(
    "linop1, linop2", cases=[case_add_combinations, case_symmetric_combinations]
)
def test_sub(
    linop1,
    linop2,
):
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 - linop2).todense(), matrix1 - matrix2)


@pytest_cases.parametrize_with_cases(
    "linop1, linop2", cases=[case_mul_combinations, case_symmetric_combinations]
)
def test_mul(
    linop1,
    linop2,
):
    linop1, matrix1 = linop1
    linop2, matrix2 = linop2
    assert jnp.allclose((linop1 @ linop2).todense(), matrix1 @ matrix2)


@pytest_cases.parametrize_with_cases("linop, matrix", cases=special_linops)
def test_transpose(linop, matrix):
    assert jnp.allclose(linop.T.todense(), matrix.T)


# TODO(2bys): Add test for inverse.
