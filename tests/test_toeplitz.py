# test_toeplitz.py

import time

import jax
import jax.numpy as jnp
import pytest
import pytest_cases
from scipy.linalg import solve_toeplitz

import linox
from linox._toeplitz import Toeplitz

DType = jnp.float32
CaseType = tuple[linox.LinearOperator, jax.Array]
KeyType = jax.random.PRNGKey


jax.config.update("jax_enable_x64", True)


def sample_toeplitz(size: int) -> CaseType:
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    c = jax.random.normal(key1, (size,))

    op = Toeplitz(c)
    matrix = jnp.array(jax.scipy.linalg.toeplitz(c))

    assert op.shape == matrix.shape, "Shape mismatch"

    return op, matrix


@pytest.mark.parametrize("size", [8, 100, 1000])
def case_toeplitz(size: int) -> CaseType:
    linop, matrix = sample_toeplitz(size)
    return linop, matrix


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 22, 278]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


# ============================================================================
# Basic Arithmetic Operations Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_toeplitz)
def test_todense(linop: linox.Toeplitz, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix), "Dense matrix does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_toeplitz)
def test_mv(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    vector = jax.random.normal(key, matrix.shape[-1])
    assert jnp.allclose(linop @ vector, matrix @ vector, atol=1e-7)
