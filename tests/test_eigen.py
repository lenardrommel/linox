# test_eigen.py

import jax
import jax.numpy as jnp
import pytest

import linox
from linox._eigen import EigenD
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Matrix
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


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 22, 278]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture
def square_spd_eigend(key: jax.random.PRNGKey) -> tuple[EigenD, jax.Array]:
    size = 5
    A = jax.random.normal(key, (size, size))
    A = A @ A.T + jnp.eye(size) * 1e-6
    eigenvalues, eigenvectors = linox.leigh(Matrix(A))
    eigend = EigenD(eigenvectors, Diagonal(eigenvalues))
    return eigend, A


def test_eigend_todense(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    assert jnp.allclose(eigend.todense(), matrix), "Dense matrix does not match"


def test_eigend_shape(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    assert eigend.shape == matrix.shape, "Shape does not match"


def test_eigend_matmul(
    square_spd_eigend: tuple[EigenD, jax.Array], key: jax.random.PRNGKey
) -> None:
    eigend, matrix = square_spd_eigend
    vec = jax.random.normal(key, (matrix.shape[1],))
    result = eigend @ vec
    expected = matrix @ vec
    assert jnp.allclose(result, expected), "Matmul does not match"


def test_eigend_inverse(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    inv_eigend = linox.linverse(eigend)
    result = inv_eigend.todense()
    expected = jnp.linalg.inv(matrix)
    assert jnp.allclose(result, expected, atol=1e-6), "Inverse does not match"


def test_eigend_sqrt(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    sqrt_eigend = linox.lsqrt(eigend)
    result = (sqrt_eigend @ sqrt_eigend.T).todense()
    assert jnp.allclose(result, matrix, atol=1e-6), "Sqrt does not match"


def test_eigend_cholesky(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    L = linox.lcholesky(eigend)
    result = (L @ L.T).todense()
    assert jnp.allclose(result, matrix, atol=1e-6), "Cholesky does not match"


def test_eigend_leigh(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, _ = square_spd_eigend
    eigenvalues, eigenvectors = linox.leigh(eigend)
    assert jnp.allclose(eigend.eigenvalues, eigenvalues), "Eigenvalues do not match"
    assert jnp.allclose(eigenvectors.todense(), eigend.Q.todense()), (
        "Eigenvectors do not match"
    )


def test_eigend_diagonal(square_spd_eigend: tuple[EigenD, jax.Array]) -> None:
    eigend, matrix = square_spd_eigend
    result = linox.diagonal(eigend)
    expected = jnp.diag(matrix)
    assert jnp.allclose(result, expected, atol=1e-6), "Diagonal does not match"
