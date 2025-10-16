# test_decompositions.py

import jax
import pytest
import pytest_cases
from jax import numpy as jnp

import linox
from linox.typing import ShapeLike, ShapeType
from linox.utils import as_dense

CaseType = tuple[linox.LinearOperator, jax.Array]
KeyType = jax.random.PRNGKey

symmetric_shapes = [
    (2, 2),
    (3, 3),
]

kron_shapes = [
    ((2, 2), (2, 2)),
    ((3, 3), (2, 2)),
    ((2, 2), (3, 3)),
]


# ============================================================================
# Helper functions
# ============================================================================


def sample_spd(shape: ShapeLike) -> CaseType:
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, shape)
    matrix = A @ A.T + jnp.eye(shape[0]) * 1e-3
    op = linox.utils.as_linop(matrix)
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


def sample_scaled(shape: ShapeLike, scalar: float) -> CaseType:
    op, matrix = sample_spd(shape)
    op = scalar * op
    matrix *= scalar
    return op, matrix


def sample_add(shape: ShapeLike) -> CaseType:
    op1, matrix1 = sample_spd(shape)
    op2, matrix2 = sample_spd(shape)
    op = op1 + op2
    matrix = matrix1 + matrix2
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


def sample_product(shape: ShapeLike) -> CaseType:
    op1, matrix1 = sample_spd(shape)
    op2, matrix2 = sample_spd(shape)
    op = op1 @ op2
    matrix = matrix1 @ matrix2
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


def sample_diagonal(shape: ShapeLike) -> CaseType:
    key = jax.random.PRNGKey(1)
    arr = jax.random.normal(key, shape)
    return linox.Diagonal(arr), linox._matrix._batch_jnp_diag(arr)


def sample_kronecker(shape: ShapeLike) -> CaseType:
    (shapeA, shapeB) = shape

    opA, matrixA = sample_spd(shapeA)
    opB, matrixB = sample_spd(shapeB)

    op = linox._kronecker.Kronecker(opA, opB)
    matrix = jnp.kron(matrixA, matrixB)

    assert op.shape == matrix.shape, "Shape mismatch"

    return op, matrix


def sample_isotropicadd(shape: ShapeLike, scalar: float) -> CaseType:
    op, matrix = sample_spd(shape)
    op = linox._isotropicadd.IsotropicAdditiveLinearOperator(scalar, op)
    matrix += jnp.eye(shape[0]) * scalar

    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


# ============================================================================
# Cases
# ============================================================================


@pytest.mark.parametrize("shape", symmetric_shapes)
def case_matrix(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_spd(shape)
    return linop, matrix


@pytest.mark.parametrize("shape", symmetric_shapes)
def case_add(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_add(shape)
    return linop, matrix


@pytest.mark.parametrize("shape", symmetric_shapes)
def case_product(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_product(shape)
    return linop, matrix


@pytest.mark.parametrize("shape", symmetric_shapes)
def case_scaled_linop(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_scaled(shape, scalar=2.0)
    return linop, matrix


@pytest.mark.parametrize("shape", symmetric_shapes)
def case_diagonal(
    shape: ShapeType,
) -> CaseType:
    linop, arr = sample_diagonal((shape[0],))
    return linop, arr


@pytest.mark.parametrize("shape", kron_shapes)
def case_kronecker(
    shape: ShapeType,
) -> CaseType:
    linop, matrix = sample_kronecker(shape)
    return linop, matrix


@pytest.mark.parametrize("shape", symmetric_shapes)
@pytest.mark.parametrize("scalar", [0.1, 1.0, 1e-8])
def case_isotropicadd(shape: ShapeType, scalar: float) -> CaseType:
    linop, matrix = sample_isotropicadd(shape, scalar=scalar)
    return linop, matrix


@pytest.mark.parametrize("shape", kron_shapes)
@pytest.mark.parametrize("scalar", [0.1, 1.0, 1e-8])
def case_isotropicadd_kron(shape: ShapeType, scalar: float) -> CaseType:
    linop, matrix = sample_kronecker(shape)
    linop = linox._isotropicadd.IsotropicAdditiveLinearOperator(scalar, linop)
    matrix += jnp.eye(matrix.shape[0]) * scalar
    return linop, matrix


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [2, 43, 257]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[pytest.param(ncols, id=f"ncols{ncols}") for ncols in [1, 3, 5]],
)
def ncols(request: pytest.FixtureRequest) -> int:
    return request.param


# ============================================================================
# Tests
# ============================================================================


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=[case_matrix, case_scaled_linop, case_product, case_add, case_isotropicadd],
)
def test_eigh(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    lin_eigvals, lin_eigvecs = linox.leigh(linop)
    assert jnp.allclose(eigvals, as_dense(lin_eigvals), atol=1e-5), (
        "Eigenvalues do not match"
    )
    for i in range(eigvecs.shape[1]):
        vec1 = eigvecs[:, i]
        vec2 = lin_eigvecs.todense()[:, i]
        if not jnp.allclose(vec1, vec2, atol=1e-5):
            vec2 = -vec2  # Account for sign ambiguity
        assert jnp.allclose(vec1, vec2, atol=1e-5), "Eigenvectors do not match"


@pytest_cases.parametrize_with_cases(
    "linop,matrix", cases=[case_kronecker, case_isotropicadd_kron]
)
def test_eigh_kronecker(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    eigvals = jnp.linalg.eigvalsh(matrix)
    lin_eigvals, eigvecs = linox.leigh(linop)
    assert jnp.allclose(eigvals, jnp.sort(as_dense(lin_eigvals)), atol=1e-5), (
        "Eigenvalues do not match"
    )
    D_lin = jnp.diag(as_dense(lin_eigvals))
    assert jnp.allclose(as_dense(eigvecs @ D_lin @ eigvecs.T), matrix, atol=1e-5), (
        "Eigen decomposition does not match"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_isotropicadd)
def test_eigh_isotropicadd(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    lin_eigvals, lin_eigvecs = linox.leigh(linop)
    assert jnp.allclose(eigvals, lin_eigvals.todense(), atol=1e-5), (
        "Eigenvalues do not match"
    )
    for i in range(eigvecs.shape[1]):
        vec1 = eigvecs[:, i]
        vec2 = lin_eigvecs.todense()[:, i]
        if not jnp.allclose(vec1, vec2, atol=1e-5):
            vec2 = -vec2
        assert jnp.allclose(vec1, vec2, atol=1e-5), "Eigenvectors do not match"

