# test_kronecker.py

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox._kronecker import Kronecker
from tests.test_linox_cases._kronecker_cases import (
    case_add,
    case_kronecker,
    case_matmul,
)

CaseType = tuple[linox.Kronecker, jax.Array]
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
def square_spd_kronecker(key: jax.random.PRNGKey) -> tuple[Kronecker, jax.Array]:
    """Generate a square symmetric positive definite matrix for testing."""
    sizeA = 2
    sizeB = 2
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (sizeA, sizeA))
    A = A @ A.T + jnp.eye(sizeA) * 1e-6
    B = jax.random.normal(key2, (sizeB, sizeB))
    B = B @ B.T + jnp.eye(sizeB) * 1e-6
    op = Kronecker(A, B)
    matrix = jnp.kron(A, B)
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


@pytest.fixture
def square_spd_nested_kronecker(key: jax.random.PRNGKey) -> tuple[Kronecker, jax.Array]:
    """Generate a square symmetric positive definite matrix for testing."""
    sizeA = 4
    sizeB = 3
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (sizeA, sizeA))
    A = A @ A.T + jnp.eye(sizeA)
    B = jax.random.normal(key2, (sizeB, sizeB))
    B = B @ B.T + jnp.eye(sizeB)
    op = Kronecker(Kronecker(A, A), Kronecker(B, B))
    matrix = jnp.kron(jnp.kron(A, A), jnp.kron(B, B))
    assert op.shape == matrix.shape, "Shape mismatch"
    assert jnp.allclose(op.todense(), matrix), "Dense matrix does not match"
    return op, matrix


@pytest.fixture
def square_kronecker(key: jax.random.PRNGKey) -> tuple[Kronecker, jax.Array]:
    """Generate a square matrix for testing."""
    sizeA = 4
    sizeB = 3
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (sizeA, sizeA))
    B = jax.random.normal(key2, (sizeB, sizeB))
    op = Kronecker(A, B)
    matrix = jnp.kron(A, B)
    assert op.shape == matrix.shape, "Shape mismatch"
    return op, matrix


# ============================================================================
# Basic Arithmetic Operations Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
def test_to_dense(linop: linox.Kronecker, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix), "Dense matrix does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
def test_shape(linop: linox.Kronecker, matrix: jax.Array) -> None:
    assert linop.shape == matrix.shape, "Shape does not match"
    assert linop.todense().shape == matrix.shape, "Dense shape does not match"
    assert linop.todense().shape == linop.shape, "Dense shape does not match"


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
def test_mv(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    vector = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop @ vector, matrix @ vector), (
        "MatVec does not match dense matmul"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
def test_matmat(
    linop: linox.LinearOperator, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop @ vec, matrix @ vec, atol=1e-6), (
        "MatVec does not match dense matmul"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
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


# ============================================================================
# Transpose Tests
# ============================================================================
@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kronecker])
def test_transpose(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    """Test transpose operation."""
    result_linop = linox.transpose(linop)
    expected_transposed = matrix.swapaxes(-1, -2)
    assert jnp.allclose(result_linop.todense(), expected_transposed)

    result_t = linop.T
    assert jnp.allclose(result_t.todense(), expected_transposed)

    result_transpose = linop.transpose()
    assert jnp.allclose(result_transpose.todense(), expected_transposed)


# ============================================================================
# Special Linear Operator Class Tests
# ============================================================================


def test_inverse(square_spd_nested_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_nested_kronecker
    linop_inv = linox.linverse(linop)
    matrix_inv = jnp.linalg.inv(matrix)
    assert jnp.allclose(linop_inv.todense(), matrix_inv, atol=1e-6), (
        "Inverse does not match"
    )
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop_inv @ vec, matrix_inv @ vec, atol=1e-6), (
        "Inverse matvec does not match"
    )


def test_pinverse(
    square_spd_nested_kronecker: tuple[Kronecker, jax.Array],
) -> None:
    linop, matrix = square_spd_nested_kronecker
    linop_pinv = linox.lpinverse(linop)
    matrix_pinv = jnp.linalg.pinv(matrix)
    assert jnp.allclose(linop_pinv.todense(), matrix_pinv, atol=1e-6), (
        "Pseudo-inverse does not match"
    )
    key = jax.random.PRNGKey(10)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop_pinv @ vec, matrix_pinv @ vec, atol=1e-6), (
        "Pseudo-inverse matvec does not match"
    )


def test_qr(square_spd_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_kronecker
    linop_q, linop_r = linox.lqr(linop)
    matrix_q, matrix_r = jnp.linalg.qr(matrix)
    assert jnp.allclose((linop_q @ linop_r).todense(), matrix_q @ matrix_r), (
        "QR decomposition does not match"
    )
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(
        (linop_q @ linop_r) @ vec, (matrix_q @ matrix_r) @ vec, atol=1e-6
    ), "Q matvec does not match"


def test_svd(square_spd_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_kronecker
    linop_u, linop_s, linop_vh = linox.svd(linop)
    matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(matrix)
    assert jnp.allclose((linop_u @ jnp.diag(linop_s) @ linop_vh).todense(), matrix)
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(
        (linop_u @ jnp.diag(linop_s) @ linop_vh) @ vec, matrix @ vec, atol=1e-6
    ), "SVD matvec does not match"


def test_eigh(square_spd_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_kronecker
    linop_eigenvalues, linop_eigenvectors = linox.leigh(linop)

    assert jnp.allclose(
        (
            linop_eigenvectors @ jnp.diag(linop_eigenvalues) @ linop_eigenvectors.T
        ).todense(),
        matrix,
    )


def test_cholesky(
    square_spd_nested_kronecker: tuple[Kronecker, jax.Array],
) -> None:
    linop, matrix = square_spd_nested_kronecker
    Lop = linox.lcholesky(linop)

    assert jnp.allclose((Lop @ Lop.T).todense(), matrix, atol=1e-6), (
        "Cholesky does not match"
    )


def test_slogdet(square_spd_nested_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_nested_kronecker
    sign1, logdet1 = linox.slogdet(linop)
    sign2, logdet2 = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(logdet1, logdet2, atol=1e-6), "Log-determinant does not match"
    assert sign1 == sign2, "Sign of log-determinant does not match"


# ============================================================================
# JAX Tree Registration Tests
# ============================================================================


def test_pytree_registration() -> None:
    op = Kronecker(jnp.eye(3), jnp.eye(3))

    flat, tree_def = jax.tree_util.tree_flatten(op)
    reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

    test_vector = jnp.ones(op.shape[-1])
    original_result = op @ test_vector
    reconstructed_result = reconstructed @ test_vector

    assert jnp.allclose(original_result, reconstructed_result)
