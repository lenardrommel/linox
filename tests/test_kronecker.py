from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox._kronecker import Kronecker
from linox.types import ShapeLike, ShapeType

jax.config.update("jax_enable_x64", True)

CaseType = tuple[linox.Kronecker, jax.Array]

basic_shapes = [((2, 2), (4, 4)), ((4, 4), (4, 4)), ((1, 1), (3, 3)), ((1, 2), (3, 4))]

add_shapes = [
    (((2, 2), (4, 4)), ((2, 2), (4, 4))),
    (((2, 2), (4, 4)), ((4, 4), (2, 2))),
    (((2, 2), (2, 2)), ((2, 2), (2, 2))),
]

basic_dims = [1, 2, 3]

add_dims = [(1, 1), (2, 2), (1, 2), (2, 1), (3, 2), (2, 3)]


def sample_kronecker(shape: ShapeLike) -> CaseType:
    (shapeA, shapeB) = shape

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, shapeA)
    B = jax.random.normal(key2, shapeB)

    op = Kronecker(A, B)
    matrix = jnp.kron(A, B)

    assert op.shape == matrix.shape, "Shape mismatch"

    return op, matrix


def sample_nested_kronecker(dim: int, shape: ShapeLike) -> CaseType:
    op = None
    M = None
    for _ in range(dim):
        op, M = sample_kronecker(shape)
        op = Kronecker(op, op)
        M = jnp.kron(M, M)

    assert op.shape == M.shape, "Shape mismatch"

    return op, M


linops_options = [sample_kronecker, sample_nested_kronecker]

# ============================================================================
# Case Generators
# ============================================================================


@pytest.mark.parametrize("dim", basic_dims)
@pytest.mark.parametrize("shape", basic_shapes)
def case_kronecker(shape: ShapeLike, dim: int) -> CaseType:
    """Test case for Kronecker product."""
    return sample_nested_kronecker(dim, shape)


@pytest.mark.parametrize(
    ("dims", "shape", "get_linop"),
    list(
        product(
            add_dims,
            add_shapes,
            [(sample_nested_kronecker, sample_nested_kronecker)],
        )
    ),
)
def case_add(
    dims: tuple[int, int],
    shape: tuple[tuple[ShapeLike, ShapeLike], tuple[ShapeLike, ShapeLike]],
    get_linop: tuple[Callable, Callable],
) -> tuple[CaseType, CaseType]:
    dim1, dim2 = dims
    get_linop1, get_linop2 = get_linop
    shape1, shape2 = shape
    return get_linop1(dim1, shape1), get_linop2(dim2, shape2)


@pytest.mark.parametrize(
    ("dims", "shape", "get_linop"),
    list(
        product(
            add_dims,
            add_shapes,
            [(sample_nested_kronecker, sample_nested_kronecker)],
        )
    ),
)
def case_matmul(
    dims: tuple[int, int],
    shape: tuple[ShapeType, ShapeType],
    get_linop: tuple[Callable, Callable],
) -> tuple[CaseType, CaseType]:
    dim1, dim2 = dims
    get_linop1, get_linop2 = get_linop
    shape1, shape2 = shape
    return get_linop1(dim1, shape1), get_linop2(dim2, shape2)


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
