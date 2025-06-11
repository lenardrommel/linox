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


# ============================================================================
# Transpose Tests
# ============================================================================


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_transpose(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    """Test transpose operation."""
    result_linop = linox.transpose(linop)
    assert jnp.allclose(result_linop.todense(), matrix.T)
    # Test with .T property
    result_t = linop.T
    assert jnp.allclose(result_t.todense(), matrix.T)
    # Test with .transpose() method
    result_transpose = linop.transpose()
    assert jnp.allclose(result_transpose.todense(), matrix.T)


# ============================================================================
# Diagonal Tests
# ============================================================================


@pytest_cases.parametrize_with_cases("linop,matrix", cases=special_linops)
def test_diagonal(linop: linox.LinearOperator, matrix: jax.Array) -> None:
    """Test diagonal extraction."""
    if matrix.shape[0] == matrix.shape[1]:  # Only for square matrices
        result = linox.diagonal(linop)
        expected = jnp.diag(matrix)
        assert jnp.allclose(result, expected)


# ============================================================================
# Utility Function Tests
# ============================================================================


def test_is_square() -> None:
    """Test is_square utility function."""
    square_op = linox.Identity(5)
    rect_op = linox.Matrix(jnp.ones((3, 5)))

    assert linox.is_square(square_op) is True
    assert linox.is_square(rect_op) is False


def test_is_symmetric(key: jax.random.PRNGKey) -> None:
    """Test is_symmetric utility function."""
    # Create symmetric matrix
    A = jax.random.normal(key, (4, 4))
    symmetric_matrix = A + A.T
    symmetric_op = linox.Matrix(symmetric_matrix)

    # Create non-symmetric matrix
    nonsymmetric_op = linox.Matrix(jax.random.normal(key, (4, 4)))

    assert linox.is_symmetric(symmetric_op), "linop not symmetric"
    assert not linox.is_symmetric(nonsymmetric_op), "linop should be non-symmetric"
    # Note: is_symmetric might return True for randomly generated matrix by chance
    # so we don't assert False for nonsymmetric_op


def test_symmetrize(key: jax.random.PRNGKey) -> None:
    """Test symmetrize function."""
    matrix = jax.random.normal(key, (4, 4))
    op = linox.Matrix(matrix)

    symmetrized = linox.symmetrize(op)
    expected = 0.5 * (matrix + matrix.T)

    assert jnp.allclose(symmetrized.todense(), expected)


# ============================================================================
# Linear System Solving Tests
# ============================================================================


def test_lsolve(square_spd_matrix: jax.Array, key: jax.random.PRNGKey) -> None:
    """Test linear system solving."""
    op = linox.Matrix(square_spd_matrix)
    b = jax.random.normal(key, (square_spd_matrix.shape[0],)) + jnp.ones((
        square_spd_matrix.shape[0],
    ))

    x = linox.lsolve(op, b)
    expected = jax.scipy.linalg.solve(square_spd_matrix, b, assume_a="sym")

    assert jnp.allclose(x, expected, atol=1e-6)
    assert jnp.allclose(op @ x, b, atol=1e-3)


def test_lpsolve(square_matrix: jax.Array, key: jax.random.PRNGKey) -> None:
    """Test pseudo-inverse linear system solving."""
    op = linox.Matrix(square_matrix)
    b = jax.random.normal(key, (square_matrix.shape[0],))

    x = linox.lpsolve(op, b)
    expected = jnp.linalg.pinv(square_matrix) @ b

    assert jnp.allclose(x, expected, atol=1e-6)


# ============================================================================
# Decomposition Tests
# ============================================================================


def test_lcholesky(square_spd_matrix: jax.Array) -> None:
    """Test Cholesky decomposition."""
    op = linox.Matrix(square_spd_matrix)

    L = linox.lcholesky(op)
    expected = jnp.linalg.cholesky(square_spd_matrix)

    assert jnp.allclose(L, expected, atol=1e-6)


def test_lqr(square_matrix: jax.Array) -> None:
    """Test QR decomposition."""
    op = linox.Matrix(square_matrix)

    Q, R = linox.lqr(op)
    expected_Q, expected_R = jnp.linalg.qr(square_matrix)

    assert jnp.allclose(Q, expected_Q, atol=1e-6)
    assert jnp.allclose(R, expected_R, atol=1e-6)


def test_leigh(square_spd_matrix: jax.Array) -> None:
    """Test eigenvalue decomposition."""
    op = linox.Matrix(square_spd_matrix)

    eigenvals, eigenvecs = linox.leigh(op)
    expected_eigenvals, expected_eigenvecs = jnp.linalg.eigh(square_spd_matrix)

    assert jnp.allclose(eigenvals, expected_eigenvals, atol=1e-6)
    # Eigenvectors can have different signs, so check that they span the same space
    assert jnp.allclose(jnp.abs(eigenvecs), jnp.abs(expected_eigenvecs), atol=1e-6)


# ============================================================================
# Determinant Tests
# ============================================================================


def test_ldet(square_spd_matrix: jax.Array) -> None:
    """Test determinant computation."""
    op = linox.Matrix(square_spd_matrix)

    det = linox.ldet(op)
    expected = jnp.linalg.det(square_spd_matrix)

    assert jnp.allclose(det, expected, rtol=1e-6)


def test_slogdet(square_spd_matrix: jax.Array) -> None:
    """Test sign and log determinant computation."""
    op = linox.Matrix(square_spd_matrix)

    sign, logdet = linox.slogdet(op)
    expected_sign, expected_logdet = jnp.linalg.slogdet(square_spd_matrix)

    assert jnp.allclose(sign, expected_sign)
    assert jnp.allclose(logdet, expected_logdet, atol=1e-6)


# ============================================================================
# Inverse Tests
# ============================================================================


def test_linverse(square_spd_matrix: jax.Array, key: jax.random.PRNGKey) -> None:
    """Test inverse operation."""
    op = linox.Matrix(square_spd_matrix)

    inv_op = linox.linverse(op)
    expected = jnp.linalg.inv(square_spd_matrix)

    assert jnp.allclose(inv_op.todense(), expected, atol=1e-6)

    # Test that A * A^-1 = I
    identity_result = op @ inv_op
    expected_identity = jnp.eye(square_spd_matrix.shape[0])
    assert jnp.allclose(identity_result.todense(), expected_identity, atol=1e-6)


def test_lpinverse(square_matrix: jax.Array) -> None:
    """Test pseudo-inverse operation."""
    op = linox.Matrix(square_matrix)

    pinv_op = linox.lpinverse(op)
    expected = jnp.linalg.pinv(square_matrix)

    assert jnp.allclose(pinv_op.todense(), expected, atol=1e-6)


# ============================================================================
# Square Root Tests
# ============================================================================


def test_lsqrt_scaled_operator(key: jax.random.PRNGKey) -> None:
    """Test square root of scaled operators."""
    base_matrix = jnp.eye(4)
    scalar = 4.0

    base_op = linox.Matrix(base_matrix)
    scaled_op = linox.ScaledLinearOperator(base_op, scalar)

    sqrt_op = linox.lsqrt(scaled_op)

    # For a scaled identity matrix, sqrt should be sqrt(scalar) * identity
    expected_scalar = jnp.sqrt(scalar)
    expected = expected_scalar * base_matrix

    assert jnp.allclose(sqrt_op.todense(), expected)


# ============================================================================
# Congruence Transform Tests
# ============================================================================


def test_congruence_transform(key: jax.random.PRNGKey) -> None:
    """Test congruence transformation A B A^T."""
    A_matrix = jax.random.normal(key, (4, 4))
    B_matrix = jax.random.normal(key, (4, 4))

    A_op = linox.Matrix(A_matrix)
    B_op = linox.Matrix(B_matrix)

    congruence_op = linox.congruence_transform(A_op, B_op)
    expected = A_matrix @ B_matrix @ A_matrix.T

    assert jnp.allclose(congruence_op.todense(), expected, atol=1e-6)


# ============================================================================
# Special Linear Operator Class Tests
# ============================================================================


def test_scaled_linear_operator(key: jax.random.PRNGKey) -> None:
    """Test ScaledLinearOperator functionality."""
    base_matrix = jax.random.normal(key, (3, 3))
    scalar = 2.5

    base_op = linox.Matrix(base_matrix)
    scaled_op = linox.ScaledLinearOperator(base_op, scalar)

    # Test basic operations
    expected = scalar * base_matrix
    assert jnp.allclose(scaled_op.todense(), expected)

    # Test transpose
    scaled_transpose = scaled_op.transpose()
    expected_transpose = scalar * base_matrix.T
    assert jnp.allclose(scaled_transpose.todense(), expected_transpose)

    # Test matrix-vector multiplication
    vector = jax.random.normal(key, (3,))
    result = scaled_op @ vector
    expected_result = scalar * (base_matrix @ vector)
    assert jnp.allclose(result, expected_result)


def test_add_linear_operator(key: jax.random.PRNGKey) -> None:
    """Test AddLinearOperator functionality."""
    A_matrix = jax.random.normal(key, (3, 3))
    B_matrix = jax.random.normal(key, (3, 3))

    A_op = linox.Matrix(A_matrix)
    B_op = linox.Matrix(B_matrix)

    add_op = linox.AddLinearOperator(A_op, B_op)

    expected = A_matrix + B_matrix
    assert jnp.allclose(add_op.todense(), expected)

    # Test transpose
    add_transpose = add_op.transpose()
    expected_transpose = A_matrix.T + B_matrix.T
    assert jnp.allclose(add_transpose.todense(), expected_transpose)


def test_product_linear_operator(key: jax.random.PRNGKey) -> None:
    """Test ProductLinearOperator functionality."""
    A_matrix = jax.random.normal(key, (3, 4))
    B_matrix = jax.random.normal(key, (4, 5))

    A_op = linox.Matrix(A_matrix)
    B_op = linox.Matrix(B_matrix)

    product_op = linox.ProductLinearOperator(A_op, B_op)

    expected = A_matrix @ B_matrix
    assert jnp.allclose(product_op.todense(), expected, atol=1e-6)

    # Test transpose
    product_transpose = product_op.transpose()
    expected_transpose = (A_matrix @ B_matrix).T
    assert jnp.allclose(product_transpose.todense(), expected_transpose, atol=1e-6)


def test_transposed_linear_operator(key: jax.random.PRNGKey) -> None:
    """Test TransposedLinearOperator functionality."""
    matrix = jax.random.normal(key, (3, 4))
    op = linox.Matrix(matrix)

    transposed_op = linox.TransposedLinearOperator(op)

    expected = matrix.T
    assert jnp.allclose(transposed_op.todense(), expected)

    # Test double transpose
    double_transpose = transposed_op.transpose()
    assert jnp.allclose(double_transpose.todense(), matrix)


def test_inverse_linear_operator(square_spd_matrix: jax.Array) -> None:
    """Test InverseLinearOperator functionality."""
    op = linox.Matrix(square_spd_matrix)

    inv_op = linox.InverseLinearOperator(op)

    expected = jnp.linalg.inv(square_spd_matrix)
    assert jnp.allclose(inv_op.todense(), expected, atol=1e-6)

    # Test determinant of inverse
    det_inv = linox.ldet(inv_op)
    expected_det_inv = 1.0 / jnp.linalg.det(square_spd_matrix)
    assert jnp.allclose(det_inv, expected_det_inv, rtol=1e-6)


def test_pseudo_inverse_linear_operator(square_matrix: jax.Array) -> None:
    """Test PseudoInverseLinearOperator functionality."""
    op = linox.Matrix(square_matrix)

    pinv_op = linox.PseudoInverseLinearOperator(op)

    expected = jnp.linalg.pinv(square_matrix)
    assert jnp.allclose(pinv_op.todense(), expected, atol=1e-6)


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_shape_mismatch_errors(key: jax.random.PRNGKey) -> None:
    """Test that appropriate errors are raised for shape mismatches."""
    A = linox.Matrix(jax.random.normal(key, (3, 4)))
    B = linox.Matrix(jax.random.normal(key, (5, 6)))

    # Test matmul shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        linox.ProductLinearOperator(A, B)

    # Test solve shape mismatch
    b_wrong_shape = jax.random.normal(key, (5,))
    with pytest.raises(ValueError, match="Shape mismatch"):
        linox.lsolve(A, b_wrong_shape)


def test_non_square_errors(key: jax.random.PRNGKey) -> None:
    """Test that appropriate errors are raised for non-square matrices."""
    rect_matrix = jax.random.normal(key, (3, 4))
    rect_op = linox.Matrix(rect_matrix)

    with pytest.raises(ValueError, match="not square"):
        linox.ldet(rect_op)

    with pytest.raises(ValueError, match="not square"):
        linox.slogdet(rect_op)


# ============================================================================
# JAX Tree Registration Tests
# ============================================================================


def test_pytree_registration() -> None:
    """Test that all linear operators are properly registered as PyTrees."""
    # Create instances of each operator type
    base_op = linox.Identity(3)

    operators = [
        linox.ScaledLinearOperator(base_op, 2.0),
        linox.AddLinearOperator(base_op, base_op),
        linox.ProductLinearOperator(base_op, base_op),
        linox.TransposedLinearOperator(base_op),
        linox.InverseLinearOperator(base_op),
        linox.PseudoInverseLinearOperator(base_op),
    ]

    for op in operators:
        # Test that JAX can handle these as PyTrees
        flat, tree_def = jax.tree_util.tree_flatten(op)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        # Verify the reconstructed operator works the same
        test_vector = jnp.ones(3)
        original_result = op @ test_vector
        reconstructed_result = reconstructed @ test_vector

        assert jnp.allclose(original_result, reconstructed_result)
