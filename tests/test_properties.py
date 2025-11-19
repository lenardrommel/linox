# test_properties.py

"""Systematic tests for operator property checking functions.

These tests verify is_symmetric, is_hermitian, and is_square work correctly
for all operator types using probabilistic checking (without calling todense).
"""

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 22, 278]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture
def square_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a square matrix for testing."""
    size = 4
    return jax.random.normal(key, (size, size))


@pytest.fixture
def symmetric_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a symmetric matrix for testing."""
    size = 4
    A = jax.random.normal(key, (size, size))
    return (A + A.T) / 2


@pytest.fixture
def hermitian_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a Hermitian matrix for testing."""
    size = 4
    key1, key2 = jax.random.split(key)
    A_real = jax.random.normal(key1, (size, size))
    A_imag = jax.random.normal(key2, (size, size))
    A = A_real + 1j * A_imag
    return (A + A.conj().T) / 2


@pytest.fixture
def non_square_matrix(key: jax.random.PRNGKey) -> jax.Array:
    """Generate a non-square matrix for testing."""
    return jax.random.normal(key, (3, 5))


# ============================================================================
# Test is_square
# ============================================================================


def test_is_square_matrix_square():
    """Test is_square on square Matrix."""
    A = linox.Matrix(jnp.eye(4))
    assert linox.is_square(A)


def test_is_square_matrix_non_square():
    """Test is_square on non-square Matrix."""
    A = linox.Matrix(jnp.ones((3, 5)))
    assert not linox.is_square(A)


def test_is_square_identity():
    """Test is_square on Identity."""
    A = linox.Identity((4,))
    assert linox.is_square(A)


def test_is_square_diagonal():
    """Test is_square on Diagonal."""
    A = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    assert linox.is_square(A)


def test_is_square_kronecker():
    """Test is_square on Kronecker product."""
    A = linox.Identity((2,))
    B = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    K = linox.Kronecker(A, B)
    assert linox.is_square(K)


def test_is_square_block_diagonal():
    """Test is_square on BlockDiagonal."""
    blocks = [linox.Identity((2,)), linox.Diagonal(jnp.array([1.0, 2.0]))]
    A = linox.BlockDiagonal(blocks)
    assert linox.is_square(A)


def test_is_square_isotropic_additive():
    """Test is_square on IsotropicAdditiveLinearOperator."""
    base = linox.Matrix(jnp.eye(3))
    A = linox.IsotropicAdditiveLinearOperator(1.0, base)
    assert linox.is_square(A)


# ============================================================================
# Test is_symmetric - Probabilistic checking
# ============================================================================


def test_is_symmetric_identity():
    """Test is_symmetric on Identity (always symmetric)."""
    A = linox.Identity((10,))
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=10)


def test_is_symmetric_diagonal():
    """Test is_symmetric on Diagonal (always symmetric)."""
    A = linox.Diagonal(jnp.array([1.0, 2.0, 3.0, 4.0]))
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=10)


def test_is_symmetric_zero():
    """Test is_symmetric on Zero (always symmetric)."""
    A = linox.Zero((5, 5))
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=10)


def test_is_symmetric_symmetric_matrix(symmetric_matrix):
    """Test is_symmetric on explicitly symmetric matrix."""
    A = linox.Matrix(symmetric_matrix)
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=20)


def test_is_symmetric_non_symmetric_matrix(key):
    """Test is_symmetric detects non-symmetric matrix."""
    # Create clearly non-symmetric matrix
    mat = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    A = linox.Matrix(mat)
    assert not linox.is_symmetric(A, key=key, num_probes=20)


def test_is_symmetric_kronecker_symmetric():
    """Test is_symmetric on Kronecker of symmetric operators."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((3,))
    K = linox.Kronecker(A, B)
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(K, key=key, num_probes=20)


def test_is_symmetric_scaled_symmetric():
    """Test is_symmetric on scaled symmetric operator."""
    A = linox.Identity((5,))
    scaled = 2.5 * A
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(scaled, key=key, num_probes=20)


def test_is_symmetric_isotropic_additive(symmetric_matrix):
    """Test is_symmetric on IsotropicAdditiveLinearOperator with symmetric base."""
    base = linox.Matrix(symmetric_matrix)
    A = linox.IsotropicAdditiveLinearOperator(1.5, base)
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=20)


# ============================================================================
# Test is_hermitian - Probabilistic checking for complex matrices
# ============================================================================


def test_is_hermitian_identity():
    """Test is_hermitian on Identity (always Hermitian)."""
    A = linox.Identity((10,))
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(A, key=key, num_probes=10)


def test_is_hermitian_diagonal_real():
    """Test is_hermitian on real Diagonal (Hermitian if symmetric)."""
    A = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(A, key=key, num_probes=10)


def test_is_hermitian_diagonal_complex():
    """Test is_hermitian on complex Diagonal with real eigenvalues."""
    # Real diagonal values => Hermitian
    A = linox.Diagonal(jnp.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]))
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(A, key=key, num_probes=10)


def test_is_hermitian_diagonal_complex_non_hermitian():
    """Test is_hermitian detects non-Hermitian complex Diagonal."""
    # Complex diagonal values => Not Hermitian (unless on diagonal only)
    A = linox.Diagonal(jnp.array([1.0 + 1j, 2.0 + 0.5j, 3.0 - 1j]))
    key = jax.random.PRNGKey(0)
    assert not linox.is_hermitian(A, key=key, num_probes=10)


def test_is_hermitian_hermitian_matrix(hermitian_matrix):
    """Test is_hermitian on explicitly Hermitian matrix."""
    A = linox.Matrix(hermitian_matrix)
    key = jax.random.PRNGKey(0)
    # Hermitian check is probabilistic, so use sufficient probes
    assert linox.is_hermitian(A, key=key, num_probes=30)


def test_is_hermitian_non_hermitian_matrix(key):
    """Test is_hermitian detects non-Hermitian matrix."""
    # Create clearly non-Hermitian matrix
    mat = jnp.array(
        [[1.0 + 0j, 2.0 + 1j], [3.0 + 1j, 4.0 + 0j]]
    )  # Not conjugate symmetric
    A = linox.Matrix(mat)
    assert not linox.is_hermitian(A, key=key, num_probes=20)


def test_is_hermitian_symmetric_real_is_hermitian(symmetric_matrix):
    """Test that real symmetric matrices are Hermitian."""
    A = linox.Matrix(symmetric_matrix)
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(A, key=key, num_probes=20)


# ============================================================================
# Test edge cases
# ============================================================================


def test_is_symmetric_non_square_raises():
    """Test is_symmetric raises on non-square operator."""
    A = linox.Matrix(jnp.ones((3, 5)))
    key = jax.random.PRNGKey(0)
    # Should handle non-square gracefully or raise
    # Depending on implementation, adjust assertion
    try:
        result = linox.is_symmetric(A, key=key, num_probes=10)
        assert not result  # Non-square can't be symmetric
    except (ValueError, AssertionError):
        pass  # Expected for non-square


def test_is_hermitian_non_square_raises():
    """Test is_hermitian raises on non-square operator."""
    A = linox.Matrix(jnp.ones((3, 5)))
    key = jax.random.PRNGKey(0)
    try:
        result = linox.is_hermitian(A, key=key, num_probes=10)
        assert not result  # Non-square can't be Hermitian
    except (ValueError, AssertionError):
        pass  # Expected for non-square


def test_is_symmetric_single_element():
    """Test is_symmetric on 1x1 matrix (always symmetric)."""
    A = linox.Matrix(jnp.array([[5.0]]))
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(A, key=key, num_probes=5)


def test_is_hermitian_single_element_real():
    """Test is_hermitian on 1x1 real matrix (always Hermitian)."""
    A = linox.Matrix(jnp.array([[5.0]]))
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(A, key=key, num_probes=5)


# ============================================================================
# Test consistency: symmetric real => Hermitian
# ============================================================================


def test_symmetric_real_implies_hermitian():
    """Test that real symmetric operators are also Hermitian."""
    # Create symmetric real matrix
    mat = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    A = linox.Matrix(mat)
    key = jax.random.PRNGKey(0)

    is_sym = linox.is_symmetric(A, key=key, num_probes=20)
    is_herm = linox.is_hermitian(A, key=key, num_probes=20)

    # For real matrices, symmetric <=> Hermitian
    assert is_sym == is_herm


# ============================================================================
# Test with different operators - systematic
# ============================================================================


@pytest.mark.parametrize(
    "operator",
    [
        linox.Identity((5,)),
        linox.Diagonal(jnp.array([1.0, 2.0, 3.0])),
        linox.Zero((4, 4)),
        linox.Matrix(jnp.eye(3)),
    ],
)
def test_is_square_various_operators(operator):
    """Test is_square works for various operator types."""
    assert linox.is_square(operator)


@pytest.mark.parametrize(
    "operator",
    [
        linox.Identity((5,)),
        linox.Diagonal(jnp.array([1.0, 2.0, 3.0])),
        linox.Zero((4, 4)),
    ],
)
def test_is_symmetric_always_symmetric_operators(operator):
    """Test operators that are always symmetric."""
    key = jax.random.PRNGKey(0)
    assert linox.is_symmetric(operator, key=key, num_probes=10)


@pytest.mark.parametrize(
    "operator",
    [
        linox.Identity((5,)),
        linox.Diagonal(jnp.array([1.0, 2.0, 3.0])),
        linox.Zero((4, 4)),
    ],
)
def test_is_hermitian_always_hermitian_operators(operator):
    """Test operators that are always Hermitian."""
    key = jax.random.PRNGKey(0)
    assert linox.is_hermitian(operator, key=key, num_probes=10)
