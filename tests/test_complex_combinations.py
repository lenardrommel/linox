# test_complex_combinations.py

"""Tests for complex operator combinations and compositions.

Tests systematic compositions of operators:
- Kronecker products with various operator types
- IsotropicAdditiveLinearOperator with different base operators
- Nested arithmetic operations (scaled + transposed + product)
- BlockDiagonal and BlockMatrix with mixed operator types
- Deep nesting of operators
"""

import jax
import jax.numpy as jnp
import pytest

import linox


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 42]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


# ============================================================================
# Kronecker Product Combinations
# ============================================================================


def test_kronecker_identity_identity():
    """Test Kronecker product of two Identity operators."""
    A = linox.Identity((2,))
    B = linox.Identity((3,))
    K = linox.Kronecker(A, B)

    expected = jnp.eye(6)
    assert jnp.allclose(K.todense(), expected)


def test_kronecker_diagonal_diagonal():
    """Test Kronecker product of two Diagonal operators."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Diagonal(jnp.array([3.0, 4.0, 5.0]))
    K = linox.Kronecker(A, B)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = jnp.diag(jnp.array([3.0, 4.0, 5.0]))
    expected = jnp.kron(A_dense, B_dense)
    assert jnp.allclose(K.todense(), expected)


def test_kronecker_identity_diagonal():
    """Test Kronecker product of Identity and Diagonal."""
    A = linox.Identity((2,))
    B = linox.Diagonal(jnp.array([3.0, 4.0]))
    K = linox.Kronecker(A, B)

    A_dense = jnp.eye(2)
    B_dense = jnp.diag(jnp.array([3.0, 4.0]))
    expected = jnp.kron(A_dense, B_dense)
    assert jnp.allclose(K.todense(), expected)


def test_kronecker_matrix_matrix(key):
    """Test Kronecker product of two Matrix operators."""
    A_arr = jax.random.normal(key, (2, 2))
    key, subkey = jax.random.split(key)
    B_arr = jax.random.normal(subkey, (3, 3))

    A = linox.Matrix(A_arr)
    B = linox.Matrix(B_arr)
    K = linox.Kronecker(A, B)

    expected = jnp.kron(A_arr, B_arr)
    assert jnp.allclose(K.todense(), expected, atol=1e-5)


def test_kronecker_nested():
    """Test nested Kronecker products: (A ⊗ B) ⊗ C."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((2,))
    C = linox.Diagonal(jnp.array([3.0, 4.0]))

    K1 = linox.Kronecker(A, B)
    K2 = linox.Kronecker(K1, C)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = jnp.eye(2)
    C_dense = jnp.diag(jnp.array([3.0, 4.0]))
    expected = jnp.kron(jnp.kron(A_dense, B_dense), C_dense)

    assert jnp.allclose(K2.todense(), expected)


def test_kronecker_with_scaled():
    """Test Kronecker product with ScaledLinearOperator."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B_base = linox.Identity((3,))
    B = 2.0 * B_base

    K = linox.Kronecker(A, B)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = 2.0 * jnp.eye(3)
    expected = jnp.kron(A_dense, B_dense)

    assert jnp.allclose(K.todense(), expected)


# ============================================================================
# IsotropicAdditiveLinearOperator Combinations
# ============================================================================


def test_isotropic_add_with_identity():
    """Test IsotropicAdditiveLinearOperator with Identity base."""
    scalar = 2.0
    base = linox.Identity((4,))
    iso = linox.IsotropicAdditiveLinearOperator(scalar, base)

    # scalar * I + I = (scalar + 1) * I
    expected = (scalar + 1.0) * jnp.eye(4)
    assert jnp.allclose(iso.todense(), expected)


def test_isotropic_add_with_diagonal():
    """Test IsotropicAdditiveLinearOperator with Diagonal base."""
    scalar = 1.5
    diag_vals = jnp.array([1.0, 2.0, 3.0])
    base = linox.Diagonal(diag_vals)
    iso = linox.IsotropicAdditiveLinearOperator(scalar, base)

    # 1.5 * I + diag([1,2,3])
    expected = scalar * jnp.eye(3) + jnp.diag(diag_vals)
    assert jnp.allclose(iso.todense(), expected)


def test_isotropic_add_with_zero():
    """Test IsotropicAdditiveLinearOperator with Zero base."""
    scalar = 3.0
    base = linox.Zero((3, 3))
    iso = linox.IsotropicAdditiveLinearOperator(scalar, base)

    # 3 * I + 0 = 3 * I
    expected = scalar * jnp.eye(3)
    assert jnp.allclose(iso.todense(), expected)


def test_isotropic_add_with_matrix(key):
    """Test IsotropicAdditiveLinearOperator with Matrix base."""
    scalar = 0.5
    mat = jax.random.normal(key, (4, 4))
    base = linox.Matrix(mat)
    iso = linox.IsotropicAdditiveLinearOperator(scalar, base)

    expected = scalar * jnp.eye(4) + mat
    assert jnp.allclose(iso.todense(), expected, atol=1e-5)


def test_isotropic_add_with_kronecker():
    """Test IsotropicAdditiveLinearOperator with Kronecker base."""
    scalar = 1.0
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((2,))
    kron = linox.Kronecker(A, B)
    iso = linox.IsotropicAdditiveLinearOperator(scalar, kron)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = jnp.eye(2)
    kron_dense = jnp.kron(A_dense, B_dense)
    expected = scalar * jnp.eye(4) + kron_dense

    assert jnp.allclose(iso.todense(), expected)


def test_isotropic_add_nested():
    """Test nested IsotropicAdditiveLinearOperator."""
    scalar1 = 1.0
    scalar2 = 0.5
    base = linox.Identity((3,))
    iso1 = linox.IsotropicAdditiveLinearOperator(scalar1, base)
    iso2 = linox.IsotropicAdditiveLinearOperator(scalar2, iso1)

    # scalar2 * I + (scalar1 * I + I) = (scalar1 + scalar2 + 1) * I
    expected = (scalar1 + scalar2 + 1.0) * jnp.eye(3)
    assert jnp.allclose(iso2.todense(), expected)


# ============================================================================
# Nested Arithmetic Operations
# ============================================================================


def test_scaled_add_product(key):
    """Test combination: scalar * (A + B) @ C."""
    A = jax.random.normal(key, (3, 3))
    B = jax.random.normal(jax.random.split(key)[1], (3, 3))
    C = jax.random.normal(jax.random.split(key)[0], (3, 3))

    A_op = linox.Matrix(A)
    B_op = linox.Matrix(B)
    C_op = linox.Matrix(C)

    combined = 2.0 * (A_op + B_op) @ C_op

    expected = 2.0 * (A + B) @ C
    assert jnp.allclose(combined.todense(), expected, atol=1e-5)


def test_transposed_scaled_add(key):
    """Test combination: (scalar * A)^T + B."""
    A = jax.random.normal(key, (3, 4))
    B = jax.random.normal(jax.random.split(key)[1], (4, 3))

    A_op = linox.Matrix(A)
    B_op = linox.Matrix(B)

    combined = (2.5 * A_op).T + B_op

    expected = 2.5 * A.T + B
    assert jnp.allclose(combined.todense(), expected, atol=1e-5)


def test_deep_nesting(key):
    """Test deeply nested operator: ((A @ B)^T + C) @ D."""
    A = jax.random.normal(key, (2, 3))
    keys = jax.random.split(key, 4)
    B = jax.random.normal(keys[1], (3, 4))
    C = jax.random.normal(keys[2], (4, 3))
    D = jax.random.normal(keys[3], (3, 2))

    A_op = linox.Matrix(A)
    B_op = linox.Matrix(B)
    C_op = linox.Matrix(C)
    D_op = linox.Matrix(D)

    combined = ((A_op @ B_op).T + C_op) @ D_op

    expected = ((A @ B).T + C) @ D
    assert jnp.allclose(combined.todense(), expected, atol=1e-5)


def test_arithmetic_with_identity():
    """Test arithmetic operations mixing Identity with other operators."""
    I = linox.Identity((3,))
    D = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))

    # (2*I + D) @ (I - 0.5*D)
    left = 2.0 * I + D
    right = I - 0.5 * D

    combined = left @ right

    left_dense = 2.0 * jnp.eye(3) + jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    right_dense = jnp.eye(3) - 0.5 * jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    expected = left_dense @ right_dense

    assert jnp.allclose(combined.todense(), expected, atol=1e-5)


# ============================================================================
# BlockDiagonal and BlockMatrix Combinations
# ============================================================================


def test_block_diagonal_mixed_types():
    """Test BlockDiagonal with mixed operator types."""
    block1 = linox.Identity((2,))
    block2 = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    block3 = linox.Matrix(jnp.ones((2, 2)))

    B = linox.BlockDiagonal([block1, block2, block3])

    expected = jnp.block(
        [
            [jnp.eye(2), jnp.zeros((2, 3)), jnp.zeros((2, 2))],
            [jnp.zeros((3, 2)), jnp.diag(jnp.array([1.0, 2.0, 3.0])), jnp.zeros((3, 2))],
            [jnp.zeros((2, 2)), jnp.zeros((2, 3)), jnp.ones((2, 2))],
        ]
    )

    assert jnp.allclose(B.todense(), expected, atol=1e-5)


def test_block_diagonal_with_scaled():
    """Test BlockDiagonal with scaled operators."""
    block1 = 2.0 * linox.Identity((2,))
    block2 = 0.5 * linox.Diagonal(jnp.array([1.0, 2.0]))

    B = linox.BlockDiagonal([block1, block2])

    expected = jnp.block(
        [
            [2.0 * jnp.eye(2), jnp.zeros((2, 2))],
            [jnp.zeros((2, 2)), 0.5 * jnp.diag(jnp.array([1.0, 2.0]))],
        ]
    )

    assert jnp.allclose(B.todense(), expected, atol=1e-5)


def test_block_diagonal_arithmetic():
    """Test arithmetic with BlockDiagonal."""
    block1 = linox.Identity((2,))
    block2 = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.BlockDiagonal([block1, block2])

    # Add scalar identity
    combined = B + linox.Identity((4,))

    expected = jnp.block(
        [
            [jnp.eye(2), jnp.zeros((2, 2))],
            [jnp.zeros((2, 2)), jnp.diag(jnp.array([1.0, 2.0]))],
        ]
    ) + jnp.eye(4)

    assert jnp.allclose(combined.todense(), expected, atol=1e-5)


def test_block_matrix_2x2():
    """Test BlockMatrix2x2 with different operator types."""
    A11 = linox.Identity((2,))
    A12 = linox.Zero((2, 3))
    A21 = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    A22 = linox.Matrix(jnp.ones((3, 3)))

    # Note: A21 should be (3, 2) to match dimensions
    # Let me fix this
    A21 = linox.Zero((3, 2))

    B = linox.BlockMatrix2x2(A11, A12, A21, A22)

    expected = jnp.block(
        [[jnp.eye(2), jnp.zeros((2, 3))], [jnp.zeros((3, 2)), jnp.ones((3, 3))]]
    )

    assert jnp.allclose(B.todense(), expected, atol=1e-5)


# ============================================================================
# Combinations with Kronecker and IsotropicAdditive
# ============================================================================


def test_kronecker_isotropic_combination():
    """Test Kronecker product inside IsotropicAdditive."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((2,))
    K = linox.Kronecker(A, B)

    scalar = 0.5
    iso = linox.IsotropicAdditiveLinearOperator(scalar, K)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = jnp.eye(2)
    K_dense = jnp.kron(A_dense, B_dense)
    expected = scalar * jnp.eye(4) + K_dense

    assert jnp.allclose(iso.todense(), expected)


def test_isotropic_inside_kronecker():
    """Test IsotropicAdditive inside Kronecker product."""
    iso1 = linox.IsotropicAdditiveLinearOperator(
        1.0, linox.Diagonal(jnp.array([1.0, 2.0]))
    )
    iso2 = linox.IsotropicAdditiveLinearOperator(0.5, linox.Identity((2,)))

    K = linox.Kronecker(iso1, iso2)

    iso1_dense = jnp.eye(2) + jnp.diag(jnp.array([1.0, 2.0]))
    iso2_dense = 0.5 * jnp.eye(2) + jnp.eye(2)
    expected = jnp.kron(iso1_dense, iso2_dense)

    assert jnp.allclose(K.todense(), expected)


def test_scaled_kronecker_isotropic():
    """Test scaled Kronecker inside IsotropicAdditive."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((3,))
    K = linox.Kronecker(A, B)
    scaled_K = 2.0 * K

    iso = linox.IsotropicAdditiveLinearOperator(0.5, scaled_K)

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    B_dense = jnp.eye(3)
    K_dense = jnp.kron(A_dense, B_dense)
    expected = 0.5 * jnp.eye(6) + 2.0 * K_dense

    assert jnp.allclose(iso.todense(), expected)


# ============================================================================
# Matrix-vector and matrix-matrix products for complex combinations
# ============================================================================


def test_kronecker_matvec(key):
    """Test matvec with Kronecker of complex operators."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B_mat = jax.random.normal(key, (3, 3))
    B = linox.Matrix(B_mat)

    K = linox.Kronecker(A, B)

    v = jax.random.normal(jax.random.split(key)[1], (6,))

    result = K @ v
    expected = jnp.kron(jnp.diag(jnp.array([1.0, 2.0])), B_mat) @ v

    assert jnp.allclose(result, expected, atol=1e-5)


def test_isotropic_add_matvec(key):
    """Test matvec with IsotropicAdditive of scaled Kronecker."""
    A = linox.Diagonal(jnp.array([1.0, 2.0]))
    B = linox.Identity((2,))
    K = 1.5 * linox.Kronecker(A, B)

    iso = linox.IsotropicAdditiveLinearOperator(0.5, K)

    v = jax.random.normal(key, (4,))

    result = iso @ v

    A_dense = jnp.diag(jnp.array([1.0, 2.0]))
    K_dense = 1.5 * jnp.kron(A_dense, jnp.eye(2))
    expected = (0.5 * jnp.eye(4) + K_dense) @ v

    assert jnp.allclose(result, expected, atol=1e-5)


def test_nested_arithmetic_matvec(key):
    """Test matvec with deeply nested arithmetic."""
    A = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    B = linox.Identity((3,))

    # (2*A + B)^T @ v
    combined = (2.0 * A + B).T

    v = jax.random.normal(key, (3,))
    result = combined @ v

    A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    expected = (2.0 * A_dense + jnp.eye(3)).T @ v

    assert jnp.allclose(result, expected, atol=1e-5)


# ============================================================================
# Stress test: Very deep nesting
# ============================================================================


def test_very_deep_nesting():
    """Test very deep nesting of operators."""
    # Create a deeply nested structure
    base = linox.Identity((2,))

    # Layer 1: Scale
    op1 = 2.0 * base

    # Layer 2: Add
    op2 = op1 + linox.Diagonal(jnp.array([1.0, 1.0]))

    # Layer 3: Transpose
    op3 = op2.T

    # Layer 4: Product
    op4 = op3 @ linox.Identity((2,))

    # Layer 5: Scale again
    op5 = 0.5 * op4

    # Layer 6: Kronecker
    op6 = linox.Kronecker(op5, linox.Identity((2,)))

    # Layer 7: IsotropicAdditive
    op7 = linox.IsotropicAdditiveLinearOperator(1.0, op6)

    # Verify it produces correct result
    expected = jnp.eye(4) + jnp.kron(
        0.5 * (2.0 * jnp.eye(2) + jnp.diag(jnp.array([1.0, 1.0]))), jnp.eye(2)
    )

    assert jnp.allclose(op7.todense(), expected, atol=1e-5)
