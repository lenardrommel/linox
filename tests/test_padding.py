import jax.numpy as jnp
import pytest

from linox import Matrix, SymmetricCroppingOperator


@pytest.fixture
def dense_operator():
    """Fixture for creating a simple dense operator."""
    matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return Matrix(matrix)


def test_symmetric_cropping_operator_matmul(dense_operator):
    """Test the matmul operation of SymmetricCroppingOperator."""
    padding = 1
    op = SymmetricCroppingOperator(dense_operator, padding=padding)

    # Create a vector to apply the operator
    vector = jnp.array([[2.0]])

    # Expected output
    extended_vector = jnp.array([[0.0], [2.0], [0.0]])
    expected_result = (dense_operator @ extended_vector)[padding:-padding]

    # Actual result
    result = op @ vector

    assert jnp.allclose(result, expected_result), "Matmul result is incorrect."


def test_symmetric_cropping_operator_todense(dense_operator):
    """Test the dense representation of SymmetricCroppingOperator."""
    padding = 1
    op = SymmetricCroppingOperator(dense_operator, padding=padding)

    # Expected dense matrix
    expected_dense = dense_operator.todense()[padding:-padding, padding:-padding]

    # Actual dense matrix
    dense_matrix = op.todense()

    assert jnp.allclose(dense_matrix, expected_dense), "Dense matrix is incorrect."


def test_symmetric_cropping_operator_shape(dense_operator):
    """Test the shape of SymmetricCroppingOperator."""
    padding = 1
    op = SymmetricCroppingOperator(dense_operator, padding=padding)

    # Expected shape
    expected_shape = (
        dense_operator.shape[0] - 2 * padding,
        dense_operator.shape[1] - 2 * padding,
    )

    assert op.shape == expected_shape, "Shape of operator is incorrect."


# def test_symmetric_cropping_operator_zero_padding(dense_operator):
#     """Test SymmetricCroppingOperator with zero padding."""
#     padding = 0
#     op = SymmetricCroppingOperator(dense_operator, padding=padding)

#     # Expected dense matrix is the same as the original operator
#     expected_dense = dense_operator.todense()
#     print(op.todense())
#     print(expected_dense)
#     assert np.testing.assert_allclose(
#         op.todense(), expected_dense
#     ), "Zero padding result is incorrect."
