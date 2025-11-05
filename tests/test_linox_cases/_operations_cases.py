# _operations_cases.py
import jax
from jax import numpy as jnp

from linox import LinearOperator
from linox._arithmetic import iso, kron, linverse, lpinverse
from linox.utils import as_linop

CaseType = tuple[LinearOperator, jax.Array]


def sample_matrix(shape, dtype, psd) -> CaseType:
    """Generate a sample linear operator for testing."""
    A = jax.random.normal(jax.random.PRNGKey(0), shape, dtype=dtype)
    if psd:
        A = A @ A.T + jnp.eye(shape[0], dtype=dtype) * 1e-3

    return as_linop(A), A


def sample_scaled_operator(linop, matrix, scalar) -> CaseType:
    """Generate a sample scaled linear operator for testing."""
    scaled_linop = scalar * linop
    scaled_matrix = scalar * matrix
    return scaled_linop, scaled_matrix


def sample_add_operator(linop1, matrix1, linop2, matrix2) -> CaseType:
    """Generate a sample addition linear operator for testing."""
    add_linop = linop1 + linop2
    add_matrix = matrix1 + matrix2
    return add_linop, add_matrix


def sample_product_operator(linop1, matrix1, linop2, matrix2) -> CaseType:
    """Generate a sample product linear operator for testing."""
    product_linop = linop1 @ linop2
    product_matrix = matrix1 @ matrix2
    return product_linop, product_matrix


def sample_transposed_operator(linop, matrix) -> CaseType:
    """Generate a sample transposed linear operator for testing."""
    transposed_linop = linop.T
    transposed_matrix = matrix.T
    return transposed_linop, transposed_matrix


def sample_inverse_operator(linop, matrix) -> CaseType:
    """Generate a sample inverse linear operator for testing."""
    inverse_linop = linverse(linop)
    inverse_matrix = jnp.linalg.inv(matrix)
    return inverse_linop, inverse_matrix


def sample_pseudoinverse_operator(linop, matrix) -> CaseType:
    """Generate a sample pseudoinverse linear operator for testing."""
    pinverse_linop = lpinverse(linop)
    pinverse_matrix = jnp.linalg.pinv(matrix)
    return pinverse_linop, pinverse_matrix


def sample_kron_operator(linop1, matrix1, linop2, matrix2) -> CaseType:
    """Generate a sample Kronecker product linear operator for testing."""
    kron_linop = kron(linop1, linop2)
    kron_matrix = jnp.kron(matrix1, matrix2)
    return kron_linop, kron_matrix


def sample_iso_operator(linop, matrix, scalar) -> CaseType:
    """Generate a sample isotropic additive linear operator for testing."""
    iso_linop = iso(scalar, linop)
    iso_matrix = scalar * jnp.eye(matrix.shape[0]) + matrix
    return iso_linop, iso_matrix
