# test_operations.py

"""Property-based tests for operations on combinations of linear operators.

This module tests various operations (eigendecomposition, SVD, inverse, etc.)
on randomly combined linear operators to ensure they produce correct results.

## Operations Tested
- leigh: Eigendecomposition for symmetric/Hermitian matrices
- svd: Singular value decomposition
- linverse: Matrix inverse
- lpinverse: Pseudoinverse
- lcholesky: Cholesky decomposition
- ldet: Determinant
- lsolve: Solve linear systems

## Test Strategy
Uses the combination helpers from test_linox_cases/_operations_cases.py to:
1. Generate base SPD matrices
2. Apply random combinations of operations (scaled, add, product, transpose, kron, iso)
3. Verify that linox operations match NumPy/JAX equivalents
"""

import jax
import jax.numpy as jnp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import linox
from linox._arithmetic import (
    diagonal,
    iso,
    kron,
    lcholesky,
    ldet,
    leigh,
    linverse,
    lpinverse,
    svd,
)
from linox.utils import as_dense
from tests.test_linox_cases._operations_cases import (
    CaseType,
    sample_add_operator,
    sample_iso_operator,
    sample_matrix,
    sample_product_operator,
    sample_scaled_operator,
    sample_transposed_operator,
)

DTYPE = jnp.float64
FLOATS = st.floats(
    min_value=0.0001,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
)
SMALL_FLOATS = st.floats(
    min_value=0.1,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
)


# ============================================================================
# Strategies for generating SPD operator combinations
# ============================================================================


@st.composite
def spd_operator_strategy(
    draw: st.DrawFn,
    n: int = 4,
    max_depth: int = 3,
) -> CaseType:
    """Generate a random SPD linear operator combination.

    Args:
        draw: Hypothesis draw function
        n: Matrix dimension
        max_depth: Maximum nesting depth of operations

    Returns:
        Tuple of (LinearOperator, dense_matrix)
    """
    # Start with a base SPD matrix
    linop, matrix = sample_matrix((n, n), DTYPE, psd=True)

    if max_depth == 0:
        return linop, matrix

    # Choose a random combination of operations
    operation = draw(
        st.sampled_from([
            "scaled",
            "iso",
            "add",
            "product",
        ])
    )

    if operation == "scaled":
        scalar = draw(SMALL_FLOATS)
        if scalar == 0.0:
            scalar = 1.0
        return sample_scaled_operator(linop, matrix, scalar)

    if operation == "iso":
        scalar = draw(SMALL_FLOATS)
        if scalar == 0.0:
            scalar = 1.0
        return sample_iso_operator(linop, matrix, scalar)

    if operation == "add":
        # Add another SPD matrix
        linop2, matrix2 = draw(spd_operator_strategy(n=n, max_depth=max_depth - 1))
        return sample_add_operator(linop, matrix, linop2, matrix2)

    if operation == "product":
        # Product: A @ A.T to maintain SPD
        linop_t, matrix_t = sample_transposed_operator(linop, matrix)
        return sample_product_operator(linop, matrix, linop_t, matrix_t)

    return linop, matrix


@st.composite
def general_operator_strategy(
    draw: st.DrawFn,
    n: int = 4,
    max_depth: int = 3,
) -> CaseType:
    """Generate a random (possibly non-SPD) linear operator combination.

    Args:
        draw: Hypothesis draw function
        n: Matrix dimension
        max_depth: Maximum nesting depth of operations

    Returns:
        Tuple of (LinearOperator, dense_matrix)
    """
    # Start with a base matrix (not necessarily SPD)
    linop, matrix = sample_matrix((n, n), DTYPE, psd=True)

    if max_depth == 0:
        return linop, matrix

    # Choose a random combination of operations
    operation = draw(
        st.sampled_from([
            "scaled",
            "add",
            "product",
            "transpose",
        ])
    )

    if operation == "scaled":
        scalar = draw(FLOATS)
        return sample_scaled_operator(linop, matrix, scalar)

    if operation == "add":
        linop2, matrix2 = draw(general_operator_strategy(n=n, max_depth=max_depth - 1))
        return sample_add_operator(linop, matrix, linop2, matrix2)

    if operation == "product":
        linop2, matrix2 = draw(general_operator_strategy(n=n, max_depth=max_depth - 1))
        return sample_product_operator(linop, matrix, linop2, matrix2)

    if operation == "transpose":
        return sample_transposed_operator(linop, matrix)

    return linop, matrix


# ============================================================================
# Tests for eigendecomposition (leigh)
# ============================================================================


@given(spd_operator_strategy(n=4, max_depth=2))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_leigh_on_combinations(operator_case: CaseType) -> None:
    """Test eigendecomposition on random SPD operator combinations."""
    linop, matrix = operator_case

    # Compute eigendecomposition
    eigenvalues_linop, eigenvectors_linop = leigh(linop)
    eigenvalues_np, eigenvectors_np = jnp.linalg.eigh(matrix)

    # Sort eigenvalues and eigenvectors for comparison
    idx_linop = jnp.argsort(as_dense(eigenvalues_linop))
    idx_np = jnp.argsort(eigenvalues_np)

    sorted_evals_linop = as_dense(eigenvalues_linop)[idx_linop]
    sorted_evecs_linop = as_dense(eigenvectors_linop)[:, idx_linop]

    sorted_evals_np = eigenvalues_np[idx_np]
    sorted_evecs_np = eigenvectors_np[:, idx_np]

    # Check eigenvalues match
    assert jnp.allclose(sorted_evals_linop, sorted_evals_np, atol=1e-4, rtol=1e-4), (
        f"Eigenvalues mismatch:\nLinop: {sorted_evals_linop}\nNumPy: {sorted_evals_np}"
    )

    # Check that eigenvectors span the same space (allow sign flips)
    for i in range(len(sorted_evals_np)):
        dot_product = jnp.abs(jnp.dot(sorted_evecs_linop[:, i], sorted_evecs_np[:, i]))
        assert jnp.allclose(dot_product, 1.0, atol=1e-4), (
            f"Eigenvector {i} mismatch: dot product = {dot_product}"
        )


# ============================================================================
# Tests for Cholesky decomposition (lcholesky)
# ============================================================================


@given(spd_operator_strategy(n=4, max_depth=2))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_lcholesky_on_combinations(operator_case: CaseType) -> None:
    """Test Cholesky decomposition on random SPD operator combinations."""
    linop, matrix = operator_case

    # Compute Cholesky
    L_linop = lcholesky(linop)

    # Verify L @ L.T = A (the mathematical property)
    reconstructed = as_dense(L_linop @ L_linop.T)

    assert jnp.allclose(reconstructed, matrix, atol=1e-4, rtol=1e-4), (
        f"Cholesky reconstruction failed:\nL @ L.T:\n{reconstructed}\n"
        f"Original:\n{matrix}"
    )


# ============================================================================
# Tests for inverse (linverse)
# ============================================================================


@given(spd_operator_strategy(n=4, max_depth=2))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_linverse_on_combinations(operator_case: CaseType) -> None:
    """Test matrix inverse on random SPD operator combinations."""
    linop, matrix = operator_case

    # Compute inverse
    inv_linop = linverse(linop)
    inv_np = jnp.linalg.inv(matrix)

    # Check inverse matches
    assert jnp.allclose(as_dense(inv_linop), inv_np, atol=1e-4, rtol=1e-4), (
        f"Inverse mismatch:\nLinop:\n{as_dense(inv_linop)}\nNumPy:\n{inv_np}"
    )

    # Verify A @ A^-1 = I
    identity_test = (linop @ inv_linop).todense()
    expected_identity = jnp.eye(matrix.shape[0])
    assert jnp.allclose(identity_test, expected_identity, atol=1e-4, rtol=1e-4), (
        f"A @ A^-1 != I:\n{identity_test}"
    )


# ============================================================================
# Tests for pseudoinverse (lpinverse)
# ============================================================================


@given(general_operator_strategy(n=4, max_depth=2))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_lpinverse_on_combinations(operator_case: CaseType) -> None:
    """Test pseudoinverse on random operator combinations."""
    linop, matrix = operator_case

    # Compute pseudoinverse
    pinv_linop = lpinverse(linop)
    pinv_np = jnp.linalg.pinv(matrix)

    # Check pseudoinverse matches
    assert jnp.allclose(as_dense(pinv_linop), pinv_np, atol=1e-3, rtol=1e-3), (
        f"Pseudoinverse mismatch:\nLinop:\n{as_dense(pinv_linop)}\nNumPy:\n{pinv_np}"
    )


# ============================================================================
# Tests for determinant (ldet)
# ============================================================================


@given(spd_operator_strategy(n=4, max_depth=2))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ldet_on_combinations(operator_case: CaseType) -> None:
    """Test determinant on random SPD operator combinations."""
    linop, matrix = operator_case

    # Compute determinant
    det_linop = ldet(linop)
    det_np = jnp.linalg.det(matrix)

    # Check determinant matches
    assert jnp.allclose(det_linop, det_np, atol=1e-3, rtol=1e-3), (
        f"Determinant mismatch: linop={det_linop}, numpy={det_np}"
    )


# ============================================================================
# Tests for SVD
# ============================================================================


@given(general_operator_strategy(n=4, max_depth=2))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_svd_on_combinations(operator_case: CaseType) -> None:
    """Test SVD on random operator combinations."""
    linop, matrix = operator_case

    # Compute SVD
    U_linop, S_linop, Vt_linop = svd(linop)
    _, S_np, _ = jnp.linalg.svd(matrix)

    # Sort singular values for comparison
    idx_linop = jnp.argsort(as_dense(S_linop))[::-1]
    idx_np = jnp.argsort(S_np)[::-1]

    sorted_S_linop = as_dense(S_linop)[idx_linop]
    sorted_S_np = S_np[idx_np]

    # Check singular values match
    assert jnp.allclose(sorted_S_linop, sorted_S_np, atol=1e-4, rtol=1e-4), (
        f"Singular values mismatch:\nLinop: {sorted_S_linop}\nNumPy: {sorted_S_np}"
    )

    # Verify reconstruction: U @ diag(S) @ Vt = A
    reconstructed = as_dense(U_linop @ linox.Diagonal(S_linop) @ Vt_linop)
    assert jnp.allclose(reconstructed, matrix, atol=1e-4, rtol=1e-4), (
        f"SVD reconstruction failed:\nU @ S @ Vt:\n{reconstructed}\nOriginal:\n{matrix}"
    )


# ============================================================================
# Tests for diagonal extraction
# ============================================================================


@given(general_operator_strategy(n=4, max_depth=2))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_diagonal_on_combinations(operator_case: CaseType) -> None:
    """Test diagonal extraction on random operator combinations."""
    linop, matrix = operator_case

    # Extract diagonal
    diag_linop = diagonal(linop)
    diag_np = jnp.diag(matrix)

    # Check diagonal matches
    assert jnp.allclose(diag_linop, diag_np, atol=1e-5, rtol=1e-5), (
        f"Diagonal mismatch:\nLinop: {diag_linop}\nNumPy: {diag_np}"
    )


# ============================================================================
# Specific complex combination tests
# ============================================================================


def test_complex_iso_kron_combination() -> None:
    """Test operations on (A ⊗ B) + sI."""
    n = 2

    # Create base operators
    A_dense = jax.random.normal(jax.random.PRNGKey(0), (n, n))
    A_dense = A_dense @ A_dense.T + jnp.eye(n) * 0.1
    A = linox.Matrix(A_dense)

    B_dense = jax.random.normal(jax.random.PRNGKey(1), (n, n))
    B_dense = B_dense @ B_dense.T + jnp.eye(n) * 0.1
    B = linox.Matrix(B_dense)

    # Create Kronecker product
    kron_linop = kron(A, B)
    kron_matrix = jnp.kron(A_dense, B_dense)

    # Add isotropic term
    s = 0.5
    iso_linop = iso(s, kron_linop)
    iso_matrix = s * jnp.eye(n * n) + kron_matrix

    # Test eigendecomposition
    eigenvalues_linop, _ = leigh(iso_linop)
    eigenvalues_np, _ = jnp.linalg.eigh(iso_matrix)

    idx_linop = jnp.argsort(as_dense(eigenvalues_linop))
    idx_np = jnp.argsort(eigenvalues_np)

    assert jnp.allclose(
        as_dense(eigenvalues_linop)[idx_linop],
        eigenvalues_np[idx_np],
        atol=1e-4,
        rtol=1e-4,
    )

    # Test inverse
    inv_linop = linverse(iso_linop)
    inv_np = jnp.linalg.inv(iso_matrix)

    assert jnp.allclose(as_dense(inv_linop), inv_np, atol=1e-4, rtol=1e-4)

    # Test determinant
    det_linop = ldet(iso_linop)
    det_np = jnp.linalg.det(iso_matrix)

    assert jnp.allclose(det_linop, det_np, atol=1e-3, rtol=1e-3)


def test_nested_operations_combination() -> None:
    """Test operations on nested combinations: ((A + B) @ C^T) + sI."""
    n = 3
    key = jax.random.PRNGKey(42)

    # Create base SPD matrices
    A_dense = jax.random.normal(key, (n, n))
    A_dense = A_dense @ A_dense.T + jnp.eye(n) * 0.1
    A = linox.Matrix(A_dense)

    B_dense = jax.random.normal(jax.random.split(key)[1], (n, n))
    B_dense = B_dense @ B_dense.T + jnp.eye(n) * 0.1
    B = linox.Matrix(B_dense)

    C_dense = jax.random.normal(jax.random.split(key)[0], (n, n))
    C_dense = C_dense @ C_dense.T + jnp.eye(n) * 0.1
    C = linox.Matrix(C_dense)

    # Build: ((A + B) @ C^T) + sI
    sum_linop = A + B
    sum_matrix = A_dense + B_dense

    product_linop = sum_linop @ C.T
    product_matrix = sum_matrix @ C_dense.T

    # Make it SPD by doing: result = product @ product^T
    spd_linop = product_linop @ product_linop.T
    spd_matrix = product_matrix @ product_matrix.T

    s = 0.3
    final_linop = iso(s, spd_linop)
    final_matrix = s * jnp.eye(n) + spd_matrix

    # Test inverse
    inv_linop = linverse(final_linop)
    inv_np = jnp.linalg.inv(final_matrix)

    assert jnp.allclose(as_dense(inv_linop), inv_np, atol=1e-4, rtol=1e-4)

    # Test Cholesky property
    L = lcholesky(final_linop)
    reconstructed = (L @ L.T).todense()

    assert jnp.allclose(reconstructed, final_matrix, atol=1e-4, rtol=1e-4)
