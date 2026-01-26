# test_kronecker.py

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox._kronecker import (
    Kronecker,
    KroneckerSelectedEigenvectors,
    extract_kronecker_factors,
    topk_eigh,
)
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


def test_solve(square_spd_nested_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_nested_kronecker
    key = jax.random.PRNGKey(10)
    vec = jax.random.normal(key, (matrix.shape[-1],))

    linop_result = linox.lsolve(linop, vec)
    matrix_result = jnp.linalg.solve(matrix, vec)
    assert jnp.allclose(linop_result, matrix_result, atol=1e-6), "Solve does not match"


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


def test_psolve(square_spd_nested_kronecker: tuple[Kronecker, jax.Array]) -> None:
    linop, matrix = square_spd_nested_kronecker
    key = jax.random.PRNGKey(10)
    vec = jax.random.normal(key, (matrix.shape[-1],))
    linop_result = linox.lpsolve(linop, vec)
    matrix_result = jnp.linalg.pinv(matrix) @ vec
    assert jnp.allclose(linop_result, matrix_result, atol=1e-6), (
        "Pseudo-solve does not match"
    )


def test_kronecker_lsqrt_preserves_matmul(
    square_spd_kronecker: tuple[Kronecker, jax.Array],
    key: jax.random.PRNGKey,
    ncols: int,
) -> None:
    linop, matrix = square_spd_kronecker
    scale = 1.7
    scaled = linox.ScaledLinearOperator(linop, scale)
    sqrt_op = linox.lsqrt(scaled)
    rhs = jax.random.normal(key, (matrix.shape[-1], ncols))
    jitter_A = 1e-10 if linop.A.dtype == jnp.float64 else 1e-6  # type: ignore[attr-defined]
    jitter_B = 1e-10 if linop.B.dtype == jnp.float64 else 1e-6  # type: ignore[attr-defined]
    chol_A = jnp.linalg.cholesky(
        linop.A.todense() + jitter_A * jnp.eye(linop.A.shape[0], dtype=linop.A.dtype)
    )
    chol_B = jnp.linalg.cholesky(
        linop.B.todense() + jitter_B * jnp.eye(linop.B.shape[0], dtype=linop.B.dtype)
    )
    dense_sqrt = jnp.sqrt(scale) * jnp.kron(chol_A, chol_B)
    assert sqrt_op.shape == (matrix.shape[0], matrix.shape[0])
    matmul_result = sqrt_op @ rhs
    assert matmul_result.shape == (matrix.shape[0], ncols)
    assert jnp.allclose(matmul_result, dense_sqrt @ rhs, atol=1e-5)


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
        (linop_eigenvectors @ linop_eigenvalues @ linop_eigenvectors.T).todense(),
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


# ============================================================================
# Nested Kronecker and topk_eigh Tests
# ============================================================================


def test_leigh_nested_kronecker() -> None:
    """Test leigh works correctly for nested Kronecker products."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Create nested Kronecker: (A ⊗ B) ⊗ (C ⊗ D)
    A = jax.random.normal(k1, (3, 3))
    A = A @ A.T + jnp.eye(3) * 0.1
    B = jax.random.normal(k2, (2, 2))
    B = B @ B.T + jnp.eye(2) * 0.1
    C = jax.random.normal(k3, (2, 2))
    C = C @ C.T + jnp.eye(2) * 0.1
    D = jax.random.normal(k4, (2, 2))
    D = D @ D.T + jnp.eye(2) * 0.1

    nested_kron = Kronecker(Kronecker(A, B), Kronecker(C, D))
    dense_kron = jnp.kron(jnp.kron(A, B), jnp.kron(C, D))

    # Get eigendecomposition
    Lambda, Q = linox.leigh(nested_kron)

    # Verify reconstruction: Q @ Lambda @ Q.T should equal original
    reconstructed = (Q @ Lambda @ Q.T).todense()
    assert jnp.allclose(reconstructed, dense_kron, atol=1e-5), (
        f"Reconstruction error: {jnp.max(jnp.abs(reconstructed - dense_kron))}"
    )


def test_extract_kronecker_factors_simple() -> None:
    """Test extract_kronecker_factors with simple Kronecker product."""
    A = linox.Matrix(jnp.eye(3))
    B = linox.Matrix(jnp.eye(4))
    kron = Kronecker(A, B)

    factors, scalar = extract_kronecker_factors(kron)

    assert len(factors) == 2
    assert scalar is None
    assert factors[0].shape == (3, 3)
    assert factors[1].shape == (4, 4)


def test_extract_kronecker_factors_nested() -> None:
    """Test extract_kronecker_factors with nested Kronecker products."""
    A = linox.Matrix(jnp.eye(2))
    B = linox.Matrix(jnp.eye(3))
    C = linox.Matrix(jnp.eye(4))

    nested_kron = Kronecker(A, Kronecker(B, C))

    factors, scalar = extract_kronecker_factors(nested_kron)

    assert len(factors) == 3
    assert scalar is None
    assert factors[0].shape == (2, 2)
    assert factors[1].shape == (3, 3)
    assert factors[2].shape == (4, 4)


def test_extract_kronecker_factors_with_scalar() -> None:
    """Test extract_kronecker_factors with ScaledLinearOperator wrapper."""
    A = linox.Matrix(jnp.eye(3))
    B = linox.Matrix(jnp.eye(4))
    kron = Kronecker(A, B)
    scaled_kron = 2.5 * kron

    factors, scalar = extract_kronecker_factors(scaled_kron)

    assert len(factors) == 2
    assert jnp.isclose(scalar, 2.5)


def test_topk_eigh_with_factors() -> None:
    """Test topk_eigh with list of factors."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    A = jax.random.normal(k1, (3, 3))
    A = A @ A.T + jnp.eye(3) * 0.1
    B = jax.random.normal(k2, (4, 4))
    B = B @ B.T + jnp.eye(4) * 0.1

    k = 5
    eigs, vecs = topk_eigh([linox.Matrix(A), linox.Matrix(B)], k=k, largest=True)

    # Verify with dense computation
    dense_kron = jnp.kron(A, B)
    dense_eigs, _ = jnp.linalg.eigh(dense_kron)
    dense_topk = jnp.sort(dense_eigs)[::-1][:k]

    assert jnp.allclose(eigs, dense_topk, atol=1e-5)
    assert isinstance(vecs, KroneckerSelectedEigenvectors)
    assert vecs.shape == (12, k)


def test_topk_eigh_with_kronecker_operator() -> None:
    """Test topk_eigh with single Kronecker LinearOperator."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    A = jax.random.normal(k1, (3, 3))
    A = A @ A.T + jnp.eye(3) * 0.1
    B = jax.random.normal(k2, (4, 4))
    B = B @ B.T + jnp.eye(4) * 0.1

    kron = Kronecker(linox.Matrix(A), linox.Matrix(B))

    k = 5
    eigs, vecs = topk_eigh(kron, k=k, largest=True)

    # Verify with dense computation
    dense_kron = jnp.kron(A, B)
    dense_eigs, _ = jnp.linalg.eigh(dense_kron)
    dense_topk = jnp.sort(dense_eigs)[::-1][:k]

    assert jnp.allclose(eigs, dense_topk, atol=1e-5)


def test_topk_eigh_with_scaled_kronecker() -> None:
    """Test topk_eigh with ScaledLinearOperator(Kronecker(...))."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    A = jax.random.normal(k1, (3, 3))
    A = A @ A.T + jnp.eye(3) * 0.1
    B = jax.random.normal(k2, (4, 4))
    B = B @ B.T + jnp.eye(4) * 0.1

    scale = 2.0
    kron = Kronecker(linox.Matrix(A), linox.Matrix(B))
    scaled_kron = scale * kron

    k = 5
    eigs, vecs = topk_eigh(scaled_kron, k=k, largest=True)

    # Verify with dense computation
    dense_kron = scale * jnp.kron(A, B)
    dense_eigs, _ = jnp.linalg.eigh(dense_kron)
    dense_topk = jnp.sort(dense_eigs)[::-1][:k]

    assert jnp.allclose(eigs, dense_topk, atol=1e-5)


def test_topk_eigh_with_nested_kronecker() -> None:
    """Test topk_eigh with nested Kronecker structure."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    A = jax.random.normal(k1, (2, 2))
    A = A @ A.T + jnp.eye(2) * 0.1
    B = jax.random.normal(k2, (3, 3))
    B = B @ B.T + jnp.eye(3) * 0.1
    C = jax.random.normal(k3, (2, 2))
    C = C @ C.T + jnp.eye(2) * 0.1

    # Nested: A ⊗ (B ⊗ C)
    nested_kron = Kronecker(linox.Matrix(A), Kronecker(linox.Matrix(B), linox.Matrix(C)))

    k = 5
    eigs, vecs = topk_eigh(nested_kron, k=k, largest=True)

    # Verify with dense computation
    dense_kron = jnp.kron(A, jnp.kron(B, C))
    dense_eigs, _ = jnp.linalg.eigh(dense_kron)
    dense_topk = jnp.sort(dense_eigs)[::-1][:k]

    assert jnp.allclose(eigs, dense_topk, atol=1e-5)


def test_topk_eigh_eigenvector_correctness() -> None:
    """Test that topk_eigh eigenvectors satisfy the eigenvalue equation."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    A = jax.random.normal(k1, (3, 3))
    A = A @ A.T + jnp.eye(3) * 0.1
    B = jax.random.normal(k2, (4, 4))
    B = B @ B.T + jnp.eye(4) * 0.1

    kron = Kronecker(linox.Matrix(A), linox.Matrix(B))
    dense_kron = jnp.kron(A, B)

    k = 3
    eigs, vecs = topk_eigh(kron, k=k, largest=True)

    # Verify Av = λv for each eigenpair
    for i in range(k):
        lam = eigs[i]
        e_i = jnp.zeros(k).at[i].set(1.0)
        v = vecs @ e_i  # Get i-th eigenvector
        Av = dense_kron @ v
        residual = jnp.linalg.norm(Av - lam * v)
        assert residual < 1e-5, f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}"
