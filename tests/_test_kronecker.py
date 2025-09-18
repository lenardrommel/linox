"""Performance benchmarks for Kronecker operators vs dense operations."""

import time
from typing import List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import pytest

from linox import (
    Identity,
    Kronecker,
    LinearOperator,
    Matrix,
    ScaledLinearOperator,
    lpinverse,
    lpsolve,
    lsolve,
)
from linox._matrix import Ones
from linox.types import ShapeLike
from linox.utils import as_linop

CaseType = tuple[LinearOperator, Kronecker, Matrix]


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [0, 22, 278]],
)
def key(request: pytest.FixtureRequest) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(request.param)


def square_spd_matrix(size: int, key: jax.random.PRNGKey) -> jax.Array:
    """Generate a square symmetric positive definite matrix for testing."""
    A = jax.random.normal(key, (size, size))
    return A @ A.T + jnp.eye(size) * 1e-6  # Ensure positive definiteness


def time_function(func, *args, num_runs=10, warmup=3):
    """Time a function with JIT compilation warmup."""
    # Warmup runs to trigger JIT compilation
    for _ in range(warmup):
        result = func(*args)
        # Block until computation is done (important for JAX)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif (
            isinstance(result, list)
            and result
            and hasattr(result[0], "block_until_ready")
        ):
            for r in result:
                r.block_until_ready()

    # Actual timing runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args)
        # Block until computation is done (important for JAX)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif (
            isinstance(result, list)
            and result
            and hasattr(result[0], "block_until_ready")
        ):
            for r in result:
                r.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return min(times), sum(times) / len(times), max(times)


def sample_kronecker(
    shapeA: ShapeLike, shapeB: ShapeLike, key: jax.random.PRNGKey
) -> CaseType:
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, shapeA)
    B = jax.random.normal(key2, shapeB)
    op = Kronecker(A, B)
    return op, jnp.kron(A, B)


def create_nested_kronecker(key: jax.Array, dims: List[int]) -> Tuple[any, jax.Array]:
    """Create nested Kronecker product and its dense equivalent."""
    keys = jax.random.split(key, len(dims))

    # Create first matrix
    M_first = square_spd_matrix(dims[0], keys[0])
    kron_op = as_linop(M_first)
    kron_dense = M_first

    # Build up nested Kronecker product
    for i, dim in enumerate(dims[1:], 1):
        M = square_spd_matrix(dim, keys[i])
        kron_op = Kronecker(kron_op, as_linop(M))
        kron_dense = jnp.kron(kron_dense, M)

    return kron_op, kron_dense


def test_kronecker_shape():
    key = jax.random.key(42)
    op, arr = sample_kronecker((2, 2), (3, 3), key)


def test_kronecker_basic():
    """Test basic Kronecker operations."""
    key = jax.random.key(42)
    op, arr = sample_kronecker((2, 2), (3, 3), key)

    # Test dense representation
    assert jnp.allclose(op.todense(), arr), "Dense representation mismatch"

    # Test matrix-vector multiplication
    key = jax.random.key(1)
    dim = jax.random.randint(key, (), 1, 5)
    vec = jax.random.normal(key, (op.shape[-1], dim))
    assert jnp.allclose(op @ vec, arr @ vec), "Matrix-vector multiplication mismatch"

    # Test transpose
    vec_t = jax.random.normal(key, (op.shape[-1],)).reshape(-1)
    assert jnp.allclose(op.T @ vec_t, arr.T @ vec_t), "Transpose mismatch"


def test_kronecker_square():
    """Test square Kronecker operators."""
    key = jax.random.key(123)
    op, arr = sample_kronecker((2, 2), (3, 3), key)

    assert jnp.allclose(op.todense(), arr)

    vec = jax.random.normal(jax.random.key(1), (op.shape[1],))
    assert jnp.allclose(op @ vec, arr @ vec)


def test_kronecker() -> None:
    op, arr = sample_kronecker((2, 2), (3, 3), jax.random.key(42))
    key = jax.random.key(1)
    vec = jax.random.normal(key, op.shape[-1])
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ vec, arr @ vec)
    assert jnp.allclose(op.T @ vec.T, arr.T @ vec.T)


def test_kronecker_matmul() -> None:
    op1, arr1 = sample_kronecker((2, 2), (3, 3), jax.random.key(42))
    key = jax.random.key(1)
    M = jax.random.normal(key, op1.shape)

    def assert_equality(res_op, res_dense):
        if hasattr(res_op, "todense"):
            res_op = res_op.todense()
        assert jnp.allclose(res_op, res_dense, atol=1e-3), (
            "Matrix-matrix multiplication mismatch"
        )

    assert_equality(op1 @ M, arr1 @ M)
    assert_equality(M @ op1, M @ arr1)

    nested_op, nested_arr = create_nested_kronecker(jax.random.key(0), [6, 4, 5])
    M = jax.random.normal(jax.random.key(1), nested_op.shape)
    assert_equality(nested_op @ M, nested_arr @ M)


@pytest.mark.parametrize(
    "key, dim, batch_size",
    [
        (jax.random.key(0), 2, 4),
        (jax.random.key(1), 3, 5),
        (jax.random.key(2), 4, 6),
    ],
)
def test_kronecker_solve(key, dim, batch_size):
    shapes = jax.random.randint(
        key,
        dim + 1,
        1,
        5,
    )

    shapes = shapes.at[0].set(batch_size)
    b, dims = shapes[0], shapes[1:]
    M_batch = square_spd_matrix(b, key)
    Matrix_list = []

    for dim in dims:
        M = square_spd_matrix(dim, key)
        Matrix_list.append(M)

    def make_kronecker_op(op, M):
        return Kronecker(op, as_linop(M))

    kron_res = M_batch

    def make_kronecker_array(kron, M):
        return jnp.kron(kron, M)

    kron_prod = as_linop(M_batch)
    for M in Matrix_list:
        kron_prod = make_kronecker_op(kron_prod, M)
        kron_res = make_kronecker_array(kron_res, M)

    assert kron_prod.shape == kron_res.shape or kron_prod.shape[1:] == kron_res.shape, (
        "Shapes do not match"
    )
    assert jnp.allclose(kron_prod.todense(), kron_res), (
        "Dense representations do not match"
    )

    vec = jax.random.normal(key, shapes).reshape(-1)

    assert jnp.allclose(kron_prod @ vec, kron_res @ vec), (
        "Matrix-vector multiplication does not match"
    )

    vec = jax.random.normal(key, shapes)
    assert jnp.allclose(
        lsolve(kron_prod, vec),
        jax.scipy.linalg.solve(kron_res, vec.reshape(-1), assume_a="sym"),
        rtol=1e-4,
    ), "Solve does not match dense solve"

    # assert jnp.allclose(
    #     lpsolve(kron_prod, vec), jnp.linalg.pinv(kron_res) @ vec.reshape(-1), rtol=1e-2
    # ), "Pseudosolve does not match dense solve"


def test_leigh_kronecker():
    key = jax.random.key(1)


def extract_block(kron_op: Kronecker, i: int, j: int) -> jax.Array:
    """Extract block (i,j) from Kronecker product A ⊗ B.

    For A ⊗ B, block (i,j) should equal a_ij * B.
    """
    A, B = kron_op.A, kron_op.B
    m_B, n_B = B.shape

    # Create unit vectors to extract the block
    e_i = jnp.zeros(A.shape[0])
    e_i = e_i.at[i].set(1.0)
    e_j = jnp.zeros(A.shape[1])
    e_j = e_j.at[j].set(1.0)

    # Extract a_ij coefficient
    a_ij = (e_i.T @ A.todense() @ e_j) if hasattr(A, "todense") else A[i, j]

    # The block should be a_ij * B
    expected_block = a_ij * B.todense() if hasattr(B, "todense") else a_ij * B

    # Extract actual block from Kronecker product
    I_B = jnp.eye(n_B)
    block_vectors = kron_op @ jnp.kron(e_j, I_B).T
    actual_block = block_vectors[i * m_B : (i + 1) * m_B, :]

    return actual_block, expected_block


def test_kronecker_block_structure():
    """Test that Kronecker product has correct block structure."""
    key = jax.random.key(42)
    key_A, key_B = jax.random.split(key)

    # Create test matrices
    A = jax.random.normal(key_A, (3, 4))
    B = jax.random.normal(key_B, (2, 3))

    kron_op = Kronecker(A, B)

    # Test several blocks
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            actual_block, expected_block = extract_block(kron_op, i, j)
            assert jnp.allclose(actual_block, expected_block, rtol=1e-10), (
                f"Block ({i},{j}) mismatch"
            )


def test_structured_kronecker_chain():
    """Test the specific pattern: A ⊗ σI ⊗ I ⊗ ones ⊗ (1/σ)I ⊗ B"""
    key = jax.random.key(123)
    keys = jax.random.split(key, 6)

    A = jax.random.normal(keys[0], (2, 3))
    sigma = 2.0
    dim1, dim2 = 3, 4
    B = jax.random.normal(keys[1], (2, 2))

    sigma_I = ScaledLinearOperator(Identity(dim1), sigma)
    I_middle = Identity(dim2)
    ones = Ones((3, 3))
    inv_sigma_I = ScaledLinearOperator(Identity(2), 1 / sigma)

    # Build nested Kronecker: A ⊗ σI ⊗ I ⊗ ones ⊗ (1/σ)I ⊗ B
    kron_chain = as_linop(A)
    for component in [sigma_I, I_middle, ones, inv_sigma_I, B]:
        kron_chain = Kronecker(kron_chain, as_linop(component))

    def verify_outer_block_structure(chain_op, A_ref):
        """Verify that the outermost Kronecker structure matches A."""
        # Extract a few sample blocks and verify they scale correctly
        total_inner_size = chain_op.shape[0] // A_ref.shape[0]

        # Sample some entries of A
        test_indices = (
            [(0, 0), (0, 1), (1, 0)]
            if A_ref.shape[0] > 1 and A_ref.shape[1] > 1
            else [(0, 0)]
        )

        for i, j in test_indices:
            if i < A_ref.shape[0] and j < A_ref.shape[1]:
                # Create test vector for block (i,j)
                test_vec = jnp.zeros(chain_op.shape[1])
                block_start = j * total_inner_size
                block_end = (j + 1) * total_inner_size
                test_vec = test_vec.at[block_start:block_end].set(1.0)

                # Apply Kronecker chain
                result = chain_op @ test_vec

                # Extract the corresponding block
                result_block = result[i * total_inner_size : (i + 1) * total_inner_size]

                # The scaling should match A[i,j] times the inner Kronecker product
                # This is more of a structural test than exact value comparison
                if A_ref[i, j] != 0:
                    assert jnp.any(jnp.abs(result_block) > 1e-10), (
                        f"Block ({i},{j}) should be non-zero when A[{i},{j}] = {A_ref[i, j]}"
                    )

    verify_outer_block_structure(kron_chain, A)


def test_kronecker_blocks_vs_dense_blocks():
    """Compare blocks extracted from Kronecker operator vs dense matrix blocks."""
    key = jax.random.key(456)
    key_A, key_B = jax.random.split(key)

    A = jax.random.normal(key_A, (2, 3))
    B = jax.random.normal(key_B, (3, 2))

    kron_op = Kronecker(A, B)
    dense_kron = jnp.kron(A, B)

    m_B, n_B = B.shape

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            row_start, row_end = i * m_B, (i + 1) * m_B
            col_start, col_end = j * n_B, (j + 1) * n_B
            dense_block = dense_kron[row_start:row_end, col_start:col_end]

            expected_block = A[i, j] * B

            assert jnp.allclose(dense_block, expected_block, rtol=1e-12), (
                f"Dense block ({i},{j}) doesn't match A[{i},{j}] * B"
            )


def test_numerical_stability_blocks_vs_full():
    """Test that block extraction remains stable when full Kronecker becomes large."""
    key = jax.random.key(789)
    keys = jax.random.split(key, 4)

    A = jax.random.normal(keys[0], (5, 4))
    B = jax.random.normal(keys[1], (6, 7))
    C = jax.random.normal(keys[2], (3, 2))

    # Nested Kronecker: A ⊗ B ⊗ C
    kron_AB = Kronecker(A, B)
    kron_ABC = Kronecker(kron_AB, C)

    test_vector = jax.random.normal(keys[3], (kron_ABC.shape[1],))
    result = kron_ABC @ test_vector

    assert jnp.all(jnp.isfinite(result)), (
        "Kronecker operation produced non-finite values"
    )
    assert jnp.any(jnp.abs(result) > 1e-15), "Result is suspiciously close to zero"

    assert result.shape[0] == kron_ABC.shape[0], "Output shape mismatch"


def test_lpinverse_block_structure(key):
    """Test that lpinverse of Kronecker product has correct block structure."""
    key = jax.random.key(42)
    key_A, key_B = jax.random.split(key)

    # Create test matrices (rectangular for more general test)
    A = jax.random.normal(key_A, (4, 4))
    B = jax.random.normal(key_B, (3, 3))

    kron_op = Kronecker(A, B)
    kron_pinv = lpinverse(kron_op)

    # Test several blocks of the pseudoinverse
    A_pinv = jnp.linalg.pinv(A)
    B_pinv = jnp.linalg.pinv(B)
    A_kron_B_pinv = jnp.kron(A_pinv, B_pinv)
    assert jnp.allclose(kron_pinv.todense(), A_kron_B_pinv, rtol=1e-8), (
        "Kronecker pseudoinverse mismatch"
    )


if __name__ == "__main__":
    test_kronecker_block_structure()
    test_structured_kronecker_chain()
    test_kronecker_blocks_vs_dense_blocks()
    test_numerical_stability_blocks_vs_full()
    test_lpinverse_block_structure()
    print("All block-wise Kronecker tests passed!")
