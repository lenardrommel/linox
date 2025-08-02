"""Performance benchmarks for Kronecker operators vs dense operations."""

import time
from typing import List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import pytest

from linox import Kronecker, LinearOperator, Matrix, lpsolve, lsolve
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


def test_kronecker_basic():
    """Test basic Kronecker operations."""
    key = jax.random.key(42)
    op, arr = sample_kronecker((2, 2), (3, 3), key)

    # Test dense representation
    assert jnp.allclose(op.todense(), arr), "Dense representation mismatch"

    # Test matrix-vector multiplication
    key = jax.random.key(1)
    vec = jax.random.normal(key, (op.shape[1],))
    assert jnp.allclose(op @ vec, arr @ vec), "Matrix-vector multiplication mismatch"

    # Test transpose
    vec_t = jax.random.normal(key, (op.shape[0],)).reshape(-1)
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
    vec = jax.random.normal(key, op.shape[::-1])
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ vec, arr @ vec)
    assert jnp.allclose(op.T @ vec.T, arr.T @ vec.T)
    assert jnp.allclose(op @ vec[..., 0], arr @ vec[..., 0])


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

    assert kron_prod.shape == kron_res.shape, "Shapes do not match"
    assert jnp.allclose(kron_prod.todense(), kron_res), (
        "Dense representations do not match"
    )

    vec = jax.random.normal(key, shapes).reshape(-1)

    assert jnp.allclose(kron_prod @ vec, kron_res @ vec), (
        "Matrix-vector multiplication does not match"
    )
