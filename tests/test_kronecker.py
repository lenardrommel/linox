"""Performance benchmarks for Kronecker operators vs dense operations."""

import time
from typing import List, Tuple, Type, Union

import jax
import jax.numpy as jnp

from linox import Kronecker, LinearOperator, Matrix, lpsolve, lsolve
from linox.types import ShapeLike
from linox.utils import as_linop

CaseType = tuple[LinearOperator, Kronecker, Matrix]


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


def make_psd(m: jax.Array) -> jax.Array:
    """Make a positive semidefinite matrix."""
    return m @ m.T + m.shape[0] * jnp.eye(m.shape[0]) * 1e-6


def sample_kronecker(shapeA: ShapeLike, shapeB: ShapeLike) -> CaseType:
    key = jax.random.key(5)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, shapeA)
    B = jax.random.normal(key2, shapeB)
    op = Kronecker(A, B)
    return op, jnp.kron(A, B)


def create_nested_kronecker(key: jax.Array, dims: List[int]) -> Tuple[any, jax.Array]:
    """Create nested Kronecker product and its dense equivalent."""
    keys = jax.random.split(key, len(dims))

    # Create first matrix
    M_first = make_psd(jax.random.uniform(keys[0], (dims[0], dims[0])))
    kron_op = as_linop(M_first)
    kron_dense = M_first

    # Build up nested Kronecker product
    for i, dim in enumerate(dims[1:], 1):
        M = make_psd(jax.random.uniform(keys[i], (dim, dim)))
        kron_op = Kronecker(kron_op, as_linop(M))
        kron_dense = jnp.kron(kron_dense, M)

    return kron_op, kron_dense


def test_kronecker_basic():
    """Test basic Kronecker operations."""
    key = jax.random.key(42)
    op, arr = sample_kronecker((2, 3), (3, 2), key)

    # Test dense representation
    assert jnp.allclose(op.todense(), arr, rtol=1e-10)

    # Test matrix-vector multiplication
    key = jax.random.key(1)
    vec = jax.random.normal(key, (op.shape[1],))
    assert jnp.allclose(op @ vec, arr @ vec, rtol=1e-10)

    # Test transpose
    vec_t = jax.random.normal(key, (op.shape[0],))
    assert jnp.allclose(op.T @ vec_t, arr.T @ vec_t, rtol=1e-10)


def test_kronecker_square():
    """Test square Kronecker operators."""
    key = jax.random.key(123)
    op, arr = sample_kronecker((2, 2), (3, 3), key)

    assert jnp.allclose(op.todense(), arr, rtol=1e-10)

    vec = jax.random.normal(jax.random.key(1), (op.shape[1],))
    assert jnp.allclose(op @ vec, arr @ vec, rtol=1e-10)


def test_kronecker() -> None:
    op, arr = sample_kronecker((2, 2), (3, 2))
    key = jax.random.key(1)
    vec = jax.random.normal(key, op.shape[::-1])
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ vec, arr @ vec)
    assert jnp.allclose(op.T @ vec.T, arr.T @ vec.T)
    assert jnp.allclose(op @ vec[..., 0], arr @ vec[..., 0])


def test_kronecker_solve(key, dim, batch_size):
    shapes = jax.random.randint(
        key,
        dim + 1,
        1,
        5,
    )

    shapes = shapes.at[0].set(batch_size)
    b, dims = shapes[0], shapes[1:]
    M_batch = make_psd(jax.random.uniform(key, (b, b)))
    Matrix_list = []

    for dim in dims:
        M = make_psd(jax.random.uniform(key, (dim, dim)))
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


def benchmark_matmul(dims: List[int], num_vectors: int = 5, num_runs: int = 10):
    """Benchmark matrix-vector multiplication."""
    key = jax.random.key(42)
    key_op, key_vec = jax.random.split(key)

    # Create operators
    kron_op, kron_dense = create_nested_kronecker(key_op, dims)

    # Create test vectors
    vec_keys = jax.random.split(key_vec, num_vectors)
    vectors = [jax.random.normal(k, (kron_op.shape[1],)) for k in vec_keys]

    print(f"\nBenchmarking dims={dims}, total_size={kron_op.shape}")
    print(
        f"Memory: Kronecker stores {sum(d * d for d in dims) * 8 / 1024:.1f}KB, "
        f"Dense would store {kron_op.shape[0] * kron_op.shape[1] * 8 / 1024 / 1024:.1f}MB"
    )

    # Benchmark Kronecker operator
    def kron_matmul():
        return [kron_op @ v for v in vectors]

    # Benchmark dense matrix
    def dense_matmul():
        return [kron_dense @ v for v in vectors]

    # Time both approaches
    kron_min, kron_avg, kron_max = time_function(kron_matmul, num_runs=num_runs)
    dense_min, dense_avg, dense_max = time_function(dense_matmul, num_runs=num_runs)

    speedup = dense_avg / kron_avg

    print(
        f"Dense:     {dense_avg * 1000:.2f}ms ± {(dense_max - dense_min) * 1000 / 2:.2f}ms"
    )
    print(f"Speedup:   {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")

    # Verify correctness
    kron_results = kron_matmul()
    dense_results = dense_matmul()
    max_error = max(
        jnp.max(jnp.abs(kr - dr)) for kr, dr in zip(kron_results, dense_results)
    )
    print(f"Max error: {max_error:.2e}")

    return {
        "dims": dims,
        "total_size": kron_op.shape[0],
        "kron_time": kron_avg,
        "dense_time": dense_avg,
        "speedup": speedup,
        "memory_ratio": (kron_op.shape[0] * kron_op.shape[1])
        / sum(d * d for d in dims),
    }


test_kronecker_solve(jax.random.key(0), 3, batch_size=2)


def test_performance_crossover():
    """Find the crossover point where Kronecker becomes faster."""
    print("Finding performance crossover point...")

    # Test progressively larger sizes
    crossover_found = False
    dims_list = [
        [13, 13, 13],  # 27x27
        [14, 14, 14],  # 64x64
        [15, 15, 15],  # 100x100
        [16, 16, 16],  # 180x180
        [18, 18, 18],  # 240x240
        [18, 18, 18],  # 384x384
        [10, 8, 6],  # 480x480
        [10, 10, 8],  # 800x800
    ]

    for dims in dims_list:
        result = benchmark_matmul(dims, num_vectors=1, num_runs=3)
        if result["speedup"] > 1.0 and not crossover_found:
            print(f"\n🎯 CROSSOVER FOUND at dims={dims}, size={result['total_size']}")
            crossover_found = True

        if result["total_size"] > 1000:  # Stop at reasonable size
            break

    return crossover_found


test_performance_crossover()
