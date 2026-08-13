# test_kernel.py


import jax
import jax.numpy as jnp
import linox as lo
import pytest
import pytest_cases
from linox._types import ShapeType
from linox.utils.debug import inspect_run

CaseType = tuple[lo.LinearOperator, jax.Array]


def _inner_product_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
    return jnp.dot(x1, x2) + 1e-8


def _rbf_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
    diff = x1 - x2
    return jnp.exp(-0.5 * jnp.dot(diff, diff))


def sample_kernel(shape: ShapeType) -> CaseType:
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, shape)
    y = jax.random.normal(key, shape)
    linop = lo.ArrayKernel(_inner_product_kernel, x, y)
    matrix = jnp.dot(x, y.T)
    return linop, matrix


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


basic_shapes = [1, 2, 10, 100]


@pytest.mark.parametrize("shape", basic_shapes)
def case_kernel(shape: ShapeType) -> CaseType:
    linop, matrix = sample_kernel((shape, shape))
    return linop, matrix


# ============================================================================
# TIER A: Small-n Correctness Tests (OK to densify for verification)
# ============================================================================
@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_to_dense(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    assert jnp.allclose(linop.todense(), matrix, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_mv(linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey) -> None:
    vector = jax.random.normal(key, (matrix.shape[-1],))
    assert jnp.allclose(linop @ vector, matrix @ vector, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop, matrix", cases=[case_kernel])
def test_shape(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    assert linop.shape == matrix.shape


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_matmat(
    linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey, ncols: int
) -> None:
    mat = jax.random.normal(key, (matrix.shape[-1], ncols))
    assert jnp.allclose(linop @ mat, matrix @ mat, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_scalar_mul(
    linop: lo.ArrayKernel, matrix: jax.Array, key: jax.random.PRNGKey
) -> None:
    scalar = jax.random.normal(key, ())
    assert jnp.allclose((scalar * linop).todense(), scalar * matrix, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_transpose(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    result_linop = linop.T
    expected_transposed = matrix.swapaxes(-1, -2)
    assert jnp.allclose(result_linop.todense(), expected_transposed, atol=1e-6)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=[case_kernel])
def test_positive_definiteness(linop: lo.ArrayKernel, matrix: jax.Array) -> None:
    leigh_vals = lo.leigh(linop)[0]
    eigvals = jnp.linalg.eigvalsh(matrix)
    assert jnp.all(leigh_vals >= 0), f"Leigh decomposition failed for {linop}"
    assert jnp.all(eigvals >= 0), f"Eigenvalues of {matrix} are not positive"
    assert jnp.allclose(leigh_vals, eigvals, atol=1e-6), (
        f"Leigh decomposition and eigenvalues do not match for {linop}"
    )


@pytest.mark.parametrize("n", [10, 50, 100])
def test_toeplitzkernel_correctness(n: int) -> None:
    x = jnp.linspace(0, 1, n).reshape(-1, 1)
    K_toeplitz = lo.ToeplitzKernel(_rbf_kernel, x)
    kernel_fn = jax.vmap(
        jax.vmap(_rbf_kernel, in_axes=(None, 0)),
        in_axes=(0, None),
    )
    K_dense = kernel_fn(x, x)
    assert jnp.allclose(K_toeplitz.todense(), K_dense, atol=1e-6)
    key = jax.random.PRNGKey(42)
    v = jax.random.normal(key, (n,))
    assert jnp.allclose(K_toeplitz @ v, K_dense @ v, atol=1e-6)


def test_toeplitzkernel_transpose_is_self() -> None:
    n = 50
    x = jnp.linspace(0, 1, n).reshape(-1, 1)
    K = lo.ToeplitzKernel(_rbf_kernel, x)
    assert K.transpose() is K
    K_dense = K.todense()
    assert jnp.allclose(K_dense, K_dense.T, atol=1e-6)


# ============================================================================
# Factory Selection Tests (no device sync)
# ============================================================================
def test_kernel_operator_selects_toeplitz_for_uniform_self_cov() -> None:
    n = 100
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.kernel_operator(_rbf_kernel, x, is_stationary=True, assume_uniform=True)
    assert isinstance(K, lo.ToeplitzKernel)


def test_kernel_operator_selects_toeplitz_with_identity_x1() -> None:
    n = 100
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.kernel_operator(
        _rbf_kernel, x, x1=x, is_stationary=True, assume_uniform=True
    )
    assert isinstance(K, lo.ToeplitzKernel)


def test_kernel_operator_selects_arraykernel_for_different_x1() -> None:
    n = 100
    x0 = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    x1 = jnp.arange(n, 2 * n).reshape(-1, 1).astype(jnp.float32)
    K = lo.kernel_operator(_rbf_kernel, x0, x1, is_stationary=True, assume_uniform=True)
    assert isinstance(K, lo.ArrayKernel)
    assert not isinstance(K, lo.ToeplitzKernel)


def test_kernel_operator_selects_arraykernel_for_nonstationary() -> None:
    n = 100
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.kernel_operator(_rbf_kernel, x, is_stationary=False, assume_uniform=True)
    assert isinstance(K, lo.ArrayKernel)
    assert not isinstance(K, lo.ToeplitzKernel)


def test_kernel_operator_no_device_sync_for_large_n() -> None:
    n = 50000
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.kernel_operator(_rbf_kernel, x, is_stationary=True, assume_uniform=True)
    assert isinstance(K, lo.ToeplitzKernel)


# ============================================================================
# TIER B: Large-n "No Densification" Tests
# Use monkeypatch to make _todense raise AssertionError
# ============================================================================
class DensificationError(Exception):
    pass


def _raise_on_densify(*args, **kwargs):
    msg = "Attempted to densify!"
    raise DensificationError(msg)


@pytest.mark.parametrize("n", [5000, 10000])
def test_arraykernel_matmul_does_not_densify(n: int, monkeypatch) -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x)
    monkeypatch.setattr(lo.ArrayKernel, "_todense", _raise_on_densify)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))
    result = K @ v
    assert result.shape == (n,)


@pytest.mark.parametrize("n", [10000, 20000])
def test_toeplitzkernel_matmul_does_not_densify(n: int, monkeypatch) -> None:
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.ToeplitzKernel(_rbf_kernel, x)
    monkeypatch.setattr(lo.ToeplitzKernel, "_todense", _raise_on_densify)
    monkeypatch.setattr(lo.Toeplitz, "_todense", _raise_on_densify)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))
    result = K @ v
    assert result.shape == (n,)


@pytest.mark.parametrize("n", [5000])
def test_arraykernel_transpose_matmul_does_not_densify(n: int, monkeypatch) -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, 3))
    y = jax.random.normal(jax.random.PRNGKey(1), (n + 100, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x, y)
    K_T = K.T
    monkeypatch.setattr(lo.ArrayKernel, "_todense", _raise_on_densify)
    v = jax.random.normal(jax.random.PRNGKey(2), (n,))
    result = K_T @ v
    assert result.shape == (n + 100,)


# ============================================================================
# Tests using inspect_run to verify no densify events during matmul
# ============================================================================
def test_arraykernel_matmul_no_densify_event() -> None:
    n = 1000
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))

    def do_matmul():
        return K @ v

    result, report = inspect_run(do_matmul)
    assert result.shape == (n,)
    densify_events = [e for e in report.events if e.kind == "densify"]
    assert len(densify_events) == 0, f"Unexpected densify events: {densify_events}"


def test_toeplitzkernel_matmul_no_densify_event() -> None:
    n = 1000
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.ToeplitzKernel(_rbf_kernel, x)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))

    def do_matmul():
        return K @ v

    result, report = inspect_run(do_matmul)
    assert result.shape == (n,)
    densify_events = [e for e in report.events if e.kind == "densify"]
    assert len(densify_events) == 0, f"Unexpected densify events: {densify_events}"


def test_todense_emits_densify_event() -> None:
    n = 100
    x = jax.random.normal(jax.random.PRNGKey(0), (n, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x)

    def do_todense():
        return K.todense()

    result, report = inspect_run(do_todense)
    assert result.shape == (n, n)
    densify_events = [e for e in report.events if e.kind == "densify"]
    assert len(densify_events) == 1


def test_large_todense_emits_warn_event() -> None:
    n = 600
    x = jax.random.normal(jax.random.PRNGKey(0), (n, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x)

    def do_todense():
        return K._todense()

    result, report = inspect_run(do_todense)
    assert result.shape == (n, n)
    warn_events = [e for e in report.events if e.kind == "warn"]
    assert len(warn_events) >= 1, "Expected warning for large densification"


# ============================================================================
# JIT Compatibility Tests
# ============================================================================
def test_arraykernel_matmul_jit_compatible() -> None:
    n = 500
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))

    @jax.jit
    def matmul_fn(vec):
        return K @ vec

    result = matmul_fn(v)
    assert result.shape == (n,)
    result2 = matmul_fn(v)
    assert jnp.allclose(result, result2)


def test_toeplitzkernel_matmul_jit_compatible() -> None:
    n = 500
    x = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K = lo.ToeplitzKernel(_rbf_kernel, x)
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))

    @jax.jit
    def matmul_fn(vec):
        return K @ vec

    result = matmul_fn(v)
    assert result.shape == (n,)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================
def test_toeplitzkernel_rejects_different_x1() -> None:
    x0 = jnp.arange(10).reshape(-1, 1).astype(jnp.float32)
    x1 = jnp.arange(10, 20).reshape(-1, 1).astype(jnp.float32)
    with pytest.raises(ValueError, match="self-covariance"):
        lo.ToeplitzKernel(_rbf_kernel, x0, x1)


def test_arraykernel_cross_covariance_shape() -> None:
    n0, n1 = 50, 100
    x0 = jax.random.normal(jax.random.PRNGKey(0), (n0, 3))
    x1 = jax.random.normal(jax.random.PRNGKey(1), (n1, 3))
    K = lo.ArrayKernel(_inner_product_kernel, x0, x1)
    assert K.shape == (n0, n1)
    v = jax.random.normal(jax.random.PRNGKey(2), (n1,))
    result = K @ v
    assert result.shape == (n0,)


def test_toeplitzkernel_1d_flat_input() -> None:
    n = 100
    x = jnp.linspace(0, 1, n)

    def scalar_rbf(x1, x2):
        return jnp.exp(-0.5 * (x1 - x2) ** 2)

    K = lo.ToeplitzKernel(scalar_rbf, x)
    assert K.shape == (n, n)
    v = jax.random.normal(jax.random.PRNGKey(0), (n,))
    result = K @ v
    assert result.shape == (n,)


def test_arraykernel_different_chunk_sizes_same_result() -> None:
    n = 100
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, 3))
    v = jax.random.normal(jax.random.PRNGKey(1), (n,))
    K1 = lo.ArrayKernel(_inner_product_kernel, x, chunk_size=64)
    K2 = lo.ArrayKernel(_inner_product_kernel, x, chunk_size=512)
    result1 = K1 @ v
    result2 = K2 @ v
    assert jnp.allclose(result1, result2, atol=1e-6)



@pytest.mark.parametrize("n", [100])
def test_kronecker_of_toeplitz_matmul_does_not_densify(n: int, monkeypatch) -> None:
    x1 = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    x2 = jnp.arange(n).reshape(-1, 1).astype(jnp.float32)
    K1 = lo.ToeplitzKernel(_rbf_kernel, x1)
    K2 = lo.ToeplitzKernel(_rbf_kernel, x2)
    K_kron = lo.Kronecker(K1, K2)
    monkeypatch.setattr(lo.ToeplitzKernel, "_todense", _raise_on_densify)
    monkeypatch.setattr(lo.Toeplitz, "_todense", _raise_on_densify)
    total_size = n * n
    v = jax.random.normal(jax.random.PRNGKey(0), (total_size,))
    result = K_kron @ v
    assert result.shape == (total_size,)
