import jax
import jax.numpy as jnp
import pytest

import linox as lo
from linox.kernels.kernel import L2InnerProductKernel, Matern32Kernel

# --- Mock Infrastructure based on calibrate.py ---


class SobolevInnerProductKernel:
    """Mock H1 Kernel."""

    def __init__(self, bias=1.0) -> None:
        self.bias = bias

    def __call__(self, x, y=None):
        if y is None:
            y = x
        # Mocking H1 as L2 + something, for testing algebra validity
        return jnp.sum(x * y) + self.bias


def _make_additive_kernel(k1, k2):
    def k(x, y=None):
        return k1(x, y) + k2(x, y)

    return k


# Minimal registry
def get_kernel_fn(name, **params):
    if name == "M32":
        return Matern32Kernel(
            learnable=False, lengthscale=params.get("lengthscale", 1.0)
        )
    if name == "L2":
        return L2InnerProductKernel(learnable=False, bias=params.get("bias", 1.0))
    if name == "H1":
        return SobolevInnerProductKernel(bias=params.get("bias", 1.0))
    if name == "SM32_H1":
        return _make_additive_kernel(
            Matern32Kernel(learnable=False, lengthscale=params.get("lengthscale", 1.0)),
            SobolevInnerProductKernel(bias=params.get("bias", 1.0)),
        )
    msg = f"Unknown kernel: {name}"
    raise ValueError(msg)


# --- Test Cases ---


def build_1d_problem(n_spatial=10, n_time=5):
    """1D case from calibrate.py:
    x (M32)
    u (L2)
    time (M32) => Identity fallback if no channel_dims logic in simplified build.

    Structure:
    K_spatial = M32(x)
    K_function = L2(u)
    K_time = Identity(T) (simplified as per build_time_kernel fallback)

    K_base = K_function kron K_spatial
    K_base = K_base kron K_time
    K = output_scale * K_base + noise * I
    """
    # Grid/Data
    x = jnp.linspace(0, 1, n_spatial)[:, None]  # Spatial grid
    u = jnp.array([1.0])[:, None]  # Function input (dummy 1D feature)

    # Params
    params = {
        "spatial_x": {"lengthscale": 0.5},
        "function_u": {"bias": 0.1},
        "output_scale": 1.0,
        "noise_variance": 1e-3,
    }

    # 1. Spatial Kernel (Toeplitz or ArrayKernel)
    # x is uniform, M32 is stationary -> Toeplitz
    k_sp_fn = get_kernel_fn("M32", **params["spatial_x"])
    # K_spatial = lo.kernel_operator(k_sp_fn, x, is_stationary=True, assume_uniform=False)
    # We use ArrayKernel for simplicity unless we specifically want to test Toeplitz detection logic
    # But lo.kernel_operator tries to be smart. Let's explicitly use lo.kernel_operator
    K_spatial = lo.kernel_operator(k_sp_fn, x, is_stationary=True, assume_uniform=False)

    # 2. Function Kernel
    k_fn_fn = get_kernel_fn("L2", **params["function_u"])
    K_function = lo.kernel_operator(k_fn_fn, u, is_stationary=False)

    # 3. Base Kernel
    # calibrate.py: K_base = K_spatial kron K_function (logic varies, let's follow
    # "K_base = K_spatial kron K_function" mentioned in get_kernel_structure_string?)
    # Lines say:
    # if spatial and function: K_base = K_spatial kron K_function (WAIT: build_kernel does Kronecker(fop, sop))
    # So Fop (Function) comes FIRST in KroneckerArgs?
    # lo.Kronecker(fop, sop) -> A=fop, B=sop.

    K_base_sf = lo.Kronecker(K_function, K_spatial)

    # 4. Time Kernel
    # calibrate.py: base = lo.Kronecker(base, self.build_time_kernel(params))
    # If time dims exist. Let's assume M32 time kernel as per "1" config in get_structure_config_for_data
    t = jnp.arange(n_time, dtype=float)[:, None]
    k_time_fn = get_kernel_fn("M32", lengthscale=2.0)
    K_time = lo.kernel_operator(k_time_fn, t, is_stationary=True, assume_uniform=False)

    K_total_base = lo.Kronecker(K_base_sf, K_time)

    # 5. Scale and Noise
    K_scaled = lo.ScaledLinearOperator(K_total_base, jnp.array(params["output_scale"]))
    K_final = lo.IsotropicAdditiveLinearOperator(
        jnp.array(params["noise_variance"]), K_scaled
    )

    return K_final, (n_spatial * 1 * n_time)  # 1 is dim of u


def build_2d_problem(n_spatial=8, n_time=3):
    """2D case from calibrate.py:
    x (M32), y (M32) => Spatial
    u (SM32_H1)      => Function
    time (M32)       => Time.
    """
    # Grid/Data (2D grid, flattened for build_axis_kernel?)
    # lo.kernel_operator handles data.
    # If we have 2 spatial dims (x, y), build_axis_kernel receives them as list of coords.
    # And produces Kronecker(Kx, Ky).

    # Coords
    x = jnp.linspace(0, 1, n_spatial)[:, None]
    y = jnp.linspace(0, 1, n_spatial)[:, None]

    u = jnp.array([1.0, -1.0])[:, None]  # Function input

    params = {
        "spatial_x": {"lengthscale": 0.5},
        "spatial_y": {"lengthscale": 0.8},
        "function_u": {"lengthscale": 1.5, "bias": 0.5},
    }

    # Spatial: Kx (kron) Ky
    kx_fn = get_kernel_fn("M32", **params["spatial_x"])
    Kx = lo.kernel_operator(kx_fn, x, is_stationary=True, assume_uniform=False)

    ky_fn = get_kernel_fn("M32", **params["spatial_y"])
    Ky = lo.kernel_operator(ky_fn, y, is_stationary=True, assume_uniform=False)

    K_spatial = lo.Kronecker(Kx, Ky)

    # Function: SM32_H1
    ku_fn = get_kernel_fn("SM32_H1", **params["function_u"])
    K_function = lo.kernel_operator(
        ku_fn, u, is_stationary=False
    )  # Not stationary because H1/L2 part

    # Base: Func kron Spatial
    K_base_sf = lo.Kronecker(K_function, K_spatial)

    # Time
    t = jnp.arange(n_time, dtype=float)[:, None]
    kt_fn = get_kernel_fn("M32", lengthscale=1.0)
    K_time = lo.kernel_operator(kt_fn, t, is_stationary=True, assume_uniform=False)

    K_total_base = lo.Kronecker(K_base_sf, K_time)

    # Scale/Noise
    output_scale = 2.0
    noise_var = 1e-4

    K_scaled = lo.ScaledLinearOperator(K_total_base, jnp.array(output_scale))
    K_final = lo.IsotropicAdditiveLinearOperator(jnp.array(noise_var), K_scaled)

    # Total size = size(u) * size(x) * size(y) * size(t) = 2 * 8 * 8 * 3
    total_size = 2 * n_spatial * n_spatial * n_time

    return K_final, total_size


class TestGPKernels1D:
    def test_matmul_shapes(self) -> None:
        n_spatial = 10
        n_time = 5
        K, size = build_1d_problem(n_spatial, n_time)

        assert K.shape == (size, size)

        # Test matmul
        rng = jax.random.PRNGKey(0)
        v = jax.random.normal(rng, (size,))
        res = K @ v
        assert res.shape == (size,)
        assert jnp.all(jnp.isfinite(res))

    def test_lsolve(self) -> None:
        n_spatial = 8
        n_time = 2
        K, size = build_1d_problem(n_spatial, n_time)

        rng = jax.random.PRNGKey(1)
        y = jax.random.normal(rng, (size,))

        # Solve K x = y
        x_sol = lo.lsolve(K, y)

        # Verify: K @ x_sol approx y
        y_recon = K @ x_sol
        assert jnp.allclose(y, y_recon, atol=1e-4, rtol=1e-4)

    def test_slogdet(self) -> None:
        n_spatial = 5
        n_time = 2
        K, _size = build_1d_problem(n_spatial, n_time)

        sign, logdet = lo.slogdet(K)

        assert sign == 1.0  # SPD
        assert jnp.isfinite(logdet)

        # Check against dense (for small size)
        K_dense = K.todense()
        _sign_d, logdet_d = jnp.linalg.slogdet(K_dense)
        assert jnp.allclose(logdet, logdet_d, atol=1e-3)


class TestGPKernels2D:
    def test_matmul_shapes(self) -> None:
        n_spatial = 4
        n_time = 2
        K, size = build_2d_problem(n_spatial, n_time)  # 2 * 4*4 * 2 = 64

        assert K.shape == (size, size)

        rng = jax.random.PRNGKey(2)
        v = jax.random.normal(rng, (size,))
        res = K @ v
        assert res.shape == (size,)

    def test_prediction_pipeline_ops(self) -> None:
        """Mock the operations in predict()."""
        # prediction involves:
        # build_kernel(train) -> K_train
        # lsolve(K_train, y) -> alpha
        # predict mean = K_cross @ alpha
        # predict cov = K_test - K_cross @ linverse(K_train) @ K_cross.T

        n_spatial = 4
        n_time = 2
        K_train, size = build_2d_problem(n_spatial, n_time)

        rng = jax.random.PRNGKey(3)
        y = jax.random.normal(rng, (size,))

        # 1. lsolve
        alpha = lo.lsolve(K_train, y)
        assert alpha.shape == (size,)

        # 2. linverse
        Ki = lo.linverse(K_train)

        # Verify inverse
        # Ki @ K_train approx I
        # Only check matvec for efficiency
        v = jax.random.normal(rng, (size,))
        res = Ki @ (K_train @ v)
        assert jnp.allclose(res, v, atol=1e-3)

    def test_sample_posterior_ops(self) -> None:
        """Mock operations in sample_posterior."""
        # Uses lsqrt(IsotropicAdditiveLinearOperator(1e-6, pred_cov))

        n_spatial = 4
        n_time = 2
        K_train, size = build_2d_problem(n_spatial, n_time)

        # lsqrt of the kernel itself (as a proxy for posterior cov)
        L = lo.lsqrt(K_train)

        # Check L @ L.T = K
        # Stochastic check via matvecs
        rng = jax.random.PRNGKey(4)
        v = jax.random.normal(rng, (size,))

        # K v
        Kv = K_train @ v

        # L L.T v
        LLtv = L @ (L.T @ v)

        assert jnp.allclose(Kv, LLtv, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
