import jax.numpy as jnp

import linox as lo
from linox._kronecker import Kronecker
from linox._toeplitz import Toeplitz


def test_debug_shapes() -> None:
    # K_time (2x2 Toeplitz)
    kt_v = jnp.array([1.0, 0.5])
    K_time = Toeplitz(kt_v)

    # K_sf (32x32) - Mock with Matrix for simplicity first, then Kronecker
    # K_sf = Kronecker(K_func(2x2), K_spatial(16x16))

    # Mock K_sf as Identity(32) to isolate Toeplitz in Kronecker(K_sf, K_time)
    K_sf = lo.Identity(32)

    K_total = Kronecker(K_sf, K_time)

    # Vec (64, 1)
    v = jnp.ones((64, 1))

    try:
        K_total @ v
    except Exception:
        import traceback

        traceback.print_exc()

    # Now reproduction of 2D case structure
    # K_func (2x2)
    K_func = lo.Identity(2)  # Matrix
    # K_spatial (16x16)
    K_spatial = lo.Identity(16)  # Matrix

    K_sf_real = Kronecker(K_func, K_spatial)  # 32x32

    K_total_real = Kronecker(K_sf_real, K_time)  # 64x64

    try:
        K_total_real @ v
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    test_debug_shapes()
