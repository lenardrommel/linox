
import jax
import jax.numpy as jnp
import linox as lo
from linox._kronecker import Kronecker
from linox._toeplitz import Toeplitz
from linox._matrix import Matrix

def test_debug_shapes():
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
    
    print("Testing K_total @ v...")
    try:
        res = K_total @ v
        print(f"Result shape: {res.shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Now reproduction of 2D case structure
    # K_func (2x2)
    K_func = lo.Identity(2) # Matrix
    # K_spatial (16x16)
    K_spatial = lo.Identity(16) # Matrix
    
    K_sf_real = Kronecker(K_func, K_spatial) # 32x32
    
    K_total_real = Kronecker(K_sf_real, K_time) # 64x64
    
    print("\nTesting Real Structure @ v...")
    try:
        res = K_total_real @ v
        print(f"Result shape: {res.shape}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_shapes()
