
import jax
import jax.numpy as jnp

import linox.config as config
from linox import Matrix, inverse, solve
from linox.operators import InverseLinearOperator


def test_approx_inverse():
    # Create a simple SPD matrix
    key = jax.random.PRNGKey(0)
    A_dense = jax.random.normal(key, (10, 10))
    A_dense = A_dense @ A_dense.T + 1e-3 * jnp.eye(10)
    A = Matrix(A_dense)
    
    # 1. Test exact inverse (default)
    inv_exact = inverse(A, method="exact")
    # Exact inverse of Matrix returns Matrix (dense)
    assert isinstance(inv_exact, (InverseLinearOperator, Matrix))
    if isinstance(inv_exact, InverseLinearOperator):
        assert inv_exact.method == "exact"
    
    # 2. Test approx inverse
    # Increase maxiter to ensure convergence
    inv_approx = inverse(A, method="approx", maxiter=100)
    assert isinstance(inv_approx, InverseLinearOperator)
    assert inv_approx.method == "lsmr"
    
    # 3. Test explicit method
    inv_cg = inverse(A, method="cg", maxiter=50)
    assert inv_cg.method == "cg"
    assert inv_cg.solver_options == {"maxiter": 50}
    
    # 4. Run matmul
    x = jnp.ones(10)
    b = A @ x
    
    # Approx solve via inverse
    x_est = inv_approx @ b
    
    error = jnp.linalg.norm(x - x_est)
    print(f"Approx inverse error: {error}")
    assert error < 1e-3

    # CG solve via inverse
    x_cg = inv_cg @ b
    error_cg = jnp.linalg.norm(x - x_cg)
    print(f"CG inverse error: {error_cg}")
    assert error_cg < 1e-3
    
    print("All tests passed!")

if __name__ == "__main__":
    test_approx_inverse()
