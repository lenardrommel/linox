
import jax
import jax.numpy as jnp
from linox.linalg.woodbury import woodbury_chol_solve, woodbury_solve

jax.config.update("jax_enable_x64", True)

def test_woodbury_solve():
    key = jax.random.PRNGKey(0)
    n = 10
    k = 3
    
    U = jax.random.normal(key, (n, k))
    s = jax.random.uniform(key, (k,)) + 0.1
    d = jax.random.uniform(key, (n,)) + 0.1
    v = jax.random.normal(key, (n,))
    
    D = jnp.diag(d)
    S = jnp.diag(s)
    A = U @ S @ U.T + D
    
    # Expected solution
    x_expected = jnp.linalg.solve(A, v)
    
    # Woodbury solution
    x_woodbury = woodbury_solve(U, s, d, v)
    
    assert jnp.allclose(x_expected, x_woodbury, atol=1e-5)

def test_woodbury_chol_solve():
    key = jax.random.PRNGKey(1)
    n = 10
    k = 3
    
    L = jax.random.normal(key, (n, k))
    d = jax.random.uniform(key, (n,)) + 0.1
    v = jax.random.normal(key, (n,))
    
    D = jnp.diag(d)
    A = L @ L.T + D
    
    # Expected solution
    x_expected = jnp.linalg.solve(A, v)
    
    # Woodbury solution
    x_woodbury = woodbury_chol_solve(L, d, v)
    
    assert jnp.allclose(x_expected, x_woodbury, atol=1e-5)
    
def test_woodbury_solve_scalar_d():
    key = jax.random.PRNGKey(2)
    n = 10
    k = 3
    
    U = jax.random.normal(key, (n, k))
    s = jax.random.uniform(key, (k,)) + 0.1
    d = 0.5
    v = jax.random.normal(key, (n,))
    
    D = jnp.eye(n) * d
    S = jnp.diag(s)
    A = U @ S @ U.T + D
    
    x_expected = jnp.linalg.solve(A, v)
    x_woodbury = woodbury_solve(U, s, d, v)
    
    assert jnp.allclose(x_expected, x_woodbury, atol=1e-5)

def test_isotropic_add_woodbury_integration():
    from linox.operators.arithmetic import lsolve
    from linox.operators.isotropic import IsotropicAdditiveLinearOperator
    from linox.operators.lowrank import SymmetricLowRank
    
    key = jax.random.PRNGKey(3)
    n = 10
    k = 3
    
    U = jax.random.normal(key, (n, k))
    s_vec = jax.random.uniform(key, (k,)) + 0.1
    s_scalar = 0.5
    v = jax.random.normal(key, (n,))
    
    A_lowrank = SymmetricLowRank(U, s_vec)
    iso_op = IsotropicAdditiveLinearOperator(s_scalar, A_lowrank)
    
    # Expected: (s I + U S U^T) v
    D = jnp.eye(n) * s_scalar
    S = jnp.diag(s_vec)
    A_dense = D + U @ S @ U.T
    x_expected = jnp.linalg.solve(A_dense, v)
    
    # Via lsolve (should use woodbury path)
    x_iso = lsolve(iso_op, v)
    
    assert jnp.allclose(x_expected, x_iso, atol=1e-5)

def test_positive_diag_plus_lowrank_woodbury_integration():
    from linox.operators.arithmetic import lsolve
    from linox.operators.diagonal import Diagonal
    from linox.operators.lowrank import (
        PositiveDiagonalPlusSymmetricLowRank,
        SymmetricLowRank,
    )
    
    key = jax.random.PRNGKey(4)
    n = 10
    k = 3
    
    U = jax.random.normal(key, (n, k))
    s_vec = jax.random.uniform(key, (k,)) + 0.1
    d_vec = jax.random.uniform(key, (n,)) + 0.1
    v = jax.random.normal(key, (n,))
    
    diag_op = Diagonal(d_vec)
    lowrank_op = SymmetricLowRank(U, s_vec)
    pd_op = PositiveDiagonalPlusSymmetricLowRank(diag_op, lowrank_op)
    
    # Expected
    D = jnp.diag(d_vec)
    S = jnp.diag(s_vec)
    A_dense = D + U @ S @ U.T
    x_expected = jnp.linalg.solve(A_dense, v)
    
    # Via lsolve (should use woodbury path)
    x_pd = lsolve(pd_op, v)
    
    assert jnp.allclose(x_expected, x_pd, atol=1e-5)
