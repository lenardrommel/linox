
import jax
import jax.numpy as jnp
import pytest

import linox.linalg.functions as lfuncs  # import directly to check
from linox import Matrix
from linox.api import slogdet, solve, sqrt


def test_slq_logdet_approx():
    key = jax.random.PRNGKey(0)
    n = 50
    # Symmetric positive definite matrix
    # A = Q D Q^T. LogDet A = sum log D_i
    diag = jnp.linspace(1.0, 10.0, n)
    # A = diag 
    A = Matrix(jnp.diag(diag))
    
    true_logdet = jnp.sum(jnp.log(diag))
    
    sign, approx = slogdet(A, method="slq", key=key, num_samples=30, m=20)
    
    assert sign == 1.0
    rel_err = jnp.abs(approx - true_logdet) / true_logdet
    # SLQ is stochastic, 20% err is a safe bound for small samples
    assert rel_err < 0.2

def test_lsmr_solve():
    key = jax.random.PRNGKey(1)
    n = 40
    A_mat = jax.random.normal(key, (n, n)) + jnp.eye(n) * 3
    A = Matrix(A_mat)
    b = jnp.ones(n)
    
    # Exact solve
    x_exact = jnp.linalg.solve(A_mat, b)
    
    # LSMR solve
    x_approx = solve(A, b, method="lsmr", maxiter=100) # lsmr default
    
    rel_err = jnp.linalg.norm(x_approx - x_exact) / jnp.linalg.norm(x_exact)
    assert rel_err < 1e-3

def test_sqrt_lanczos():
    # Test sqrt(A) * v ≈ A^{1/2} v
    key = jax.random.PRNGKey(2)
    n = 30
    # Make A SPD
    X = jax.random.normal(key, (n, n))
    A_mat = X @ X.T + jnp.eye(n) * 0.1
    A = Matrix(A_mat)
    
    # Exact sqrt
    w, V = jnp.linalg.eigh(A_mat)
    sqrt_A_dense = V @ jnp.diag(jnp.sqrt(w)) @ V.T
    
    v = jax.random.normal(key, (n,))
    expected = sqrt_A_dense @ v
    
    # Approx sqrt operator
    # method="lanczos" implies MatrixFunction wrapper
    S = sqrt(A, method="lanczos", num_iters=25)
    
    approx = S @ v
    
    rel_err = jnp.linalg.norm(approx - expected) / jnp.linalg.norm(expected)
    # Lanczos on small matrix with iter ~ size should be very good
    assert rel_err < 0.05

def test_eigh_lanczos():
    from linox.api import eigh
    key = jax.random.PRNGKey(3)
    n = 40
    # Symmetric matrix (SPD to be safe for sorting)
    X = jax.random.normal(key, (n, n))
    A_mat = X @ X.T
    A = Matrix(A_mat)
    
    # Exact top 5
    w_true, v_true = jax.scipy.linalg.eigh(A_mat)
    # eigh returns ascending order
    # top 5 (largest algebraic) are last 5
    top_k = 5
    w_expected = w_true[-top_k:]
    
    # Approx
    w_approx, v_approx = eigh(A, k=top_k, method="lanczos", num_iters=30)
    
    # Check eigenvalues match (sorting might differ or be same)
    # lanczos_eigh usually returns sorted?
    w_approx_sorted = jnp.sort(w_approx)
    w_expected_sorted = jnp.sort(w_expected)
    
    rel_err = jnp.linalg.norm(w_approx_sorted - w_expected_sorted) / jnp.linalg.norm(w_expected_sorted)
    assert rel_err < 0.05
