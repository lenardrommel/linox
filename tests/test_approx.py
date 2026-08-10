
import jax
import jax.numpy as jnp
import linox
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
    w_true, _v_true = jax.scipy.linalg.eigh(A_mat)
    # eigh returns ascending order
    # top 5 (largest algebraic) are last 5
    top_k = 5
    w_expected = w_true[-top_k:]
    
    # Approx
    w_approx, _v_approx = eigh(A, k=top_k, method="lanczos", num_iters=30)
    
    # Check eigenvalues match (sorting might differ or be same)
    # lanczos_eigh usually returns sorted?
    w_approx_sorted = jnp.sort(w_approx)
    w_expected_sorted = jnp.sort(w_expected)
    
    rel_err = jnp.linalg.norm(w_approx_sorted - w_expected_sorted) / jnp.linalg.norm(w_expected_sorted)
    assert rel_err < 0.05


def test_sqrt_exact_returns_a_valid_factor():
    """`sqrt` returns a factor S with S @ S.T == A (not necessarily A^(1/2))."""
    key = jax.random.PRNGKey(5)
    n = 12
    X = jax.random.normal(key, (n, n))
    A_mat = X @ X.T + jnp.eye(n) * 0.1

    S = linox.todense(sqrt(Matrix(A_mat), method="exact"))
    assert jnp.linalg.norm(S @ S.T - A_mat) < 1e-8


def test_sqrt_lanczos_is_not_silently_downgraded_to_exact():
    """An explicit method= request must be honoured.

    Regression: `sqrt(A, method="lanczos")` used to try the exact structured
    dispatch first, so it returned a dense Cholesky factor and silently
    ignored both `method=` and `num_iters`.
    """
    key = jax.random.PRNGKey(2)
    n = 30
    X = jax.random.normal(key, (n, n))
    A_mat = X @ X.T + jnp.eye(n) * 0.1

    S = sqrt(Matrix(A_mat), method="lanczos", num_iters=25)

    # The Krylov path stays lazy rather than densifying to a Matrix.
    assert type(S).__name__ != "Matrix"

    # And it approximates the principal square root, not a Cholesky factor.
    w, V = jnp.linalg.eigh(A_mat)
    principal = V @ jnp.diag(jnp.sqrt(w)) @ V.T
    chol = jnp.linalg.cholesky(A_mat)

    v = jax.random.normal(key, (n,))
    err_principal = jnp.linalg.norm(S @ v - principal @ v) / jnp.linalg.norm(
        principal @ v
    )
    err_chol = jnp.linalg.norm(S @ v - chol @ v) / jnp.linalg.norm(chol @ v)

    assert err_principal < 0.05
    assert err_principal < err_chol


def test_unknown_method_is_rejected():
    """Typos in method= must raise rather than silently taking the default."""
    import pytest

    from linox.api import eigh, inverse, slogdet, solve, trace

    A = Matrix(jnp.eye(6) * 2.0)

    for fn in (sqrt, inverse, eigh, trace, slogdet):
        with pytest.raises(ValueError, match="Unknown method"):
            fn(A, method="totally-bogus-method")

    with pytest.raises(ValueError, match="Unknown method"):
        solve(A, jnp.ones(6), method="totally-bogus-method")
