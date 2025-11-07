"""Tests for matrix-free algorithms inspired by matfree.

These tests verify the correctness of:
- Lanczos and Arnoldi iterations
- Hutchinson trace estimation
- LSMR solver
- Matrix function approximations
- Stochastic Lanczos quadrature
"""

import jax
import jax.numpy as jnp
import pytest

import linox
from linox import Matrix
from linox._algorithms import (
    arnoldi_iteration,
    arnoldi_matrix_function,
    chebyshev_matrix_function,
    hutchinson_diagonal,
    hutchinson_trace,
    hutchinson_trace_and_diagonal,
    lanczos_eigh,
    lanczos_matrix_function,
    lanczos_tridiag,
    lsmr_solve,
    stochastic_lanczos_quadrature,
)


class TestLanczosArnoldi:
    """Tests for Lanczos and Arnoldi iterations."""

    def test_lanczos_tridiag_shape(self):
        """Test that Lanczos returns correct shapes."""
        n = 100
        A = Matrix(jnp.eye(n))
        v0 = jnp.ones(n)
        num_iters = 10

        Q, alpha, beta = lanczos_tridiag(A, v0, num_iters)

        assert Q.shape == (n, num_iters)
        assert alpha.shape == (num_iters,)
        assert beta.shape == (num_iters - 1,)

    def test_lanczos_orthogonality(self):
        """Test that Lanczos vectors are orthonormal."""
        n = 50
        key = jax.random.PRNGKey(0)
        A_dense = jax.random.normal(key, (n, n))
        A_dense = (A_dense + A_dense.T) / 2  # Make symmetric
        A = Matrix(A_dense)
        v0 = jax.random.normal(jax.random.PRNGKey(1), (n,))
        num_iters = 10

        Q, alpha, beta = lanczos_tridiag(A, v0, num_iters, reortho=True)

        # Check orthonormality
        QTQ = Q.T @ Q
        assert jnp.allclose(QTQ, jnp.eye(num_iters), atol=1e-6)

    def test_lanczos_eigenvalues_identity(self):
        """Test Lanczos eigenvalue computation on scaled identity."""
        n = 100
        A = Matrix(3.0 * jnp.eye(n))  # Scaled identity with eigenvalue 3
        v0 = jnp.ones(n) / jnp.sqrt(n)
        num_iters = 10
        k = 1  # Identity has one distinct eigenvalue

        eigs, vecs = lanczos_eigh(A, v0, num_iters, k=k, which="LA")

        assert eigs.shape == (k,)
        assert vecs.shape == (n, k)
        # Eigenvalue should be 3 (one-dimensional Krylov subspace)
        assert jnp.allclose(eigs[0], 3.0, atol=1e-6)

    def test_lanczos_eigenvalues_diagonal(self):
        """Test Lanczos eigenvalue computation on diagonal matrix."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        v0 = jnp.ones(n)
        num_iters = 30  # Increased for better convergence
        k = 5

        eigs, vecs = lanczos_eigh(A, v0, num_iters, k=k, which="LA")

        # Should get top k eigenvalues (approximate, not exact)
        expected = diag_vals[-k:][::-1]
        assert jnp.allclose(eigs, expected, atol=1.0)  # Relaxed tolerance for Lanczos approximation

    def test_arnoldi_shape(self):
        """Test that Arnoldi returns correct shapes."""
        n = 100
        A = Matrix(jnp.eye(n))
        v0 = jnp.ones(n)
        num_iters = 10

        Q, H = arnoldi_iteration(A, v0, num_iters)

        assert Q.shape == (n, num_iters)
        assert H.shape == (num_iters + 1, num_iters)

    def test_arnoldi_orthogonality(self):
        """Test that Arnoldi vectors are orthonormal."""
        n = 50
        key = jax.random.PRNGKey(0)
        A = Matrix(jax.random.normal(key, (n, n)))
        v0 = jax.random.normal(jax.random.PRNGKey(1), (n,))
        num_iters = 10

        Q, H = arnoldi_iteration(A, v0, num_iters)

        # Check orthonormality
        QTQ = Q.T @ Q
        assert jnp.allclose(QTQ, jnp.eye(num_iters), atol=1e-6)


class TestHutchinsonTrace:
    """Tests for Hutchinson trace estimation."""

    def test_trace_identity(self):
        """Test trace estimation on identity matrix."""
        n = 100
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = hutchinson_trace(A, key, num_samples=200)

        # True trace is n
        assert jnp.abs(trace_est - n) <= 3 * trace_std  # Within 3 sigma
        assert jnp.abs(trace_est - n) <= 5.0  # Should be quite accurate

    def test_trace_diagonal(self):
        """Test trace estimation on diagonal matrix."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = hutchinson_trace(A, key, num_samples=300)

        true_trace = jnp.sum(diag_vals)
        assert jnp.abs(trace_est - true_trace) <= 3 * trace_std

    def test_diagonal_estimation(self):
        """Test diagonal estimation."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        diag_est, diag_std = hutchinson_diagonal(A, key, num_samples=500)

        assert jnp.allclose(diag_est, diag_vals, atol=0.5)  # Stochastic estimate

    def test_trace_and_diagonal_joint(self):
        """Test joint trace and diagonal estimation."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        result = hutchinson_trace_and_diagonal(A, key, num_samples=500)

        trace_est, trace_std = result["trace"]
        diag_est, diag_std = result["diagonal"]

        true_trace = jnp.sum(diag_vals)
        assert jnp.abs(trace_est - true_trace) <= 3 * trace_std
        assert jnp.allclose(diag_est, diag_vals, atol=0.5)

    def test_rademacher_vs_normal(self):
        """Test that both distributions give reasonable estimates."""
        n = 100
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_rad, _ = hutchinson_trace(A, key, num_samples=200, distribution="rademacher")
        trace_norm, _ = hutchinson_trace(A, key, num_samples=200, distribution="normal")

        # Both should be close to true trace (n=100)
        assert jnp.abs(trace_rad - n) < 10.0
        assert jnp.abs(trace_norm - n) < 10.0


class TestLSMR:
    """Tests for LSMR solver."""

    def test_lsmr_identity(self):
        """Test LSMR on identity system."""
        n = 50
        A = Matrix(jnp.eye(n))
        b = jnp.ones(n)

        x, info = lsmr_solve(A, b, atol=1e-8, btol=1e-8)

        assert jnp.allclose(x, b, atol=1e-6)
        assert info["istop"] in [1, 2, 3]  # Converged

    def test_lsmr_diagonal(self):
        """Test LSMR on diagonal system."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        x_true = jnp.ones(n)
        b = A @ x_true

        x, info = lsmr_solve(A, b, atol=1e-10, btol=1e-10)

        assert jnp.allclose(x, x_true, atol=1e-6)
        assert info["istop"] in [1, 2, 3]

    def test_lsmr_overdetermined(self):
        """Test LSMR on overdetermined least-squares problem."""
        m, n = 100, 50
        key = jax.random.PRNGKey(0)
        A_dense = jax.random.normal(key, (m, n))
        A = Matrix(A_dense)
        x_true = jax.random.normal(jax.random.PRNGKey(1), (n,))
        b = A @ x_true + 0.01 * jax.random.normal(jax.random.PRNGKey(2), (m,))

        x, info = lsmr_solve(A, b, atol=1e-8, btol=1e-8, maxiter=200)

        # Should get close to true solution
        assert jnp.linalg.norm(x - x_true) < 0.5
        # Residual should be small
        assert info["normr"] < 1.0

    def test_lsmr_with_damping(self):
        """Test LSMR with Tikhonov regularization."""
        n = 50
        A = Matrix(jnp.eye(n))
        b = jnp.ones(n)
        damp = 0.1

        x, info = lsmr_solve(A, b, damp=damp, atol=1e-8, btol=1e-8)

        # With damping, solution should be slightly smaller than b
        assert jnp.allclose(x, b / (1 + damp**2), atol=1e-4)


class TestMatrixFunctions:
    """Tests for matrix function approximations."""

    def test_lanczos_exp_identity(self):
        """Test matrix exponential on -I."""
        n = 100
        A = Matrix(-jnp.eye(n))
        v = jnp.ones(n)
        num_iters = 10

        result = lanczos_matrix_function(A, v, jnp.exp, num_iters)

        # exp(-I)v = exp(-1) * v
        expected = jnp.exp(-1.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_lanczos_log_identity(self):
        """Test matrix logarithm on scaled identity."""
        n = 50
        scale = 2.0
        A = Matrix(scale * jnp.eye(n))
        v = jnp.ones(n)
        num_iters = 5

        result = lanczos_matrix_function(A, v, jnp.log, num_iters)

        # log(2I)v = log(2) * v
        expected = jnp.log(scale) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_arnoldi_exp_triangular(self):
        """Test Arnoldi matrix exponential on upper triangular matrix."""
        n = 10
        A_dense = jnp.triu(jnp.ones((n, n)) * 0.1)
        A = Matrix(A_dense)
        v = jnp.ones(n)
        num_iters = n

        result = arnoldi_matrix_function(A, v, jnp.exp, num_iters)

        # Compare with dense computation
        expected = jax.scipy.linalg.expm(A_dense) @ v
        assert jnp.allclose(result, expected, atol=1e-2)

    def test_chebyshev_exp_small_matrix(self):
        """Test Chebyshev matrix exponential."""
        n = 20
        A = Matrix(0.5 * jnp.eye(n))  # Eigenvalues in [-1, 1]
        v = jnp.ones(n)
        num_terms = 20

        result = chebyshev_matrix_function(A, v, jnp.exp, num_terms, bounds=(-1.0, 1.0))

        # exp(0.5*I)v = exp(0.5) * v
        expected = jnp.exp(0.5) * v
        assert jnp.allclose(result, expected, atol=1e-2)


class TestStochasticLanczosQuadrature:
    """Tests for stochastic Lanczos quadrature (SLQ)."""

    def test_slq_logdet_identity(self):
        """Test log-determinant estimation on identity."""
        n = 50
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        logdet_est, logdet_std = stochastic_lanczos_quadrature(
            A, jnp.log, key, num_samples=100, num_iters=5
        )

        # log|I| = 0
        assert jnp.abs(logdet_est) < 3 * logdet_std
        assert jnp.abs(logdet_est) < 1.0

    def test_slq_logdet_diagonal(self):
        """Test log-determinant estimation on diagonal matrix."""
        n = 20
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        logdet_est, logdet_std = stochastic_lanczos_quadrature(
            A, jnp.log, key, num_samples=100, num_iters=20
        )

        true_logdet = jnp.sum(jnp.log(diag_vals))
        # Should be within a few standard errors
        assert jnp.abs(logdet_est - true_logdet) < 5 * logdet_std

    def test_slq_trace_exp(self):
        """Test trace(exp(A)) estimation."""
        n = 20
        A = Matrix(-jnp.eye(n))  # -I
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = stochastic_lanczos_quadrature(
            A, jnp.exp, key, num_samples=50, num_iters=5
        )

        # trace(exp(-I)) = n * exp(-1)
        true_trace = n * jnp.exp(-1.0)
        # Use max of 3*sigma and small absolute tolerance for numerical precision
        assert jnp.abs(trace_est - true_trace) <= max(3 * trace_std, 1e-10)


class TestArithmeticIntegration:
    """Tests for integration with linox arithmetic module."""

    def test_ltrace_basic(self):
        """Test ltrace function from linox."""
        n = 100
        A = linox.Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = linox.ltrace(A, key=key, num_samples=200)

        assert jnp.abs(trace_est - n) < 5.0

    def test_ltrace_default_key(self):
        """Test ltrace with default key."""
        n = 50
        A = linox.Matrix(jnp.eye(n))

        trace_est, _ = linox.ltrace(A, num_samples=100)

        assert jnp.abs(trace_est - n) < 10.0

    def test_lexp_with_vector(self):
        """Test lexp function with vector."""
        n = 50
        A = linox.Matrix(-jnp.eye(n))
        v = jnp.ones(n)

        result = linox.lexp(A, v=v, num_iters=10)

        expected = jnp.exp(-1.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_llog_with_vector(self):
        """Test llog function with vector."""
        n = 30
        A = linox.Matrix(2.0 * jnp.eye(n))
        v = jnp.ones(n)

        result = linox.llog(A, v=v, num_iters=5)

        expected = jnp.log(2.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_lpow_with_vector(self):
        """Test lpow function with vector."""
        n = 30
        A = linox.Matrix(4.0 * jnp.eye(n))
        v = jnp.ones(n)
        power = 0.5

        result = linox.lpow(A, power=power, v=v, num_iters=5)

        expected = 2.0 * v  # sqrt(4) = 2
        assert jnp.allclose(result, expected, atol=1e-3)


# Test that algorithms work with JAX transformations
class TestJAXTransformations:
    """Test that algorithms are compatible with JAX transformations."""

    def test_hutchinson_trace_jit(self):
        """Test that Hutchinson trace can be JIT compiled."""

        @jax.jit
        def traced_func(A_dense, key):
            A = Matrix(A_dense)
            return hutchinson_trace(A, key, num_samples=10)

        A_dense = jnp.eye(10)
        key = jax.random.PRNGKey(0)

        trace_est, _ = traced_func(A_dense, key)
        assert jnp.abs(trace_est - 10.0) < 5.0

    def test_lanczos_tridiag_jit(self):
        """Test that Lanczos can be JIT compiled."""

        @jax.jit
        def lanczos_func(A_dense, v0):
            A = Matrix(A_dense)
            return lanczos_tridiag(A, v0, num_iters=5)

        A_dense = jnp.eye(20)
        v0 = jnp.ones(20)

        Q, alpha, beta = lanczos_func(A_dense, v0)
        assert Q.shape == (20, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
