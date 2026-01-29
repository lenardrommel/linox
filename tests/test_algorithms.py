# test_algorithms.py

"""Tests for matrix-free algorithms inspired by matfree.

These tests verify the correctness of:
- Lanczos and Arnoldi iterations
- Hutchinson trace estimation
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
    hutchinson_diagonal,
    hutchinson_trace,
    hutchinson_trace_and_diagonal,
    lanczos_eigh,
    lanczos_matrix_function,
    lanczos_tridiag,
    stochastic_lanczos_quadrature,
)


class TestLanczosArnoldi:
    """Tests for Lanczos and Arnoldi iterations."""

    def test_lanczos_tridiag_shape(self) -> None:
        """Test that Lanczos returns correct shapes."""
        n = 100
        A = Matrix(jnp.eye(n))
        v0 = jnp.ones(n)
        num_iters = 10

        Q, alpha, beta = lanczos_tridiag(A, v0, num_iters)

        assert Q.shape == (n, num_iters)
        assert alpha.shape == (num_iters,)
        assert beta.shape == (num_iters - 1,)

    def test_lanczos_orthogonality(self) -> None:
        """Test that Lanczos vectors are orthonormal."""
        n = 50
        key = jax.random.PRNGKey(0)
        A_dense = jax.random.normal(key, (n, n))
        A_dense = (A_dense + A_dense.T) / 2  # Make symmetric
        A = Matrix(A_dense)
        v0 = jax.random.normal(jax.random.PRNGKey(1), (n,))
        num_iters = 10

        Q, _alpha, _beta = lanczos_tridiag(A, v0, num_iters, reortho=True)

        # Check orthonormality
        QTQ = Q.T @ Q
        assert jnp.allclose(QTQ, jnp.eye(num_iters), atol=1e-6)

    def test_lanczos_eigenvalues_identity(self) -> None:
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

    def test_lanczos_eigenvalues_diagonal(self) -> None:
        """Test Lanczos eigenvalue computation on diagonal matrix."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        v0 = jnp.ones(n)
        num_iters = 30  # Increased for better convergence
        k = 5

        eigs, _vecs = lanczos_eigh(A, v0, num_iters, k=k, which="LA")

        # Should get top k eigenvalues (approximate, not exact)
        expected = diag_vals[-k:][::-1]
        assert jnp.allclose(
            eigs, expected, atol=1.0
        )  # Relaxed tolerance for Lanczos approximation

    def test_arnoldi_shape(self) -> None:
        """Test that Arnoldi returns correct shapes."""
        n = 100
        A = Matrix(jnp.eye(n))
        v0 = jnp.ones(n)
        num_iters = 10

        Q, H = arnoldi_iteration(A, v0, num_iters)

        assert Q.shape == (n, num_iters)
        assert H.shape == (num_iters + 1, num_iters)

    def test_arnoldi_orthogonality(self) -> None:
        """Test that Arnoldi vectors are orthonormal."""
        n = 50
        key = jax.random.PRNGKey(0)
        A = Matrix(jax.random.normal(key, (n, n)))
        v0 = jax.random.normal(jax.random.PRNGKey(1), (n,))
        num_iters = 10

        Q, _H = arnoldi_iteration(A, v0, num_iters)

        # Check orthonormality
        QTQ = Q.T @ Q
        assert jnp.allclose(QTQ, jnp.eye(num_iters), atol=1e-6)


class TestHutchinsonTrace:
    """Tests for Hutchinson trace estimation."""

    def test_trace_identity(self) -> None:
        """Test trace estimation on identity matrix."""
        n = 100
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = hutchinson_trace(A, key, num_samples=200)

        # True trace is n
        assert jnp.abs(trace_est - n) <= 3 * trace_std  # Within 3 sigma
        assert jnp.abs(trace_est - n) <= 5.0  # Should be quite accurate

    def test_trace_diagonal(self) -> None:
        """Test trace estimation on diagonal matrix."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = hutchinson_trace(A, key, num_samples=300)

        true_trace = jnp.sum(diag_vals)
        assert jnp.abs(trace_est - true_trace) <= 3 * trace_std

    def test_diagonal_estimation(self) -> None:
        """Test diagonal estimation."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        diag_est, _diag_std = hutchinson_diagonal(A, key, num_samples=500)

        assert jnp.allclose(diag_est, diag_vals, atol=0.5)  # Stochastic estimate

    def test_trace_and_diagonal_joint(self) -> None:
        """Test joint trace and diagonal estimation."""
        n = 50
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        result = hutchinson_trace_and_diagonal(A, key, num_samples=500)

        trace_est, trace_std = result["trace"]
        diag_est, _diag_std = result["diagonal"]

        true_trace = jnp.sum(diag_vals)
        assert jnp.abs(trace_est - true_trace) <= 3 * trace_std
        assert jnp.allclose(diag_est, diag_vals, atol=0.5)

    def test_rademacher_vs_normal(self) -> None:
        """Test that both distributions give reasonable estimates."""
        n = 100
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_rad, _ = hutchinson_trace(
            A, key, num_samples=200, distribution="rademacher"
        )
        trace_norm, _ = hutchinson_trace(A, key, num_samples=200, distribution="normal")

        # Both should be close to true trace (n=100)
        assert jnp.abs(trace_rad - n) < 10.0
        assert jnp.abs(trace_norm - n) < 10.0


class TestMatrixFunctions:
    """Tests for matrix function approximations."""

    def test_lanczos_exp_identity(self) -> None:
        """Test matrix exponential on -I."""
        n = 100
        A = Matrix(-jnp.eye(n))
        v = jnp.ones(n)
        num_iters = 10

        result = lanczos_matrix_function(A, v, jnp.exp, num_iters)

        # exp(-I)v = exp(-1) * v
        expected = jnp.exp(-1.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_lanczos_log_identity(self) -> None:
        """Test matrix logarithm on scaled identity."""
        n = 50
        scale = 2.0
        A = Matrix(scale * jnp.eye(n))
        A = linox.IsotropicAdditiveLinearOperator(1e-4, A)
        v = jnp.ones(n)
        num_iters = 5

        result = lanczos_matrix_function(A, v, jnp.log, num_iters)

        # log(2I)v = log(2) * v
        expected = jnp.log(scale) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_arnoldi_exp_triangular(self) -> None:
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


class TestStochasticLanczosQuadrature:
    """Tests for stochastic Lanczos quadrature (SLQ)."""

    def test_slq_logdet_identity(self) -> None:
        """Test log-determinant estimation on identity."""
        n = 50
        A = Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        logdet_est, logdet_std = stochastic_lanczos_quadrature(
            A, jnp.log, key, num_samples=100, num_iters=5
        )

        # log|I| = 0
        assert jnp.abs(logdet_est) < 3 * logdet_std + 1e-6
        assert jnp.abs(logdet_est) < 1.0

    def test_slq_logdet_diagonal(self) -> None:
        """Test log-determinant estimation on diagonal matrix."""
        n = 20
        diag_vals = jnp.arange(1.0, n + 1.0)
        A = Matrix(jnp.diag(diag_vals))
        key = jax.random.PRNGKey(0)

        logdet_est, logdet_std = stochastic_lanczos_quadrature(
            A, jnp.log, key, num_samples=100, num_iters=20
        )

        true_logdet = jnp.sum(jnp.log(diag_vals))
        # Should be within a few standard errors (or numerically exact)
        assert jnp.abs(logdet_est - true_logdet) <= 5 * logdet_std + 1e-12

    def test_slq_trace_exp(self) -> None:
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

    def test_ltrace_basic(self) -> None:
        """Test ltrace function from linox."""
        n = 100
        A = linox.Matrix(jnp.eye(n))
        key = jax.random.PRNGKey(0)

        trace_est, _trace_std = linox.ltrace(A, key=key, num_samples=200)

        assert jnp.abs(trace_est - n) < 5.0

    def test_ltrace_default_key(self) -> None:
        """Test ltrace with default key."""
        n = 50
        A = linox.Matrix(jnp.eye(n))

        trace_est, _ = linox.ltrace(A, num_samples=100)

        assert jnp.abs(trace_est - n) < 10.0

    def test_lexp_with_vector(self) -> None:
        """Test lexp function with vector."""
        n = 50
        A = linox.Matrix(-jnp.eye(n))
        v = jnp.ones(n)

        result = linox.lexp(A, v=v, num_iters=10)

        expected = jnp.exp(-1.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_llog_with_vector(self) -> None:
        """Test llog function with vector."""
        n = 30
        A = linox.Matrix(2.0 * jnp.eye(n))
        v = jnp.ones(n)

        result = linox.llog(A, v=v, num_iters=5)

        expected = jnp.log(2.0) * v
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_lpow_with_vector(self) -> None:
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

    def test_hutchinson_trace_jit(self) -> None:
        """Test that Hutchinson trace can be JIT compiled."""

        @jax.jit
        def traced_func(A_dense, key):
            A = Matrix(A_dense)
            return hutchinson_trace(A, key, num_samples=10)

        A_dense = jnp.eye(10)
        key = jax.random.PRNGKey(0)

        trace_est, _ = traced_func(A_dense, key)
        assert jnp.abs(trace_est - 10.0) < 5.0

    def test_lanczos_tridiag_jit(self) -> None:
        """Test that Lanczos can be JIT compiled."""

        @jax.jit
        def lanczos_func(A_dense, v0):
            A = Matrix(A_dense)
            return lanczos_tridiag(A, v0, num_iters=5)

        A_dense = jnp.eye(20)
        v0 = jnp.ones(20)

        Q, _alpha, _beta = lanczos_func(A_dense, v0)
        assert Q.shape == (20, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
