"""Tests for matrix-free SVD algorithms."""

import jax
import jax.numpy as jnp
import pytest

import linox
from linox import Matrix
from linox._algorithms._svd import lanczos_bidiag, svd_partial


class TestLanczosBidiagonalization:
    """Tests for Lanczos bidiagonalization."""

    def test_bidiag_shapes(self):
        """Test that bidiagonalization returns correct shapes."""
        m, n = 100, 50
        A = Matrix(jnp.random.randn(m, n))
        u0 = jnp.ones(m)
        num_iters = 10

        U, V, alpha, beta = lanczos_bidiag(A, u0, num_iters)

        assert U.shape == (m, num_iters)
        assert V.shape == (n, num_iters)
        assert alpha.shape == (num_iters,)
        assert beta.shape == (num_iters - 1,)

    def test_bidiag_orthogonality(self):
        """Test that U and V are orthonormal."""
        m, n = 80, 60
        A = Matrix(jnp.random.randn(m, n))
        u0 = jnp.ones(m)
        num_iters = 10

        U, V, alpha, beta = lanczos_bidiag(A, u0, num_iters)

        # Check U is orthonormal
        UTU = U.T @ U
        assert jnp.allclose(UTU, jnp.eye(num_iters), atol=1e-10)

        # Check V is orthonormal
        VTV = V.T @ V
        assert jnp.allclose(VTV, jnp.eye(num_iters), atol=1e-10)

    def test_bidiag_reconstruction(self):
        """Test that A ≈ U B V^T."""
        m, n = 50, 40
        key = jax.random.PRNGKey(0)
        A_dense = jax.random.normal(key, (m, n))
        A = Matrix(A_dense)
        u0 = jnp.ones(m)
        num_iters = min(m, n)

        U, V, alpha, beta = lanczos_bidiag(A, u0, num_iters)

        # Construct bidiagonal matrix B
        B = jnp.diag(alpha)
        if beta.size > 0:
            B = B + jnp.diag(beta, k=1)

        # Reconstruction
        A_approx = U @ B @ V.T

        # Should be close to original (within Krylov subspace)
        error = jnp.linalg.norm(A_dense - A_approx, "fro")
        assert error < 1.0  # Reasonable approximation


class TestPartialSVD:
    """Tests for partial SVD computation."""

    def test_svd_shapes(self):
        """Test that partial SVD returns correct shapes."""
        m, n = 100, 50
        A = Matrix(jnp.random.randn(m, n))
        k = 5

        U, S, Vt = svd_partial(A, k)

        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_svd_singular_values_descending(self):
        """Test that singular values are in descending order."""
        m, n = 80, 60
        A = Matrix(jnp.random.randn(m, n))
        k = 10

        U, S, Vt = svd_partial(A, k)

        # Check descending order
        assert jnp.all(S[:-1] >= S[1:])

    def test_svd_identity_matrix(self):
        """Test SVD on identity matrix."""
        n = 50
        A = Matrix(jnp.eye(n))
        k = 10

        U, S, Vt = svd_partial(A, k, num_iters=15)

        # All singular values should be 1
        assert jnp.allclose(S, jnp.ones(k), atol=1e-6)

    def test_svd_diagonal_matrix(self):
        """Test SVD on diagonal matrix."""
        n = 50
        diag_vals = jnp.arange(n, 0, -1, dtype=float)  # Descending
        A = Matrix(jnp.diag(diag_vals))
        k = 10

        U, S, Vt = svd_partial(A, k, num_iters=20)

        # Should get top k singular values
        expected = diag_vals[:k]
        assert jnp.allclose(S, expected, atol=1e-4)

    def test_svd_reconstruction(self):
        """Test low-rank reconstruction A ≈ U S Vt."""
        m, n = 100, 80
        key = jax.random.PRNGKey(42)
        A_dense = jax.random.normal(key, (m, n))
        A = Matrix(A_dense)
        k = 15

        U, S, Vt = svd_partial(A, k, num_iters=30)

        # Low-rank reconstruction
        A_approx = U @ jnp.diag(S) @ Vt

        # Compare with full SVD (top k components)
        U_full, S_full, Vt_full = jnp.linalg.svd(A_dense, full_matrices=False)
        A_full_k = U_full[:, :k] @ jnp.diag(S_full[:k]) @ Vt_full[:k, :]

        # Should be close to full SVD reconstruction
        error = jnp.linalg.norm(A_approx - A_full_k, "fro")
        assert error < 1.0

    def test_svd_orthogonality(self):
        """Test that U and Vt are orthonormal."""
        m, n = 80, 60
        A = Matrix(jnp.random.randn(m, n))
        k = 10

        U, S, Vt = svd_partial(A, k)

        # Check U orthonormality
        UTU = U.T @ U
        assert jnp.allclose(UTU, jnp.eye(k), atol=1e-8)

        # Check Vt orthonormality
        VVt = Vt @ Vt.T
        assert jnp.allclose(VVt, jnp.eye(k), atol=1e-8)


class TestLSVDIntegration:
    """Tests for lsvd function in arithmetic module."""

    def test_lsvd_basic(self):
        """Test basic lsvd usage."""
        m, n = 100, 50
        A = Matrix(jnp.random.randn(m, n))
        k = 5

        U, S, Vt = linox.lsvd(A, k)

        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_lsvd_vs_full_svd(self):
        """Compare lsvd with full SVD."""
        m, n = 80, 60
        key = jax.random.PRNGKey(123)
        A_dense = jax.random.normal(key, (m, n))
        A = Matrix(A_dense)
        k = 10

        # Partial SVD
        U_partial, S_partial, Vt_partial = linox.lsvd(A, k, num_iters=25)

        # Full SVD (for comparison)
        U_full, S_full, Vt_full = jnp.linalg.svd(A_dense, full_matrices=False)

        # Top k singular values should match
        assert jnp.allclose(S_partial, S_full[:k], atol=1e-3)

    def test_lsvd_with_initial_vector(self):
        """Test lsvd with custom initial vector."""
        m, n = 50, 40
        A = Matrix(jnp.random.randn(m, n))
        k = 5
        u0 = jax.random.normal(jax.random.PRNGKey(0), (m,))

        U, S, Vt = linox.lsvd(A, k, u0=u0)

        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_lsvd_large_matrix(self):
        """Test lsvd on larger matrix."""
        m, n = 1000, 500
        # Create low-rank matrix for faster test
        key = jax.random.PRNGKey(99)
        rank = 20
        U_true = jax.random.normal(key, (m, rank))
        V_true = jax.random.normal(jax.random.PRNGKey(100), (rank, n))
        A_dense = U_true @ V_true
        A = Matrix(A_dense)
        k = 10

        U, S, Vt = linox.lsvd(A, k, num_iters=30)

        # Should successfully compute
        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

        # Singular values should be positive
        assert jnp.all(S > 0)


class TestJAXCompatibility:
    """Test JAX transformations."""

    def test_lsvd_jit(self):
        """Test that lsvd can be JIT compiled."""

        @jax.jit
        def compute_svd(A_dense, k):
            A = Matrix(A_dense)
            return linox.lsvd(A, k, num_iters=15)

        m, n = 50, 40
        A_dense = jax.random.normal(jax.random.PRNGKey(0), (m, n))
        k = 5

        U, S, Vt = compute_svd(A_dense, k)

        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_lsvd_grad(self):
        """Test that lsvd is differentiable."""

        def loss(A_dense, k):
            A = Matrix(A_dense)
            U, S, Vt = linox.lsvd(A, k, num_iters=10)
            return jnp.sum(S)

        m, n = 30, 20
        A_dense = jax.random.normal(jax.random.PRNGKey(0), (m, n))
        k = 3

        # Compute gradient
        grad_fn = jax.grad(loss)
        grad = grad_fn(A_dense, k)

        assert grad.shape == (m, n)
        # Gradient should be non-zero
        assert jnp.linalg.norm(grad) > 0
