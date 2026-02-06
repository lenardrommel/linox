"""Tests for structured operator dispatches of matrix-free functions.

Tests that ltrace, lexp, llog, lpow work correctly and exploit structure for:
- Diagonal
- Identity
- Kronecker
- EigenD
- IsotropicAdditive
"""

import jax
import jax.numpy as jnp
import pytest

import linox
from linox import Diagonal, Identity, Matrix
from linox._eigen import EigenD
from linox._isotropicadd import IsotropicAdditiveLinearOperator
from linox._kronecker import Kronecker


class TestDiagonalDispatches:
    """Tests for Diagonal operator dispatches."""

    def test_ltrace_diagonal(self) -> None:
        """Test exact trace for diagonal matrices."""
        diag_vals = jnp.arange(1.0, 11.0)
        A = Diagonal(diag_vals)
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = linox.ltrace(A, key=key)

        true_trace = jnp.sum(diag_vals)
        assert jnp.allclose(trace_est, true_trace)
        assert trace_std == 0.0  # Exact computation

    def test_lexp_diagonal_with_vector(self) -> None:
        """Test matrix exponential of diagonal with vector."""
        diag_vals = jnp.array([1.0, 2.0, 3.0])
        A = Diagonal(diag_vals)
        v = jnp.ones(3)

        result = linox.lexp(A, v=v)

        expected = jnp.exp(diag_vals) * v
        assert jnp.allclose(result, expected)

    def test_lexp_diagonal_lazy(self) -> None:
        """Test matrix exponential of diagonal returns lazy operator."""
        diag_vals = jnp.array([1.0, 2.0, 3.0])
        A = Diagonal(diag_vals)

        exp_A = linox.lexp(A, v=None)

        assert isinstance(exp_A, Diagonal)
        assert jnp.allclose(exp_A.diag, jnp.exp(diag_vals))

    def test_llog_diagonal(self) -> None:
        """Test matrix logarithm of diagonal."""
        diag_vals = jnp.array([1.0, 2.0, 4.0])
        A = Diagonal(diag_vals)
        v = jnp.ones(3)

        result = linox.llog(A, v=v)

        expected = jnp.log(diag_vals) * v
        assert jnp.allclose(result, expected)

    def test_lpow_diagonal(self) -> None:
        """Test matrix power of diagonal."""
        diag_vals = jnp.array([1.0, 4.0, 9.0])
        A = Diagonal(diag_vals)
        v = jnp.ones(3)

        result = linox.lpow(A, power=0.5, v=v)

        expected = jnp.sqrt(diag_vals) * v
        assert jnp.allclose(result, expected)


class TestIdentityDispatches:
    """Tests for Identity operator dispatches."""

    def test_ltrace_identity(self) -> None:
        """Test exact trace for identity matrix."""
        n = 50
        A = Identity((n,))
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = linox.ltrace(A, key=key)

        assert jnp.allclose(trace_est, n)
        assert trace_std == 0.0

    def test_lexp_identity(self) -> None:
        """Test matrix exponential of identity: exp(I) = e*I."""
        n = 10
        A = Identity((n,))
        v = jnp.ones(n)

        result = linox.lexp(A, v=v)

        expected = jnp.exp(1.0) * v
        assert jnp.allclose(result, expected)

    def test_llog_identity(self) -> None:
        """Test matrix logarithm of identity: log(I) = 0."""
        n = 10
        A = Identity((n,))
        v = jnp.ones(n)

        result = linox.llog(A, v=v)

        expected = jnp.zeros(n)
        assert jnp.allclose(result, expected)

    def test_lpow_identity(self) -> None:
        """Test matrix power of identity: I^p = I."""
        n = 10
        A = Identity((n,))
        v = jnp.ones(n)

        result = linox.lpow(A, power=2.5, v=v)

        expected = v  # I^p = I, so I^p @ v = v
        assert jnp.allclose(result, expected)


class TestKroneckerDispatches:
    """Tests for Kronecker product dispatches."""

    def test_ltrace_kronecker(self) -> None:
        """Test trace of Kronecker product: trace(A⊗B) = trace(A)*trace(B)."""
        A = Diagonal(jnp.array([1.0, 2.0, 3.0]))
        B = Diagonal(jnp.array([4.0, 5.0]))
        K = Kronecker(A, B)
        key = jax.random.PRNGKey(0)

        trace_est, _trace_std = linox.ltrace(K, key=key)

        # trace(A⊗B) = trace(A) * trace(B) = 6 * 9 = 54
        true_trace = 6.0 * 9.0
        assert jnp.allclose(trace_est, true_trace)

    def test_lexp_kronecker(self) -> None:
        """Test exp(A⊗B) = exp(A)⊗exp(B)."""
        A = Diagonal(jnp.array([1.0, 2.0]))
        B = Diagonal(jnp.array([3.0, 4.0]))
        K = Kronecker(A, B)

        exp_K = linox.lexp(K, v=None)

        assert isinstance(exp_K, Kronecker)
        # Check structure: should be Kronecker(exp(A), exp(B))
        assert isinstance(exp_K.A, Diagonal)
        assert isinstance(exp_K.B, Diagonal)

    def test_lpow_kronecker(self) -> None:
        """Test (A⊗B)^p = A^p⊗B^p."""
        A = Diagonal(jnp.array([1.0, 4.0]))
        B = Diagonal(jnp.array([9.0, 16.0]))
        K = Kronecker(A, B)

        pow_K = linox.lpow(K, power=0.5, v=None)

        assert isinstance(pow_K, Kronecker)
        # Check that it's sqrt(A) ⊗ sqrt(B)
        assert isinstance(pow_K.A, Diagonal)
        assert isinstance(pow_K.B, Diagonal)
        assert jnp.allclose(pow_K.A.diag, jnp.array([1.0, 2.0]))
        assert jnp.allclose(pow_K.B.diag, jnp.array([3.0, 4.0]))


class TestEigenDDispatches:
    """Tests for EigenD (eigendecomposition) dispatches."""

    def test_ltrace_eigend(self) -> None:
        """Test trace from eigenvalues: trace(A) = sum(λ)."""
        # Create symmetric matrix
        A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        A = EigenD(A_dense)
        key = jax.random.PRNGKey(0)

        trace_est, trace_std = linox.ltrace(A, key=key)

        true_trace = 6.0
        assert jnp.allclose(trace_est, true_trace)
        assert trace_std == 0.0  # Exact from eigenvalues

    def test_lexp_eigend(self) -> None:
        """Test matrix exponential using eigendecomposition."""
        # Create diagonal matrix (easy to verify)
        A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        A = EigenD(A_dense)
        v = jnp.ones(3)

        result = linox.lexp(A, v=v)

        # exp(diag(1,2,3)) @ ones = [e^1, e^2, e^3]
        expected = jnp.exp(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, expected)

    def test_llog_eigend(self) -> None:
        """Test matrix logarithm using eigendecomposition."""
        A_dense = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        A = EigenD(A_dense)
        v = jnp.ones(3)

        result = linox.llog(A, v=v)

        expected = jnp.log(jnp.array([1.0, 2.0, 4.0]))
        assert jnp.allclose(result, expected)

    def test_lpow_eigend(self) -> None:
        """Test matrix power using eigendecomposition."""
        A_dense = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
        A = EigenD(A_dense)
        v = jnp.ones(3)

        result = linox.lpow(A, power=0.5, v=v)

        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)


class TestIsotropicAdditiveDispatches:
    """Tests for IsotropicAdditive (sI + A) dispatches."""

    def test_ltrace_isotropic(self) -> None:
        """Test trace(sI + A) = s*n + trace(A)."""
        n = 10
        s = 2.0
        A = Diagonal(jnp.ones(n))
        iso_A = IsotropicAdditiveLinearOperator(s, A)
        key = jax.random.PRNGKey(0)

        trace_est, _trace_std = linox.ltrace(iso_A, key=key)

        # trace(2I + I) = 2*10 + 10 = 30
        true_trace = 30.0
        assert jnp.allclose(trace_est, true_trace)

    def test_lexp_isotropic(self) -> None:
        """Test exp(sI + A) using eigendecomposition."""
        n = 3
        s = 1.0
        A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        A = Matrix(A_dense)
        iso_A = IsotropicAdditiveLinearOperator(s, A)
        v = jnp.ones(n)

        result = linox.lexp(iso_A, v=v)

        # exp(I + diag(1,2,3)) = exp(diag(2,3,4))
        expected = jnp.exp(jnp.array([2.0, 3.0, 4.0]))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_llog_isotropic(self) -> None:
        """Test log(sI + A) using eigendecomposition."""
        n = 3
        s = 1.0
        A_dense = jnp.diag(jnp.array([1.0, 3.0, 7.0]))
        A = Matrix(A_dense)
        iso_A = IsotropicAdditiveLinearOperator(s, A)
        v = jnp.ones(n)

        result = linox.llog(iso_A, v=v)

        # log(I + diag(1,3,7)) = log(diag(2,4,8))
        expected = jnp.log(jnp.array([2.0, 4.0, 8.0]))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_lpow_isotropic(self) -> None:
        """Test (sI + A)^p using eigendecomposition."""
        n = 3
        s = 1.0
        A_dense = jnp.diag(jnp.array([0.0, 3.0, 8.0]))
        A = Matrix(A_dense)
        iso_A = IsotropicAdditiveLinearOperator(s, A)
        v = jnp.ones(n)

        result = linox.lpow(iso_A, power=0.5, v=v)

        # (I + diag(0,3,8))^0.5 = diag(1,4,9)^0.5 = diag(1,2,3)
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected, atol=1e-5)


class TestStructureExploitation:
    """Tests that verify structure is actually being exploited (not falling back to general algorithms)."""

    def test_diagonal_faster_than_general(self) -> None:
        """Diagonal should use O(n) operations, not O(n^2)."""
        # This is more of a sanity check - we verify the correct type is returned
        A = Diagonal(jnp.arange(100.0))

        exp_A = linox.lexp(A, v=None)

        # Should return Diagonal, not a dense matrix
        assert isinstance(exp_A, Diagonal)

    def test_kronecker_preserves_structure(self) -> None:
        """Kronecker operations should preserve structure."""
        A = Diagonal(jnp.array([1.0, 2.0]))
        B = Diagonal(jnp.array([3.0, 4.0]))
        K = Kronecker(A, B)

        exp_K = linox.lexp(K, v=None)

        # Should return Kronecker of diagonals, not dense
        assert isinstance(exp_K, Kronecker)

    def test_eigend_uses_cached_eigendecomposition(self) -> None:
        """EigenD should use cached eigenvalues, not recompute."""
        A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        A = EigenD(A_dense)

        # First call caches eigendecomposition
        result1 = linox.lexp(A, v=jnp.ones(3))

        # Second call should use cache (can't directly test, but no errors)
        result2 = linox.llog(A, v=jnp.ones(3))

        # Both should work without recomputing
        assert result1.shape == (3,)
        assert result2.shape == (3,)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identity_of_size_1(self) -> None:
        """Test identity matrix of size 1."""
        A = Identity((1,))
        v = jnp.array([5.0])

        result = linox.lexp(A, v=v)

        expected = jnp.exp(1.0) * v
        assert jnp.allclose(result, expected)

    def test_diagonal_with_zeros(self) -> None:
        """Test diagonal matrix with zero entries."""
        A = Diagonal(jnp.array([0.0, 1.0, 2.0]))
        v = jnp.ones(3)

        result = linox.lexp(A, v=v)

        expected = jnp.exp(jnp.array([0.0, 1.0, 2.0]))
        assert jnp.allclose(result, expected)

    def test_negative_power(self) -> None:
        """Test negative powers (matrix inverse)."""
        A = Diagonal(jnp.array([1.0, 2.0, 4.0]))
        v = jnp.ones(3)

        result = linox.lpow(A, power=-1.0, v=v)

        expected = jnp.array([1.0, 0.5, 0.25])
        assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
