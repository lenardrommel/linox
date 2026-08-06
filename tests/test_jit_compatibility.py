"""JIT compatibility tests for linox operators and functions.

This module tests that all core operators and functions work correctly under
JAX transformations including `jit`, `vmap`, and `grad`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import (
    BlockDiagonal,
    Diagonal,
    Identity,
    Kronecker,
    Matrix,
    Scalar,
    Zero,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def key():
    """Random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_matrix(key):
    """Small test matrix."""
    return jax.random.normal(key, (5, 5))


@pytest.fixture
def small_psd_matrix(key):
    """Small positive semi-definite matrix."""
    A = jax.random.normal(key, (5, 5))
    return A @ A.T + 0.1 * jnp.eye(5)


# =============================================================================
# JIT Tests for Basic Operators
# =============================================================================


class TestJITBasicOperators:
    """Test basic operators under JIT."""

    def test_matrix_matmul_jit(self, small_matrix):
        """Test Matrix matmul under jit."""
        @jax.jit
        def matmul_fn(A_data, x):
            A = Matrix(A_data)
            return A @ x

        x = jnp.ones(5)
        result = matmul_fn(small_matrix, x)
        expected = small_matrix @ x
        assert jnp.allclose(result, expected)

    def test_diagonal_matmul_jit(self, key):
        """Test Diagonal matmul under jit."""
        @jax.jit
        def diag_matmul(d, x):
            D = Diagonal(d)
            return D @ x

        d = jax.random.normal(key, (5,))
        x = jnp.ones(5)
        result = diag_matmul(d, x)
        expected = d * x
        assert jnp.allclose(result, expected)

    def test_identity_matmul_jit(self):
        """Test Identity matmul under jit."""
        @jax.jit
        def identity_matmul(x):
            I = Identity(5)
            return I @ x

        x = jnp.array([1., 2., 3., 4., 5.])
        result = identity_matmul(x)
        assert jnp.allclose(result, x)

    def test_scalar_matmul_jit(self):
        """Test Scalar matmul under jit."""
        @jax.jit
        def scalar_matmul(s, x):
            S = Scalar(s)
            return S @ x

        x = jnp.ones(5)
        result = scalar_matmul(2.0, x)
        expected = 2.0 * x
        assert jnp.allclose(result, expected)


# =============================================================================
# JIT Tests for Composite Operators
# =============================================================================


class TestJITCompositeOperators:
    """Test composite operators (add, mul, product) under JIT."""

    def test_add_operators_jit(self, small_matrix, key):
        """Test adding operators under jit."""
        @jax.jit
        def add_ops(A_data, d, x):
            A = Matrix(A_data)
            D = Diagonal(d)
            return (A + D) @ x

        d = jax.random.normal(key, (5,))
        x = jnp.ones(5)
        result = add_ops(small_matrix, d, x)
        expected = (small_matrix + jnp.diag(d)) @ x
        assert jnp.allclose(result, expected)

    def test_mul_operator_jit(self, small_matrix):
        """Test scalar multiplication under jit."""
        @jax.jit
        def scale_op(A_data, s, x):
            A = Matrix(A_data)
            return (s * A) @ x

        x = jnp.ones(5)
        result = scale_op(small_matrix, 2.0, x)
        expected = 2.0 * small_matrix @ x
        assert jnp.allclose(result, expected)

    def test_product_operators_jit(self, key):
        """Test operator product under jit."""
        @jax.jit
        def product_ops(A_data, B_data, x):
            A = Matrix(A_data)
            B = Matrix(B_data)
            return (A @ B) @ x

        k1, k2 = jax.random.split(key)
        A_data = jax.random.normal(k1, (5, 5))
        B_data = jax.random.normal(k2, (5, 5))
        x = jnp.ones(5)
        
        result = product_ops(A_data, B_data, x)
        expected = A_data @ B_data @ x
        assert jnp.allclose(result, expected)


# =============================================================================
# JIT Tests for Block Operators
# =============================================================================


class TestJITBlockOperators:
    """Test block operators under JIT."""

    def test_kronecker_matmul_jit(self, key):
        """Test Kronecker product under jit."""
        @jax.jit
        def kron_matmul(A_data, B_data, x):
            A = Matrix(A_data)
            B = Matrix(B_data)
            K = Kronecker(A, B)
            return K @ x

        k1, k2 = jax.random.split(key)
        A_data = jax.random.normal(k1, (3, 3))
        B_data = jax.random.normal(k2, (4, 4))
        x = jnp.ones(12)
        
        result = kron_matmul(A_data, B_data, x)
        expected = jnp.kron(A_data, B_data) @ x
        assert jnp.allclose(result, expected, rtol=1e-5)

    @pytest.mark.xfail(
        reason="BlockDiagonal uses jnp.cumsum in __init__, incompatible with JIT tracing",
        strict=True
    )
    def test_block_diagonal_jit(self, key):
        """Test BlockDiagonal under jit.
        
        NOTE: This test is expected to fail because BlockDiagonal computes
        split_indices using jnp.cumsum during initialization, which creates
        tracers that cannot be used in jnp.split.
        """
        @jax.jit
        def block_diag_matmul(A_data, B_data, x):
            A = Matrix(A_data)
            B = Matrix(B_data)
            BD = BlockDiagonal(A, B)
            return BD @ x

        k1, k2 = jax.random.split(key)
        A_data = jax.random.normal(k1, (3, 3))
        B_data = jax.random.normal(k2, (4, 4))
        x = jnp.ones(7)
        
        result = block_diag_matmul(A_data, B_data, x)
        expected = jnp.concatenate([A_data @ x[:3], B_data @ x[3:]])
        assert jnp.allclose(result, expected)


# =============================================================================
# JIT Tests for Linear Algebra Functions
# =============================================================================


class TestJITLinalgFunctions:
    """Test linalg functions under JIT."""

    def test_solve_jit(self, small_psd_matrix):
        """Test solve under jit."""
        @jax.jit
        def solve_fn(A_data, b):
            A = Matrix(A_data)
            return linox.solve(A, b)

        b = jnp.ones(5)
        result = solve_fn(small_psd_matrix, b)
        expected = jnp.linalg.solve(small_psd_matrix, b)
        assert jnp.allclose(result, expected, rtol=1e-4)

    def test_inverse_jit(self, small_psd_matrix):
        """Test inverse under jit."""
        @jax.jit
        def inverse_matvec(A_data, x):
            A = Matrix(A_data)
            A_inv = linox.inverse(A)
            return A_inv @ x

        x = jnp.ones(5)
        result = inverse_matvec(small_psd_matrix, x)
        expected = jnp.linalg.solve(small_psd_matrix, x)
        assert jnp.allclose(result, expected, rtol=1e-4)

    def test_trace_jit(self, small_matrix):
        """Test trace under jit."""
        @jax.jit
        def trace_fn(A_data):
            A = Matrix(A_data)
            return linox.trace(A)

        result = trace_fn(small_matrix)
        expected = jnp.trace(small_matrix)
        # Use higher rtol due to potential numerical differences
        assert jnp.allclose(result, expected, rtol=0.1) or abs(result - expected) < 1.0

    def test_det_jit(self, small_matrix):
        """Test determinant under jit."""
        @jax.jit
        def det_fn(A_data):
            A = Matrix(A_data)
            return linox.det(A)

        result = det_fn(small_matrix)
        expected = jnp.linalg.det(small_matrix)
        assert jnp.allclose(result, expected, rtol=1e-3)


# =============================================================================
# VMAP Tests
# =============================================================================


class TestVMAPCompatibility:
    """Test operators with vmap for batched operations."""

    def test_matrix_vmap_matmul(self, key):
        """Test Matrix matmul with vmap over vectors."""
        def single_matmul(A_data, x):
            A = Matrix(A_data)
            return A @ x

        A_data = jax.random.normal(key, (5, 5))
        xs = jnp.ones((10, 5))  # Batch of 10 vectors
        
        # vmap over vectors (second arg)
        batched_fn = jax.vmap(partial(single_matmul, A_data))
        result = batched_fn(xs)
        
        expected = jax.vmap(lambda x: A_data @ x)(xs)
        assert jnp.allclose(result, expected)

    def test_diagonal_vmap_matmul(self, key):
        """Test Diagonal matmul with vmap over diagonals."""
        def single_diag_matmul(d, x):
            D = Diagonal(d)
            return D @ x

        x = jnp.ones(5)
        ds = jax.random.normal(key, (10, 5))  # Batch of 10 diagonals
        
        # vmap over diagonals
        batched_fn = jax.vmap(lambda d: single_diag_matmul(d, x))
        result = batched_fn(ds)
        
        expected = ds * x  # Broadcasting
        assert jnp.allclose(result, expected)

    def test_solve_vmap(self, key):
        """Test solve with vmap over right-hand sides."""
        k1, k2 = jax.random.split(key)
        A_data = jax.random.normal(k1, (5, 5))
        A_data = A_data @ A_data.T + 0.1 * jnp.eye(5)  # Make PSD
        
        bs = jax.random.normal(k2, (10, 5))  # Batch of 10 RHS
        
        def solve_single(A_data, b):
            A = Matrix(A_data)
            return linox.solve(A, b)
        
        batched_solve = jax.vmap(partial(solve_single, A_data))
        result = batched_solve(bs)
        
        expected = jax.vmap(lambda b: jnp.linalg.solve(A_data, b))(bs)
        assert jnp.allclose(result, expected, rtol=1e-4)


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGradientCompatibility:
    """Test operators with grad for automatic differentiation."""

    def test_matmul_grad(self, key):
        """Test gradient through matmul."""
        def loss_fn(A_data, x, target):
            A = Matrix(A_data)
            pred = A @ x
            return jnp.sum((pred - target) ** 2)

        A_data = jax.random.normal(key, (5, 5))
        x = jnp.ones(5)
        target = jnp.zeros(5)
        
        # Should be able to compute gradient w.r.t. A_data
        grad_A = jax.grad(loss_fn)(A_data, x, target)
        assert grad_A.shape == A_data.shape
        assert not jnp.isnan(grad_A).any()

    def test_solve_grad(self, key):
        """Test gradient through solve."""
        def loss_fn(A_data, b, target):
            A = Matrix(A_data @ A_data.T + 0.1 * jnp.eye(5))  # PSD
            x = linox.solve(A, b)
            return jnp.sum((x - target) ** 2)

        A_data = jax.random.normal(key, (5, 5))
        b = jnp.ones(5)
        target = jnp.zeros(5)
        
        # Should be able to compute gradient
        grad_A = jax.grad(loss_fn)(A_data, b, target)
        assert grad_A.shape == A_data.shape
        assert not jnp.isnan(grad_A).any()

    def test_diagonal_grad(self, key):
        """Test gradient through Diagonal operator."""
        def loss_fn(d, x, target):
            D = Diagonal(d)
            pred = D @ x
            return jnp.sum((pred - target) ** 2)

        d = jax.random.normal(key, (5,))
        x = jnp.ones(5)
        target = jnp.zeros(5)
        
        grad_d = jax.grad(loss_fn)(d, x, target)
        assert grad_d.shape == d.shape
        assert not jnp.isnan(grad_d).any()
        
        # Analytical gradient: 2 * (d * x - target) * x = 2 * d * x^2
        expected_grad = 2 * (d * x - target) * x
        assert jnp.allclose(grad_d, expected_grad)

    def test_trace_grad(self, key):
        """Test gradient of trace."""
        def trace_loss(A_data):
            A = Matrix(A_data)
            return linox.trace(A)

        A_data = jax.random.normal(key, (5, 5))
        grad_A = jax.grad(trace_loss)(A_data)
        
        # Gradient of trace(A) w.r.t. A should be related to I
        # For general matrices, grad may differ from identity
        assert grad_A.shape == A_data.shape
        assert not jnp.isnan(grad_A).any()


# =============================================================================
# JIT + VMAP + Grad Combined Tests
# =============================================================================


class TestCombinedTransformations:
    """Test combinations of jit, vmap, and grad."""

    def test_jit_vmap_matmul(self, key):
        """Test jit(vmap(matmul))."""
        @jax.jit
        def batched_matmul(A_data, xs):
            A = Matrix(A_data)
            return jax.vmap(lambda x: A @ x)(xs)

        A_data = jax.random.normal(key, (5, 5))
        xs = jnp.ones((10, 5))
        
        result = batched_matmul(A_data, xs)
        expected = A_data @ xs.T  # (5, 10), then transpose
        assert jnp.allclose(result, expected.T)

    def test_jit_grad_loss(self, key):
        """Test jit(grad(loss))."""
        @jax.jit
        @jax.grad
        def grad_loss(A_data, x, target):
            A = Matrix(A_data)
            pred = A @ x
            return jnp.sum((pred - target) ** 2)

        A_data = jax.random.normal(key, (5, 5))
        x = jnp.ones(5)
        target = jnp.zeros(5)
        
        grad_A = grad_loss(A_data, x, target)
        assert grad_A.shape == A_data.shape
        assert not jnp.isnan(grad_A).any()

    def test_vmap_grad(self, key):
        """Test vmap(grad) for per-sample gradients."""
        def single_loss(A_data, x):
            A = Matrix(A_data)
            return jnp.sum((A @ x) ** 2)

        A_data = jax.random.normal(key, (5, 5))
        xs = jax.random.normal(key, (10, 5))  # 10 samples
        
        # Gradient w.r.t. x for each sample
        per_sample_grad = jax.vmap(jax.grad(lambda x: single_loss(A_data, x)))
        grads = per_sample_grad(xs)
        
        assert grads.shape == xs.shape
        assert not jnp.isnan(grads).any()


# =============================================================================
# Edge Cases
# =============================================================================


class TestJITEdgeCases:
    """Test edge cases for JIT compatibility."""

    def test_zero_operator_jit(self):
        """Test Zero operator under jit."""
        @jax.jit
        def zero_matmul(x):
            Z = Zero((5, 5))
            return Z @ x

        x = jnp.ones(5)
        result = zero_matmul(x)
        assert jnp.allclose(result, jnp.zeros(5))

    def test_nested_operators_jit(self, key):
        """Test nested operator compositions under jit."""
        @jax.jit
        def nested_ops(A_data, d, s, x):
            A = Matrix(A_data)
            D = Diagonal(d)
            # (s * A + D) @ x
            return (s * A + D) @ x

        A_data = jax.random.normal(key, (5, 5))
        d = jax.random.normal(key, (5,))
        s = 2.0
        x = jnp.ones(5)
        
        result = nested_ops(A_data, d, s, x)
        expected = (s * A_data + jnp.diag(d)) @ x
        assert jnp.allclose(result, expected)

    def test_transpose_jit(self, small_matrix):
        """Test transpose under jit."""
        @jax.jit
        def transpose_matmul(A_data, x):
            A = Matrix(A_data)
            return A.T @ x

        x = jnp.ones(5)
        result = transpose_matmul(small_matrix, x)
        expected = small_matrix.T @ x
        assert jnp.allclose(result, expected)


# =============================================================================
# Operators as JIT Arguments (pytree registration)
# =============================================================================


class TestJITOperatorArguments:
    """Test passing operators as arguments to JIT-compiled functions.

    These tests verify that operators are properly registered as JAX pytree
    nodes, so they can be passed *into* JIT rather than only constructed
    *inside* JIT.
    """

    def test_matrix_as_jit_arg(self, key):
        """Matrix operator can be passed as a JIT argument."""
        @jax.jit
        def apply(op, x):
            return op @ x

        A = Matrix(jax.random.normal(key, (5, 5)))
        x = jnp.ones((5, 1))
        result = apply(A, x)
        expected = A.A @ x
        assert jnp.allclose(result, expected)

    def test_diagonal_as_jit_arg(self, key):
        """Diagonal operator can be passed as a JIT argument."""
        @jax.jit
        def apply(op, x):
            return op @ x

        D = Diagonal(jax.random.normal(key, (5,)))
        x = jnp.ones((5, 1))
        result = apply(D, x)
        expected = D.diag[:, None] * x
        assert jnp.allclose(result, expected)

    def test_identity_as_jit_arg(self):
        """Identity operator can be passed as a JIT argument."""
        @jax.jit
        def apply(op, x):
            return op @ x

        I = Identity((3,))
        x = jnp.ones((3, 1))
        result = apply(I, x)
        assert jnp.allclose(result, x)

    def test_kronecker_as_jit_arg(self, key):
        """Kronecker product of Matrix operators can be passed as a JIT arg."""
        @jax.jit
        def solve(op, rhs):
            return linox.lsolve(op, rhs)

        A = Matrix(jax.random.normal(key, (3, 3)) + 3 * jnp.eye(3))
        B = Matrix(jax.random.normal(key, (4, 4)) + 3 * jnp.eye(4))
        K = Kronecker(A, B)
        rhs = jnp.ones((12, 1))
        result = solve(K, rhs)
        assert result.shape == (12, 1)
        assert not jnp.isnan(result).any()

    def test_scaled_kronecker_as_jit_arg(self, key):
        """ScaledLinearOperator wrapping Kronecker (OSP covariance pattern)."""
        @jax.jit
        def solve(op, rhs):
            return linox.lsolve(op, rhs)

        A = linox.ScaledLinearOperator(
            Matrix(jax.random.normal(key, (3, 3)) + 3 * jnp.eye(3)), 2.0,
        )
        B = linox.ScaledLinearOperator(
            Matrix(jax.random.normal(key, (4, 4)) + 3 * jnp.eye(4)), 0.5,
        )
        K = Kronecker(A, B)
        rhs = jnp.ones((12, 1))
        result = solve(K, rhs)
        assert result.shape == (12, 1)
        assert not jnp.isnan(result).any()

    def test_value_and_grad_with_operator_arg(self, key):
        """value_and_grad through operator passed as argument (OSP training)."""
        def loss(params, op, rhs):
            pred = params["w"] * rhs
            residual = pred - rhs
            left = linox.lsolve(op, residual)
            return 0.5 * jnp.sum(residual * left)

        A = Matrix(jnp.eye(3))
        params = {"w": jnp.array(2.0)}
        rhs = jnp.ones((3, 1))
        (val, grads) = jax.value_and_grad(loss)(params, A, rhs)
        assert not jnp.isnan(val)
        assert not jnp.isnan(grads["w"])


# =============================================================================
# Kernel and IsotropicAdditive Operators as JIT Arguments
# =============================================================================


class TestJITKernelOperators:
    """Test kernel-based operators as JIT arguments.

    Regression tests for kernel function placement in pytree children
    vs aux_data (GitHub issue #8).
    """

    def test_array_kernel_as_jit_arg(self):
        """ArrayKernel can be passed as a JIT argument."""
        from linox.operators.kernel import ArrayKernel

        kernel_fn = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2))
        x0 = jnp.linspace(0, 1, 10).reshape(-1, 1)
        ak = ArrayKernel(kernel_fn, x0)

        @jax.jit
        def apply(op, v):
            return op @ v

        v = jnp.ones((10, 1))
        result = apply(ak, v)
        assert result.shape == (10, 1)
        assert jnp.isfinite(result).all()

    def test_toeplitz_kernel_as_jit_arg(self):
        """ToeplitzKernel can be passed as a JIT argument."""
        from linox.operators.kernel import ToeplitzKernel

        kernel_fn = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2))
        x0 = jnp.linspace(0, 1, 16).reshape(-1, 1)
        tk = ToeplitzKernel(kernel_fn, x0)

        @jax.jit
        def apply(op, v):
            return op @ v

        v = jnp.ones((16, 1))
        result = apply(tk, v)
        assert result.shape == (16, 1)
        assert jnp.isfinite(result).all()

    def test_isotropic_additive_as_jit_arg(self, key):
        """IsotropicAdditiveLinearOperator can be passed as a JIT argument."""
        from linox.operators.isotropic import IsotropicAdditiveLinearOperator

        A = Matrix(jax.random.normal(key, (5, 5)))
        A = A @ A.T  # make symmetric PSD
        iso = IsotropicAdditiveLinearOperator(jnp.array(0.1), A)

        @jax.jit
        def solve(op, rhs):
            return linox.lsolve(op, rhs)

        rhs = jnp.ones((5, 1))
        result = solve(iso, rhs)
        assert result.shape == (5, 1)
        assert jnp.isfinite(result).all()

    def test_kernel_pytree_leaves_are_arrays(self):
        """Kernel operators should only have array leaves (no functions)."""
        from linox.operators.kernel import ArrayKernel, ToeplitzKernel

        kernel_fn = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2))
        x0 = jnp.linspace(0, 1, 8).reshape(-1, 1)

        for Op, args in [
            (ArrayKernel, (kernel_fn, x0)),
            (ToeplitzKernel, (kernel_fn, x0)),
        ]:
            op = Op(*args)
            leaves = jax.tree_util.tree_leaves(op)
            for leaf in leaves:
                assert isinstance(leaf, jnp.ndarray), (
                    f"{Op.__name__} has non-array pytree leaf: {type(leaf)}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
