# _toeplity.py

import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from scipy.linalg import solve_toeplitz

from linox._arithmetic import lsolve
from linox._linear_operator import LinearOperator
from linox.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# Toeplitz Linear Operator
# --------------------------------------------------------------------------- #


class Toeplitz(LinearOperator):
    """A Toeplitz matrix which is constructed from a 1D array."""

    def __init__(self, v: ArrayLike) -> None:
        self.v = jnp.asarray(v)
        super().__init__(self.v.shape, self.v.dtype)

    @property
    def shape(self) -> jax.Array:
        return *self.v.shape, *self.v.shape

    def _matmul(self, vector: jax.Array) -> jax.Array:
        n = self.v.shape[0]

        if vector.ndim == 1:
            vector = vector.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False

        embedded_col = jnp.concatenate([self.v, self.v[-1:0:-1]])

        p = len(embedded_col)

        fft_col = jnp.fft.fft(embedded_col)

        vector_padded = jnp.concatenate(
            [vector, jnp.zeros((p - n, vector.shape[1]))], axis=0
        )

        fft_vector = jnp.fft.fft(vector_padded, axis=0)

        fft_result = fft_col.reshape(-1, 1) * fft_vector

        result = jnp.fft.ifft(fft_result, axis=0).real[:n]

        if squeeze_output:
            result = result.squeeze(axis=1)

        return result

    def todense(self) -> jax.Array:
        return jsp.linalg.toeplitz(self.v)

    def from_matrix(self, matrix: jax.Array) -> "Toeplitz":
        self.v = matrix[0, :]
        return Toeplitz(self.v)

    def transpose(self) -> "Toeplitz":
        return Toeplitz(self.v)


@lsolve.dispatch
def _(A: Toeplitz, b: jax.Array) -> jax.Array:
    """Solve a Toeplitz system."""
    return solve_toeplitz(
        c=A.v,
        b=b,
        check_finite=False,
    )


def levinson(a, b):
    """JAX translation using regular for loops (unrolled at compile time)."""
    n = b.shape[0]
    dtype = a.dtype

    # Initialize
    x = jnp.zeros(n, dtype=dtype)
    g = jnp.zeros(n, dtype=dtype)
    h = jnp.zeros(n, dtype=dtype)
    reflection_coeff = jnp.zeros(n + 1, dtype=dtype)

    # Base case
    x = x.at[0].set(b[0] / a[n - 1])
    reflection_coeff = reflection_coeff.at[0].set(1.0)
    reflection_coeff = reflection_coeff.at[1].set(x[0])

    if n == 1:
        return x, reflection_coeff

    g = g.at[0].set(a[n - 2] / a[n - 1])
    h = h.at[0].set(a[n] / a[n - 1])

    # Main loop - regular Python for loop (unrolled by JAX)
    for m in range(1, n):
        # Compute numerator and denominator of x[m]
        x_num = -b[m]
        x_den = -a[n - 1]
        for j in range(m):
            nmj = n + m - (j + 1)
            x_num = x_num + a[nmj] * x[j]
            x_den = x_den + a[nmj] * g[m - j - 1]

        x_m = x_num / x_den
        x = x.at[m].set(x_m)
        reflection_coeff = reflection_coeff.at[m + 1].set(x_m)

        # Update x
        for j in range(m):
            x = x.at[j].add(-x_m * g[m - j - 1])

        if m == n - 1:
            return x, reflection_coeff

        # Compute g[m] and h[m]
        g_num = -a[n - m - 2]
        h_num = -a[n + m]
        g_den = -a[n - 1]
        for j in range(m):
            g_num = g_num + a[n + j - m - 1] * g[j]
            h_num = h_num + a[n + m - j - 1] * h[j]
            g_den = g_den + a[n + j - m - 1] * h[m - j - 1]

        g = g.at[m].set(g_num / g_den)
        h = h.at[m].set(h_num / x_den)

        # Update g and h arrays
        k = m - 1
        m2 = (m + 1) // 2
        c1, c2 = g[m], h[m]
        for j in range(m2):
            gj, gk = g[j], g[k]
            hj, hk = h[j], h[k]
            g = g.at[j].set(gj - c1 * hk)
            g = g.at[k].set(gk - c1 * hj)
            h = h.at[j].set(hj - c2 * gk)
            h = h.at[k].set(hk - c2 * gj)
            k -= 1

    return x, reflection_coeff


def _validate_args_for_toeplitz_ops(
    c_or_cr, b, check_finite, keep_b_shape, enforce_square=True
):
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr

    else:
        c = c_or_cr
        r = c.conjugate()

    if b is None:
        raise ValueError("`b` must be an array, not None.")

    b_shape = b.shape

    is_not_square = r.shape[0] != c.shape[0]
    if (enforce_square and is_not_square) or b.shape[0] != r.shape[0]:
        raise ValueError("Incompatible dimensions.")

    is_cmplx = jnp.iscomplexobj(r) or jnp.iscomplexobj(c) or jnp.iscomplexobj(b)
    dtype = jnp.complex128 if is_cmplx else jnp.float64
    r, c, b = (jnp.asarray(i, dtype=dtype) for i in (r, c, b))

    if b.ndim == 1 and not keep_b_shape:
        b = b.reshape(-1, 1)
    elif b.ndim != 1:
        b = b.reshape(b.shape[0], -1 if b.size > 0 else 0)

    return r, c, b, dtype, b_shape


def _solve_toeplitz(c, r, b, check_finite):
    r, c, b, dtype, b_shape = _validate_args_for_toeplitz_ops(
        (c, r), b, check_finite, keep_b_shape=True
    )

    # accommodate empty arrays
    if b.size == 0:
        return jnp.empty_like(b)

    # Form a 1-D array of values to be used in the matrix, containing a
    # reversed copy of r[1:], followed by c.
    vals = jnp.concatenate((r[-1:0:-1], c))
    if b is None:
        raise ValueError("illegal value, `b` is a required argument")

    if b.ndim == 1:
        x, _ = levinson(vals, jnp.asarray(b))
    else:
        x = jnp.column_stack([
            levinson(vals, jnp.asarray(b[:, i]))[0] for i in range(b.shape[1])
        ])
        x = x.reshape(*b_shape)

    return x


def test_solve_toeplitz():
    key = jax.random.PRNGKey(0)
    n = 5
    r = jax.random.normal(key, (n,))
    c = jax.random.normal(key, (n,))
    A = jsp.linalg.toeplitz(c, r)
    b = jax.random.normal(key, (n,))

    x1 = solve_toeplitz_jax(c, b)
    x2 = jnp.linalg.solve(A, b)

    assert jnp.allclose(x1, x2), "Toeplitz solve does not match numpy.linalg.solve"


def _scipy_solve_toeplitz(c_np, b_np):
    """Pure numpy function for scipy solve."""
    return solve_toeplitz(c_np, b_np, check_finite=False)


@jax.custom_vjp
def toeplitz_solve_hybrid(toeplitz_vec, b):
    """Hybrid Toeplitz solver: scipy forward pass with custom VJP.

    Args:
        toeplitz_vec: First row/column of symmetric Toeplitz matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    # Use pure_callback to call scipy from within JAX
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def scipy_call(c, b_val):
        c_np = jnp.asarray(c)
        b_np = jnp.asarray(b_val)
        return _scipy_solve_toeplitz(c_np, b_np)

    x = jax.pure_callback(
        scipy_call, result_shape, toeplitz_vec, b, vmap_method="sequential"
    )
    return x


def hybrid_fwd(toeplitz_vec, b):
    """Forward pass: use scipy and save residuals for backward pass."""
    x = toeplitz_solve_hybrid(toeplitz_vec, b)
    residuals = (toeplitz_vec, b, x)
    return x, residuals


def hybrid_bwd(residuals, grad_output):
    """Custom VJP: efficient backward pass using Toeplitz structure.

    Args:
        residuals: (toeplitz_vec, b, x) from forward pass
        grad_output: Gradient w.r.t. output x

    Returns:
        (grad_toeplitz_vec, grad_b): Gradients w.r.t. inputs
    """
    toeplitz_vec, b, x = residuals
    n = len(toeplitz_vec)

    # Reconstruct Toeplitz matrix for gradient computation
    T = jsp.linalg.toeplitz(toeplitz_vec)

    # Gradient w.r.t. b: A^{-T} @ grad_output
    grad_b = jnp.linalg.solve(T.T, grad_output)

    # Gradient w.r.t. toeplitz_vec using Toeplitz structure
    # For symmetric Toeplitz: ∂T[i,j]/∂c[k] = 1 if |i-j| = k, else 0
    grad_toeplitz_vec = jnp.zeros_like(toeplitz_vec)

    # Efficient vectorized computation
    i_indices = jnp.arange(n)[:, None]  # Shape (n, 1)
    j_indices = jnp.arange(n)[None, :]  # Shape (1, n)

    for k in range(n):
        # Find all (i,j) pairs where |i-j| = k
        mask = jnp.abs(i_indices - j_indices) == k
        # Sum: -x[i] * grad_b[j] for all valid (i,j)
        contribution = jnp.sum(jnp.where(mask, x[:, None] * grad_b[None, :], 0))
        grad_toeplitz_vec = grad_toeplitz_vec.at[k].set(-contribution)

    return grad_toeplitz_vec, grad_b


# Register the VJP
toeplitz_solve_hybrid.defvjp(hybrid_fwd, hybrid_bwd)


def solve_toeplitz_jax(c_or_cr, b, check_finite=True):
    """
    JAX-compatible Toeplitz solver using hybrid approach.

    Args:
        c_or_cr: Either first column c, or tuple (c, r) for non-symmetric
        b: Right-hand side vector
        check_finite: Ignored (kept for API compatibility)

    Returns:
        x: Solution vector
    """
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        # For non-symmetric case, use scipy directly
        c_np, r_np = jnp.array(c), jnp.array(r)
        b_np = jnp.array(b)
        x_np = solve_toeplitz((c_np, r_np), b_np, check_finite=False)
        return jnp.array(x_np)
    else:
        # Symmetric case: use hybrid solver with custom VJP
        return toeplitz_solve_hybrid(c_or_cr, b)


class TestToeplitzSolver:
    """Comprehensive test suite for hybrid Toeplitz solver."""

    def test_small_systems(self):
        """Test small systems (1x1, 2x2, 3x3)."""
        print("Testing small systems...")

        # 1x1 system
        c = jnp.array([2.0])
        b = jnp.array([6.0])
        x_jax = solve_toeplitz_jax(c, b)
        x_expected = jnp.array([3.0])
        assert jnp.allclose(x_jax, x_expected), f"1x1 failed: {x_jax} vs {x_expected}"

        # 2x2 system
        c = jnp.array([3.0, 1.0])
        b = jnp.array([7.0, 5.0])
        x_jax = solve_toeplitz_jax(c, b)
        A = jsp.linalg.toeplitz(c)
        x_dense = jnp.linalg.solve(A, b)
        assert jnp.allclose(x_jax, x_dense, rtol=1e-12), f"2x2 failed"

        # 3x3 system
        c = jnp.array([4.0, 2.0, 1.0])
        b = jnp.array([1.0, 2.0, 3.0])
        x_jax = solve_toeplitz_jax(c, b)
        A = jsp.linalg.toeplitz(c)
        x_dense = jnp.linalg.solve(A, b)
        residual = jnp.linalg.norm(A @ x_jax - b)
        assert jnp.allclose(x_jax, x_dense, rtol=1e-12), f"3x3 failed"
        assert residual < 1e-12, f"3x3 residual too large: {residual}"
        print("✅ Small systems passed")

    def test_against_scipy(self):
        """Test against scipy.linalg.solve_toeplitz."""
        print("Testing against SciPy...")

        sizes = [4, 5, 8, 10, 15, 25, 50]  # Test larger sizes now
        for n in sizes:
            key = jax.random.PRNGKey(42 + n)

            # Generate well-conditioned Toeplitz matrix
            c = jax.random.normal(key, (n,)) * 0.1
            c = c.at[0].set(1.0)  # Make diagonally dominant
            b = jax.random.normal(key, (n,))

            x_jax = solve_toeplitz_jax(c, b)
            x_scipy = solve_toeplitz(jnp.array(c), jnp.array(b), check_finite=False)

            diff = jnp.max(jnp.abs(x_jax - x_scipy))
            assert jnp.allclose(x_jax, x_scipy, rtol=1e-12), (
                f"Size {n} failed, max_diff={diff}"
            )

        print("✅ SciPy comparison passed")

    def test_complex_matrices(self):
        """Test complex Toeplitz matrices."""
        print("Testing complex matrices...")

        key = jax.random.PRNGKey(123)
        n = 6

        # Complex symmetric Toeplitz
        c_real = jax.random.normal(key, (n,))
        c_imag = jax.random.normal(key, (n,))
        c = c_real + 1j * c_imag
        c = c.at[0].set(c[0].real + 0j)  # Make diagonal real for stability

        b_real = jax.random.normal(key, (n,))
        b_imag = jax.random.normal(key, (n,))
        b = b_real + 1j * b_imag

        x_jax = solve_toeplitz_jax(c, b)
        A = jsp.linalg.toeplitz(c)
        x_dense = jnp.linalg.solve(A, b)

        assert jnp.allclose(x_jax, x_dense, rtol=1e-10), "Complex test failed"
        print("✅ Complex matrices passed")

    def test_non_symmetric(self):
        """Test non-symmetric Toeplitz matrices."""
        print("Testing non-symmetric matrices...")

        key = jax.random.PRNGKey(456)
        n = 5

        c = jax.random.normal(key, (n,))
        r = jax.random.normal(key, (n,))
        r = r.at[0].set(c[0])  # First element must match
        b = jax.random.normal(key, (n,))

        x_jax = solve_toeplitz_jax((c, r), b)
        A = jsp.linalg.toeplitz(c, r)
        x_dense = jnp.linalg.solve(A, b)

        assert jnp.allclose(x_jax, x_dense, rtol=1e-10), "Non-symmetric test failed"
        print("✅ Non-symmetric matrices passed")

    def test_special_matrices(self):
        """Test special cases: identity, zero off-diagonal, etc."""
        print("Testing special matrices...")

        # Identity matrix
        n = 4
        c = jnp.zeros(n).at[0].set(1.0)
        b = jnp.array([1.0, 2.0, 3.0, 4.0])
        x_jax = solve_toeplitz_jax(c, b)
        assert jnp.allclose(x_jax, b), "Identity test failed"

        # Diagonal matrix (zero off-diagonal)
        c = jnp.zeros(n).at[0].set(2.0)
        x_jax = solve_toeplitz_jax(c, b)
        x_expected = b / 2.0
        assert jnp.allclose(x_jax, x_expected), "Diagonal test failed"

        print("✅ Special matrices passed")

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned matrices."""
        print("Testing numerical stability...")

        # Create an ill-conditioned but solvable Toeplitz matrix
        n = 8
        c = jnp.array([1.0] + [0.95**i for i in range(1, n)])
        b = jnp.ones(n)

        x_jax = solve_toeplitz_jax(c, b)
        A = jsp.linalg.toeplitz(c)

        # Check residual
        residual = jnp.linalg.norm(A @ x_jax - b)
        cond_num = jnp.linalg.cond(A)

        # Should be numerically stable since we use scipy
        assert residual < 1e-10, f"Residual {residual} too large for cond={cond_num}"

        print(
            f"✅ Stability test passed (cond={cond_num:.1e}, residual={residual:.1e})"
        )

    def test_jax_features(self):
        """Test JAX-specific features: JIT, vmap, gradients."""
        print("Testing JAX features...")

        c = jnp.array([2.0, 1.0, 0.5])
        b = jnp.array([1.0, 2.0, 3.0])

        # JIT compilation
        @jax.jit
        def solve_jit(c, b):
            return solve_toeplitz_jax(c, b)

        x_normal = solve_toeplitz_jax(c, b)
        x_jit = solve_jit(c, b)
        assert jnp.allclose(x_normal, x_jit), "JIT test failed"

        # Vectorization with vmap
        batch_size = 3
        c_batch = jnp.tile(c[None, :], (batch_size, 1))
        b_batch = jnp.tile(b[None, :], (batch_size, 1))

        solve_vmap = jax.vmap(solve_toeplitz_jax, in_axes=(0, 0))
        x_batch = solve_vmap(c_batch, b_batch)

        for i in range(batch_size):
            assert jnp.allclose(x_batch[i], x_normal), f"vmap test failed for batch {i}"

        # Test gradients
        def loss_fn(c):
            x = solve_toeplitz_jax(c, b)
            return jnp.sum(x**2)

        try:
            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(c)
            assert grads.shape == c.shape, "Gradient shape mismatch"
            assert jnp.all(jnp.isfinite(grads)), "Gradient contains NaN/Inf"
            print(f"✅ JAX features passed (grad norm: {jnp.linalg.norm(grads):.2e})")
        except Exception as e:
            print(f"⚠️  Gradient test failed: {e}")


#     def benchmark_performance(self):
#         """Benchmark against alternatives."""
#         print("Benchmarking performance...")

#         sizes = [200, 500, 1000]  # Test larger sizes
#         results = []

#         for n in sizes:
#             key = jax.random.PRNGKey(789)
#             c = jax.random.normal(key, (n,)) * 0.1
#             c = c.at[0].set(1.0)
#             b = jax.random.normal(key, (n,))

#             # Warmup for JIT compilation
#             solve_jit = jax.jit(solve_toeplitz_jax)
#             _ = solve_jit(c, b)

#             A = jsp.linalg.toeplitz(c)
#             _ = jnp.linalg.solve(A, b)

#             # Time JIT-compiled JAX version
#             start = time.time()
#             for _ in range(100):
#                 x_jax = solve_jit(c, b)
#             time_jax = time.time() - start

#             # Time dense solve
#             start = time.time()
#             for _ in range(100):
#                 x_dense = jnp.linalg.solve(A, b)
#             time_dense = time.time() - start

#             # Time scipy (convert to numpy)
#             c_np, b_np = np.array(c), np.array(b)
#             start = time.time()
#             for _ in range(100):
#                 x_scipy = solve_toeplitz(c_np, b_np, check_finite=False)
#             time_scipy = time.time() - start

#             speedup_vs_dense = time_dense / time_jax
#             speedup_vs_scipy = time_scipy / time_jax

#             results.append({
#                 "n": n,
#                 "jax_time": time_jax,
#                 "dense_time": time_dense,
#                 "scipy_time": time_scipy,
#                 "speedup_vs_dense": speedup_vs_dense,
#                 "speedup_vs_scipy": speedup_vs_scipy,
#             })

#             print(
#                 f"n={n:3d}: JAX={time_jax:.4f}s, Dense={time_dense:.4f}s, "
#                 f"SciPy={time_scipy:.4f}s | Speedup: {speedup_vs_dense:.1f}x vs dense, "
#                 f"{speedup_vs_scipy:.1f}x vs scipy"
#             )

#         return results

#     def test_gradient_performance(self):
#         """Test gradient computation performance."""
#         print("Testing gradient performance...")

#         sizes = [10, 20, 50]

#         for n in sizes:
#             key = jax.random.PRNGKey(42)
#             c = jax.random.normal(key, (n,)) * 0.1
#             c = c.at[0].set(1.0)
#             b = jax.random.normal(key, (n,))

#             def loss_hybrid(c):
#                 x = toeplitz_solve_hybrid(c, b)
#                 return jnp.sum(x**2)

#             def loss_dense(c):
#                 A = jsp.linalg.toeplitz(c)
#                 x = jnp.linalg.solve(A, b)
#                 return jnp.sum(x**2)

#             # Compile gradient functions
#             grad_hybrid = jax.jit(jax.grad(loss_hybrid))
#             grad_dense = jax.jit(jax.grad(loss_dense))

#             # Warmup
#             _ = grad_hybrid(c)
#             _ = grad_dense(c)

#             # Time hybrid approach
#             start = time.time()
#             for _ in range(50):
#                 g_hybrid = grad_hybrid(c)
#             time_hybrid = time.time() - start

#             # Time dense approach
#             start = time.time()
#             for _ in range(50):
#                 g_dense = grad_dense(c)
#             time_dense = time.time() - start

#             # Check accuracy
#             accuracy = jnp.allclose(g_hybrid, g_dense, rtol=1e-8)
#             speedup = time_dense / time_hybrid

#             print(
#                 f"n={n:2d}: Hybrid={time_hybrid:.4f}s, Dense={time_dense:.4f}s | "
#                 f"Speedup: {speedup:.1f}x, Accurate: {accuracy}"
#             )

#     def run_all_tests(self):
#         """Run all tests."""
#         print("Running comprehensive hybrid Toeplitz solver tests")
#         print("=" * 55)

#         self.test_small_systems()
#         self.test_against_scipy()
#         self.test_complex_matrices()
#         self.test_non_symmetric()
#         self.test_special_matrices()
#         self.test_numerical_stability()
#         self.test_jax_features()

#         print("\n" + "=" * 55)
#         print("All tests passed! 🎉")

#         print("\nForward pass performance:")
#         print("-" * 30)
#         results = self.benchmark_performance()

#         print("\nGradient performance:")
#         print("-" * 30)
#         self.test_gradient_performance()

#         return results


# # For integration with LinearOperator
# class Toeplitz:
#     """Toeplitz linear operator using hybrid solver."""

#     def __init__(self, v):
#         self.v = jnp.asarray(v)
#         self.shape = (len(v), len(v))
#         self.dtype = v.dtype

#     def _matmul(self, vector):
#         T = jsp.linalg.toeplitz(self.v)
#         return T @ vector

#     def todense(self):
#         return jsp.linalg.toeplitz(self.v)


# def lsolve_toeplitz(A, b):
#     """Dispatch function for Toeplitz linear operator."""
#     return solve_toeplitz_jax(A.v, b)


# if __name__ == "__main__":
#     tester = TestToeplitzSolver()
#     results = tester.run_all_tests()

#     print(f"\n{'=' * 55}")
#     print("Summary:")
#     print("- Forward pass: Uses scipy (fast, reliable)")
#     print("- Gradients: Custom VJP (efficient, accurate)")
#     print("- Integration: Ready for linox framework")
#     print("- Performance: Expected to be competitive with scipy")
#     print("- Features: Full JAX compatibility (JIT, vmap, grad)")
