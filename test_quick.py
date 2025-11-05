#!/usr/bin/env python3
"""Quick test script for linox algorithms."""

import sys
sys.path.insert(0, '/home/user/linox')

import jax
import jax.numpy as jnp
from linox import Matrix
import linox
from linox._algorithms import (
    lanczos_tridiag,
    lanczos_eigh,
    hutchinson_trace,
    lanczos_matrix_function,
    stochastic_lanczos_quadrature,
)

print("=" * 70)
print("Testing Linox Matrix-Free Algorithms")
print("=" * 70)

# Test 1: Lanczos Tridiagonalization
print("\n[Test 1] Lanczos Tridiagonalization")
print("-" * 70)
n = 50
# Use diagonal matrix with different eigenvalues (not identity!)
A_dense = jnp.diag(jnp.arange(1.0, n + 1.0))
A = Matrix(A_dense)
v0 = jnp.ones(n)
num_iters = 10

Q, alpha, beta = lanczos_tridiag(A, v0, num_iters, reortho=True)
print(f"✅ Lanczos completed: Q shape = {Q.shape}, alpha shape = {alpha.shape}")

# Check orthogonality
QTQ = Q.T @ Q
orthogonality_error = jnp.max(jnp.abs(QTQ - jnp.eye(num_iters)))
print(f"   Orthogonality error: {orthogonality_error:.2e}")
assert orthogonality_error < 1e-5, "Orthogonality test failed!"

# Test 2: Eigenvalue Computation
print("\n[Test 2] Lanczos Eigenvalue Computation")
print("-" * 70)
diag_vals = jnp.arange(1.0, 21.0)
A = Matrix(jnp.diag(diag_vals))
v0 = jnp.ones(20)

eigs, vecs = lanczos_eigh(A, v0, num_iters=20, k=5, which='LA')
print(f"✅ Computed top 5 eigenvalues")
print(f"   Eigenvalues: {eigs}")
print(f"   Expected: {diag_vals[-5:][::-1]}")
error = jnp.max(jnp.abs(eigs - diag_vals[-5:][::-1]))
print(f"   Error: {error:.2e}")
assert error < 1e-3, "Eigenvalue test failed!"

# Test 3: Hutchinson Trace Estimation
print("\n[Test 3] Hutchinson Trace Estimation")
print("-" * 70)
n = 100
# Use a more interesting matrix
diag_vals = jnp.arange(1.0, n + 1.0)
A = Matrix(jnp.diag(diag_vals))
key = jax.random.PRNGKey(0)

trace_est, trace_std = hutchinson_trace(A, key, num_samples=300)
true_trace = jnp.sum(diag_vals)
print(f"✅ Trace estimation completed")
print(f"   Estimated trace: {trace_est:.2f} ± {trace_std:.2f}")
print(f"   True trace: {true_trace:.2f}")
print(f"   Error: {abs(trace_est - true_trace):.2f}")
# For diagonal matrices, Hutchinson gives exact result with zero variance
assert abs(trace_est - true_trace) < max(3 * trace_std, 10.0), "Trace estimation test failed!"

# Test 4: Matrix Exponential
print("\n[Test 4] Matrix Exponential f(A)v")
print("-" * 70)
n = 50
A = Matrix(-jnp.eye(n))
v = jnp.ones(n)

exp_Av = lanczos_matrix_function(A, v, jnp.exp, num_iters=10)
expected = jnp.exp(-1.0) * v
error = jnp.max(jnp.abs(exp_Av - expected))
print(f"✅ Matrix exponential completed")
print(f"   Result[0]: {exp_Av[0]:.6f}")
print(f"   Expected[0]: {expected[0]:.6f}")
print(f"   Max error: {error:.2e}")
assert error < 1e-3, "Matrix exponential test failed!"

# Test 5: Stochastic Lanczos Quadrature (Log-Determinant)
print("\n[Test 5] Stochastic Lanczos Quadrature (Log-Det)")
print("-" * 70)
n = 20
diag_vals = jnp.arange(1.0, n + 1.0)
A = Matrix(jnp.diag(diag_vals))
key = jax.random.PRNGKey(42)

logdet_est, logdet_std = stochastic_lanczos_quadrature(
    A, jnp.log, key, num_samples=50, num_iters=20
)
true_logdet = jnp.sum(jnp.log(diag_vals))
print(f"✅ SLQ log-determinant completed")
print(f"   Estimated log|A|: {logdet_est:.2f} ± {logdet_std:.2f}")
print(f"   True log|A|: {true_logdet:.2f}")
print(f"   Error: {abs(logdet_est - true_logdet):.2f}")
# Handle case where std=0 (perfect estimate for diagonal matrices)
assert abs(logdet_est - true_logdet) < max(5 * logdet_std, 1.0), "SLQ test failed!"

# Test 6: Linox Arithmetic Integration
print("\n[Test 6] Linox Arithmetic API (ltrace, lexp, llog)")
print("-" * 70)
A = Matrix(2.0 * jnp.eye(30))
v = jnp.ones(30)
key = jax.random.PRNGKey(0)

# Test ltrace
trace_est, _ = linox.ltrace(A, key=key, num_samples=100)
print(f"✅ linox.ltrace: {trace_est:.2f} (expected: {2.0 * 30})")
assert abs(trace_est - 60.0) < 10.0, "ltrace test failed!"

# Test lexp
exp_result = linox.lexp(A, v=v, num_iters=5)
print(f"✅ linox.lexp: result[0] = {exp_result[0]:.4f} (expected: {jnp.exp(2.0):.4f})")
assert jnp.allclose(exp_result, jnp.exp(2.0) * v, atol=1e-2), "lexp test failed!"

# Test llog
log_result = linox.llog(A, v=v, num_iters=5)
print(f"✅ linox.llog: result[0] = {log_result[0]:.4f} (expected: {jnp.log(2.0):.4f})")
assert jnp.allclose(log_result, jnp.log(2.0) * v, atol=1e-2), "llog test failed!"

# Test lpow (use positional args for plum dispatch)
pow_result = linox.lpow(A, 0.5, v, 5)
print(f"✅ linox.lpow: result[0] = {pow_result[0]:.4f} (expected: {jnp.sqrt(2.0):.4f})")
assert jnp.allclose(pow_result, jnp.sqrt(2.0) * v, atol=1e-2), "lpow test failed!"

print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print("\nThe matrix-free algorithms are working correctly!")
print("\nKey features tested:")
print("  ✅ Lanczos tridiagonalization (orthogonality preserved)")
print("  ✅ Eigenvalue computation (accurate top-k eigenvalues)")
print("  ✅ Hutchinson trace estimation (stochastic, unbiased)")
print("  ✅ Matrix functions (exp, log, pow via Krylov methods)")
print("  ✅ Stochastic Lanczos quadrature (log-determinant)")
print("  ✅ Linox arithmetic integration (ltrace, lexp, llog, lpow)")
print("\nYou can now use these algorithms for:")
print("  • GP inference (log-determinant estimation)")
print("  • Large eigenvalue problems (few eigenvalues)")
print("  • Matrix functions without densification")
print("  • Trace estimation for large operators")
