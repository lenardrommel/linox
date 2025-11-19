#!/usr/bin/env python3
"""Quick test script for linox algorithms."""

import sys

sys.path.insert(0, "/home/user/linox")

import jax
import jax.numpy as jnp

import linox
from linox import Matrix
from linox._algorithms import (
    hutchinson_trace,
    lanczos_eigh,
    lanczos_matrix_function,
    lanczos_tridiag,
    stochastic_lanczos_quadrature,
)

# Test 1: Lanczos Tridiagonalization
n = 50
# Use diagonal matrix with different eigenvalues (not identity!)
A_dense = jnp.diag(jnp.arange(1.0, n + 1.0))
A = Matrix(A_dense)
v0 = jnp.ones(n)
num_iters = 10

Q, alpha, beta = lanczos_tridiag(A, v0, num_iters, reortho=True)

# Check orthogonality
QTQ = Q.T @ Q
orthogonality_error = jnp.max(jnp.abs(QTQ - jnp.eye(num_iters)))
assert orthogonality_error < 1e-5, "Orthogonality test failed!"

# Test 2: Eigenvalue Computation
diag_vals = jnp.arange(1.0, 21.0)
A = Matrix(jnp.diag(diag_vals))
v0 = jnp.ones(20)

eigs, vecs = lanczos_eigh(A, v0, num_iters=20, k=5, which="LA")
error = jnp.max(jnp.abs(eigs - diag_vals[-5:][::-1]))
assert error < 1e-3, "Eigenvalue test failed!"

# Test 3: Hutchinson Trace Estimation
n = 100
# Use a more interesting matrix
diag_vals = jnp.arange(1.0, n + 1.0)
A = Matrix(jnp.diag(diag_vals))
key = jax.random.PRNGKey(0)

trace_est, trace_std = hutchinson_trace(A, key, num_samples=300)
true_trace = jnp.sum(diag_vals)
# For diagonal matrices, Hutchinson gives exact result with zero variance
assert abs(trace_est - true_trace) < max(3 * trace_std, 10.0), (
    "Trace estimation test failed!"
)

# Test 4: Matrix Exponential
n = 50
A = Matrix(-jnp.eye(n))
v = jnp.ones(n)

exp_Av = lanczos_matrix_function(A, v, jnp.exp, num_iters=10)
expected = jnp.exp(-1.0) * v
error = jnp.max(jnp.abs(exp_Av - expected))
assert error < 1e-3, "Matrix exponential test failed!"

# Test 5: Stochastic Lanczos Quadrature (Log-Determinant)
n = 20
diag_vals = jnp.arange(1.0, n + 1.0)
A = Matrix(jnp.diag(diag_vals))
key = jax.random.PRNGKey(42)

logdet_est, logdet_std = stochastic_lanczos_quadrature(
    A, jnp.log, key, num_samples=50, num_iters=20
)
true_logdet = jnp.sum(jnp.log(diag_vals))
# Handle case where std=0 (perfect estimate for diagonal matrices)
assert abs(logdet_est - true_logdet) < max(5 * logdet_std, 1.0), "SLQ test failed!"

# Test 6: Linox Arithmetic Integration
A = Matrix(2.0 * jnp.eye(30))
v = jnp.ones(30)
key = jax.random.PRNGKey(0)

# Test ltrace
trace_est, _ = linox.ltrace(A, key=key, num_samples=100)
assert abs(trace_est - 60.0) < 10.0, "ltrace test failed!"

# Test lexp
exp_result = linox.lexp(A, v=v, num_iters=5)
assert jnp.allclose(exp_result, jnp.exp(2.0) * v, atol=1e-2), "lexp test failed!"

# Test llog
log_result = linox.llog(A, v=v, num_iters=5)
assert jnp.allclose(log_result, jnp.log(2.0) * v, atol=1e-2), "llog test failed!"

# Test lpow (use positional args for plum dispatch)
pow_result = linox.lpow(A, 0.5, v, 5)
assert jnp.allclose(pow_result, jnp.sqrt(2.0) * v, atol=1e-2), "lpow test failed!"
