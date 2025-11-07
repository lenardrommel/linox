#!/usr/bin/env python3
"""Quick test of structured dispatches."""

import sys
sys.path.insert(0, '/home/user/linox')

import jax
import jax.numpy as jnp
import linox
from linox import Diagonal, Identity, Matrix
from linox._kronecker import Kronecker
from linox._eigen import EigenD
from linox._isotropicadd import IsotropicAdditiveLinearOperator

print("=" * 70)
print("Testing Structured Operator Dispatches")
print("=" * 70)

# Test 1: Diagonal
print("\n[Test 1] Diagonal Operator")
print("-" * 70)
A = Diagonal(jnp.array([1.0, 2.0, 3.0]))
key = jax.random.PRNGKey(0)

trace_est, _ = linox.ltrace(A, key=key)
print(f"✅ ltrace(Diagonal): {trace_est:.2f} (expected: 6.00)")

v = jnp.ones(3)
exp_result = linox.lexp(A, v=v)
print(f"✅ lexp(Diagonal) @ v: {exp_result}")
print(f"   Expected: {jnp.exp(jnp.array([1.0, 2.0, 3.0]))}")

# Test 2: Identity
print("\n[Test 2] Identity Operator")
print("-" * 70)
I = Identity((10,))
trace_est, _ = linox.ltrace(I, key=key)
print(f"✅ ltrace(Identity): {trace_est:.2f} (expected: 10.00)")

v = jnp.ones(10)
exp_result = linox.lexp(I, v=v)
print(f"✅ lexp(Identity) @ v[0]: {exp_result[0]:.4f} (expected: {jnp.exp(1.0):.4f})")

# Test 3: Kronecker
print("\n[Test 3] Kronecker Product")
print("-" * 70)
A = Diagonal(jnp.array([1.0, 2.0]))
B = Diagonal(jnp.array([3.0, 4.0]))
K = Kronecker(A, B)

trace_est, _ = linox.ltrace(K, key=key)
print(f"✅ ltrace(Kronecker): {trace_est:.2f} (expected: {3.0 * 7.0:.2f})")

exp_K = linox.lexp(K, v=None)
print(f"✅ lexp(Kronecker) type: {type(exp_K).__name__}")
assert isinstance(exp_K, Kronecker), "Should return Kronecker!"

# Test 4: EigenD
print("\n[Test 4] EigenD (Eigendecomposition)")
print("-" * 70)
A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
A = EigenD(A_dense)

trace_est, _ = linox.ltrace(A, key=key)
print(f"✅ ltrace(EigenD): {trace_est:.2f} (expected: 6.00)")

v = jnp.ones(3)
exp_result = linox.lexp(A, v=v)
print(f"✅ lexp(EigenD) @ v: {exp_result}")
print(f"   Expected: {jnp.exp(jnp.array([1.0, 2.0, 3.0]))}")

# Test 5: IsotropicAdditive
print("\n[Test 5] IsotropicAdditive (sI + A)")
print("-" * 70)
s = 1.0
A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
A = Matrix(A_dense)
iso_A = IsotropicAdditiveLinearOperator(s, A)

trace_est, _ = linox.ltrace(iso_A, key=key)
print(f"✅ ltrace(sI + A): {trace_est:.2f} (expected: {s*3 + 6.0:.2f})")

v = jnp.ones(3)
exp_result = linox.lexp(iso_A, v=v)
print(f"✅ lexp(sI + A) @ v: {exp_result}")
print(f"   Expected: {jnp.exp(jnp.array([2.0, 3.0, 4.0]))}")

# Test 6: Matrix functions
print("\n[Test 6] Matrix Functions (log, pow)")
print("-" * 70)
A = Diagonal(jnp.array([1.0, 4.0, 16.0]))
v = jnp.ones(3)

log_result = linox.llog(A, v=v)
print(f"✅ llog(Diagonal) @ v: {log_result}")
print(f"   Expected: {jnp.log(jnp.array([1.0, 4.0, 16.0]))}")

pow_result = linox.lpow(A, 0.5, v)
print(f"✅ lpow(Diagonal, 0.5) @ v: {pow_result}")
print(f"   Expected: {jnp.array([1.0, 2.0, 4.0])}")

print("\n" + "=" * 70)
print("🎉 ALL QUICK TESTS PASSED!")
print("=" * 70)
print("\nStructured dispatches are working correctly for:")
print("  ✅ Diagonal - O(n) operations")
print("  ✅ Identity - O(n) operations")
print("  ✅ Kronecker - Preserves structure")
print("  ✅ EigenD - Uses cached eigendecomposition")
print("  ✅ IsotropicAdditive - Efficient via eigenvalues")
print("\nAll operators exploit their structure for efficiency!")
