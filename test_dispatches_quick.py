#!/usr/bin/env python3
"""Quick test of structured dispatches."""

import sys

sys.path.insert(0, "/home/user/linox")

import jax
import jax.numpy as jnp

import linox
from linox import Diagonal, Identity, Matrix
from linox._eigen import EigenD
from linox._isotropicadd import IsotropicAdditiveLinearOperator
from linox._kronecker import Kronecker

# Test 1: Diagonal
A = Diagonal(jnp.array([1.0, 2.0, 3.0]))
key = jax.random.PRNGKey(0)

trace_est, _ = linox.ltrace(A, key=key)

v = jnp.ones(3)
exp_result = linox.lexp(A, v=v)

# Test 2: Identity
I = Identity((10,))
trace_est, _ = linox.ltrace(I, key=key)

v = jnp.ones(10)
exp_result = linox.lexp(I, v=v)

# Test 3: Kronecker
A = Diagonal(jnp.array([1.0, 2.0]))
B = Diagonal(jnp.array([3.0, 4.0]))
K = Kronecker(A, B)

trace_est, _ = linox.ltrace(K, key=key)

exp_K = linox.lexp(K, v=None)
assert isinstance(exp_K, Kronecker), "Should return Kronecker!"

# Test 4: EigenD
A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
A = EigenD(A_dense)

trace_est, _ = linox.ltrace(A, key=key)

v = jnp.ones(3)
exp_result = linox.lexp(A, v=v)

# Test 5: IsotropicAdditive
s = 1.0
A_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
A = Matrix(A_dense)
iso_A = IsotropicAdditiveLinearOperator(s, A)

trace_est, _ = linox.ltrace(iso_A, key=key)

v = jnp.ones(3)
exp_result = linox.lexp(iso_A, v=v)

# Test 6: Matrix functions
A = Diagonal(jnp.array([1.0, 4.0, 16.0]))
v = jnp.ones(3)

log_result = linox.llog(A, v=v)

pow_result = linox.lpow(A, 0.5, v)
