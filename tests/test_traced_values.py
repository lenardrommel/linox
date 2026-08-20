# test_traced_values.py

import jax
import linox
from jax import numpy as jnp


def create_operator():
    key = jax.random.key(0)
    shape = (100, 100)
    A_m = jax.random.uniform(key=key, shape=shape)
    A = 1 * linox.utils.as_linop(A_m)
    B_m = jax.random.uniform(key=key, shape=(shape))
    B = 1 * linox.utils.as_linop(B_m)
    C_m = jax.random.uniform(key=key, shape=(shape))
    1 * linox.utils.as_linop(C_m)
    return A + B, A_m + B_m


op, M = create_operator()
op_inv = linox.linverse(op)
res = op_inv @ jnp.zeros(op.shape[0])
