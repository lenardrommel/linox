"""Stochastic Lanczos Quadrature (SLQ).

Approximates trace(f(A)) using stochastic probes and Lanczos tridiagonalization.
"""

import jax
import jax.numpy as jnp
from jax import random

from linox.linalg.approx.lanczos import lanczos_matrix_function
from linox.utils.array import LinearOperatorLike


def slq(
    A: LinearOperatorLike,
    func_scalar: callable,  # f: scalar -> scalar (e.g. jnp.log)
    key: jax.Array,
    num_samples: int = 10,
    m: int = 20,  # Krylov iterations
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate trace(f(A)) using SLQ.

    trace(f(A)) ≈ (1/M) * sum_i v_i^T f(A) v_i
    v_i^T f(A) v_i ≈ ||v_i||^2 * e_1^T f(T_m) e_1
    """
    n = A.shape[-1]

    # 1. Generate probes
    # Batched generation
    if distribution == "rademacher":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: 2 * random.bernoulli(k, shape=(n,)) - 1.0)(keys)
    elif distribution == "normal":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: random.normal(k, shape=(n,)))(keys)
    else:
        msg = f"Unknown distribution: {distribution}"
        raise ValueError(msg)

    # 2. Lanczos on each probe
    # Note: Lanczos is inherently sequential per vector, so we vmap the entire lanczos process.

    def process_probe(v):
        # f(A)v approx
        w = lanczos_matrix_function(A, v, func_scalar, m)
        # v^T f(A) v
        return jnp.dot(v, w)

    # vmap over samples
    estimates = jax.vmap(process_probe)(V)

    return jnp.mean(estimates), jnp.std(estimates, ddof=1) / jnp.sqrt(num_samples)


def slq_logdet(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 10,
    m: int = 20,
) -> tuple[jax.Array, jax.Array]:
    """Estimate log-determinant using SLQ.

    logdet(A) = trace(log(A))
    """
    return slq(A, jnp.log, key, num_samples, m)
