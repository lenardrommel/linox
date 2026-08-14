"""Hutchinson's stochastic trace estimator.

Approximate trace and diagonal using stochastic probes (Hutchinson's method).
Optimized for batched execution via matrix-matrix multiplication (A @ Z).
"""

import jax
import jax.numpy as jnp
from jax import random

from linox.utils.array import LinearOperatorLike


def hutchinson_trace(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate trace of a linear operator using Hutchinson's method.

    Computes Monte Carlo estimate: trace(A) ≈ (1/M) * sum(v_i^T A v_i)
    Uses batched execution (A @ Z) for efficiency.

    Args:
        A: LinearOperator or array (n, n)
        key: PRNG key
        num_samples: Number of random probes
        distribution: 'rademacher' (default) or 'normal'

    Returns
    -------
        (estimate, std_error)
    """
    n = A.shape[-1]
    Z = _generate_probes(key, n, num_samples, distribution)  # (n, num_samples)

    # Batched matmul: (n, n) @ (n, samples) -> (n, samples)
    AZ = A @ Z

    # v^T A v = sum(v * Av) for each column
    samples = jnp.sum(Z * AZ, axis=0)  # (samples,)

    return jnp.mean(samples), jnp.std(samples, ddof=1) / jnp.sqrt(num_samples)


def hutchinson_diagonal(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate diagonal of a linear operator using Hutchinson's method.

    Computes: diag(A) ≈ (1/M) * sum(v_i ⊙ (A v_i))

    Args:
        A: LinearOperator or array (n, n)
        key: PRNG key
        num_samples: Number of probes
        distribution: 'rademacher' (default) or 'normal'

    Returns
    -------
        (estimate, std_error) each of shape (n,)
    """
    n = A.shape[-1]
    Z = _generate_probes(key, n, num_samples, distribution)  # (n, num_samples)

    AZ = A @ Z

    samples = Z * AZ  # (n, samples)

    # Mean over samples axis=1
    mean = jnp.mean(samples, axis=1)
    std = jnp.std(samples, axis=1, ddof=1) / jnp.sqrt(num_samples)

    return mean, std


def hutchinson_trace_and_diagonal(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Jointly estimate trace and diagonal using shared probes."""
    n = A.shape[-1]
    Z = _generate_probes(key, n, num_samples, distribution)

    AZ = A @ Z

    # Diagonal samples: (n, samples)
    diag_samples = Z * AZ

    # Trace samples: sum over n -> (samples,)
    trace_samples = jnp.sum(diag_samples, axis=0)

    trace_mean = jnp.mean(trace_samples)
    trace_std = jnp.std(trace_samples, ddof=1) / jnp.sqrt(num_samples)

    diag_mean = jnp.mean(diag_samples, axis=1)
    diag_std = jnp.std(diag_samples, axis=1, ddof=1) / jnp.sqrt(num_samples)

    return {"trace": (trace_mean, trace_std), "diagonal": (diag_mean, diag_std)}


def _generate_probes(key, n, num_samples, distribution):
    """Generate (n, num_samples) probe matrix."""
    if distribution == "rademacher":
        # Bernoulli in {0, 1} -> { -1, 1 }
        return 2 * random.bernoulli(key, shape=(n, num_samples)) - 1.0
    if distribution == "normal":
        return random.normal(key, shape=(n, num_samples))
    msg = f"Unknown distribution: {distribution}"
    raise ValueError(msg)
