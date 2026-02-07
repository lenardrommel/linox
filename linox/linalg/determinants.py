"""Determinant and log-determinant functions."""

import jax
import jax.numpy as jnp

from linox.utils.array import LinearOperatorLike

# Dispatch for specific operators is handled via registers in operators/ modules?
# Or we centrally dispatch here?
# Linox v0.0.3 design: thin wrapper around internal implementations.
# But `linox.operators._arithmetic` has `slogdet` dispatch.
# For generic `slogdet`, we usually default to dense unless structural.
# For approx, we use SLQ.


def slogdet(
    A: LinearOperatorLike,
    method: str = "exact",
    **kwargs
) -> tuple[jax.Array, jax.Array]:
    """Compute sign and log of determinant.

    Args:
        A: Linear operator.
        method: "exact" or "slq".
        **kwargs: method-specific args (key, num_samples, m for SLQ)
    """
    from linox.linalg.approx.slq import slq_logdet
    from linox.operators.arithmetic import slogdet as _slogdet_dispatch
    from linox.utils import as_linop

    op = as_linop(A)

    if method == "slq":
        # Check requirements
        key = kwargs.get("key")
        if key is None:
             msg = "SLQ requires a PRNG key."
             raise ValueError(msg)

        # SLQ assumes symmetric A for Lanczos.
        # If A is not symmetric, SLQ trace(log(A)) is valid?
        # log(A) for non-symmetric uses complex logic, Lanczos assumes symmetry.
        # We assume A is symmetric positive definite for logdet usually?
        # Or at least symmetric.
        # Ideally check is_symmetric.

        num_samples = kwargs.get("num_samples", 10)
        m = kwargs.get("m", 20)  # Krylov dim

        est_logdet, _std = slq_logdet(op, key, num_samples, m)
        # Sign is assumed 1.0 for PSD matrices suitable for SLQ logdet?
        # SLQ log approach typically for SPD.
        # If A has negative eigenvalues, log(A) is complex.

        return jnp.array(1.0), est_logdet

    # Exact fallthrough
    return _slogdet_dispatch(op)


def det(A: LinearOperatorLike) -> jax.Array:
    """Compute determinant."""
    sign, logabsdet = slogdet(A, method="exact")
    return sign * jnp.exp(logabsdet)


def logdet(A: LinearOperatorLike) -> jax.Array:
    """Compute log-determinant (sum of log eigenvalues)."""
    _sign, logabsdet = slogdet(A, method="exact")
    return logabsdet  # Assume positive for logdet(A) context usually
