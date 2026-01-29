"""Stochastic trace estimation using Hutchinson's method.

This module implements Hutchinson's stochastic trace estimator and related
algorithms for computing traces of large linear operators without explicit
matrix construction or densification.

The implementation is inspired by the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al.

Key algorithms:
- Hutchinson trace estimation: Stochastic estimation of trace(A)
- Diagonal estimation: Stochastic estimation of diag(A)
- Trace and diagonal estimation: Joint estimation

References:
----------
.. [1] M. F. Hutchinson, "A stochastic estimator of the trace of the influence
       matrix for Laplacian smoothing splines," Communications in Statistics -
       Simulation and Computation, vol. 19, no. 2, pp. 433-450, 1990.

.. [2] matfree: Matrix-free linear algebra in JAX
       https://github.com/pnkraemer/matfree

.. [3] N. Krämer, M. Schober, and P. Hennig, "Gradients of functions of large matrices,"
       arXiv preprint arXiv:2405.17277, 2024.
"""

import jax
import jax.numpy as jnp
from jax import random

from linox.typing import LinearOperatorLike


def hutchinson_trace(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate trace of a linear operator using Hutchinson's method.

    Computes a Monte Carlo estimate of trace(A) using random test vectors:
        trace(A) ≈ (1/num_samples) * sum_i v_i^T A v_i

    This avoids the need to densify the operator or compute all diagonal
    elements explicitly. The estimator is unbiased and has variance that
    decreases as O(1/num_samples).

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator whose trace we want to estimate. Should support
        matrix-vector multiplication via `A @ v`.
    key : jax.Array
        JAX random key for generating test vectors.
    num_samples : int, optional
        Number of random test vectors to use. More samples give more accurate
        estimates but require more matrix-vector products. Default is 100.
    distribution : str, optional
        Distribution for test vectors:
        - 'rademacher': Uniform random ±1 (recommended, minimal variance)
        - 'normal': Standard Gaussian N(0, 1)
        Default is 'rademacher'.

    Returns:
    -------
    trace_estimate : jax.Array
        Monte Carlo estimate of trace(A).
    trace_std : jax.Array
        Standard error of the estimate (std / sqrt(num_samples)).

    Examples:
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.eye(1000) + 0.1 * jnp.ones((1000, 1000)))
    >>> key = jax.random.PRNGKey(0)
    >>> trace_est, trace_std = hutchinson_trace(A, key, num_samples=100)
    >>> # True trace is 1000 + 0.1*1000 = 1100
    >>> print(f"Estimate: {trace_est:.2f} ± {trace_std:.2f}")

    Notes:
    -----
    The Rademacher distribution (±1 with equal probability) is theoretically
    optimal for trace estimation in the sense of minimizing variance [1].

    For GPU/TPU acceleration, consider increasing num_samples and using
    jax.vmap to parallelize the matrix-vector products.

    References:
    ----------
    Based on Hutchinson (1990) [1] and matfree.stochtrace [2].
    """
    # Get operator shape
    if hasattr(A, "shape"):
        n = A.shape[0]
    else:
        # If A is a JAX array
        n = A.shape[0]

    # Generate test vectors
    if distribution == "rademacher":
        # Rademacher: uniform ±1
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: 2 * random.bernoulli(k, shape=(n,)) - 1)(keys)
    elif distribution == "normal":
        # Standard Gaussian
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: random.normal(k, shape=(n,)))(keys)
    else:
        msg = f"Unknown distribution: {distribution}. Use 'rademacher' or 'normal'."
        raise ValueError(msg)

    # Compute v^T A v for each test vector
    def compute_quadratic_form(v):
        Av = A @ v
        return jnp.dot(v, Av)

    quadratic_forms = jax.vmap(compute_quadratic_form)(V)

    # Monte Carlo estimate
    trace_estimate = jnp.mean(quadratic_forms)
    trace_std = jnp.std(quadratic_forms, ddof=1) / jnp.sqrt(num_samples)

    return trace_estimate, trace_std


def hutchinson_diagonal(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Estimate diagonal of a linear operator using Hutchinson's method.

    Computes a Monte Carlo estimate of diag(A) using random test vectors:
        diag(A) ≈ (1/num_samples) * sum_i v_i ⊙ (A v_i)

    where ⊙ denotes element-wise multiplication. This avoids densifying
    the operator.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator whose diagonal we want to estimate.
    key : jax.Array
        JAX random key for generating test vectors.
    num_samples : int, optional
        Number of random test vectors to use. Default is 100.
    distribution : str, optional
        Distribution for test vectors ('rademacher' or 'normal').
        Default is 'rademacher'.

    Returns:
    -------
    diagonal_estimate : jax.Array, shape (n,)
        Monte Carlo estimate of diag(A).
    diagonal_std : jax.Array, shape (n,)
        Standard error of each diagonal element estimate.

    Examples:
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.diag(jnp.arange(100.0)) + 0.1 * jnp.ones((100, 100)))
    >>> key = jax.random.PRNGKey(0)
    >>> diag_est, diag_std = hutchinson_diagonal(A, key, num_samples=200)
    >>> # Compare with true diagonal
    >>> true_diag = jnp.diag(A.todense())
    >>> print(f"Max error: {jnp.max(jnp.abs(diag_est - true_diag)):.4f}")

    Notes:
    -----
    For operators with explicit diagonal() methods, use those instead as they
    are exact. This is useful when the diagonal is not directly accessible or
    would require densification.
    """
    # Get operator shape
    n = A.shape[0] if hasattr(A, "shape") else A.shape[0]

    # Generate test vectors
    if distribution == "rademacher":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: 2 * random.bernoulli(k, shape=(n,)) - 1.0)(keys)
    elif distribution == "normal":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: random.normal(k, shape=(n,)))(keys)
    else:
        msg = f"Unknown distribution: {distribution}. Use 'rademacher' or 'normal'."
        raise ValueError(msg)

    # Compute v ⊙ (A v) for each test vector
    def compute_diagonal_sample(v):
        Av = A @ v
        return v * Av

    diagonal_samples = jax.vmap(compute_diagonal_sample)(V)

    # Monte Carlo estimate
    diagonal_estimate = jnp.mean(diagonal_samples, axis=0)
    diagonal_std = jnp.std(diagonal_samples, axis=0, ddof=1) / jnp.sqrt(num_samples)

    return diagonal_estimate, diagonal_std


def hutchinson_trace_and_diagonal(
    A: LinearOperatorLike,
    key: jax.Array,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Jointly estimate trace and diagonal using Hutchinson's method.

    Efficiently computes both trace(A) and diag(A) estimates using the same
    set of random test vectors. This is more efficient than calling both
    hutchinson_trace and hutchinson_diagonal separately.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator to estimate properties of.
    key : jax.Array
        JAX random key for generating test vectors.
    num_samples : int, optional
        Number of random test vectors to use. Default is 100.
    distribution : str, optional
        Distribution for test vectors ('rademacher' or 'normal').
        Default is 'rademacher'.

    Returns:
    -------
    result : dict
        Dictionary with keys:
        - 'trace': (estimate, std) for trace(A)
        - 'diagonal': (estimate, std) for diag(A)

    Examples:
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.eye(100) + 0.1 * jnp.ones((100, 100)))
    >>> key = jax.random.PRNGKey(0)
    >>> result = hutchinson_trace_and_diagonal(A, key, num_samples=100)
    >>> trace_est, trace_std = result['trace']
    >>> diag_est, diag_std = result['diagonal']

    Notes:
    -----
    This function reuses the same matrix-vector products for both estimates,
    making it approximately 2x faster than calling the functions separately.
    """
    # Get operator shape
    n = A.shape[0] if hasattr(A, "shape") else A.shape[0]

    # Generate test vectors
    if distribution == "rademacher":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: 2 * random.bernoulli(k, shape=(n,)) - 1.0)(keys)
    elif distribution == "normal":
        keys = random.split(key, num_samples)
        V = jax.vmap(lambda k: random.normal(k, shape=(n,)))(keys)
    else:
        msg = f"Unknown distribution: {distribution}. Use 'rademacher' or 'normal'."
        raise ValueError(msg)

    # Compute both quadratic form and diagonal samples
    def compute_samples(v):
        Av = A @ v
        trace_sample = jnp.dot(v, Av)
        diagonal_sample = v * Av
        return trace_sample, diagonal_sample

    trace_samples, diagonal_samples = jax.vmap(compute_samples)(V)

    # Trace estimates
    trace_estimate = jnp.mean(trace_samples)
    trace_std = jnp.std(trace_samples, ddof=1) / jnp.sqrt(num_samples)

    # Diagonal estimates
    diagonal_estimate = jnp.mean(diagonal_samples, axis=0)
    diagonal_std = jnp.std(diagonal_samples, axis=0, ddof=1) / jnp.sqrt(num_samples)

    return {
        "trace": (trace_estimate, trace_std),
        "diagonal": (diagonal_estimate, diagonal_std),
    }
