"""Matrix function approximations using Krylov subspace methods.

This module implements matrix-function-vector products f(A)v and trace
estimation trace(f(A)) using Lanczos, Arnoldi, and Chebyshev approximations.

The implementations are inspired by the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al.

Key algorithms:
- Lanczos-based matrix functions for symmetric operators
- Arnoldi-based matrix functions for general operators
- Chebyshev polynomial approximations
- Stochastic Lanczos quadrature for trace(f(A))

References
----------
.. [1] N. Krämer, M. Schober, and P. Hennig, "Gradients of functions of large matrices,"
       arXiv preprint arXiv:2405.17277, 2024.

.. [2] matfree: Matrix-free linear algebra in JAX
       https://github.com/pnkraemer/matfree

.. [3] Y. Saad, "Analysis of some Krylov subspace approximations to the matrix
       exponential operator," SIAM Journal on Numerical Analysis, vol. 29, no. 1,
       pp. 209-228, 1992.

.. [4] N. J. Higham, "Functions of Matrices: Theory and Computation," SIAM, 2008.
"""

import jax
import jax.numpy as jnp

from linox._types import LinearOperatorLike
from linox.linalg.approx.lanczos import lanczos_matrix_function


def stochastic_lanczos_quadrature(
    A: LinearOperatorLike,
    func: callable,
    key: jax.Array,
    num_samples: int = 100,
    num_iters: int = 20,
    distribution: str = "rademacher",
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Estimate trace(f(A)) using stochastic Lanczos quadrature (SQL).

    Combines Hutchinson trace estimation with Lanczos matrix function
    approximation to efficiently estimate trace(f(A)) for large symmetric
    operators. This is particularly useful for log-determinant estimation
    (f = log) and other matrix function traces in GPs and statistical models.

    The estimator is:
        trace(f(A)) ≈ (1/num_samples) * sum_i v_i^T f(A) v_i

    where each f(A) v_i is computed using Lanczos approximation.

    Parameters
    ----------
    A : LinearOperatorLike
        Symmetric linear operator whose trace we want to estimate.
    func : callable
        Matrix function. Should accept a matrix and return f(matrix).
        Common choice: jnp.log for log-determinant estimation.
    key : jax.Array
        JAX random key for generating test vectors.
    num_samples : int, optional
        Number of random test vectors (Hutchinson samples). Default is 100.
    num_iters : int, optional
        Number of Lanczos iterations per sample. Default is 20.
    distribution : str, optional
        Distribution for test vectors ('rademacher' or 'normal').
        Default is 'rademacher'.
    reortho : bool, optional
        Whether to use full reorthogonalization in Lanczos. Default is True.

    Returns
    -------
    trace_estimate : jax.Array
        Estimate of trace(f(A)).
    trace_std : jax.Array
        Standard error of the estimate.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> # Estimate log-determinant
    >>> A = Matrix(jnp.diag(jnp.arange(1.0, 101.0)))
    >>> key = jax.random.PRNGKey(0)
    >>> # True log-det = sum(log(1), log(2), ..., log(100))
    >>> trace_est, trace_std = stochastic_lanczos_quadrature(
    ...     A, jnp.log, key, num_samples=50, num_iters=10
    ... )
    >>> true_logdet = jnp.sum(jnp.log(jnp.arange(1.0, 101.0)))
    >>> print(f"Estimate: {trace_est:.2f} ± {trace_std:.2f}")
    >>> print(f"True value: {true_logdet:.2f}")

    Notes
    -----
    This is one of the most important algorithms for GP inference, as it
    allows computing log|K| where K is a large GP covariance matrix without
    forming K explicitly or computing its full eigendecomposition.

    The method combines:
    1. Hutchinson's stochastic trace estimation (unbiased)
    2. Lanczos approximation for f(A)v (deterministic approximation)

    Increasing num_samples reduces variance (stochastic error).
    Increasing num_iters reduces bias (Lanczos approximation error).

    References
    ----------
    Based on matfree.funm.integrand_funm_sym and matfree.stochtrace [1, 2].
    This method is central to the approach in Ubaru et al. (2017).
    """
    from jax import random

    # Get operator shape
    n = A.shape[0]

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

    # Compute v^T f(A) v for each sample using Lanczos
    def compute_quadratic_form(v):
        fAv = lanczos_matrix_function(A, v, func, num_iters, reortho=reortho)
        return jnp.dot(v, fAv)

    quadratic_forms = jax.vmap(compute_quadratic_form)(V)

    # Monte Carlo estimate
    trace_estimate = jnp.mean(quadratic_forms)
    trace_std = jnp.std(quadratic_forms, ddof=1) / jnp.sqrt(num_samples)

    return trace_estimate, trace_std


# --- Operator Wrappers ---


def MatrixFunction(A, func, **kwargs):
    """Create a lazy matrix function operator."""
    from linox.operators.functional import MatrixFunctionLinearOperator

    return MatrixFunctionLinearOperator(A, func, **kwargs)


def sqrt(A: LinearOperatorLike, **kwargs) -> LinearOperatorLike:
    """Lazy matrix square root."""
    return MatrixFunction(A, jnp.sqrt, **kwargs)


def log(A: LinearOperatorLike, **kwargs) -> LinearOperatorLike:
    """Lazy matrix logarithm."""
    return MatrixFunction(A, jnp.log, **kwargs)


def exp(A: LinearOperatorLike, **kwargs) -> LinearOperatorLike:
    """Lazy matrix exponential."""
    return MatrixFunction(A, jnp.exp, **kwargs)


def pow(A: LinearOperatorLike, p: float, **kwargs) -> LinearOperatorLike:
    """Lazy matrix power."""
    return MatrixFunction(A, lambda x: jnp.power(x, p), **kwargs)
