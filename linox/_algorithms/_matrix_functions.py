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
from jax import lax

from linox._algorithms._lanczos_arnoldi import arnoldi_iteration, lanczos_tridiag
from linox.typing import ArrayLike, LinearOperatorLike


def lanczos_matrix_function(
    A: LinearOperatorLike,
    v: ArrayLike,
    func: callable,
    num_iters: int,
    reortho: bool = True,
) -> jax.Array:
    """Approximate f(A)v using Lanczos tridiagonalization.

    Computes an approximation to f(A)v for a symmetric linear operator A
    and vector v, where f is a scalar function. This avoids computing the
    full matrix function f(A).

    The method uses Lanczos tridiagonalization to project A onto a small
    Krylov subspace, applies f to the small tridiagonal matrix, and projects
    back to the original space.

    Parameters
    ----------
    A : LinearOperatorLike
        Symmetric linear operator or matrix.
    v : ArrayLike
        Vector to multiply by f(A).
    func : callable
        Matrix function to apply. Should accept a matrix and return f(matrix).
        Common choices: jnp.exp, jnp.log, jnp.sqrt, or custom functions.
    num_iters : int
        Number of Lanczos iterations. Larger values give better approximations
        but require more matrix-vector products. Typically 10-50 is sufficient.
    reortho : bool, optional
        Whether to use full reorthogonalization. Default is True.

    Returns
    -------
    result : jax.Array
        Approximation to f(A) @ v.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> # Matrix exponential
    >>> A = Matrix(-jnp.eye(100))  # -I (eigenvalues all -1)
    >>> v = jnp.ones(100)
    >>> # exp(-I)v ≈ exp(-1) * v ≈ 0.368 * v
    >>> result = lanczos_matrix_function(A, v, jnp.exp, num_iters=10)
    >>> print(f"Approximation: {result[0]:.4f}")  # Should be ~0.368

    Notes
    -----
    For symmetric operators, this is the most efficient method for computing
    f(A)v. The approximation improves as num_iters increases.

    Common matrix functions:
    - Matrix exponential: jnp.exp (for ODEs, diffusion)
    - Matrix logarithm: jnp.log (for divergences, log-det)
    - Matrix square root: jnp.sqrt (for preconditioning)
    - Inverse square root: lambda M: jnp.linalg.inv(sqrtm(M))
    - Power: lambda M: jnp.linalg.matrix_power(M, p)

    References
    ----------
    Based on matfree.funm.funm_lanczos_sym [1, 2] and Saad (1992) [3].
    """
    v = jnp.asarray(v)
    v_norm = jnp.linalg.norm(v)
    v_normalized = v / v_norm

    # Perform Lanczos tridiagonalization
    Q, alpha, beta = lanczos_tridiag(A, v_normalized, num_iters, reortho=reortho)

    # Construct tridiagonal matrix
    T = jnp.diag(alpha)
    if beta.size > 0:
        T = T + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)

    # Apply function to tridiagonal matrix using eigendecomposition
    # (converts element-wise function to matrix function)
    eigvals, eigvecs = jnp.linalg.eigh(T)
    fT = eigvecs @ jnp.diag(func(eigvals)) @ eigvecs.T

    # Extract first row (corresponds to initial vector contribution)
    e1 = jnp.zeros(num_iters)
    e1 = e1.at[0].set(1.0)

    # Project back: f(A)v ≈ ||v|| * Q * f(T) * e1
    result = v_norm * (Q @ (fT @ e1))

    return result


def arnoldi_matrix_function(
    A: LinearOperatorLike,
    v: ArrayLike,
    func: callable,
    num_iters: int,
) -> jax.Array:
    """Approximate f(A)v using Arnoldi iteration.

    Computes an approximation to f(A)v for a general (possibly non-symmetric)
    linear operator A and vector v. This is the non-symmetric generalization
    of lanczos_matrix_function.

    Parameters
    ----------
    A : LinearOperatorLike
        General linear operator or matrix.
    v : ArrayLike
        Vector to multiply by f(A).
    func : callable
        Matrix function to apply. Should accept a matrix and return f(matrix).
    num_iters : int
        Number of Arnoldi iterations.

    Returns
    -------
    result : jax.Array
        Approximation to f(A) @ v.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> # Non-symmetric matrix
    >>> A = Matrix(jnp.array([[1., 2.], [0., 1.]]))
    >>> v = jnp.array([1., 1.])
    >>> result = arnoldi_matrix_function(A, v, jnp.exp, num_iters=2)

    Notes
    -----
    For symmetric operators, lanczos_matrix_function is more efficient.
    Use this function when the operator is known to be non-symmetric.

    References
    ----------
    Based on matfree.funm.funm_arnoldi [1, 2].
    """
    v = jnp.asarray(v)
    v_norm = jnp.linalg.norm(v)
    v_normalized = v / v_norm

    # Perform Arnoldi iteration
    Q, H = arnoldi_iteration(A, v_normalized, num_iters)

    # Apply function to Hessenberg matrix (use only upper part)
    H_upper = H[:num_iters, :]
    # Use Schur decomposition for general matrices
    eigvals, eigvecs = jnp.linalg.eig(H_upper)
    fH = eigvecs @ jnp.diag(func(eigvals)) @ jnp.linalg.inv(eigvecs)

    # Extract first row
    e1 = jnp.zeros(num_iters)
    e1 = e1.at[0].set(1.0)

    # Project back
    result = v_norm * (Q @ (fH @ e1))

    return result


def chebyshev_matrix_function(
    A: LinearOperatorLike,
    v: ArrayLike,
    func: callable,
    num_terms: int,
    bounds: tuple[float, float] | None = None,
) -> jax.Array:
    """Approximate f(A)v using Chebyshev polynomial expansion.

    Approximates f(A)v by expanding f in Chebyshev polynomials and applying
    the polynomial to A. This is particularly efficient for matrix functions
    that are smooth on the spectrum of A.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator or matrix. The spectrum should be contained in the
        interval [a, b] specified by bounds.
    v : ArrayLike
        Vector to multiply by f(A).
    func : callable
        Scalar function to apply. Should map scalars to scalars.
    num_terms : int
        Number of Chebyshev terms to use in the expansion.
    bounds : tuple[float, float], optional
        Interval [a, b] containing the spectrum of A. If None, assumes [-1, 1].
        Default is None.

    Returns
    -------
    result : jax.Array
        Approximation to f(A) @ v.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> # Matrix with spectrum in [-1, 1]
    >>> A = Matrix(0.9 * jnp.eye(100))
    >>> v = jnp.ones(100)
    >>> # Approximate exp(A)v
    >>> result = chebyshev_matrix_function(
    ...     A, v, jnp.exp, num_terms=20, bounds=(-1.0, 1.0)
    ... )

    Notes
    -----
    The Chebyshev expansion is most accurate when:
    1. The function f is smooth on the spectrum of A
    2. The bounds tightly contain the spectrum of A
    3. Sufficient terms are used (typically 10-50)

    This method may be more efficient than Lanczos/Arnoldi when the same
    function needs to be applied to multiple vectors, as the Chebyshev
    coefficients can be precomputed.

    References
    ----------
    Based on matfree.funm.funm_chebyshev [2] and Higham (2008) [4].
    """
    v = jnp.asarray(v)

    if bounds is None:
        a, b = -1.0, 1.0
    else:
        a, b = bounds

    # Compute Chebyshev nodes in [a, b]
    nodes = jnp.array(
        [jnp.cos(jnp.pi * (2 * k + 1) / (2 * num_terms)) for k in range(num_terms)]
    )
    # Map from [-1, 1] to [a, b]
    nodes_scaled = 0.5 * ((b - a) * nodes + (b + a))

    # Evaluate function at Chebyshev nodes
    func_vals = jax.vmap(func)(nodes_scaled)

    # Compute Chebyshev coefficients
    def compute_coeff(j):
        return (2.0 / num_terms) * jnp.sum(
            func_vals
            * jnp.array([jnp.cos(jnp.pi * j * (2 * k + 1) / (2 * num_terms)) for k in range(num_terms)])
        )

    coeffs = jax.vmap(compute_coeff)(jnp.arange(num_terms))

    # Scale operator to [-1, 1]
    # A_scaled = 2(A - (a+b)/2 I) / (b-a)
    def matvec_scaled(v):
        c = (a + b) / 2
        s = (b - a) / 2
        return (A @ v - c * v) * (2.0 / s)

    # Evaluate Chebyshev polynomial at A using Clenshaw recursion
    # T_0(A)v = v, T_1(A)v = A_scaled @ v
    # T_{n+1}(A)v = 2 A_scaled @ T_n(A)v - T_{n-1}(A)v

    def clenshaw_step(carry, coeff_k):
        b_k, b_km1 = carry
        b_kp1 = coeff_k * v + 2 * matvec_scaled(b_k) - b_km1
        return (b_kp1, b_k), None

    # Initialize
    b_0 = jnp.zeros_like(v)
    b_1 = jnp.zeros_like(v)

    # Run Clenshaw recursion (backwards from highest degree)
    (b_final, _), _ = lax.scan(clenshaw_step, (b_0, b_1), coeffs[::-1])

    # Final step: result = 0.5 * c_0 * v + A_scaled @ b_final - b_prev
    # For simplicity, we reconstruct using standard Chebyshev evaluation
    T_prev = jnp.zeros_like(v)
    T_curr = v.copy()
    result = coeffs[0] * T_curr

    for k in range(1, num_terms):
        T_next = 2 * matvec_scaled(T_curr) - T_prev
        result = result + coeffs[k] * T_next
        T_prev = T_curr
        T_curr = T_next

    return result


def stochastic_lanczos_quadrature(
    A: LinearOperatorLike,
    func: callable,
    key: jax.Array,
    num_samples: int = 100,
    num_iters: int = 20,
    distribution: str = "rademacher",
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Estimate trace(f(A)) using stochastic Lanczos quadrature (SLQ).

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
    if hasattr(A, "shape"):
        n = A.shape[0]
    else:
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
