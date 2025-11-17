"""Matrix-free SVD via Lanczos bidiagonalization.

This module implements matrix-free singular value decomposition (SVD) using
Lanczos bidiagonalization for large-scale problems where only a few singular
values and vectors are needed.

The implementation is inspired by the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al.

Key algorithms:
- Lanczos bidiagonalization: Reduces matrix to bidiagonal form
- Partial SVD: Computes k largest singular values/vectors

References
----------
.. [1] N. Krämer, M. Schober, and P. Hennig, "Gradients of functions of large matrices,"
       arXiv preprint arXiv:2405.17277, 2024.
       https://arxiv.org/abs/2405.17277

.. [2] matfree: Matrix-free linear algebra in JAX
       https://github.com/pnkraemer/matfree

.. [3] G. H. Golub and C. F. Van Loan, "Matrix Computations," 4th ed., Johns Hopkins, 2013.
"""

import jax
import jax.numpy as jnp
from jax import lax

from linox.typing import ArrayLike, LinearOperatorLike


def lanczos_bidiag(
    A: LinearOperatorLike,
    u0: ArrayLike,
    num_iters: int,
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Lanczos bidiagonalization for SVD computation.

    Reduces a matrix A to bidiagonal form: A ≈ U B V^T, where U and V are
    orthonormal and B is bidiagonal.

    This is the foundation for computing partial SVD of large matrices where
    only a few singular values/vectors are needed.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator or matrix of shape (m, n). Should support both
        `A @ v` and `A.T @ u` operations.
    u0 : ArrayLike
        Initial vector of shape (m,) for the bidiagonalization process.
        Will be normalized internally.
    num_iters : int
        Number of bidiagonalization iterations (size of bidiagonal matrix).
        Should be much smaller than min(m, n).
    reortho : bool, optional
        Whether to use full reorthogonalization. This significantly improves
        numerical stability at the cost of O(num_iters^2) operations.
        Default is True (recommended).

    Returns
    -------
    U : jax.Array, shape (m, num_iters)
        Left orthonormal basis vectors (columns).
    V : jax.Array, shape (n, num_iters)
        Right orthonormal basis vectors (columns).
    alpha : jax.Array, shape (num_iters,)
        Diagonal elements of bidiagonal matrix B.
    beta : jax.Array, shape (num_iters-1,)
        Super-diagonal elements of bidiagonal matrix B.

    Notes
    -----
    The bidiagonal matrix B has the form:
        B = [[alpha[0], beta[0],    0,       ...],
             [0,        alpha[1], beta[1],   ...],
             [0,        0,        alpha[2],  ...],
             [...]]

    This is related to Golub-Kahan bidiagonalization and is used in
    algorithms like LSMR and partial SVD computation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> key = jax.random.PRNGKey(0)
    >>> A = Matrix(jax.random.normal(key, (100, 50)))
    >>> u0 = jnp.ones(100)
    >>> U, V, alpha, beta = lanczos_bidiag(A, u0, num_iters=10)
    >>> # U and V contain orthonormal vectors
    >>> # B = diag(alpha) + diag(beta, 1) is bidiagonal

    References
    ----------
    Inspired by matfree.decomp.bidiag [1, 2] and the Golub-Kahan process [3].
    """
    u0 = jnp.asarray(u0)
    m = u0.shape[0]
    n = A.shape[1]

    # Normalize initial vector
    beta_0 = jnp.linalg.norm(u0)
    u0 = u0 / beta_0

    # Pre-allocate arrays
    U = jnp.zeros((m, num_iters))
    V = jnp.zeros((n, num_iters))
    alpha = jnp.zeros(num_iters)
    beta = jnp.zeros(num_iters - 1) if num_iters > 1 else jnp.zeros(0)

    # Initialize first left vector
    U = U.at[:, 0].set(u0)

    def bidiag_step(k, carry):
        U_curr, V_curr, alpha_curr, beta_curr = carry

        # Get current left vector
        u = U_curr[:, k]

        # Compute right vector: v = A^T u
        v = A.T @ u

        # Orthogonalize against previous right vector if k > 0
        # In Golub-Kahan: v = A^T u - beta_{k-1} * v_{k-1}
        v_orth = lax.cond(
            k > 0,
            lambda: v - beta_curr[k - 1] * V_curr[:, k - 1],
            lambda: v,
        )

        # Full reorthogonalization against all previous V vectors
        if reortho:
            def reortho_v(v_in):
                # Compute V[:, :k].T @ v using a masked operation
                # Create a mask for the first k columns
                mask = jnp.arange(num_iters) < k
                V_masked = V_curr * mask[None, :]  # Broadcast mask over rows

                # Reorthogonalize twice for better numerical stability (matfree does this)
                coeffs = V_masked.T @ v_in
                v_out = v_in - V_masked @ coeffs
                coeffs = V_masked.T @ v_out
                v_out = v_out - V_masked @ coeffs
                return v_out

            v_orth = lax.cond(
                k > 0,
                reortho_v,
                lambda x: x,
                v_orth
            )

        # Compute alpha_k and normalize v
        alpha_k = jnp.linalg.norm(v_orth)
        alpha_curr = alpha_curr.at[k].set(alpha_k)
        v_norm = v_orth / (alpha_k + 1e-16)

        # Store v
        V_curr = V_curr.at[:, k].set(v_norm)

        # Compute next left vector: u = A v - alpha_k * u
        # Only do this if not last iteration
        def compute_next_u():
            u_new = A @ v_norm - alpha_k * u

            # Full reorthogonalization against all previous U vectors
            if reortho:
                # Create a mask for the first k+1 columns
                mask = jnp.arange(num_iters) < (k + 1)
                U_masked = U_curr * mask[None, :]  # Broadcast mask over rows

                # Reorthogonalize twice for better numerical stability
                coeffs = U_masked.T @ u_new
                u_new = u_new - U_masked @ coeffs
                coeffs = U_masked.T @ u_new
                u_new = u_new - U_masked @ coeffs

            beta_k = jnp.linalg.norm(u_new)
            u_norm = u_new / (beta_k + 1e-16)
            return u_norm, beta_k

        u_next, beta_k = lax.cond(
            k < num_iters - 1,
            compute_next_u,
            lambda: (jnp.zeros(m), 0.0),
        )

        # Store u_next and beta if not last iteration
        U_next = lax.cond(
            k < num_iters - 1,
            lambda: U_curr.at[:, k + 1].set(u_next),
            lambda: U_curr,
        )

        beta_next = lax.cond(
            k < num_iters - 1,
            lambda: beta_curr.at[k].set(beta_k),
            lambda: beta_curr,
        )

        return (U_next, V_curr, alpha_curr, beta_next)

    # Run bidiagonalization iterations
    U, V, alpha, beta = lax.fori_loop(0, num_iters, bidiag_step, (U, V, alpha, beta))

    return U, V, alpha, beta


def svd_partial(
    A: LinearOperatorLike,
    k: int,
    num_iters: int | None = None,
    u0: ArrayLike | None = None,
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute partial SVD using Lanczos bidiagonalization.

    Computes the k largest singular values and corresponding singular vectors
    of a large matrix without forming the full matrix explicitly.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator or matrix of shape (m, n).
    k : int
        Number of singular values/vectors to compute.
    num_iters : int, optional
        Number of Lanczos iterations. Should be larger than k for good
        approximation. If None, uses min(2*k, min(m, n)). Default is None.
    u0 : ArrayLike, optional
        Initial vector of shape (m,) for bidiagonalization. If None,
        uses vector of ones. Default is None.
    reortho : bool, optional
        Whether to use full reorthogonalization in bidiagonalization.
        This significantly improves numerical stability. Default is True.

    Returns
    -------
    U : jax.Array, shape (m, k)
        Left singular vectors (columns).
    S : jax.Array, shape (k,)
        Singular values in descending order.
    Vt : jax.Array, shape (k, n)
        Right singular vectors (rows).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> # Large matrix
    >>> key = jax.random.PRNGKey(0)
    >>> A_dense = jax.random.normal(key, (1000, 500))
    >>> A = Matrix(A_dense)
    >>> # Compute top 10 singular values/vectors
    >>> U, S, Vt = svd_partial(A, k=10)
    >>> print(f"Top 10 singular values: {S}")
    >>> # Verify: A ≈ U @ diag(S) @ Vt
    >>> A_approx = U @ jnp.diag(S) @ Vt
    >>> error = jnp.linalg.norm(A_dense - A_approx)

    Notes
    -----
    This is a matrix-free alternative to jnp.linalg.svd for computing
    a few singular values/vectors of large sparse or structured matrices.

    The algorithm:
    1. Performs Lanczos bidiagonalization: A ≈ U_bi B V_bi^T
    2. Computes SVD of small bidiagonal matrix B
    3. Projects back to get singular vectors of A

    References
    ----------
    Inspired by matfree library [1, 2].
    """
    m = A.shape[0]
    n = A.shape[1]

    if num_iters is None:
        num_iters = min(2 * k, min(m, n))

    if u0 is None:
        u0 = jnp.ones(m)

    # Perform bidiagonalization
    U_bi, V_bi, alpha, beta = lanczos_bidiag(A, u0, num_iters, reortho=reortho)

    # Construct bidiagonal matrix
    B = jnp.diag(alpha)
    if beta.size > 0:
        B = B + jnp.diag(beta, k=1)

    # Compute SVD of small bidiagonal matrix
    U_small, S_small, Vt_small = jnp.linalg.svd(B, full_matrices=False)

    # Select top k singular values/vectors
    U_small = U_small[:, :k]
    S = S_small[:k]
    Vt_small = Vt_small[:k, :]

    # Project back to original space
    U = U_bi @ U_small
    Vt = Vt_small @ V_bi.T

    return U, S, Vt
