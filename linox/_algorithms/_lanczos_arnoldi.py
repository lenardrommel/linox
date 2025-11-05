"""Lanczos and Arnoldi iterations for eigenvalue and matrix decompositions.

This module implements Lanczos tridiagonalization for symmetric matrices and
Arnoldi iteration for general matrices. These are core Krylov subspace methods
for large-scale eigenvalue problems and matrix function approximations.

The implementations are inspired by and follow similar patterns to the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al., with adaptations
for linox's LinearOperator abstraction.

Key algorithms:
- Lanczos tridiagonalization: Reduces symmetric operators to tridiagonal form
- Arnoldi iteration: Reduces general operators to Hessenberg form

References
----------
.. [1] N. Krämer, M. Schober, and P. Hennig, "Gradients of functions of large matrices,"
       arXiv preprint arXiv:2405.17277, 2024.
       https://arxiv.org/abs/2405.17277

.. [2] matfree: Matrix-free linear algebra in JAX
       https://github.com/pnkraemer/matfree

.. [3] Y. Saad, "Iterative Methods for Sparse Linear Systems," 2nd ed., SIAM, 2003.
"""

import jax
import jax.numpy as jnp
from jax import lax

from linox.typing import ArrayLike, LinearOperatorLike


def lanczos_tridiag(
    A: LinearOperatorLike,
    v0: ArrayLike,
    num_iters: int,
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Lanczos tridiagonalization for symmetric operators.

    Computes a tridiagonal reduction of a symmetric linear operator using the
    Lanczos algorithm. Returns the orthonormal Lanczos vectors Q and the
    tridiagonal matrix T such that A ≈ Q T Q^T on the Krylov subspace.

    This is the foundation for efficient eigenvalue computation and matrix
    function approximation for symmetric operators.

    Parameters
    ----------
    A : LinearOperatorLike
        Symmetric linear operator or matrix. Should support matrix-vector
        multiplication via `A @ v`.
    v0 : ArrayLike
        Initial vector for the Krylov process. Will be normalized internally.
    num_iters : int
        Number of Lanczos iterations (size of Krylov subspace).
    reortho : bool, optional
        Whether to perform full reorthogonalization to maintain numerical
        stability. Recommended for num_iters > 50. Default is True.

    Returns
    -------
    Q : jax.Array, shape (n, num_iters)
        Orthonormal Lanczos vectors (columns).
    alpha : jax.Array, shape (num_iters,)
        Diagonal elements of the tridiagonal matrix T.
    beta : jax.Array, shape (num_iters-1,)
        Off-diagonal elements of the tridiagonal matrix T.

    Notes
    -----
    The tridiagonal matrix T has the form:
        T = [[alpha[0], beta[0],    0,       ...],
             [beta[0],  alpha[1], beta[1],   ...],
             [0,        beta[1],  alpha[2],  ...],
             [...]]

    When reortho=False, uses the classical three-term Lanczos recurrence which
    may lose orthogonality for large num_iters. When reortho=True, performs
    full Gram-Schmidt reorthogonalization at each step.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.array([[4., 1.], [1., 3.]]))
    >>> v0 = jnp.array([1., 0.])
    >>> Q, alpha, beta = lanczos_tridiag(A, v0, num_iters=2)
    >>> # Q contains orthonormal Lanczos vectors
    >>> # T = diag(alpha) + diag(beta, 1) + diag(beta, -1)

    References
    ----------
    Inspired by matfree.decomp.tridiag_sym [1, 2].
    """
    v0 = jnp.asarray(v0)
    n = v0.size

    # Normalize initial vector
    beta_0 = jnp.linalg.norm(v0)
    v0 = v0 / beta_0

    # Pre-allocate arrays
    Q = jnp.zeros((n, num_iters))
    alpha = jnp.zeros(num_iters)
    beta = jnp.zeros(num_iters - 1) if num_iters > 1 else jnp.zeros(0)

    # Initialize first vector
    Q = Q.at[:, 0].set(v0)

    def lanczos_step(carry, _):
        Q_curr, alpha_curr, beta_curr, k = carry

        # Matrix-vector product
        v = Q_curr[:, k]
        w = A @ v

        # Compute diagonal element
        alpha_k = jnp.dot(w, v)
        alpha_curr = alpha_curr.at[k].set(alpha_k)

        # Update w
        w = w - alpha_k * v
        if k > 0:
            w = w - beta_curr[k - 1] * Q_curr[:, k - 1]

        # Reorthogonalization (full Gram-Schmidt)
        if reortho and k > 0:
            for j in range(k + 1):
                w = w - jnp.dot(w, Q_curr[:, j]) * Q_curr[:, j]

        # Compute off-diagonal element
        beta_k = jnp.linalg.norm(w)

        # Store next vector (if not last iteration)
        Q_next = lax.cond(
            k < num_iters - 1,
            lambda: Q_curr.at[:, k + 1].set(w / (beta_k + 1e-16)),
            lambda: Q_curr,
        )

        # Store beta (if not last iteration)
        beta_next = lax.cond(
            k < num_iters - 1,
            lambda: beta_curr.at[k].set(beta_k),
            lambda: beta_curr,
        )

        return (Q_next, alpha_curr, beta_next, k + 1), None

    # Run Lanczos iterations
    (Q, alpha, beta, _), _ = lax.scan(
        lanczos_step, (Q, alpha, beta, 0), None, length=num_iters
    )

    return Q, alpha, beta


def arnoldi_iteration(
    A: LinearOperatorLike,
    v0: ArrayLike,
    num_iters: int,
) -> tuple[jax.Array, jax.Array]:
    """Arnoldi iteration for general (non-symmetric) operators.

    Computes a Hessenberg reduction of a general linear operator using the
    Arnoldi iteration. Returns the orthonormal Arnoldi vectors Q and the
    upper Hessenberg matrix H such that A ≈ Q H Q^T on the Krylov subspace.

    This is the generalization of Lanczos for non-symmetric operators and is
    fundamental for GMRES and eigenvalue computation of general matrices.

    Parameters
    ----------
    A : LinearOperatorLike
        General linear operator or matrix. Should support matrix-vector
        multiplication via `A @ v`.
    v0 : ArrayLike
        Initial vector for the Krylov process. Will be normalized internally.
    num_iters : int
        Number of Arnoldi iterations (size of Krylov subspace).

    Returns
    -------
    Q : jax.Array, shape (n, num_iters)
        Orthonormal Arnoldi vectors (columns).
    H : jax.Array, shape (num_iters+1, num_iters)
        Upper Hessenberg matrix. The last row contains the residual norms.

    Notes
    -----
    The Hessenberg matrix H is upper Hessenberg (zeros below the first subdiagonal).
    The algorithm performs full reorthogonalization at each step to maintain
    numerical stability.

    For symmetric operators, use lanczos_tridiag instead as it is more efficient.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.array([[1., 2.], [3., 4.]]))
    >>> v0 = jnp.array([1., 0.])
    >>> Q, H = arnoldi_iteration(A, v0, num_iters=2)
    >>> # Q contains orthonormal Arnoldi vectors
    >>> # H is upper Hessenberg

    References
    ----------
    Inspired by matfree.decomp.hessenberg [1, 2].
    """
    v0 = jnp.asarray(v0)
    n = v0.size

    # Normalize initial vector
    beta_0 = jnp.linalg.norm(v0)
    v0 = v0 / beta_0

    # Pre-allocate arrays
    Q = jnp.zeros((n, num_iters))
    H = jnp.zeros((num_iters + 1, num_iters))

    # Initialize first vector
    Q = Q.at[:, 0].set(v0)

    def arnoldi_step(carry, _):
        Q_curr, H_curr, k = carry

        # Matrix-vector product
        v = Q_curr[:, k]
        w = A @ v

        # Full Gram-Schmidt orthogonalization
        for j in range(k + 1):
            h_jk = jnp.dot(w, Q_curr[:, j])
            H_curr = H_curr.at[j, k].set(h_jk)
            w = w - h_jk * Q_curr[:, j]

        # Compute residual norm
        h_next = jnp.linalg.norm(w)
        H_curr = H_curr.at[k + 1, k].set(h_next)

        # Store next vector (if not last iteration)
        Q_next = lax.cond(
            k < num_iters - 1,
            lambda: Q_curr.at[:, k + 1].set(w / (h_next + 1e-16)),
            lambda: Q_curr,
        )

        return (Q_next, H_curr, k + 1), None

    # Run Arnoldi iterations
    (Q, H, _), _ = lax.scan(arnoldi_step, (Q, H, 0), None, length=num_iters)

    return Q, H


def lanczos_eigh(
    A: LinearOperatorLike,
    v0: ArrayLike,
    num_iters: int,
    k: int | None = None,
    which: str = "LM",
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Compute a few eigenvalues/eigenvectors using Lanczos method.

    Uses Lanczos tridiagonalization followed by dense eigenvalue decomposition
    of the small tridiagonal matrix to compute a subset of eigenvalues and
    eigenvectors of a large symmetric operator.

    This is much more efficient than full eigenvalue decomposition when
    k << n (only a few eigenvalues needed).

    Parameters
    ----------
    A : LinearOperatorLike
        Symmetric linear operator or matrix.
    v0 : ArrayLike
        Initial vector for the Krylov process. Should be chosen to have
        components in the desired eigenspace.
    num_iters : int
        Number of Lanczos iterations. Should be >> k for good approximations.
    k : int, optional
        Number of eigenvalues/eigenvectors to return. If None, returns all
        num_iters eigenvalues. Default is None.
    which : str, optional
        Which eigenvalues to return:
        - 'LM': Largest in magnitude
        - 'LA': Largest algebraic (most positive)
        - 'SA': Smallest algebraic (most negative)
        Default is 'LM'.
    reortho : bool, optional
        Whether to use full reorthogonalization in Lanczos. Default is True.

    Returns
    -------
    eigenvalues : jax.Array, shape (k,)
        Approximate eigenvalues.
    eigenvectors : jax.Array, shape (n, k)
        Approximate eigenvectors (columns).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.eye(100) + 0.1 * jnp.random.randn(100, 100))
    >>> A = Matrix((A.todense() + A.todense().T) / 2)  # Make symmetric
    >>> v0 = jnp.ones(100)
    >>> # Compute top 5 eigenvalues
    >>> eigs, vecs = lanczos_eigh(A, v0, num_iters=20, k=5, which='LA')

    Notes
    -----
    This is the matrix-free alternative to jnp.linalg.eigh for large sparse
    symmetric operators where only a few eigenvalues are needed.
    """
    if k is None:
        k = num_iters

    # Perform Lanczos tridiagonalization
    Q, alpha, beta = lanczos_tridiag(A, v0, num_iters, reortho=reortho)

    # Construct tridiagonal matrix
    T = jnp.diag(alpha)
    if beta.size > 0:
        T = T + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)

    # Dense eigenvalue decomposition of small tridiagonal matrix
    eig_vals, eig_vecs = jnp.linalg.eigh(T)

    # Select k eigenvalues based on 'which'
    if which == "LM":
        idx = jnp.argsort(jnp.abs(eig_vals))[::-1][:k]
    elif which == "LA":
        idx = jnp.argsort(eig_vals)[::-1][:k]
    elif which == "SA":
        idx = jnp.argsort(eig_vals)[:k]
    else:
        msg = f"Invalid 'which' parameter: {which}. Must be 'LM', 'LA', or 'SA'."
        raise ValueError(msg)

    # Project eigenvectors back to original space
    eigenvalues = eig_vals[idx]
    eigenvectors = Q @ eig_vecs[:, idx]

    return eigenvalues, eigenvectors
