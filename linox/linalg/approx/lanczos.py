# _lanzcos.py

import jax
import jax.numpy as jnp
from jax import lax

from linox._types import ArrayLike, LinearOperatorLike


def lanczos_solve_sqrt(
    A: LinearOperatorLike,
    b: ArrayLike,
    tol=1e-5,
    min_eta=1e-14,
    max_iter=10,
    overwrite_b=False,
) -> jax.Array:
    """Build a low-rank inverse factor for a PSD operator using CG/Lanczos.

    Returns a skinny matrix D whose columns are A-conjugate directions
    (normalized by sqrt of the Rayleigh quotient), such that
        D @ D.T ≈ A^{-1}
    on the generated Krylov subspace. This acts like an "inverse sqrt"
    factor usable in Kronecker products, preconditioners, or low-rank
    approximations. Note this is not the symmetric A^{-1/2}; it is a
    factor whose Gram approximates A^{-1}.

    Parameters
    ----------
    A : array-like or linear operator supporting `A @ x`
        Positive semi-definite operator.
    b : array
        Start vector for the Krylov process (will be normalized).
    tol : float
        Relative tolerance for residual norm stopping.
    min_eta : float
        Minimum step Rayleigh quotient to continue (guard against breakdown).
    max_iter : int
        Maximum number of Lanczos/CG iterations (columns in the factor).
    overwrite_b : bool
        If True, may reuse the buffer of `b` for the search direction.
    """

    @jax.jit
    def _step(values):
        ds, rs, rs_norm_sq, p, eta, k = values
        # Compute search direction
        true_fn = lambda _p: rs[:, k] + rs_norm_sq[k] / rs_norm_sq[k - 1] * _p  # noqa: E731
        false_fn = lambda _p: _p  # noqa: E731
        p = jax.lax.cond(k > 0, true_fn, false_fn, p)

        # Compute modified Lanzcos vector
        w = A @ p
        eta = p @ w
        ds = ds.at[:, k].set(p / jnp.sqrt(eta))

        # Update residual
        mu = rs_norm_sq[k] / eta
        rs_prev_k = rs  # rs[:, :k]
        rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)

        # Full reorthogonalization of residual (double Gram-Schmidt)
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )

        rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1].T @ rs[:, k + 1])

        return ds, rs, rs_norm_sq, p, eta, k + 1

    def _cond_fun(values):
        _ds, _, rs_norm_sq, _, _eta, k = values
        return (rs_norm_sq[k] > sqtol) & (k < max_iter)

    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter))
    rs = jnp.zeros((b.size, max_iter + 1))
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)

    # Initialize loop variables
    sqtol = tol**2
    min_eta = min_eta
    eta = jnp.inf
    rs = rs.at[:, 0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k = jax.lax.while_loop(
        _cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0)
    )

    return ds[:, :k]


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

    Parameters
    ----------
    A : LinearOperatorLike
        Symmetric linear operator or matrix.
    v0 : ArrayLike
        Initial vector for the Krylov process.
    num_iters : int
        Number of Lanczos iterations.
    reortho : bool, optional
        Whether to perform full reorthogonalization. Default is True.

    Returns
    -------
    Q : jax.Array, shape (n, num_iters)
        Orthonormal Lanczos vectors.
    alpha : jax.Array, shape (num_iters,)
        Diagonal elements.
    beta : jax.Array, shape (num_iters-1,)
        Off-diagonal elements.
    """
    v0 = jnp.asarray(v0)
    n = v0.size

    # Normalize initial vector
    beta_0 = jnp.linalg.norm(v0)
    v0 /= beta_0

    # Pre-allocate arrays
    Q = jnp.zeros((n, num_iters))
    alpha = jnp.zeros(num_iters)
    beta = jnp.zeros(num_iters - 1) if num_iters > 1 else jnp.zeros(0)

    # Initialize first vector
    Q = Q.at[:, 0].set(v0)

    def lanczos_step(k, carry):
        Q_curr, alpha_curr, beta_curr = carry

        # Matrix-vector product
        v = Q_curr[:, k]
        w = A @ v

        # Compute diagonal element
        alpha_k = jnp.dot(w, v)
        alpha_curr = alpha_curr.at[k].set(alpha_k)

        # Update w (three-term recurrence)
        w -= alpha_k * v
        # Subtract previous vector (if k > 0)
        prev_contrib = lax.cond(
            k > 0,
            lambda: beta_curr[k - 1] * Q_curr[:, k - 1],
            lambda: jnp.zeros_like(w),
        )
        w -= prev_contrib

        # Reorthogonalization (full Gram-Schmidt)
        if reortho:
            def reorth_body(j, w_state):
                proj = jnp.dot(w_state, Q_curr[:, j])
                return w_state - proj * Q_curr[:, j]

            w = lax.cond(
                k > 0,
                lambda w_val: lax.fori_loop(0, k, reorth_body, w_val),
                lambda w_val: w_val,
                w,
            )

        # Compute off-diagonal element
        beta_k = jnp.linalg.norm(w)

        # Store next vector
        Q_next = lax.cond(
            k < num_iters - 1,
            lambda: Q_curr.at[:, k + 1].set(w / (beta_k + 1e-16)),
            lambda: Q_curr,
        )

        # Store beta
        beta_next = lax.cond(
            k < num_iters - 1,
            lambda: beta_curr.at[k].set(beta_k),
            lambda: beta_curr,
        )

        return (Q_next, alpha_curr, beta_next)

    # Run Lanczos iterations
    Q, alpha, beta = lax.fori_loop(0, num_iters, lanczos_step, (Q, alpha, beta))

    return Q, alpha, beta


def lanczos_matrix_function(
    A: LinearOperatorLike,
    v: ArrayLike,
    func: callable,
    num_iters: int,
    reortho: bool = True,
) -> jax.Array:
    """Approximate f(A)v using Lanczos tridiagonalization."""
    v = jnp.asarray(v)
    v_norm = jnp.linalg.norm(v)
    v_normalized = v / v_norm

    # Perform Lanczos tridiagonalization
    Q, alpha, beta = lanczos_tridiag(A, v_normalized, num_iters, reortho=reortho)

    # Construct tridiagonal matrix
    T = jnp.diag(alpha)
    if beta.size > 0:
        T = T + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    # Only the first column of f(T) is ever used, and it is the Gauss
    # quadrature sum  sum_i (e1^T u_i) f(theta_i) u_i.  Forming f(T) in full
    # and then multiplying by e1 is both wasteful and numerically fragile.
    #
    # Fragile because Lanczos *breaks down* once the Krylov space is
    # exhausted: for an operator with a degenerate spectrum (Identity being
    # the extreme case) beta hits zero after one step, and the remaining rows
    # of T are numerical noise whose eigenvalues sit at or below zero. Those
    # spurious modes carry essentially no quadrature weight, but evaluating
    # `func` on them produces +/-inf for functions like log, and the
    # subsequent `0 * inf` turns the entire result into NaN.
    #
    # So: drop the modes that carry no weight, and never evaluate `func` on
    # their eigenvalues at all -- substituting a safe value inside the
    # `where` rather than masking afterwards, since `jnp.where` still
    # evaluates both branches.
    weights = eigvecs[0, :]
    eps = jnp.finfo(eigvals.dtype).eps
    significant = jnp.abs(weights) > eps * jnp.maximum(jnp.max(jnp.abs(weights)), 1.0)

    safe_eigvals = jnp.where(significant, eigvals, jnp.ones_like(eigvals))
    contributions = jnp.where(significant, weights * func(safe_eigvals), 0.0)

    # Project back: v_norm * Q @ (f(T) e1)
    return v_norm * (Q @ (eigvecs @ contributions))


def lanczos_eigh(
    A: LinearOperatorLike,
    v0: ArrayLike,
    num_iters: int,
    k: int | None = None,
    which: str = "LM",
    reortho: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Compute a few eigenvalues/eigenvectors using Lanczos method."""
    if k is None:
        k = num_iters

    # Perform Lanczos tridiagonalization
    Q, alpha, beta = lanczos_tridiag(A, v0, num_iters, reortho=reortho)

    # Construct tridiagonal matrix
    T = jnp.diag(alpha)
    if beta.size > 0:
        T = T + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)

    # Dense eigenvalue decomposition
    eig_vals, eig_vecs = jnp.linalg.eigh(T)

    # Select k eigenvalues based on 'which'
    if which == "LM":
        idx = jnp.argsort(jnp.abs(eig_vals))[::-1][:k]
    elif which == "LA":
        idx = jnp.argsort(eig_vals)[::-1][:k]
    elif which == "SA":
        idx = jnp.argsort(eig_vals)[:k]
    else:
        msg = f"Invalid 'which' parameter: {which}"
        raise ValueError(msg)

    # Project eigenvectors back
    eigenvalues = eig_vals[idx]
    eigenvectors = Q @ eig_vecs[:, idx]

    return eigenvalues, eigenvectors
