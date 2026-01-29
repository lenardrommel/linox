# _lanzcos.py

import jax
import jax.numpy as jnp

from linox.typing import ArrayLike, LinearOperatorLike


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
