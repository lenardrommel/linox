"""LSMR iterative solver for least-squares problems.

This module implements the LSMR (Least Squares Minimal Residual) algorithm for
solving large-scale least-squares and linear systems in a matrix-free manner.

The implementation closely follows the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al., which
itself follows the original LSMR algorithm by Fong and Saunders.

Key features:
- Matrix-free: Only requires matrix-vector products
- Handles over-determined, under-determined, and rank-deficient systems
- JAX-compatible with automatic differentiation support
- Iterative with configurable stopping criteria

References
----------
.. [1] D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for sparse
       least-squares problems," SIAM Journal on Scientific Computing, vol. 33,
       no. 5, pp. 2950-2971, 2011.

.. [2] matfree: Matrix-free linear algebra in JAX
       https://github.com/pnkraemer/matfree

.. [3] A. Roy, N. Krämer, V. De Bortoli, and A. Doucet, "Gradients of Stochastic
       Trace Estimators via Differentiable Matrix-Free Linear Solvers,"
       arXiv preprint, 2025.
"""

import jax
import jax.numpy as jnp
from jax import lax

from linox.typing import ArrayLike, LinearOperatorLike


_LARGE_VALUE = 1e10
"""A placeholder for np.inf for stopping criteria."""


def _sym_ortho(a, b):
    """Stable implementation of Givens rotation (like scipy/matfree)."""
    # Determine which branch to take
    idx = 3  # The "else" branch
    idx = jnp.where(jnp.abs(b) > jnp.abs(a), 2, idx)
    idx = jnp.where(a == 0, 1, idx)
    idx = jnp.where(b == 0, 0, idx)

    branches = [_sym_ortho_0, _sym_ortho_1, _sym_ortho_2, _sym_ortho_3]
    return lax.switch(idx, branches, a, b)


def _sym_ortho_0(a, _b):
    """Branch: b == 0."""
    zero = jnp.zeros((), dtype=a.dtype)
    return jnp.sign(a), zero, jnp.abs(a)


def _sym_ortho_1(_a, b):
    """Branch: a == 0."""
    zero = jnp.zeros((), dtype=b.dtype)
    return zero, jnp.sign(b), jnp.abs(b)


def _sym_ortho_2(a, b):
    """Branch: |b| > |a|."""
    tau = a / b
    s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
    c = s * tau
    r = b / s
    return c, s, r


def _sym_ortho_3(a, b):
    """Branch: else (|a| >= |b| and neither is zero)."""
    tau = b / a
    c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
    s = c * tau
    r = a / c
    return c, s, r


def lsmr_solve(
    A: LinearOperatorLike,
    b: ArrayLike,
    atol: float = 1e-6,
    btol: float = 1e-6,
    ctol: float = 1e-8,
    maxiter: int | None = None,
    damp: float = 0.0,
    x0: ArrayLike | None = None,
) -> tuple[jax.Array, dict]:
    """Solve least-squares problem min ||Ax - b||_2 using LSMR algorithm.

    LSMR is an iterative method for solving:
    - Linear systems Ax = b (when A is square and full rank)
    - Least-squares min ||Ax - b||_2 (when A is over-determined)
    - Minimum-norm solutions (when A is under-determined or rank-deficient)

    The algorithm only requires matrix-vector products with A and A^T, making
    it suitable for large sparse or structured matrices.

    This implementation follows matfree's approach, which matches SciPy's LSMR.

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator or matrix of shape (m, n). Should support both
        `A @ v` and `A.T @ v` operations.
    b : ArrayLike
        Right-hand side vector of shape (m,).
    atol : float, optional
        Absolute tolerance for convergence. Default is 1e-6.
    btol : float, optional
        Relative tolerance for convergence. Default is 1e-6.
    ctol : float, optional
        Condition number tolerance. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. If None, uses min(m, n).
        Default is None.
    damp : float, optional
        Damping parameter for regularization. Solves the problem
        min ||[A; damp*I] x - [b; 0]||_2 instead. Default is 0.0 (no damping).
    x0 : ArrayLike, optional
        Initial guess for the solution. If None, starts with zero vector.
        Default is None.

    Returns
    -------
    x : jax.Array, shape (n,)
        Solution vector.
    info : dict
        Dictionary containing:
        - 'istop': Stopping condition (0-9)
        - 'itn': Number of iterations performed
        - 'normr': Final residual norm ||r||
        - 'normar': Final ||A^T r||
        - 'normA': Estimate of ||A||
        - 'condA': Estimate of cond(A)
        - 'normx': Norm of solution ||x||

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from linox import Matrix
    >>> A = Matrix(jnp.eye(50))
    >>> b = jnp.ones(50)
    >>> x, info = lsmr_solve(A, b)
    >>> assert jnp.allclose(x, b)

    Notes
    -----
    This implementation closely follows matfree's LSMR, which is based on
    Fong and Saunders (2011) and matches SciPy's implementation.

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for
           sparse least-squares problems," SIAM J. Sci. Comput., 2011.
    .. [2] https://github.com/pnkraemer/matfree
    """
    b = jnp.asarray(b)
    m = b.shape[0]
    n = A.shape[1]

    if maxiter is None:
        maxiter = min(m, n)

    # Initialize x
    if x0 is None:
        x = jnp.zeros(n, dtype=b.dtype)
        Ax = jnp.zeros(m, dtype=b.dtype)
        u = b.copy()
        beta = jnp.linalg.norm(u)
    else:
        x = jnp.asarray(x0)
        Ax = A @ x
        u = b - Ax
        beta = jnp.linalg.norm(u)

    normb = jnp.linalg.norm(b)

    # Normalize u
    u = u / jnp.where(beta > 0, beta, 1.0)

    # Apply A^T to u
    v = A.T @ u
    alpha = jnp.linalg.norm(v)
    v = v / jnp.where(alpha > 0, alpha, 1.0)
    v = jnp.where(beta == 0, jnp.zeros_like(v), v)
    alpha = jnp.where(beta == 0, jnp.zeros_like(alpha), alpha)

    # Initialize variables for 1st iteration
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.copy()
    hbar = jnp.zeros_like(x)

    # Initialize variables for estimation of ||r||
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    # Initialize variables for estimation of ||A|| and cond(A)
    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e10
    normA = jnp.sqrt(normA2)
    condA = 1.0
    normx = 0.0

    # Items for use in stopping rules
    normr = beta
    normar = alpha * beta

    # Iteration loop using lax.while_loop
    def cond_fun(carry):
        itn, istop, *_ = carry
        return (istop == 0) & (itn < maxiter)

    def body_fun(carry):
        (
            itn,
            istop,
            u,
            v,
            alpha,
            beta,
            alphabar,
            rhobar,
            rho,
            zeta,
            sbar,
            cbar,
            zetabar,
            hbar,
            h,
            x,
            betadd,
            thetatilde,
            rhodold,
            betad,
            tautildeold,
            d,
            normA2,
            maxrbar,
            minrbar,
            normar,
            normr,
            normA,
            condA,
            normx,
        ) = carry

        # Perform the next step of the bidiagonalization
        Av = A @ v
        u = Av - alpha * u
        beta = jnp.linalg.norm(u)
        u = u / jnp.where(beta > 0, beta, 1.0)

        v = A.T @ u - beta * v
        alpha = jnp.linalg.norm(v)
        v = v / jnp.where(alpha > 0, alpha, 1.0)

        # Construct rotation Qhat_{k,2k+1}
        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, h_hat, x
        hbar = h - hbar * (thetabar * rho / (rhoold * rhobarold))
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - h * (thetanew / rho)

        # Estimate of ||r||
        # Apply rotation Qhat_{k,2k+1}
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = jnp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||
        normA2 = normA2 + beta * beta
        normA = jnp.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A)
        maxrbar = jnp.maximum(maxrbar, rhobarold)
        minrbar = jnp.where(itn > 1, jnp.minimum(minrbar, rhobarold), minrbar)
        condA = jnp.maximum(maxrbar, rhotemp) / jnp.minimum(minrbar, rhotemp)

        # Compute norms for convergence testing
        normar = jnp.abs(zetabar)
        normx = jnp.linalg.norm(x)

        # Check whether we should stop
        itn = itn + 1
        test1 = normr / normb
        z = normA * normr
        z_safe = jnp.where(z != 0, z, 1.0)
        test2 = jnp.where(z != 0, normar / z_safe, _LARGE_VALUE)
        test3 = 1.0 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # Determine stopping condition (following matfree/scipy order)
        istop = 0
        istop = jnp.where(normar == 0, 9, istop)
        istop = jnp.where(normb == 0, 8, istop)
        istop = jnp.where(itn >= maxiter, 7, istop)
        istop = jnp.where(1 + test3 <= 1, 6, istop)
        istop = jnp.where(1 + test2 <= 1, 5, istop)
        istop = jnp.where(1 + t1 <= 1, 4, istop)
        istop = jnp.where(test3 <= ctol, 3, istop)
        istop = jnp.where(test2 <= atol, 2, istop)
        istop = jnp.where(test1 <= rtol, 1, istop)

        return (
            itn,
            istop,
            u,
            v,
            alpha,
            beta,
            alphabar,
            rhobar,
            rho,
            zeta,
            sbar,
            cbar,
            zetabar,
            hbar,
            h,
            x,
            betadd,
            thetatilde,
            rhodold,
            betad,
            tautildeold,
            d,
            normA2,
            maxrbar,
            minrbar,
            normar,
            normr,
            normA,
            condA,
            normx,
        )

    init_carry = (
        0,  # itn
        0,  # istop
        u,
        v,
        alpha,
        beta,
        alphabar,
        rhobar,
        rho,
        zeta,
        sbar,
        cbar,
        zetabar,
        hbar,
        h,
        x,
        betadd,
        thetatilde,
        rhodold,
        betad,
        tautildeold,
        d,
        normA2,
        maxrbar,
        minrbar,
        normar,
        normr,
        normA,
        condA,
        normx,
    )

    final_carry = lax.while_loop(cond_fun, body_fun, init_carry)

    # Extract final values
    (
        itn,
        istop,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        x,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        normar,
        normr,
        normA,
        condA,
        normx,
    ) = final_carry

    info = {
        "istop": int(istop),
        "itn": int(itn),
        "normr": float(normr),
        "normar": float(normar),
        "normA": float(normA),
        "condA": float(condA),
        "normx": float(normx),
    }

    return x, info
