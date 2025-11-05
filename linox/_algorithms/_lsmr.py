"""LSMR iterative solver for least-squares problems.

This module implements the LSMR (Least Squares Minimal Residual) algorithm for
solving large-scale least-squares and linear systems in a matrix-free manner.

The implementation is inspired by the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Krämer et al., and follows
the original LSMR algorithm by Fong and Saunders.

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


def lsmr_solve(
    A: LinearOperatorLike,
    b: ArrayLike,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
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

    Parameters
    ----------
    A : LinearOperatorLike
        Linear operator or matrix of shape (m, n). Should support both
        `A @ v` and `A.T @ v` operations.
    b : ArrayLike
        Right-hand side vector of shape (m,).
    atol : float, optional
        Absolute tolerance for convergence:
        ||A^T r|| / ||A|| ||r|| ≤ atol, where r = b - Ax.
        Default is 1e-6.
    btol : float, optional
        Relative tolerance for convergence:
        ||r|| / ||b|| ≤ btol.
        Default is 1e-6.
    conlim : float, optional
        Condition number limit. If cond(A) exceeds this, the algorithm
        may terminate. Default is 1e8.
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
        - 'istop': Stopping condition (0=not converged, 1=atol, 2=btol,
          3=both, 4=ill-conditioned, 5=maxiter)
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
    >>> # Over-determined system (least squares)
    >>> A = Matrix(jnp.random.randn(100, 50))
    >>> x_true = jnp.random.randn(50)
    >>> b = A @ x_true + 0.01 * jnp.random.randn(100)
    >>> x, info = lsmr_solve(A, b, atol=1e-6, btol=1e-6)
    >>> print(f"Converged in {info['itn']} iterations")
    >>> print(f"Residual norm: {info['normr']:.6f}")

    Notes
    -----
    LSMR is based on the Golub-Kahan bidiagonalization process and is
    mathematically equivalent to MINRES applied to the normal equations
    A^T A x = A^T b, but is more numerically stable.

    For square, symmetric positive-definite systems, use CG instead.
    For general square systems, GMRES may be more appropriate.

    The algorithm is matrix-free and only requires A @ v and A^T @ v operations.

    References
    ----------
    Based on Fong and Saunders (2011) [1] and matfree.lstsq [2, 3].
    """
    b = jnp.asarray(b)
    m = b.shape[0]

    # Get matrix shape
    if hasattr(A, "shape"):
        n = A.shape[1]
    else:
        n = A.shape[1]

    if maxiter is None:
        maxiter = min(m, n)

    # Initialize x
    if x0 is None:
        x = jnp.zeros(n)
        u = b.copy()
        beta = jnp.linalg.norm(u)
    else:
        x = jnp.asarray(x0)
        u = b - A @ x
        beta = jnp.linalg.norm(u)

    # Handle zero RHS
    if beta == 0:
        return x, {
            "istop": 0,
            "itn": 0,
            "normr": 0.0,
            "normar": 0.0,
            "normA": 0.0,
            "condA": 0.0,
            "normx": jnp.linalg.norm(x),
        }

    u = u / beta

    # Apply A^T to u
    if hasattr(A, "T"):
        v = A.T @ u
    else:
        v = A.T @ u

    alpha = jnp.linalg.norm(v)

    if alpha == 0:
        return x, {
            "istop": 0,
            "itn": 0,
            "normr": beta,
            "normar": 0.0,
            "normA": 0.0,
            "condA": 0.0,
            "normx": jnp.linalg.norm(x),
        }

    v = v / alpha

    # Initialize variables
    alphabar = alpha
    zetabar = alpha * beta
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.copy()
    hbar = jnp.zeros(n)

    # Norms and stopping criteria
    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = jnp.sqrt(normA2)
    condA = 1.0
    normx = 0.0

    # Iteration state
    def lsmr_step(carry, _):
        (
            x,
            u,
            v,
            alpha,
            beta,
            alphabar,
            zetabar,
            rho,
            rhobar,
            cbar,
            sbar,
            h,
            hbar,
            normA2,
            maxrbar,
            minrbar,
            k,
        ) = carry

        # Bidiagonalization
        u = A @ v - alpha * u
        beta = jnp.linalg.norm(u)
        u = lax.cond(beta > 0, lambda: u / beta, lambda: u)

        normA2 = normA2 + alpha * alpha + beta * beta + damp * damp

        if hasattr(A, "T"):
            v = A.T @ u - beta * v
        else:
            v = A.T @ u - beta * v

        alpha = jnp.linalg.norm(v)
        v = lax.cond(alpha > 0, lambda: v / alpha, lambda: v)

        # Construct and apply rotation
        rhobar1 = jnp.sqrt(rhobar * rhobar + damp * damp)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * alphabar
        alphabar = cs1 * alphabar

        # Second rotation
        rho = jnp.sqrt(rhobar1 * rhobar1 + beta * beta)
        cs = rhobar1 / rho
        sn = beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * zetabar
        zetabar = sn * zetabar

        # Update h, hbar, and x
        hbar = h - (theta * rhobar / (rho * rho)) * hbar
        x = x + (phi / rho) * hbar
        h = v - (theta / rho) * h

        # Estimate norms
        normA = jnp.sqrt(normA2)
        condA = normA * jnp.sqrt(jnp.abs(maxrbar / minrbar))
        normr = jnp.abs(zetabar)
        normx = jnp.linalg.norm(x)

        # Update min/max rbar
        maxrbar = jnp.maximum(maxrbar, rhobar)
        minrbar = lax.cond(k == 0, lambda: rhobar, lambda: jnp.minimum(minrbar, rhobar))

        return (
            x,
            u,
            v,
            alpha,
            beta,
            alphabar,
            zetabar,
            rho,
            rhobar,
            cbar,
            sbar,
            h,
            hbar,
            normA2,
            maxrbar,
            minrbar,
            k + 1,
        ), None

    # Convergence condition
    def converged(carry):
        (
            _,
            _,
            _,
            _,
            _,
            _,
            zetabar,
            _,
            _,
            _,
            _,
            _,
            _,
            normA2,
            _,
            _,
            k,
        ) = carry
        normA = jnp.sqrt(normA2)
        normr = jnp.abs(zetabar)
        # Test convergence
        test1 = normr / (normA * jnp.linalg.norm(x) + jnp.linalg.norm(b))
        test2 = normr / jnp.linalg.norm(b) if jnp.linalg.norm(b) > 0 else normr

        converged = (test1 <= atol) | (test2 <= btol)
        continue_iter = (~converged) & (k < maxiter)

        return continue_iter

    # Run iterations
    init_carry = (
        x,
        u,
        v,
        alpha,
        beta,
        alphabar,
        zetabar,
        rho,
        rhobar,
        cbar,
        sbar,
        h,
        hbar,
        normA2,
        maxrbar,
        minrbar,
        0,
    )

    final_carry, _ = lax.while_loop(converged, lambda c: lsmr_step(c, None)[0], init_carry)

    (
        x,
        _,
        _,
        _,
        _,
        _,
        zetabar,
        _,
        _,
        _,
        _,
        _,
        _,
        normA2,
        maxrbar,
        minrbar,
        itn,
    ) = final_carry

    # Determine stopping condition
    normA = jnp.sqrt(normA2)
    condA = normA * jnp.sqrt(jnp.abs(maxrbar / minrbar))
    normr = jnp.abs(zetabar)
    normar = jnp.abs(zetabar) * normA
    normx = jnp.linalg.norm(x)

    test1 = normr / (normA * normx + jnp.linalg.norm(b))
    test2 = normr / jnp.linalg.norm(b) if jnp.linalg.norm(b) > 0 else normr

    if itn >= maxiter:
        istop = 5
    elif test1 <= atol and test2 <= btol:
        istop = 3
    elif test1 <= atol:
        istop = 1
    elif test2 <= btol:
        istop = 2
    elif condA >= conlim:
        istop = 4
    else:
        istop = 0

    info = {
        "istop": istop,
        "itn": itn,
        "normr": float(normr),
        "normar": float(normar),
        "normA": float(normA),
        "condA": float(condA),
        "normx": float(normx),
    }

    return x, info
