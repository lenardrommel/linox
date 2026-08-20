"""Preconditioned conjugate gradients for symmetric positive-definite systems.

`jax.scipy.sparse.linalg.cg` reports no convergence information, which left
:func:`linox.solve` unable to say whether a CG solve had actually succeeded --
it had to fall back to a loose residual guard. This implementation reports a
termination code in the same style as :mod:`linox.linalg.approx.lsmr`, and
accepts a preconditioner.

Only matrix-vector products are used, so a matrix-free operator stays
matrix-free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from linox.utils.array import LinearOperatorLike, as_linop

__all__ = ["CG_CONVERGED", "CG_NOT_CONVERGED", "cg_solve"]

#: Termination codes, mirroring the `istop` convention used by `lsmr_solve`.
CG_CONVERGED = 1
#: Reported for every non-convergence, whether the iteration cap was reached or
#: the recurrence broke down. The two cannot be distinguished from outside: the
#: loop runs inside `custom_linear_solve`, so only the final residual is
#: observable. A breakdown leaves a large residual, so it lands here too.
CG_NOT_CONVERGED = 2


def cg_solve(
    A: LinearOperatorLike,
    b: jax.Array,
    *,
    preconditioner: LinearOperatorLike | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    x0: jax.Array | None = None,
    track_iterations: bool = False,
) -> tuple[jax.Array, dict]:
    r"""Solve ``A x = b`` for symmetric positive-definite ``A``.

    Parameters
    ----------
    A:
        Symmetric positive-definite operator. Only ``A @ v`` is used.
    b:
        Right-hand side, shape ``(n,)``.
    preconditioner:
        Operator approximating ``A^{-1}``, applied as ``M @ r``. Must itself be
        symmetric positive-definite for the method to remain valid. ``None``
        means no preconditioning.
    rtol, atol:
        Convergence is declared when
        ``||r|| <= max(rtol * ||b||, atol)``.
    maxiter:
        Iteration cap. Defaults to ``10 * n``, matching SciPy and the LSMR
        implementation here.
    x0:
        Initial guess. Defaults to zeros.
    track_iterations:
        Report the exact iteration count as ``info["itn"]``, at the cost of
        reverse-mode differentiability. See the note below; either mode costs
        exactly one CG run.

    Returns
    -------
    x:
        The solution.
    info:
        ``istop`` (1 converged, 2 not converged), ``normr`` and the
        convergence threshold ``atol_eff``.

    Notes on differentiability
    --------------------------
    The iteration is a ``lax.while_loop``, which has no reverse-mode rule, and
    its counter is only observable from inside it. Those two facts pull in
    opposite directions, so both modes exist and each costs one CG run:

    * ``track_iterations=False`` (default) routes the solution through
      :func:`jax.lax.custom_linear_solve`, which supplies the adjoint
      ``while_loop`` lacks -- for symmetric ``A`` the cotangent is itself a
      solve against ``A``, which is how ``jax.scipy.sparse.linalg.cg`` manages
      it too. ``jax.grad`` works; the loop runs inside the callable, so ``itn``
      is not observable.
    * ``track_iterations=True`` runs the loop directly, so ``info["itn"]`` is
      exact. Reverse-mode differentiation then raises, ``while_loop`` having no
      VJP; forward mode and ``jax.jit`` are unaffected.

    Notes
    -----
    The iteration guards ``p^T A p <= 0``, which cannot happen for a positive
    definite ``A`` and therefore signals that the operator is not positive
    definite -- or is too ill-conditioned to behave as though it is. The guard
    halts the recurrence rather than dividing and returning NaN; the caller
    sees non-convergence and a large residual.
    """
    A = as_linop(A)
    b = jnp.asarray(b)
    n = A.shape[-1]

    if b.ndim != 1:
        msg = f"cg_solve expects a vector right-hand side, got shape {b.shape}. Solve each column separately, or use jax.vmap."
        raise ValueError(msg)

    dtype = jnp.result_type(A.dtype, b.dtype)
    b = b.astype(dtype)
    maxiter = 10 * n if maxiter is None else maxiter

    if preconditioner is None:

        def apply_M(residual: jax.Array) -> jax.Array:
            return residual

    else:
        M = as_linop(preconditioner)

        def apply_M(residual: jax.Array) -> jax.Array:
            return M @ residual

    # `x` is the initial guess; the residual and search direction are built
    # inside `run`, which is what `custom_linear_solve` actually invokes.
    x = jnp.zeros((n,), dtype=dtype) if x0 is None else jnp.asarray(x0, dtype=dtype)

    # A zero right-hand side has x = 0 as its exact solution; the threshold
    # then falls back to `atol` so the loop exits immediately rather than
    # chasing a relative tolerance against zero.
    atol_eff = jnp.maximum(rtol * jnp.linalg.norm(b), atol)

    def cond(state):
        _x, r, _z, _rz, itn, istop = state
        return (istop == 0) & (itn < maxiter) & (jnp.linalg.norm(r) > atol_eff)

    def body(state):
        x, r, p, rz, itn, istop = state
        Ap = A @ p
        pAp = jnp.vdot(p, Ap)

        # Guard the division rather than producing NaN.
        broken = ~jnp.isfinite(pAp) | (pAp <= 0)
        safe_pAp = jnp.where(broken, jnp.ones_like(pAp), pAp)
        alpha = rz / safe_pAp

        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = apply_M(r_new)
        rz_new = jnp.vdot(r_new, z_new)
        beta = rz_new / jnp.where(rz == 0, jnp.ones_like(rz), rz)
        p_new = z_new + beta * p

        istop_new = jnp.where(broken, CG_NOT_CONVERGED, istop)
        keep = ~broken
        return (
            jnp.where(keep, x_new, x),
            jnp.where(keep, r_new, r),
            jnp.where(keep, p_new, p),
            jnp.where(keep, rz_new, rz),
            itn + 1,
            istop_new,
        )

    def run(_matvec, rhs: jax.Array) -> jax.Array:
        """One CG run, returning only the solution.

        `custom_linear_solve` passes the matvec as the first argument; we
        close over `A` directly instead, so it is unused.
        """
        r0 = rhs - A @ x
        z0 = apply_M(r0)
        init = (x, r0, z0, jnp.vdot(r0, z0), jnp.asarray(0), jnp.asarray(0))
        solution, *_ = jax.lax.while_loop(cond, body, init)
        return solution

    extra: dict[str, jax.Array] = {}
    if track_iterations:
        # Run the loop directly: `itn` is observable, reverse mode is not.
        r0 = b - A @ x
        z0 = apply_M(r0)
        init = (x, r0, z0, jnp.vdot(r0, z0), jnp.asarray(0), jnp.asarray(0))
        x, _r, _z, _rz, itn, _istop = jax.lax.while_loop(cond, body, init)
        extra["itn"] = itn
    else:
        x = jax.lax.custom_linear_solve(lambda v: A @ v, b, run, symmetric=True)

    # One extra matvec, computed identically in both modes so the reported
    # outcome never depends on which was chosen.
    normr = jnp.linalg.norm(b - A @ x)
    istop = jnp.where(normr <= atol_eff, CG_CONVERGED, CG_NOT_CONVERGED)

    return x, {"istop": istop, "normr": normr, "atol_eff": atol_eff, **extra}
