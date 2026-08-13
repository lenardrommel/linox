"""Woodbury-identity solvers for diagonal-plus-low-rank operators."""

# _woodbury.py

import jax
import jax.numpy as jnp

from linox._types import ArrayLike, ScalarLike
from linox.operators.dense import Matrix


def _row_broadcast(d: jax.Array, v: jax.Array) -> jax.Array:
    """Reshape a diagonal ``d`` of shape ``(n,)`` to divide ``v`` row-wise.

    ``v`` may be a vector ``(n,)`` or a matrix of right-hand sides ``(n, k)``.
    Plain ``v / d`` broadcasts along the *last* axis, which silently computes
    ``v[i, j] / d[j]`` for a square ``v`` and raises otherwise.
    """
    return d.reshape(-1, *((1,) * (v.ndim - 1)))


def woodbury_solve(U: Matrix, s: ArrayLike, d: ScalarLike | ArrayLike, v: ArrayLike):
    """Solve ``(L L^T + D) x = v`` via the Woodbury matrix identity.

    ``A = L L^T + D`` and
    ``A^{-1} v = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v``.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns
    -------
      Solution of the linear system.
    """
    d = jnp.asarray(d)
    v = jnp.asarray(v)
    is_scalar_d = d.ndim == 0

    D_inv_v = v / d if is_scalar_d else v / _row_broadcast(d, v)
    D_inv_U = U / d if is_scalar_d else U / d[:, None]

    # Capacitance matrix: C = diag(1/s) + U^T D^{-1} U
    C = U.T @ D_inv_U
    C = C.at[jnp.diag_indices(len(s))].add(1.0 / s)

    return D_inv_v - D_inv_U @ jax.scipy.linalg.cho_solve(
        (jnp.linalg.cholesky(C), True),
        U.T @ D_inv_v,
    )


def woodbury_chol_solve(L: Matrix, d: ScalarLike | ArrayLike, v: ArrayLike):
    """Solve ``(L L^T + D) x = v`` via the Woodbury matrix identity.

    ``A = L L^T + D`` and
    ``A^{-1} v = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v``.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns
    -------
      Solution of the linear system.
    """
    d = jnp.asarray(d)
    v = jnp.asarray(v)
    D_inv_v = v / (_row_broadcast(d, v) if d.ndim > 0 else d)
    D_inv_L = L / (d[:, None] if d.ndim > 0 else d)
    eye = jnp.eye(L.shape[-1])
    return D_inv_v - D_inv_L @ jax.scipy.linalg.cho_solve(
        (jnp.linalg.cholesky(eye + L.T @ D_inv_L), True), L.T @ D_inv_v
    )
