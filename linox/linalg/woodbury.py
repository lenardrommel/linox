# _woodbury.py

import jax
import jax.numpy as jnp

from linox.operators.dense import Matrix
from linox.typing import ArrayLike, ScalarLike


def woodbury_solve(U: Matrix, s: ArrayLike, d: ScalarLike | ArrayLike, v: ArrayLike):
    """Woodbury matrix identity implementation for solving specifically system of PSD plus diagonal matrix.
    A = L L^T + D
    A^{-1} b = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns
    -------
      Solution of the linear system.
    """
    d = jnp.asarray(d)
    is_scalar_d = d.ndim == 0

    D_inv_v = v / d
    D_inv_U = U / d if is_scalar_d else U / d[:, None]

    # Capacitance matrix: C = diag(1/s) + U^T D^{-1} U
    C = U.T @ D_inv_U
    C = C.at[jnp.diag_indices(len(s))].add(1.0 / s)

    return D_inv_v - D_inv_U @ jax.scipy.linalg.cho_solve(
        (jnp.linalg.cholesky(C), True),
        U.T @ D_inv_v,
    )


def woodbury_chol_solve(L: Matrix, d: ScalarLike | ArrayLike, v: ArrayLike):
    """Woodbury matrix identity implementation for solving specifically system of PSD plus diagonal matrix.
    A = L L^T + D
    A^{-1} b = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns
    -------
      Solution of the linear system.
    """
    D_inv_v = v / d
    d = jnp.asarray(d)
    D_inv_L = L / (d[:, None] if d.ndim > 0 else d)
    eye = jnp.eye(L.shape[-1])
    return D_inv_v - D_inv_L @ jax.scipy.linalg.cho_solve(
        (jnp.linalg.cholesky(eye + L.T @ D_inv_L), True), L.T @ D_inv_v
    )
