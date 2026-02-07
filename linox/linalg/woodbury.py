# _woodbury.py

import jax
import jax.numpy as jnp

from linox.operators.dense import Matrix
from linox.typing import ScalarLike, VectorLike


def woodbury_solve(U: Matrix, s: VectorLike, d: ScalarLike | VectorLike, v: VectorLike):
    """Woodbury matrix identity implementation for solving specifically system of PSD plus diagonal matrix.
    A = L L^T + D
    A^{-1} b = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns:
      Solution of the linear system.
    """
    if isinstance(d, VectorLike):
        D_inv_v = v / d
        D_inv_U = U / d[:, None]
        return D_inv_v - D_inv_U @ jax.scipy.linalg.cho_solve(
            (
                jnp.linalg.cholesky(
                    (U.T @ D_inv_U).at[jnp.diag_indices(len(s))].add(1 / s)
                ),
                True,
            ),
            U.T @ D_inv_v,
        )
    return v / d - (U / d / (d / s + 1)) @ (U.T @ v)


def woodbury_chol_solve(L: Matrix, d: ScalarLike | VectorLike, v: VectorLike):
    """Woodbury matrix identity implementation for solving specifically system of PSD plus diagonal matrix.
    A = L L^T + D
    A^{-1} b = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v.

    Args:
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns:
      Solution of the linear system.
    """
    D_inv_v = v / d
    D_inv_L = L / (d[:, None] if isinstance(d, VectorLike) else d)
    eye = jnp.eye(L.shape[-1])
    return D_inv_v - D_inv_L @ jax.scipy.linalg.cho_solve(
        (jnp.linalg.cholesky(eye + L.T @ D_inv_L), True), L.T @ D_inv_v
    )
