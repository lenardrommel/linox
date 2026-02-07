"""Arnoldi and general Krylov methods.

This module implements Arnoldi iteration for general matrices.
For symmetric matrices, use lanczos.py.
"""

import jax
import jax.numpy as jnp
from jax import lax

from linox.typing import ArrayLike, LinearOperatorLike


def arnoldi_iteration(
    A: LinearOperatorLike,
    v0: ArrayLike,
    num_iters: int,
) -> tuple[jax.Array, jax.Array]:
    """Arnoldi iteration for general (non-symmetric) operators.

    Computes a Hessenberg reduction of a general linear operator using the
    Arnoldi iteration. Returns the orthonormal Arnoldi vectors Q and the
    upper Hessenberg matrix H such that A ≈ Q H Q^T on the Krylov subspace.

    Parameters
    ----------
    A : LinearOperatorLike
        General linear operator or matrix.
    v0 : ArrayLike
        Initial vector for the Krylov process.
    num_iters : int
        Number of Arnoldi iterations.

    Returns:
    -------
    Q : jax.Array, shape (n, num_iters)
        Orthonormal Arnoldi vectors.
    H : jax.Array, shape (num_iters+1, num_iters)
        Upper Hessenberg matrix.
    """
    v0 = jnp.asarray(v0)
    n = v0.size

    # Normalize initial vector
    beta_0 = jnp.linalg.norm(v0)
    v0 /= beta_0

    # Pre-allocate arrays
    Q = jnp.zeros((n, num_iters))
    H = jnp.zeros((num_iters + 1, num_iters))

    # Initialize first vector
    Q = Q.at[:, 0].set(v0)

    def arnoldi_step(k, carry):
        Q_curr, H_curr = carry

        # Matrix-vector product
        v = Q_curr[:, k]
        w = A @ v

        # Full Gram-Schmidt orthogonalization using fori_loop
        def gs_body(j, state):
            w_state, H_state = state
            h_jk = jnp.dot(w_state, Q_curr[:, j])
            H_state = H_state.at[j, k].set(h_jk)
            w_state -= h_jk * Q_curr[:, j]
            return (w_state, H_state)

        w, H_curr = lax.fori_loop(0, k + 1, gs_body, (w, H_curr))

        # Compute residual norm
        h_next = jnp.linalg.norm(w)
        H_curr = H_curr.at[k + 1, k].set(h_next)

        # Store next vector (if not last iteration)
        Q_next = lax.cond(
            k < num_iters - 1,
            lambda: Q_curr.at[:, k + 1].set(w / (h_next + 1e-16)),
            lambda: Q_curr,
        )

        return (Q_next, H_curr)

    # Run Arnoldi iterations using fori_loop
    Q, H = lax.fori_loop(0, num_iters, arnoldi_step, (Q, H))

    return Q, H


def arnoldi_matrix_function(
    A: LinearOperatorLike,
    v: ArrayLike,
    func: callable,
    num_iters: int,
) -> jax.Array:
    """Approximate f(A)v using Arnoldi iteration."""
    v = jnp.asarray(v)
    v_norm = jnp.linalg.norm(v)
    v_normalized = v / v_norm

    # Perform Arnoldi iteration
    Q, H = arnoldi_iteration(A, v_normalized, num_iters)

    # H is (num_iters+1, num_iters), discard last row for square matrix approximation
    H_square = H[:-1, :]

    # Apply function to Hessenberg matrix
    w, V = jnp.linalg.eig(H_square)
    fH = V @ jnp.diag(func(w)) @ jnp.linalg.inv(V)

    # Extract first row
    e1 = jnp.zeros(num_iters)
    e1 = e1.at[0].set(1.0)

    # Project back
    result = v_norm * (Q @ (fH @ e1))

    return result.real
