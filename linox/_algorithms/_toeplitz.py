# _toeplitz.py (algorithms)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import solve_toeplitz


def levinson(a, b):
    """JAX translation using regular for loops (unrolled at compile time).

    Solves a Toeplitz system via Levinson recursion and returns both the
    solution and the reflection coefficients. Provided for completeness; the
    main solve path uses a hybrid SciPy+JAX approach with custom VJP.
    """
    n = b.shape[0]
    dtype = a.dtype

    # Initialize
    x = jnp.zeros(n, dtype=dtype)
    g = jnp.zeros(n, dtype=dtype)
    h = jnp.zeros(n, dtype=dtype)
    reflection_coeff = jnp.zeros(n + 1, dtype=dtype)

    # Base case
    x = x.at[0].set(b[0] / a[n - 1])
    reflection_coeff = reflection_coeff.at[0].set(1.0)
    reflection_coeff = reflection_coeff.at[1].set(x[0])

    if n == 1:
        return x, reflection_coeff

    g = g.at[0].set(a[n - 2] / a[n - 1])
    h = h.at[0].set(a[n] / a[n - 1])

    for m in range(1, n):
        # Compute numerator and denominator of x[m]
        x_num = -b[m]
        x_den = -a[n - 1]
        for j in range(m):
            nmj = n + m - (j + 1)
            x_num += a[nmj] * x[j]
            x_den += a[nmj] * g[m - j - 1]

        x_m = x_num / x_den
        x = x.at[m].set(x_m)
        reflection_coeff = reflection_coeff.at[m + 1].set(x_m)

        # Update x
        for j in range(m):
            x = x.at[j].add(-x_m * g[m - j - 1])

        if m == n - 1:
            return x, reflection_coeff

        # Compute g[m] and h[m]
        g_num = -a[n - m - 2]
        h_num = -a[n + m]
        g_den = -a[n - 1]
        for j in range(m):
            g_num += a[n + j - m - 1] * g[j]
            h_num += a[n + m - j - 1] * h[j]
            g_den += a[n + j - m - 1] * h[m - j - 1]

        g = g.at[m].set(g_num / g_den)
        h = h.at[m].set(h_num / x_den)

        # Update g and h arrays
        k = m - 1
        m2 = (m + 1) // 2
        c1, c2 = g[m], h[m]
        for j in range(m2):
            gj, gk = g[j], g[k]
            hj, hk = h[j], h[k]
            g = g.at[j].set(gj - c1 * hk)
            g = g.at[k].set(gk - c1 * hj)
            h = h.at[j].set(hj - c2 * gk)
            h = h.at[k].set(hk - c2 * gj)
            k -= 1

    return x, reflection_coeff


@jax.custom_vjp
def toeplitz_solve_hybrid(toeplitz_vec, b):
    """Hybrid Toeplitz solver: SciPy forward pass with custom VJP.

    Args:
        toeplitz_vec: First row/column of symmetric Toeplitz matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    # Use pure_callback to call SciPy from within JAX
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def scipy_call(c, b_val):
        c_np = jnp.asarray(c)
        b_np = jnp.asarray(b_val)
        return solve_toeplitz(c_np, b_np, check_finite=False)

    x = jax.pure_callback(
        scipy_call, result_shape, toeplitz_vec, b, vmap_method="sequential"
    )
    return x


def hybrid_fwd(toeplitz_vec, b):
    """Forward pass: use SciPy and save residuals for backward pass."""
    x = toeplitz_solve_hybrid(toeplitz_vec, b)
    residuals = (toeplitz_vec, b, x)
    return x, residuals


def hybrid_bwd(residuals, grad_output):
    """Custom VJP: efficient backward pass using Toeplitz structure.

    Args:
        residuals: (toeplitz_vec, b, x) from forward pass
        grad_output: Gradient w.r.t. output x

    Returns:
        (grad_toeplitz_vec, grad_b): Gradients w.r.t. inputs
    """
    toeplitz_vec, _b, x = residuals
    n = len(toeplitz_vec)

    # Reconstruct Toeplitz matrix for gradient computation
    T = jsp.linalg.toeplitz(toeplitz_vec)

    # Gradient w.r.t. b: A^{-T} @ grad_output
    grad_b = jnp.linalg.solve(T.T, grad_output)

    # Gradient w.r.t. toeplitz_vec using Toeplitz structure
    # For symmetric Toeplitz: ∂T[i,j]/∂c[k] = 1 if |i-j| = k, else 0
    grad_toeplitz_vec = jnp.zeros_like(toeplitz_vec)

    # Efficient vectorized computation
    i_indices = jnp.arange(n)[:, None]  # Shape (n, 1)
    j_indices = jnp.arange(n)[None, :]  # Shape (1, n)

    for k in range(n):
        # Find all (i,j) pairs where |i-j| = k
        mask = jnp.abs(i_indices - j_indices) == k
        # Sum: -x[i] * grad_b[j] for all valid (i,j)
        contribution = jnp.sum(jnp.where(mask, x[:, None] * grad_b[None, :], 0))
        grad_toeplitz_vec = grad_toeplitz_vec.at[k].set(-contribution)

    return grad_toeplitz_vec, grad_b


toeplitz_solve_hybrid.defvjp(hybrid_fwd, hybrid_bwd)


def solve_toeplitz_jax(c_or_cr, b, check_finite=True):  # noqa: ARG001
    """JAX-compatible Toeplitz solver using hybrid approach.

    Args:
        c_or_cr: Either first column c, or tuple (c, r) for non-symmetric
        b: Right-hand side vector
        check_finite: Ignored (kept for API compatibility)

    Returns:
        x: Solution vector
    """
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        # For non-symmetric case, use SciPy directly
        c_np, r_np = jnp.array(c), jnp.array(r)
        b_np = jnp.array(b)
        x_np = solve_toeplitz((c_np, r_np), b_np, check_finite=False)
        return jnp.array(x_np)
    # Symmetric case: use hybrid solver with custom VJP
    return toeplitz_solve_hybrid(c_or_cr, b)
