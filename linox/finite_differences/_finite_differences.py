import math

import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve


def finite_difference_coefficients(m: int, n: int, scheme: str = "central"):  # noqa: ANN202
    """Compute the coefficients for an n-point finite difference scheme.
    Args:
        m (int) : Order of derivative
        n (int) : Number of stencil points
        scheme (str) : 'forward', 'backward', or 'central'.
    """  # noqa: D205
    if n % 2 == 0:
        raise ValueError("n must be odd.")  # noqa: EM101, TRY003
    if scheme == "forward":
        points = jnp.arange(n)
    elif scheme == "backward":
        points = jnp.arange(-n + 1, 1)
    elif scheme == "central":
        points = jnp.arange(-(n // 2), n // 2 + 1)  # Changed from -n//2 to -(n//2)
    else:
        raise ValueError("Scheme must be 'forward', 'backward', or 'central'.")  # noqa: TRY003

    A = jnp.array([[p**i for p in points] for i in range(n)], dtype=jnp.float32)

    # b_j=0 for j\neq m, b_j = m! for j=m
    b = jnp.zeros(n, dtype=jnp.float32).at[m].set(math.factorial(m))

    return solve(A, b)


def _compute_coefficients(m: int, n: int):  # noqa: ANN202
    """Compute the stencil coefficients."""
    central = finite_difference_coefficients(m, n, scheme="central")

    # Number of one-sided stencils we need. For n=5 => one_sided_points=3
    # i = 0 -> 3-point scheme, i = 1 -> 4-point scheme, i = 2 -> 5-point scheme
    one_sided_points = (n + 1) // 2

    left_coeffs = []
    right_coeffs = []

    # Forward and backward stencils from size = one_sided_points up to n
    for i in range(one_sided_points):
        forward_n = one_sided_points + i
        backward_n = one_sided_points + i

        left_coeffs.append(
            finite_difference_coefficients(m, forward_n, scheme="forward")
        )
        right_coeffs.append(
            finite_difference_coefficients(m, backward_n, scheme="backward")
        )

    return {
        "central": central,
        "left": left_coeffs,
        "right": right_coeffs,
    }


def finite_difference_periodic_bc(u: jnp.ndarray, coeffs: jnp.ndarray, dx: float):
    """
    Compute the finite difference derivative with periodic boundary conditions.
    """
    n = len(coeffs)
    pad_width = (n // 2, n // 2)  #
    u_padded = jnp.pad(u, pad_width, mode="wrap")
    return jnp.convolve(u_padded, coeffs[::-1] / dx, mode="valid")


def finite_difference_adaptive_bc(u: jnp.ndarray, coeffs_dict: dict, dx: float):  # noqa: ANN202
    result = jnp.zeros_like(u)
    length = len(u)

    central_coeffs = coeffs_dict["central"]
    left_coeffs = coeffs_dict["left"]
    right_coeffs = coeffs_dict["right"]

    max_coeff_length = len(central_coeffs)
    boundary_points = (max_coeff_length - 1) // 2

    central_part = jnp.convolve(u, central_coeffs[::-1], mode="valid")
    result = result.at[boundary_points : length - boundary_points].set(central_part)

    def left_side_conv(coeffs):  # noqa: ANN202
        return jnp.convolve(u[: boundary_points * 2], coeffs[::-1], mode="valid")

    def right_side_conv(coeffs):  # noqa: ANN202
        slice_start = length - len(coeffs)
        right_slice = u[slice_start:]
        return jnp.convolve(right_slice, coeffs[::-1], mode="valid")

    # Process left boundary
    for i in range(boundary_points):
        conv_result = left_side_conv(left_coeffs[i])
        result = result.at[i].set(conv_result[i])

    # Process right boundary
    for i in range(boundary_points):
        conv_result = right_side_conv(right_coeffs[i])
        result = result.at[length - boundary_points + i].set(conv_result[0])

    return result / dx


def _compute_coefficients_vmap(m: int, n: int):  # noqa: ANN202
    """Compute the stencil coefficients using vmap (experimental).
    Args:
        m (int) : Order of derivative
        n (int) : Number of stencil points.
    """  # noqa: D205
    one_sided_points = (n + 1) // 2

    def _compute_one_sided_coefficients(i):  # noqa: ANN202
        left = finite_difference_coefficients(m, one_sided_points, scheme="forward")
        right = finite_difference_coefficients(m, one_sided_points, scheme="backward")
        return (left, right)

    i_vals = jnp.arange(one_sided_points)

    coeff_left, coeff_right = vmap(_compute_one_sided_coefficients)(i_vals)

    return {
        "central": finite_difference_coefficients(m=m, n=n, scheme="central"),
        "left": coeff_left,
        "right": coeff_right,
    }
