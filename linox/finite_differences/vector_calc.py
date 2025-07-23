from functools import partial

import jax.numpy as jnp
from jax import vmap

from linox.finite_differences._finite_differences import (
    _compute_coefficients,
    finite_difference_adaptive_bc,
    finite_difference_periodic_bc,
)


def init_coeff(stencil_points=3, derivative_order=1):  # noqa: ANN201
    """Initialize finite difference coefficients."""
    return _compute_coefficients(derivative_order, stencil_points)


def gradient_1d(u, dx, coeffs, boundary_condition="adaptive"):  # noqa: ANN201
    """Compute 1D gradient."""
    if boundary_condition == "periodic":
        # For periodic, use only central coefficients
        return finite_difference_periodic_bc(u, coeffs["central"], dx)
    # For adaptive, use full coefficients dictionary
    return finite_difference_adaptive_bc(u, coeffs, dx)


def gradient_2d(u, dx, dy, coeffs, boundary_condition="adaptive"):  # noqa: ANN201
    grad_x = vmap(lambda row: gradient_1d(row, dx, coeffs))(u)
    grad_y = vmap(lambda col: gradient_1d(col, dy, coeffs))(u.T).T

    return grad_x, grad_y


def gradient_3d(u, dx, dy, dz, coeffs, boundary_condition="adaptive"):
    """Compute 3D gradient."""
    # Create partial functions with fixed boundary condition
    grad_1d_x = partial(gradient_1d, boundary_condition=boundary_condition)
    grad_1d_y = partial(gradient_1d, boundary_condition=boundary_condition)
    grad_1d_z = partial(gradient_1d, boundary_condition=boundary_condition)

    # x-direction gradient
    grad_x = vmap(vmap(lambda row: grad_1d_x(row, dx, coeffs)))(u)

    # y-direction gradient
    u_t = jnp.transpose(u, (0, 2, 1))
    grad_y = jnp.transpose(
        vmap(vmap(lambda row: grad_1d_y(row, dy, coeffs)))(u_t), (0, 2, 1)
    )
    # z-direction gradient
    u_t = jnp.transpose(u, (2, 1, 0))
    grad_z = jnp.transpose(
        vmap(vmap(lambda row: grad_1d_z(row, dz, coeffs)))(u_t), (2, 1, 0)
    )

    return grad_x, grad_y, grad_z
