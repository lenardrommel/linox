# as_scalar(scalar, dtype)
"""Utility functions for argument types."""

import numbers

import jax
import jax.numpy as jnp

from linox._typing import DTypeLike, ScalarLike, ShapeLike, ShapeType
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix

__all__ = ["as_scalar", "as_shape", "as_linop"]


def as_shape(x: ShapeLike, ndim: numbers.Integral | None = None) -> ShapeType:
    """Convert a shape representation into a shape defined as a tuple of ints.

    Parameters
    ----------
    x
        Shape representation.
    ndim
        The required number of dimensions in the shape.

    Raises:
    ------
    TypeError
        If ``x`` is not a valid :const:`ShapeLike`.
    TypeError
        If ``x`` does not feature the required number of dimensions.
    """
    if isinstance(x, int | numbers.Integral | jnp.integer):
        shape = (int(x),)
    elif isinstance(x, tuple) and all(isinstance(item, int) for item in x):
        shape = x
    else:
        try:
            _ = iter(x)
        except TypeError as e:
            msg = f"The given shape {x} must be an integer or an iterable of integers."
            raise TypeError(msg) from e

        if not all(
            isinstance(item, int | numbers.Integral | jnp.integer) for item in x
        ):
            msg = f"The given shape {x} must only contain integer values."
            raise TypeError(msg)

        shape = tuple(int(item) for item in x)

    if isinstance(ndim, numbers.Integral) and len(shape) != ndim:
        msg = f"The given shape {shape} must have {ndim} dimensions."
        raise TypeError(msg)

    return shape


def as_scalar(x: ScalarLike, dtype: DTypeLike = None) -> jnp.ndarray:
    """Convert a scalar into a scalar JAX array.

    Parameters
    ----------
    x
        Scalar value.
    dtype
        Data type of the scalar.

    Raises:
    ------
    ValueError
        If :code:`x` can not be interpreted as a scalar.
    """
    if jnp.ndim(x) != 0:
        msg = "The given input is not a scalar."
        raise ValueError(msg)

    return jnp.asarray(x, dtype=dtype)


LinearOperatorLike = jax.Array | LinearOperator


def as_linop(A: LinearOperatorLike) -> LinearOperator:
    """Convert an object into a linear operator.

    Parameters
    ----------
    A
        Object to convert.

    Raises
    ------
    TypeError
        If ``A`` is not a valid linear operator.
    """
    if isinstance(A, LinearOperator):
        return A

    if isinstance(A, jax.Array):
        return Matrix(A)

    # Add Callable support.
    msg = f"The given object {A} is not a valid linear operator type."
    raise TypeError(msg)
