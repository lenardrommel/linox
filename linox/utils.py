# as_scalar(scalar, dtype)
"""Utility functions for argument types."""

import numbers
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator

# from linox._matrix import Matrix, Scalar
from linox.typing import ArrayLike, DTypeLike, ScalarLike, ShapeLike, ShapeType

__all__ = ["_broadcast_shapes", "as_linop", "as_scalar", "as_shape"]


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

    from linox._matrix import Scalar  # noqa: PLC0415

    return jnp.asarray(x.scalar if isinstance(x, Scalar) else x, dtype=dtype)


LinearOperatorLike = jax.Array | LinearOperator


def as_linop(A: LinearOperatorLike) -> LinearOperator:
    """Convert an object into a linear operator.

    Parameters
    ----------
    A
        Object to convert.

    Raises:
    ------
    TypeError
        If ``A`` is not a valid linear operator.
    """
    if isinstance(A, LinearOperator):
        return A

    if isinstance(A, jax.Array | jnp.ndarray):
        from linox._matrix import Matrix  # noqa: PLC0415

        return Matrix(A)

    # Add Callable support.
    msg = f"The given object {A} is not a valid linear operator type."
    raise TypeError(msg)


# inverse special behavior:
def _broadcast_shapes(shapes: Iterable[ShapeLike]) -> ShapeLike:
    try:
        return jnp.broadcast_shapes(*shapes)
    except ValueError:
        msg = f"Shapes {shapes} cannot be broadcasted."
        raise ValueError(msg)  # noqa: B904


def _broadcast_to(x: ArrayLike, shape: ShapeLike) -> jnp.ndarray:
    """Broadcast an array to a given shape."""
    if isinstance(x, jnp.ndarray):
        return x
    elif isinstance(x, LinearOperator):
        return x.toshape(shape)
    else:
        msg = f"Unsupported broadcast type {type(x)}."
        raise ValueError(msg)  # noqa: TRY004
    # except ValueError as e:
    #     msg = f"Array of shape {x.shape} cannot be broadcasted to {shape}."
    #     raise ValueError(msg) from e
