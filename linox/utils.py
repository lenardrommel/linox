# utils.py

"""Utility functions for argument types."""

import numbers
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator
from linox.typing import ArrayLike, DTypeLike, ScalarLike, ShapeLike, ShapeType

__all__ = ["_broadcast_shapes", "as_linop", "as_scalar", "as_shape"]


def as_shape(x: ShapeLike, ndim: numbers.Integral | None = None) -> ShapeType:
    """Convert a shape representation into a shape defined as a tuple of ints.

    Args:
        x: Shape representation.
        ndim: The required number of dimensions in the shape.

    Raises:
        TypeError
            If ``x`` is not a valid :const:`ShapeLike`.
        TypeError
            If ``x`` does not feature the required number of dimensions.
    """
    # Handle JAX traced values
    if isinstance(x, tuple) and all(
        isinstance(item, int | numbers.Integral | jnp.integer) or hasattr(item, "aval")
        for item in x
    ):
        if any(isinstance(item, jnp.ndarray) for item in x):
            shape = tuple(int(item) for item in x)
        else:
            shape = x
    elif isinstance(x, int | numbers.Integral | jnp.integer):
        shape = (int(x),)
    else:
        try:
            _ = iter(x)
        except TypeError as e:
            msg = f"The given shape {x} must be an integer or an iterable of integers."
            raise TypeError(msg) from e

        if not all(
            isinstance(item, int | numbers.Integral | jnp.integer)
            or hasattr(item, "aval")
            for item in x
        ):
            msg = f"The given shape {x} must only contain integer values."
            raise TypeError(msg)

        shape = tuple(item for item in x)

    if isinstance(ndim, numbers.Integral) and len(shape) != ndim:
        msg = f"The given shape {shape} must have {ndim} dimensions."
        raise TypeError(msg)

    return shape


def as_scalar(x: ScalarLike, dtype: DTypeLike = None) -> jnp.ndarray:
    """Convert a scalar into a scalar JAX array.

    Args:
        x: Scalar value.
        Scalar value.
        dtype: Data type of the scalar.

    Raises:
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

    Args:
        A: Object to convert.

    Raises:
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
    """Broadcast shapes.

    Args:
        shapes: Shapes to broadcast.

    Raises:
        ValueError
            If the shapes cannot be broadcasted.
    """
    try:
        return jnp.broadcast_shapes(*shapes)
    except ValueError:
        msg = f"Shapes {shapes} cannot be broadcasted."
        raise ValueError(msg)  # noqa: B904


def _broadcast_to(x: ArrayLike, shape: ShapeLike) -> jnp.ndarray:
    """Broadcast an array to a given shape.

    Args:
        x: Array to broadcast.
        shape: Shape to broadcast to.

    Raises:
        ValueError
            If the array cannot be broadcasted to the given shape.
    """
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, LinearOperator):
        return x.toshape(shape)
    msg = f"Unsupported broadcast type {type(x)}."
    raise ValueError(msg)


def todense(x: LinearOperatorLike) -> jnp.ndarray:
    """Convert a linear operator to a dense matrix.

    Args:
        x: Linear operator to convert.

    Returns:
        Dense matrix.
    """
    return x.todense() if isinstance(x, LinearOperator) else x


def allclose(
    a: LinearOperatorLike,
    b: LinearOperatorLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two linear operators are close to each other.

    Args:
        a: First linear operator.
        b: Second linear operator.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    Returns:
        Whether the two linear operators are close to each other.
    """
    a_dense = todense(a)
    b_dense = todense(b)
    return jnp.allclose(a_dense, b_dense, rtol=rtol, atol=atol)


def as_dense(a: LinearOperatorLike) -> jnp.ndarray:
    """Convert a linear operator to a dense matrix.

    Args:
        a: Linear operator to convert.

    Returns:
        Dense matrix.
    """
    if isinstance(a, LinearOperator):
        return a.todense()
    return a
