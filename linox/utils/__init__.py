"""Utility functions module."""

from .array import (
    allclose,
    as_dense,
    as_linop,
    as_scalar,
    as_shape,
    todense,
)
from .validation import (
    ValidationError,
    validate,
)

__all__ = [
    "ValidationError",
    "allclose",
    "as_dense",
    "as_linop",
    "as_scalar",
    "as_shape",
    "todense",
    "validate",
]
