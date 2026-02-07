"""Trace estimation for linear operators."""

import jax

from linox.linalg.approx.hutchinson import (
    hutchinson_diagonal,
    hutchinson_trace,
    hutchinson_trace_and_diagonal,
)
from linox.utils.array import LinearOperatorLike

# Export hutchinson functions as alternatives for direct usage
__all__ = [
    "hutchinson_diagonal",
    "hutchinson_trace",
    "hutchinson_trace_and_diagonal",
    "trace",
]


def trace(A: LinearOperatorLike, *, method: str = "exact", **kwargs) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute trace of a linear operator.

    Args:
        A: Linear operator.
        method: "exact" or "hutchinson".
        **kwargs: arguments for method (e.g. key, num_samples for hutchinson, return_std)
    """
    from linox.operators.arithmetic import ltrace
    from linox.utils import as_linop

    op = as_linop(A)
    
    return_std = kwargs.pop("return_std", False)

    if method == "hutchinson":
        key = kwargs.get("key")
        if key is None:
            msg = "Hutchinson trace requires a PRNG key."
            raise ValueError(msg)
        # hutchinson_trace always returns (value, std)
        result = hutchinson_trace(op, **kwargs)
        if return_std:
            return result
        return result[0]

    # Default / Exact - ltrace always returns (value, std)
    result = ltrace(op, **kwargs)
    if return_std:
        return result
    # Return just the value
    if isinstance(result, tuple):
        return result[0]
    return result
