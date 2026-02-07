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


def trace(A: LinearOperatorLike, *, method: str = "exact", **kwargs) -> jax.Array:
    """Compute trace of a linear operator.

    Args:
        A: Linear operator.
        method: "exact" or "hutchinson".
        **kwargs: arguments for method (e.g. key, num_samples for hutchinson)
    """
    from linox.operators.arithmetic import ltrace
    from linox.utils import as_linop

    op = as_linop(A)
    

    if method == "hutchinson":
        key = kwargs.get("key")
        if key is None:
            msg = "Hutchinson trace requires a PRNG key."
            raise ValueError(msg)
        # Return just the estimate value
        if kwargs.get("return_std", False):
            return hutchinson_trace(op, **kwargs)
        return hutchinson_trace(op, **kwargs)[0]

    # Default / Exact
    res = ltrace(op)
    if kwargs.get("return_std", False):
        if isinstance(res, tuple):
            return res
        else:
            return res, 0.0
    if isinstance(res, tuple):
        return res[0]
    return res
