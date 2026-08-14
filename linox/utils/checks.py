"""Preconditions that hold under ``jax.jit`` as well as eagerly.

A plain ``if not condition: raise`` cannot work on a traced value: under
``jax.jit`` the condition is a tracer with no concrete truth value, so the
check is silently skipped exactly where mistakes are hardest to spot.

:func:`require` bridges both worlds. Eagerly it raises immediately. Under a
trace it defers to a runtime callback, which raises when the computation
actually executes -- so a violated precondition is still an error rather than
a silently wrong answer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["require"]


def require(condition: bool | jax.Array, message: str) -> None:
    """Raise ``ValueError`` unless ``condition`` holds.

    Parameters
    ----------
    condition:
        A boolean or a rank-0 boolean array. Traced values are checked at
        runtime rather than at trace time.
    message:
        The message to raise with.

    Raises
    ------
    ValueError
        Eagerly, when ``condition`` is concrete and false.
    Exception
        Under ``jax.jit``, the runtime callback raises when the condition is
        violated. JAX surfaces this as an ``XlaRuntimeError`` wrapping the
        original ``ValueError`` and message.
    """
    try:
        ok = bool(condition)
    except jax.errors.ConcretizationTypeError:
        _require_at_runtime(jnp.asarray(condition), message)
        return

    if not ok:
        raise ValueError(message)


def _require_at_runtime(condition: jax.Array, message: str) -> None:
    def _check(ok: jax.Array) -> None:
        if not bool(ok):
            raise ValueError(message)

    # `lax.cond` keeps the callback off the happy path, so a satisfied
    # precondition costs a predicate and nothing else.
    jax.lax.cond(
        ~condition,
        lambda: jax.debug.callback(_check, condition),
        lambda: None,
    )
