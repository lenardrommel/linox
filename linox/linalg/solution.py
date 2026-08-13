"""Outcome reporting for linear solves.

Before this module, :func:`linox.solve` returned a bare array with no way to
tell whether it meant anything. A singular system produced finite, plausible,
wildly wrong numbers -- no exception, no warning, no NaN -- which then
propagated silently into whatever the caller did next.

The types here follow the shape of `lineax`'s ``Solution``/``RESULTS``:
solves report an outcome code, and by default a failed solve raises rather
than handing back garbage.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "RESULTS",
    "LinearSolveError",
    "Solution",
    "check_result",
    "residual_result",
]


class RESULTS(enum.IntEnum):
    """Outcome of a linear solve."""

    successful = 0
    max_steps_reached = 1
    singular = 2
    breakdown = 3
    stagnation = 4
    conlim = 5
    nonfinite_input = 6
    nonfinite_output = 7

    @property
    def message(self) -> str:
        """Human-readable explanation of this outcome."""
        return _MESSAGES[self]


_MESSAGES: dict[RESULTS, str] = {
    RESULTS.successful: "The linear solve was successful.",
    RESULTS.max_steps_reached: (
        "The iterative solver reached its maximum number of steps without "
        "converging. Increase `maxiter`, loosen the tolerance, or use a "
        "preconditioner."
    ),
    RESULTS.singular: (
        "The linear solve did not produce a solution: the operator appears to "
        "be singular or rank-deficient, so no exact solution exists. Use "
        "`linox.pinverse(a) @ b` for a least-squares/minimum-norm solution, or "
        "pass `throw=False` to accept the result as-is."
    ),
    RESULTS.breakdown: (
        "The iterative solver broke down (a division by a near-zero quantity). This usually means the operator is singular or badly conditioned."
    ),
    RESULTS.stagnation: ("The iterative solver stagnated: it stopped making progress before reaching the requested tolerance."),
    RESULTS.conlim: ("The estimated condition number exceeded the solver's limit, so the result cannot be trusted."),
    RESULTS.nonfinite_input: ("The right-hand side or operator contained NaN or infinity. The problem lies upstream of this solve."),
    RESULTS.nonfinite_output: ("The solve produced NaN or infinite values, which usually means the operator is singular or badly conditioned."),
}


class LinearSolveError(RuntimeError):
    """Raised when a linear solve fails and ``throw=True`` (the default)."""

    def __init__(self, result: RESULTS, detail: str = "") -> None:
        self.result = result
        message = f"{RESULTS(result).name}: {RESULTS(result).message}"
        if detail:
            message = f"{message}\n{detail}"
        super().__init__(message)


@dataclass(frozen=True)
class Solution:
    """The outcome of a linear solve.

    Attributes
    ----------
    value:
        The solution array. Meaningful only when ``result`` is
        ``RESULTS.successful``.
    result:
        The outcome code. May be a traced array under ``jax.jit``, in which
        case compare it against :class:`RESULTS` members with ``jnp`` ops
        rather than Python ``==``.
    stats:
        Solver-specific diagnostics, e.g. ``num_steps``, ``istop``,
        ``residual``.
    """

    value: jax.Array
    result: RESULTS | jax.Array
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool | jax.Array:
        """Whether the solve succeeded."""
        return self.result == RESULTS.successful


def residual_result(
    operator: Any,
    solution: jax.Array,
    rhs: jax.Array,
    *,
    rtol: float = 1e-5,
) -> tuple[RESULTS | jax.Array, jax.Array]:
    """Classify a solve by its relative residual ``||Ax - b|| / ||b||``.

    This is the only reliable detector for a direct solve against a singular
    operator: the output is typically finite and enormous rather than NaN, so
    a finiteness check alone misses it. Costs one extra matvec, which is cheap
    next to the factorisation it is validating.
    """
    residual = jnp.linalg.norm(jnp.asarray(operator @ solution) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(jnp.asarray(rhs).dtype).tiny)

    nonfinite = ~jnp.all(jnp.isfinite(jnp.asarray(solution)))
    failed = nonfinite | (residual > rtol)

    result = jnp.where(
        nonfinite,
        jnp.int32(RESULTS.nonfinite_output),
        jnp.where(failed, jnp.int32(RESULTS.singular), jnp.int32(RESULTS.successful)),
    )
    return result, residual


def check_result(result: RESULTS | jax.Array, *, throw: bool, detail: str = "") -> None:
    """Raise (eager) or emit a runtime error message (traced) on failure.

    Under ``jax.jit`` the outcome is a tracer, so there is nothing to raise at
    trace time. In that case a runtime callback reports the failure when it
    actually occurs, and the caller can still branch on ``Solution.result``
    inside the computation.
    """
    if not throw:
        return

    try:
        code = int(result)
    except (jax.errors.ConcretizationTypeError, TypeError):
        _report_under_trace(result, detail)
        return

    if code != RESULTS.successful:
        raise LinearSolveError(RESULTS(code), detail)


def _report_under_trace(result: jax.Array, detail: str) -> None:
    def _callback(code: jax.Array) -> None:
        outcome = RESULTS(int(code))
        if outcome is not RESULTS.successful:
            message = f"linox.solve failed: {outcome.name}: {outcome.message}"
            if detail:
                message = f"{message}\n{detail}"
            # Cannot raise from inside a traced computation; surface it loudly.
            print(f"ERROR: {message}")

    jax.lax.cond(
        jnp.asarray(result) != jnp.int32(RESULTS.successful),
        lambda: jax.debug.callback(_callback, result),
        lambda: None,
    )
