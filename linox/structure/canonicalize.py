"""Operator canonicalization pass."""

import plum

from linox.operators import Kronecker, LinearOperator, ScaledLinearOperator


@plum.dispatch
def canonicalize(op: LinearOperator) -> LinearOperator:
    """Canonicalize an operator to a standard form."""
    # Recursively canonicalize children?
    # For now, just top-level pass or shallow?
    # Ideally deep.
    return op


@canonicalize.dispatch
def _(op: ScaledLinearOperator) -> LinearOperator:
    # Recursively canonicalize the inner operator
    inner = canonicalize(op.operator)

    # 1. Flatten nested scaling: Scaled(Scaled(A, a), b) -> Scaled(A, a*b)
    if isinstance(inner, ScaledLinearOperator):
        # Flatten
        return ScaledLinearOperator(inner.operator, op.scalar * inner.scalar)

    # Re-wrap if changed
    if inner is not op.operator:
        return ScaledLinearOperator(inner, op.scalar)

    return op


@canonicalize.dispatch
def _(op: Kronecker) -> LinearOperator:
    # Kronecker is binary tree (A, B).
    # We simply recurse.
    cA = canonicalize(op.A)
    cB = canonicalize(op.B)

    if cA is not op.A or cB is not op.B:
        return Kronecker(cA, cB)

    return op
