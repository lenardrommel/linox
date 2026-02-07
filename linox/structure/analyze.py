"""Operator analysis pass."""

import plum

from linox.operators import (
    BlockDiagonal,
    IsotropicAdditiveLinearOperator,
    Kronecker,
    LinearOperator,
    PositiveDiagonalPlusSymmetricLowRank,
)
from linox.structure.ir import (
    BlockDiagonalIR,
    DenseIR,
    DiagPlusLowRankIR,
    IsotropicShiftIR,
    KroneckerIR,
    OperatorIR,
)


def _get_tags(op: LinearOperator) -> set[str]:
    """Extract tags from operator."""
    tags = set()
    # Basic property checks if they exist
    if getattr(op, "is_symmetric", False):
        tags.add("symmetric")
    if getattr(op, "is_psd", False):
        tags.add("psd")
    return tags


@plum.dispatch
def analyze(op: LinearOperator) -> OperatorIR:
    """Analyze a linear operator and return its Intermediate Representation."""
    # Default fallback: treat as dense/generic
    return DenseIR(tags=_get_tags(op), op=op)


@analyze.dispatch
def _(op: Kronecker) -> OperatorIR:
    from linox.operators.kron import extract_kronecker_factors

    factors, scalar = extract_kronecker_factors(op)
    # If scalar is not none, it means it was wrapped?
    # extract_kronecker_factors handles wrapped scaled.
    # But here we are dispatching on Kronecker directly.
    # So scalar should be None/1.0 unless Kronecker logic changes.

    return KroneckerIR(
        tags=_get_tags(op),
        scalar=1.0 if scalar is None else float(scalar),
        factors=factors,
    )


@analyze.dispatch
def _(op: IsotropicAdditiveLinearOperator) -> OperatorIR:
    return IsotropicShiftIR(
        tags=_get_tags(op), shift=float(op.scalar), base=op.operator
    )


@analyze.dispatch
def _(op: PositiveDiagonalPlusSymmetricLowRank) -> OperatorIR:
    return DiagPlusLowRankIR(
        tags=_get_tags(op),
        diag=op.diagonal,
        U=op.low_rank.U,
        S=op.low_rank.S,
        scale=float(op.low_rank_scale),
    )


@analyze.dispatch
def _(op: BlockDiagonal) -> OperatorIR:
    return BlockDiagonalIR(tags=_get_tags(op), blocks=list(op.blocks))
