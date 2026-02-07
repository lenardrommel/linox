"""Intermediate Representations (IR) for Linear Operators."""

from dataclasses import dataclass

import jax

from linox.operators import Diagonal, LinearOperator


@dataclass
class OperatorIR:
    """Base class for operator IR."""

    tags: set[str]

    @property
    def is_symmetric(self) -> bool:
        return "symmetric" in self.tags

    @property
    def is_psd(self) -> bool:
        return "psd" in self.tags


@dataclass
class DenseIR(OperatorIR):
    """IR for dense or unstructured operators."""

    op: LinearOperator


@dataclass
class KroneckerIR(OperatorIR):
    """IR for Kronecker product structure."""

    scalar: float
    factors: list[LinearOperator]


@dataclass
class IsotropicShiftIR(OperatorIR):
    """IR for s*I + Base."""

    shift: float
    base: LinearOperator


@dataclass
class DiagPlusLowRankIR(OperatorIR):
    """IR for D + scale * U S U^T."""

    diag: Diagonal
    U: jax.Array
    S: jax.Array  # vector
    scale: float


@dataclass
class BlockDiagonalIR(OperatorIR):
    """IR for BlockDiagonal operators."""

    blocks: list[LinearOperator]
