"""Linox Operators."""

import jax

from .arithmetic import (
    AddLinearOperator,
    CongruenceTransform,
    InverseLinearOperator,
    ProductLinearOperator,
    PseudoInverseLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
)
from .base import LinearOperator
from .block import BlockDiagonal, BlockMatrix, BlockMatrix2x2
from .dense import Matrix
from .diagonal import Diagonal
from .eigen import EigenD
from .factor import CholeskyFactor, PSDFromFactor, Triangular
from .isotropic import IsotropicAdditiveLinearOperator
from .kron import Kronecker
from .lowrank import (
    IsotropicScalingPlusSymmetricLowRank,
    LowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
)
from .permutation import Permutation
from .special import Identity, Ones, Scalar, Zero
from .toeplitz import Toeplitz
from .wrappers import (
    PSD,
    SPD,
    Sym,
    assume_psd,
    assume_spd,
    assume_symmetric,
)

__all__ = [
    "AddLinearOperator",
    "BlockDiagonal",
    "BlockMatrix",
    "BlockMatrix2x2",
    "CholeskyFactor",
    "CongruenceTransform",
    "Diagonal",
    "EigenD",
    "Identity",
    "InverseLinearOperator",
    "IsotropicAdditiveLinearOperator",
    "IsotropicScalingPlusSymmetricLowRank",
    "Kronecker",
    "LinearOperator",
    "LowRank",
    "Matrix",
    "Ones",
    "PSD",
    "PSDFromFactor",
    "Permutation",
    "PositiveDiagonalPlusSymmetricLowRank",
    "ProductLinearOperator",
    "PseudoInverseLinearOperator",
    "SPD",
    "Scalar",
    "ScaledLinearOperator",
    "Sym",
    "SymmetricLowRank",
    "Toeplitz",
    "TransposedLinearOperator",
    "Triangular",
    "Zero",
    "assume_psd",
    "assume_spd",
    "assume_symmetric",
]


