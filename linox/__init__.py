"""Package for Linear operators in JAX."""

__version__ = "0.0.1"

# Import functions from _arithmetic module
from ._arithmetic import (
    congruence_transform,
    diagonal,
    is_square,
    is_symmetric,
    linverse,
    lsqrt,
    symmetrize,
    transpose,
)

# Import classes from other modules
from ._block import BlockDiagonal, BlockMatrix, BlockMatrix2x2
from ._eigen import EigenD
from ._kronecker import Kronecker
from ._linear_operator import LinearOperator
from ._low_rank import (
    IsotropicScalingPlusSymmetricLowRank,
    LowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
)
from ._matrix import Diagonal, Identity, Matrix, Ones, Scalar, Zero
from ._permutation import Permutation

# Explicitly declare public API
__all__ = [
    "BlockDiagonal",
    "BlockMatrix",
    "BlockMatrix2x2",
    "Diagonal",
    "EigenD",
    "Identity",
    "IsotropicScalingPlusSymmetricLowRank",
    "Kronecker",
    "LinearOperator",
    "LowRank",
    "Matrix",
    "Ones",
    "Permutation",
    "PositiveDiagonalPlusSymmetricLowRank",
    "Scalar",
    "SymmetricLowRank",
    "Zero",
    "congruence_transform",
    "diagonal",
    "is_square",
    "is_symmetric",
    "linverse",
    "lsqrt",
    "symmetrize",
    "transpose",
]
