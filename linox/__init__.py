"""Package for Linear operators in JAX."""

__version__ = "0.0.1"

# Import functions from _arithmetic module
from ._arithmetic import (
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
from ._linear_operator import LinearOperator
from ._low_rank import IsotropicScalingPlusLowRank, LowRank
from ._matrix import Diagonal, Identity, Matrix, Ones, Zero
from ._permutation import Permutation

# Explicitly declare public API
__all__ = [
    "BlockDiagonal",
    "BlockMatrix",
    "BlockMatrix2x2",
    "Diagonal",
    "EigenD",
    "Identity",
    "IsotropicScalingPlusLowRank",
    "LinearOperator",
    "LowRank",
    "Matrix",
    "Ones",
    "Permutation",
    "Zero",
    "diagonal",
    "is_square",
    "is_symmetric",
    "linverse",
    "lsqrt",
    "symmetrize",
    "transpose",
]
