"""Package for Linear operators in JAX."""

__version__ = "0.0.1"

# Import functions from _arithmetic module
from ._arithmetic import (
    diagonal,
    is_square,
    is_symmetric,
    linverse,
    symmetrize,
    transpose,
    lsqrt,
)

# Import classes from other modules
from ._diagonal_plus_low_rank import IsotropicScalingPlusLowRank
from ._eigen import EigenD
from ._linear_operator import LinearOperator
from ._matrix import Identity, Matrix, Ones, Zero, Diagonal
from ._permutation import Permutation

# Explicitly declare public API
__all__ = [
    "Diagonal",
    "EigenD",
    "Identity",
    "IsotropicScalingPlusLowRank",
    "LinearOperator",
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
