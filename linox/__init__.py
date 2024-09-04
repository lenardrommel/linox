"""Package for Linear operators in JAX."""

__version__ = "0.0.1"

# Import functions from _arithmetic module
from ._arithmetic import (
    diagonal,
    inverse,
    is_square,
    is_symmetric,
    symmetrize,
    transpose,
)

# Import classes from other modules
from ._diagonal_plus_low_rank import IsotropicScalingPlusLowRank
from ._eigen import EigenD
from ._linear_operator import LinearOperator
from ._matrix import Identity, Matrix, Ones, Zero
from ._permutation import Permutation

# Explicitly declare public API
__all__ = [
    "EigenD",
    "Identity",
    "IsotropicScalingPlusLowRank",
    "LinearOperator",
    "Matrix",
    "Ones",
    "Permutation",
    "Zero",
    "diagonal",
    "inverse",
    "is_square",
    "is_symmetric",
    "symmetrize",
    "transpose",
]
