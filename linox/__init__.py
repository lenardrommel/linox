# __init__.py

r"""`linox`: Linear operators in JAX.

This package provides a collection of linear operators for JAX, including:

- Basic operators: :class:`Matrix`, :class:`Identity`, :class:`Diagonal`,
    :class:`Scalar`, :class:`Zero`, :class:`Ones`
- Block operators: :class:`BlockMatrix`, :class:`BlockMatrix2x2`, :class:`BlockDiagonal`
- Low rank operators: :class:`LowRank`, :class:`SymmetricLowRank`,
    :class:`IsotropicScalingPlusSymmetricLowRank`,
    :class:`PositiveDiagonalPlusSymmetricLowRank`
- Special operators: :class:`Kronecker`, :class:`Permutation`, :class:`EigenD`

Common operations:
- Arithmetic: :func:`linverse`, :func:`lsqrt`, :func:`transpose`
- Properties: :func:`is_square`, :func:`is_symmetric`
- Transformations: :func:`congruence_transform`, :func:`diagonal`, :func:`symmetrize`

All operators support lazy evaluation and can be combined to form complex linear
transformations.
"""

__version__ = "0.0.1"

# Import functions from _arithmetic module
from ._arithmetic import (
    AddLinearOperator,
    InverseLinearOperator,
    ProductLinearOperator,
    PseudoInverseLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
    congruence_transform,
    diagonal,
    is_square,
    kron,
    lcholesky,
    ldet,
    leigh,
    linverse,
    lpinverse,
    lpsolve,
    lqr,
    lsolve,
    lsqrt,
    slogdet,
    svd,
    symmetrize,
    transpose,
)

# Import classes from other modules
from ._block import BlockDiagonal, BlockMatrix, BlockMatrix2x2
from ._eigen import EigenD
from ._isotropicadd import IsotropicAdditiveLinearOperator
from ._kernel import ArrayKernel
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
from ._toeplitz import Toeplitz
from .config import is_debug, set_debug
from .utils import allclose, todense

# Explicitly declare public API
__all__ = [
    "AddLinearOperator",
    "ArrayKernel",
    "BlockDiagonal",
    "BlockMatrix",
    "BlockMatrix2x2",
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
    "Permutation",
    "PositiveDiagonalPlusSymmetricLowRank",
    "ProductLinearOperator",
    "PseudoInverseLinearOperator",
    "Scalar",
    "ScaledLinearOperator",
    "SymmetricLowRank",
    "Toeplitz",
    "TransposedLinearOperator",
    "Zero",
    "allclose",
    "congruence_transform",
    "diagonal",
    "is_square",
    "kron",
    "lcholesky",
    "ldet",
    "leigh",
    "linverse",
    "lpinverse",
    "lpsolve",
    "lqr",
    "lsolve",
    "lsqrt",
    "slogdet",
    "svd",
    "symmetrize",
    "todense",
    "transpose",
    "is_debug",
    "set_debug",
]
