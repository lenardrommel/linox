"""Linox API - Public Functional Interface.

This module provides the main functional entry points for the Linox library.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import config
from .config import is_debug, set_debug
from .linalg import functions as _functions_module
from .linalg import spectral as _spectral_module
from .linalg import trace as _trace_module

# Arithmetic operators (for export compatibility)
from .operators.arithmetic import (
    AddLinearOperator,
    InverseLinearOperator,  # Compatibility aliases
    ProductLinearOperator,
    PseudoInverseLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
    congruence_transform,
    diagonal,
    is_hermitian,
    is_square,
    is_symmetric,
    lcholesky,
    ldet,
    lexp,
    linverse,
    llog,
    lpinverse,
    lpow,
    lpsolve,
    lqr,
    symmetrize,
)
from .operators.arithmetic import InverseLinearOperator as Inverse  # Alias for inv()
from .operators.arithmetic import PseudoInverseLinearOperator as PseudoInverse
from .operators.arithmetic import leigh as _leigh_impl
from .operators.arithmetic import lsolve as _lsolve_impl
from .operators.arithmetic import lsqrt as _lsqrt_impl
from .operators.arithmetic import (
    svd as _svd_impl,
)

# Internal imports
from .operators.base import LinearOperator
from .operators.block import BlockDiagonal, BlockMatrix, BlockMatrix2x2

# Matrix classes
from .operators.dense import Matrix
from .operators.diagonal import Diagonal
from .operators.eigen import EigenD
from .operators.isotropic import IsotropicAdditiveLinearOperator
from .operators.kron import Kronecker
from .operators.lowrank import (
    IsotropicScalingPlusSymmetricLowRank,
    LowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
)
from .operators.permutation import Permutation
from .operators.special import Identity, Ones, Scalar, Zero
from .operators.toeplitz import Toeplitz
from .operators.wrappers import (
    PSD,
    SPD,
    Sym,
    assume_psd,
    assume_spd,
    assume_symmetric,
)
from .utils import array as _array_module
from .utils.array import _broadcast_shapes, allclose
from .utils.validation import ValidationError, validate

# Type aliases
ArrayLike = jax.Array | Any
LinearyOperatorLike = LinearOperator | ArrayLike
Int = int


__all__ = [
    "PSD",
    "SPD",
    # Arithmetic Classes
    "AddLinearOperator",
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
    # Core Classes
    "LinearOperator",
    "LowRank",
    # Classes Exported (Compatibility/Typing)
    "Matrix",
    "Ones",
    "Permutation",
    "PositiveDiagonalPlusSymmetricLowRank",
    "ProductLinearOperator",
    "PseudoInverseLinearOperator",
    "Scalar",
    "ScaledLinearOperator",
    "Sym",
    "SymmetricLowRank",
    "Toeplitz",
    "TransposedLinearOperator",
    # Utils / Debug / Misc
    "ValidationError",
    "Zero",
    "_broadcast_shapes",
    "allclose",
    "as_linop",
    "assume_psd",
    "assume_spd",
    "assume_symmetric",
    "block_diag",
    "bmat",
    "congruence_transform",
    "det",
    "diag",
    "diagonal",
    # Linalg Functions
    "eigh",
    "exp",
    # Creation
    "eye",
    "inv",
    "inverse",
    "is_debug",
    "is_hermitian",
    "is_square",
    "is_symmetric",
    "kron",
    "lcholesky",
    "ldet",
    "leigh",
    "lexp",
    "linverse",
    "llog",
    "log",
    "logdet",
    "lpinverse",
    "lpow",
    "lpsolve",
    "lqr",
    "lsolve",
    "lsqrt",
    "ones",
    "pinv",
    "pinverse",
    "pow",
    "set_debug",
    "slogdet",
    "solve",
    # Element-wise / Functions
    "sqrt",
    "svd",
    "symmetrize",
    "todense",
    "toeplitz",
    "trace",
    # Structure Operations
    "transpose",
    "validate",
    "zeros",
]


# --- Utilities ---


def ensure_linop(a: Any) -> LinearOperator:
    """Helper to ensure input is a LinearOperator."""
    return _array_module.as_linop(a)


def as_linop(a: Any) -> LinearOperator:
    """Convert object to LinearOperator."""
    return _array_module.as_linop(a)


def todense(a: LinearyOperatorLike) -> jax.Array:
    """Convert operator to dense matrix."""
    return _array_module.todense(a)


# --- Creation functions ---


def eye(N: int, M: int | None = None, k: int = 0, dtype: Any = None) -> LinearOperator:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If None, defaults to N.
        k: Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype: Data-type of the returned array.
    """
    if M is None:
        M = N

    if k == 0 and N == M:
        return Identity(N, dtype=dtype)

    # For off-diagonals or non-square, just use Matrix wrapping dense eye for now
    return Matrix(jnp.eye(N, M, k=k, dtype=dtype))


def zeros(dim: Int, shape: tuple[Int, Int] | None = None) -> LinearOperator:
    """Create a zero operator.

    Args:
        dim: Dimension or number of rows.
        shape: Optional shape tuple (rows, cols).
    """
    if shape is None:
        return Zero((dim, dim))
    return Zero(shape)


def ones(dim: Int, shape: tuple[Int, Int] | None = None) -> LinearOperator:
    """Create a ones operator."""
    if shape is None:
        return Ones((dim, dim))
    return Ones(shape)


def diag(v: jax.Array) -> LinearOperator:
    """Create a diagonal operator from a vector."""
    return Diagonal(v)


# --- Structure / Arithmetic wrappers ---


def transpose(a: LinearyOperatorLike) -> LinearOperator:
    """Return the transpose of a linear operator."""
    return ensure_linop(a).T


def inv(a: LinearyOperatorLike) -> LinearOperator:
    """Compute the inverse of a linear operator (lazy)."""
    return Inverse(ensure_linop(a))


def pinv(a: LinearyOperatorLike) -> LinearOperator:
    """Compute the pseudo-inverse of a linear operator (lazy)."""
    return PseudoInverse(ensure_linop(a))


def kron(a: LinearyOperatorLike, b: LinearyOperatorLike) -> LinearOperator:
    """Compute the Kronecker product of two linear operators."""
    return Kronecker(ensure_linop(a), ensure_linop(b))


def block_diag(*opers: LinearyOperatorLike) -> LinearOperator:
    """Construct a block diagonal operator from input operators."""
    return BlockDiagonal([ensure_linop(op) for op in opers])


def bmat(blocks: list[list[LinearyOperatorLike]]) -> LinearOperator:
    """Construct a block matrix from a list of lists of operators."""
    linop_blocks = [[ensure_linop(op) for op in row] for row in blocks]
    return BlockMatrix(linop_blocks)


def toeplitz(c: jax.Array, r: jax.Array | None = None) -> LinearOperator:
    """Construct a Toeplitz operator from column c and optional row r.

    If r is None, assumes symmetric Toeplitz (r = c).
    """
    if r is not None:
        msg = "Asymmetric Toeplitz not yet supported via simple wrapper."
        raise NotImplementedError(
            msg
        )
    return Toeplitz(c)


# --- Linear Algebra Functions (Canonical Wrappers) ---


def trace(a: LinearyOperatorLike, method: str = "auto", **kwargs) -> jax.Array:
    """Compute the trace of a linear operator.

    Args:
        a: Linear operator.
        method: Computation method ("auto", "exact", "hutchinson").
    """
    op = ensure_linop(a)
    m = config.resolve_method("trace", op, method)

    if m == "hutchinson":
        return _trace_module.trace(op, method="hutchinson", **kwargs)
    return _trace_module.trace(op, **kwargs)


def det(a: LinearyOperatorLike) -> jax.Array:
    """Compute determinant."""
    from linox.linalg.determinants import det as _det
    return _det(ensure_linop(a))


def slogdet(
    a: LinearyOperatorLike, method: str = "auto", **kwargs
) -> tuple[jax.Array, jax.Array]:
    """Compute sign and log of determinant.

    Args:
        a: Linear operator.
        method: Computation method ("auto", "exact", "slq").
    """
    from linox.linalg.determinants import slogdet as _slogdet

    op = ensure_linop(a)
    m = config.resolve_method("slogdet", op, method)
    return _slogdet(op, method=m, **kwargs)


def logdet(a: LinearyOperatorLike) -> jax.Array:
    """Compute log of determinant."""
    from linox.linalg.determinants import logdet as _logdet
    return _logdet(ensure_linop(a))


def solve(
    a: LinearyOperatorLike, b: jax.Array, method: str = "auto", **kwargs
) -> jax.Array:
    """Solve linear system Ax = b.

    Args:
        a: Linear operator.
        b: Right-hand side vector/matrix.
        method: Solver method ("exact", "lsmr", "cg", "auto").
    """
    op = ensure_linop(a)
    b = jnp.asarray(b)

    m = config.resolve_method("solve", op, method)

    if m == "exact":
        return _lsolve_impl(op, b, **kwargs)
    if m == "lsmr":
        from linox.linalg.approx.lsmr import lsmr_solve
        x, _ = lsmr_solve(op, b, **kwargs)
        return x
    if m in {"cg", "conjugate_gradient"}:
        x, _ = jax.scipy.sparse.linalg.cg(op, b, **kwargs)
        return x

    # If auto resolution returned something else or we fell through
    return _lsolve_impl(op, b, **kwargs)


def eigh(
    a: LinearyOperatorLike,
    k: Int | None = None,
    subset_by_index: tuple[Int, Int] | None = None,
    method: str = "auto",
    **kwargs,
) -> tuple[jax.Array, LinearOperator] | jax.Array:
    """Compute eigenvalues and eigenvectors.

    Args:
        a: Linear operator.
        k: Number of eigenvalues (for approx/partial).
        subset_by_index: Range of indices (start, end) for eigenvalues.
        method: "exact" or "lanczos".
    """
    m = config.resolve_method("eigh", ensure_linop(a), method)
    return _spectral_module.eigh(
        ensure_linop(a), k=k, subset_by_index=subset_by_index, method=m, **kwargs
    )


def svd(a: LinearyOperatorLike, **kwargs) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Singular Value Decomposition."""
    return _svd_impl(ensure_linop(a), **kwargs)


# --- Element-wise / Function Application ---


def sqrt(a: LinearyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Matrix square root.

    Args:
        a: Linear operator.
        method: "exact" or "lanczos"/"newton".
    """
    op = ensure_linop(a)
    m = config.resolve_method("sqrt", op, method)

    if m == "exact":
        return _lsqrt_impl(op)
    if m == "lanczos":
        return _functions_module.sqrt(op, method="lanczos", **kwargs)

    return _lsqrt_impl(op)


def log(a: LinearyOperatorLike, **kwargs) -> LinearOperator:
    """Matrix logarithm."""
    return _functions_module.log(ensure_linop(a), **kwargs)


def exp(a: LinearyOperatorLike, **kwargs) -> LinearOperator:
    """Matrix exponential."""
    return _functions_module.exp(ensure_linop(a), **kwargs)


def pow(a: LinearyOperatorLike, p: float, **kwargs) -> LinearOperator:
    """Matrix power."""
    return _functions_module.pow(ensure_linop(a), p, **kwargs)


# Backward compatibility aliases
leigh = _leigh_impl
lsolve = _lsolve_impl
lsqrt = _lsqrt_impl
linverse = linverse


# --- Implementation details for wrappers ---


def inverse(a: LinearyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Compute the inverse of a linear operator.

    Args:
        a: Linear operator.
        method: "exact" (lazy inverse) or "auto".
    """
    op = ensure_linop(a)
    m = config.resolve_method("inverse", op, method)

    if m == "exact":
        return linverse(op)

    if m == "approx":
        # Default to LSMR for approximate inverse
        m = "lsmr"

    # Return lazy inverse with specified solver method
    return InverseLinearOperator(op, method=m, solver_options=kwargs)


def pinverse(a: LinearyOperatorLike, method: str = "auto", **kwargs) -> LinearOperator:
    """Compute the pseudo-inverse of a linear operator."""
    op = ensure_linop(a)
    # Similar method resolution could apply
    return lpinverse(op)
