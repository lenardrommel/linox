"""The :class:`LinearOperator` base class."""

# _linear_operator.py

import operator
from functools import reduce
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from linox import config, utils
from linox._types import DTypeLike, ScalarLike, ShapeLike

BinaryOperandType = Union["LinearOperator", ScalarLike, jnp.ndarray]


class LinearOperator:
    r"""Abstract base class for `matrix-free` finite-dimensional linear operators.

    It follows in most parts the implementation of `probnum.linops.LinearOperator`
    from ProbNum
    (see also https://github.com/probabilistic-numerics/probnum/blob/main/src/probnum/linops/_linear_operator.py).

    Design choices:
    :class:`LinearOperator`\ s are defined to behave like a :class:`jax.numpy.ndarray`
    and thus, they


    * have :attr:`shape`, :attr:`dtype`, :attr:`ndim`, and :attr:`size` attributes,
    * can be matrix multiplied (:code:`@`) with a :class:`numpy.ndarray` from left and
      right, following the same broadcasting rules as :func:`numpy.matmul`,
    * can be multiplied (:code:`*`) by a scalar from the left and the right,
    * can be added to, subtracted from and matrix multiplied (:code:`@`) with other
      :class:`LinearOperator` instances with appropriate :attr:`shape`,
    * can be transposed (:attr:`T` or :meth:`transpose`), and they
    * can be type-cast (:meth:`astype`).

    This is mostly implemented lazily, i.e. the result of these operations is a new,
    composite :class:`LinearOperator`, that defers linear operations to the original
    operators and combines the results.

    Parameters
    ----------
    shape: Tuple[int]
        Shape of the linear operator.
    dtype: Type

    See Also
    --------
    aslinop : Transform into a LinearOperator

    Notes
    -----
    -   A subclass is only required to implement :meth:`_matmat`. Additionally, other
        methods like :meth:`_solve`, :meth:`_inverse`, :meth:`_transpose`,
        :meth:`_cholesky`, or :meth:`_det` should be overwritten if more performant
        implementations are available.
    -   Compared to probnum this implementation does not check for dtype to be numeric
    and not complexfloating.
    -   Matrix properties are tags.

    Important:
    ----------
    -  (...batch..., n, m) is the general shape assumption.
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
    ) -> None:
        self.__shape = utils.as_shape(shape, ndim=len(shape))

        self.__dtype = jnp.dtype(dtype)

    @property
    def shape(self) -> tuple[int]:
        """Shape of the linear operator.

        Defined as a tuple of the output and input dimension of operator.
        """
        return self.__shape

    @property
    def batch_shape(self) -> tuple[int]:
        """Shape of the batch dimensions of the linear operator."""
        return self.__shape[:-2]

    @property
    def ndim(self) -> int:
        """Number of linear operator dimensions.

        Defined analogously to numpy.ndarray.ndim.
        TODO(2bys): Check with jnp.ndarray.ndim.
        """
        return len(self.__shape)

    @property
    def batch_ndim(self) -> int:
        """Number of batch dimensions of the linear operator."""
        return len(self.__shape[:-2])

    @property
    def size(self) -> int:
        """Product of the :attr:`shape` entries."""
        return reduce(operator.mul, self.__shape, 1)

    @property
    def dtype(self) -> jnp.dtype:
        """Data type of the linear operator."""
        return self.__dtype

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric."""
        return False

    @property
    def is_psd(self) -> bool:
        """Whether the operator is positive semi-definite."""
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with shape={self.shape}, dtype={self.dtype}>"

    def graph(self, **kwargs):
        """Return this operator's structure as a tree of :class:`LinOpNode`."""
        from linox.utils.debug import linop_graph

        return linop_graph(self, **kwargs)

    def graph_str(self, **kwargs):
        """Return this operator's structure as a printable tree."""
        return self.graph(**kwargs).pretty()

    ########################################################################
    # Default Methods that should be overwritten
    ########################################################################

    def todense(self) -> jnp.ndarray:
        """Materialize this operator as a dense array."""
        if config.get_warn_on_densify():
            config.warn(f"Linear operator {self} is densed.", prefix="PerformanceWarning")

        config.emit(
            config.DebugEvent(
                kind="densify",
                msg="todense() called",
                op_type=type(self).__name__,
                op_id=id(self),
                shape=getattr(self, "shape", None),
                dtype=getattr(self, "dtype", None),
            )
        )
        return self @ jnp.eye(self.shape[-1], dtype=self.dtype)

    def _todense(self) -> jnp.ndarray:
        msg = "Subclasses must implement _todense"
        raise NotImplementedError(msg)

    def _matmul(self, other: jnp.ndarray) -> jnp.ndarray:
        return self._todense() @ other

    def transpose(self) -> "LinearOperator":
        """Return the transpose of this operator.

        Subclasses that know their own structure should override this and
        return it (``Diagonal`` returns itself, ``Kronecker`` returns a
        ``Kronecker`` of transposed factors, and so on). The default is a lazy
        wrapper that derives the adjoint from the forward matvec, so it never
        materialises the dense matrix.
        """
        from linox.operators.arithmetic import (
            TransposedLinearOperator,
        )

        return TransposedLinearOperator(self)

    @property
    def T(self) -> "LinearOperator":
        """Return the transpose of this operator, preserving structure where possible."""
        from linox.operators.arithmetic import (
            TransposedLinearOperator,
        )

        # Prefer a subclass's structured transpose (e.g. Diagonal -> itself,
        # Sym/PSD -> itself) so the operator's structure survives `.T`. The
        # base implementation returns a dense array rather than an operator,
        # in which case fall back to the lazy wrapper.
        transposed = self.transpose()
        if isinstance(transposed, LinearOperator):
            return transposed
        return TransposedLinearOperator(self)

    ########################################################################
    # Arithmetic
    ########################################################################

    def __neg__(self) -> "LinearOperator":
        from .arithmetic import lneg

        return lneg(self)

    def __add__(self, other: "LinearOperator") -> "LinearOperator":  # Here the package uses a BinaryOperandType
        from .arithmetic import ladd

        return ladd(self, other)

    def __radd__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import ladd

        # Addition is commutative, so reuse the forward dispatch: it carries
        # the (LinearOperator, Array) methods that the reversed argument order
        # would not resolve.
        return ladd(self, other)

    def __sub__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import lsub

        return lsub(self, other)

    def __rsub__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import lsub

        return lsub(other, self)

    def __mul__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import lmul

        return lmul(self, other)

    def __rmul__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import lmul

        return lmul(other, self)

    def __truediv__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import ldiv

        return ldiv(self, other)

    def __matmul__(self, other: BinaryOperandType) -> "LinearOperator":
        from .arithmetic import lmatmul

        flatten = False
        operand = other
        if isinstance(other, (jax.Array, np.ndarray)):
            operand = jnp.asarray(other)
            if operand.ndim == 1:
                operand = operand[:, None]
                flatten = True

        res = lmatmul(self, operand)
        if (
            not flatten
            and isinstance(res, jax.Array)
            and res.ndim >= 2
            and res.shape[-2] != self.shape[-2]
            and hasattr(operand, "shape")
            and res.shape[-2] == operand.shape[-1]
            and res.shape[-1] == self.shape[-2]
        ):
            res = jnp.swapaxes(res, -1, -2)
        return res if not flatten else res[..., 0]

    def __rmatmul__(self, other: BinaryOperandType) -> "LinearOperator":
        from linox.operators.arithmetic import lmatmul

        # lazy evaluation
        isLazyEvaluation = True

        if other.shape[-1] != self.shape[-2]:
            msg = f"expected other.shape[-1] to be {other.shape[-1]}, got {self.shape[-2]} instead."
            raise ValueError(msg)

        if len(other.shape) > 2:
            msg = "Only 2D arrays are supported."
            raise ValueError(msg)

        if len(other.shape) == 1:
            other = other[None, :]
            isLazyEvaluation = False

        res = lmatmul(other, self)
        return res if isLazyEvaluation else (res[0, :] if isinstance(res, jnp.ndarray) else res._todense()[0])

    def __call__(self, arr: BinaryOperandType) -> "LinearOperator":
        """Apply this operator, equivalent to ``self @ arr``."""
        return self @ arr

    @classmethod
    def tree_flatten(cls) -> tuple[tuple[any, ...], dict[str, any]]:
        """Default implementation for PyTree flattening.

        Subclasses should override this method to provide proper PyTree support.
        """
        children = ()  # No children by default
        aux_data = {}  # No auxiliary data by default
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "LinearOperator":
        """Default implementation for PyTree unflattening."""
        del children
        if cls is LinearOperator:
            msg = "Cannot unflatten the abstract LinearOperator class directly"
            raise TypeError(msg)
        return cls(**aux_data)
