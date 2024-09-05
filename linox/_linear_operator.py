from typing import Union

import jax.numpy as jnp

from linox import _utils as utils
from linox._typing import DTypeLike, ScalarLike, ShapeLike

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

    See Also:
    --------
    aslinop : Transform into a LinearOperator

    Notes:
    -----
    -   A subclass is only required to implement :meth:`_matmat`. Additionally, other
        methods like :meth:`_solve`, :meth:`_inverse`, :meth:`_transpose`,
        :meth:`_cholesky`, or :meth:`_det` should be overwritten if more performant
        implementations are available.
    -   Compared to probnum this implementation does not check for dtype to be numeric
    and not complexfloating.
    -   Matrix properties are tags.
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
    ) -> None:
        self.__shape = utils.as_shape(
            shape, ndim=2
        )  # TODO(2bys): Needs to be implemented.

        # DType
        self.__dtype = jnp.dtype(dtype)
        # possible type checks: numeric, not complexfloating
        self.mv = jnp.vectorize(self.mv, signature="(n,k)->(m,k)")

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear operator.

        Defined as a tuple of the output and input dimension of operator.
        """
        return self.__shape

    @property
    def ndim(self) -> int:
        """Number of linear operator dimensions.

        Defined analogously to numpy.ndarray.ndim. # TODO(2bys): Check with jnp.ndarray.ndim.
        """
        return 2

    @property
    def size(self) -> int:
        """Product of the :attr:`shape` entries."""
        return self.__shape[0] * self.__shape[1]

    @property
    def dtype(self) -> jnp.dtype:
        """Data type of the linear operator.

        # TODO(2bys): Check how this could be expanded to pytrees.
        """
        return self.__dtype

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with shape={self.shape}, dtype={self.dtype}>"
        )

    ########################################################################
    # Default Methods that should be overwritten
    ########################################################################

    def todense(self) -> jnp.ndarray:
        return self @ jnp.eye(self.shape[1], dtype=self.dtype)

    def mv(self, vector: jnp.ndarray) -> jnp.ndarray:
        return self.todense() @ vector

    def transpose(self) -> "LinearOperator":
        return self.todense().T

    @property
    def T(self) -> "LinearOperator":
        from linox._arithmetic import TransposedLinearOperator  # noqa: PLC0415

        return TransposedLinearOperator(self)

    ########################################################################
    # Arithmetic
    ########################################################################

    def __neg__(self) -> "LinearOperator":
        from ._arithmetic import lneg  # noqa: PLC0415

        return lneg(self)

    def __add__(
        self, other: "LinearOperator"
    ) -> "LinearOperator":  # Here the package uses a BinaryOperandType
        from ._arithmetic import ladd  # noqa: PLC0415

        return ladd(self, other)

    def __radd__(self, other: "LinearOperator") -> "LinearOperator":
        return self.__add__(other, self)

    def __sub__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import lsub  # noqa: PLC0415

        return lsub(self, other)

    def __rsub__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import lsub  # noqa: PLC0415

        return lsub(other, self)

    def __mul__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import lmul  # noqa: PLC0415

        return lmul(self, other)

    def __rmul__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import lmul  # noqa: PLC0415

        return lmul(other, self)

    def __matmul__(self, other: BinaryOperandType) -> "LinearOperator":
        # TODO(2bys): Add shape checks.
        from ._arithmetic import lmatmul  # noqa: PLC0415

        # Shape checks
        if len(other.shape) == 1:
            other = other[:, None]
            flatten = True
        else:
            flatten = False

        if other.shape[-2] != self.shape[-1]:
            msg = f"expected other.shape[-2] to be {self.shape[-1]}, got {other.shape[-2]} instead."  # noqa: E501
            raise ValueError(msg)

        # if len(other.shape) > 2:
        #     msg = "Only 2D arrays are supported."
        #     raise ValueError(msg)

        if other.dtype != self.dtype:
            msg = f"expected other.dtype to be {self.dtype}, got {other.dtype} instead."
            raise ValueError(msg)

        res = lmatmul(self, other)
        return res if not flatten else res[..., 0]

    def __rmatmul__(self, other: BinaryOperandType) -> "LinearOperator":
        # TODO(2bys): Add shape checks.
        from ._arithmetic import lmatmul  # noqa: PLC0415

        # Shape checks
        # if len(other.shape) == 1:
        #     other = other[:, None]
        #     flatten = True
        # else:
        #     flatten = False

        if other.shape[-1] != self.shape[-2]:
            msg = (
                f"expected other.shape[-1] to be {other.shape[-1]}, "
                f"got {self.shape[-2]} instead."
            )
            raise ValueError(msg)

        if len(other.shape) > 2:
            msg = "Only 2D arrays are supported."
            raise ValueError(msg)

        return lmatmul(other, self)

    def __call__(self, arr: BinaryOperandType) -> "LinearOperator":
        return self @ arr
