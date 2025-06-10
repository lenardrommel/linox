r"""Arithmetic operations for linear operators.

This module implements various arithmetic operations for linear operators, including:

- :class:`ScaledLinearOperator`: Represents :math:`\alpha A` for scalar :math:`\alpha`
    and operator :math:`A`
- :class:`AddLinearOperator`: Represents :math:`A_1 + A_2 + \ldots + A_n` for
    operators :math:`A_i`
- :class:`ProductLinearOperator`: Represents :math:`A_1A_2\ldots A_n` for operators
    :math:`A_i`
- :class:`CongruenceTransform`: Represents :math:`ABA^T` for operators
    :math:`A` and :math:`B`
- :class:`TransposedLinearOperator`: Represents :math:`A^T` for operator :math:`A`
- :class:`InverseLinearOperator`: Represents :math:`A^{-1}` for operator :math:`A`

These operators can be combined to form complex linear transformations while maintaining
efficient computation through lazy evaluation.
"""

import operator
from collections.abc import Iterable
from functools import reduce

import jax
import jax.numpy as jnp
import plum

from linox import utils
from linox._linear_operator import LinearOperator
from linox.typing import ScalarLike, ShapeLike

ArithmeticType = LinearOperator | jax.Array


# all arithmetic functions
@plum.dispatch
def ladd(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, b)


@plum.dispatch
def lsub(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, -b)


@plum.dispatch
def lmul(a: ScalarLike | jax.Array, b: LinearOperator) -> LinearOperator:
    return ScaledLinearOperator(scalar=a, operator=b)


@plum.dispatch
def lmatmul(a: LinearOperator, b: LinearOperator) -> ArithmeticType:
    return ProductLinearOperator(a, b)


# @plum.dispatch
def lneg(a: LinearOperator) -> LinearOperator:
    return ScaledLinearOperator(operator=a, scalar=-1)


@plum.dispatch
def lsqrt(a: LinearOperator) -> LinearOperator:
    msg = f"Square root of {type(a)} not implemented."
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


@plum.dispatch
def diagonal(a: LinearOperator) -> ArithmeticType:
    print(f"Warning: Linear operator {a} is densed for diagonal computation.")  # noqa: T201
    return jnp.diag(a.todense())


def transpose(a: LinearOperator) -> ArithmeticType:
    return TransposedLinearOperator(a)


@plum.dispatch
def linverse(a: LinearOperator) -> ArithmeticType:
    return InverseLinearOperator(a)


@plum.dispatch
def lpinverse(a: LinearOperator) -> ArithmeticType:
    return PseudoInverseLinearOperator(a)


@plum.dispatch
def leigh(a: LinearOperator) -> ArithmeticType:
    return jnp.linalg.eigh(a.todense())


@plum.dispatch
def lqr(a: LinearOperator) -> tuple[jax.Array, jax.Array]:
    """
    QR decomposition of a linear operator.
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    return jnp.linalg.qr(a.todense())


@plum.dispatch
def lsolve(a: LinearOperator, b: jax.Array) -> jax.Array:
    """Solve the linear system Ax = b."""
    if a.shape[-1] != b.shape[0]:
        msg = f"Shape mismatch: {a.shape} and {b.shape}"
        raise ValueError(msg)
    return jax.scipy.linalg.solve(a.todense(), b, assume_a="sym")


@plum.dispatch
def lpsolve(a: LinearOperator, b: jax.Array, rtol=1e-8) -> jax.Array:
    """Solve the linear system Ax = b."""
    if a.shape[-1] != b.shape[0]:
        msg = f"Shape mismatch: {a.shape} and {b.shape}"
        raise ValueError(msg)

    return jnp.linalg.pinv(a.to_dense(), rtol) @ b


@plum.dispatch
def lcholesky(a: LinearOperator) -> jax.Array:
    return jnp.linalg.cholesky(a.todense())


@plum.dispatch
def ldet(a: LinearOperator) -> jax.Array:
    """Compute the determinant of a linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)
    if not is_symmetric(a):
        msg = f"Operator {a} is not symmetric."
        raise ValueError(msg)

    return jnp.linalg.det(a.todense())


@plum.dispatch
def slogdet(a: LinearOperator) -> tuple[jax.Array, jax.Array]:
    """
    Compute the sign and log determinant of a linear operator.

    Returns:
        sign: Sign of the determinant
        logdet: Logarithm of the determinant
    """
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)

    return jnp.linalg.slogdet(a.todense())


# --------------------------------------------------------------------------- #
# Linear Operator checks
# --------------------------------------------------------------------------- #


def is_square(a: LinearOperator) -> bool:
    return a.shape[-1] == a.shape[-2]


def is_symmetric(a: LinearOperator) -> bool:
    return jnp.allclose(a.todense(), a.todense().T)


# --------------------------------------------------------------------------- #
# Linear Operator - Enforce tags
# --------------------------------------------------------------------------- #


def symmetrize(a: LinearOperator) -> ArithmeticType:
    return 0.5 * (a + a.transpose())


# --------------------------------------------------------------------------- #
# Dispatch - lmatmul
# --------------------------------------------------------------------------- #


@lmatmul.dispatch
def _(a: LinearOperator, b: jax.Array) -> jax.Array:
    return a._matmul(b)  # noqa: SLF001


@lmatmul.dispatch
def _(a: jax.Array, b: LinearOperator) -> LinearOperator:
    from linox._matrix import Matrix  # noqa: PLC0415

    return Matrix(a) @ b


# --------------------------------------------------------------------------- #
# Linear Operators
# --------------------------------------------------------------------------- #


class ScaledLinearOperator(LinearOperator):
    r"""Linear operator scaled with a scalar.

    For a linear operator :math:`A` and scalar :math:`\alpha`, this represents
    :math:`\alpha A` where :math:`(\alpha A)x = \alpha(Ax)` for any vector :math:`x`

    Args:
        operator: A linear operator to be scaled
        scalar: A scalar value to multiply the operator with
    """

    def __init__(self, operator: LinearOperator, scalar: ScalarLike) -> None:
        self.operator = utils.as_linop(operator)
        scalar = jnp.asarray(scalar)
        dtype = jnp.result_type(operator.dtype, scalar.dtype)
        self.scalar = utils.as_scalar(scalar, dtype)
        super().__init__(shape=operator.shape, dtype=dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return (self.operator @ arr) * self.scalar

    def todense(self) -> jax.Array:
        return self.operator.todense() * self.scalar

    def transpose(self) -> LinearOperator:
        return self.operator.transpose() * self.scalar

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.operator, self.scalar)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "ScaledLinearOperator":
        del aux_data
        operator, scalar = children
        return cls(operator=operator, scalar=scalar)


@lsqrt.dispatch
def _(a: ScaledLinearOperator) -> LinearOperator:
    return ScaledLinearOperator(lsqrt(a.operator), jnp.sqrt(a.scalar))


@diagonal.dispatch
def _(a: ScaledLinearOperator) -> jax.Array:
    return a.scalar * diagonal(a.operator)


@lsolve.dispatch
def _(a: ScaledLinearOperator, b: jax.Array) -> jax.Array:
    return lsolve(a.operator, b) / a.scalar


@lpsolve.dispatch
def _(a: ScaledLinearOperator, b: jax.Array, rtol=1e-8) -> jax.Array:
    return lpsolve(a.operator, b) / a.scalar


@lcholesky.dispatch
def _(a: ScaledLinearOperator) -> ScaledLinearOperator:
    return ScaledLinearOperator(lcholesky(a.operator), jnp.sqrt(a.scalar))


@ldet.dispatch
def _(a: ScaledLinearOperator) -> jax.Array:
    """Compute the determinant of a scaled linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)
    if not is_symmetric(a):
        msg = f"Operator {a} is not symmetric."
        raise ValueError(msg)

    return a.scalar ** a.shape[-1] * ldet(a.operator)


@slogdet.dispatch
def _(a: ScaledLinearOperator) -> tuple[jax.Array, jax.Array]:
    """Compute the sign and log determinant of a scaled linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)

    sign, logdet = slogdet(a.operator)
    return sign, logdet + (jnp.log(a.scalar) * a.shape[-1])


# inverse special behavior:
# ScaledLinearOperator(inverse(operator), inverse(Scalar))
def _broadcast_shapes(shapes: Iterable[ShapeLike]) -> ShapeLike:
    try:
        return jnp.broadcast_shapes(*shapes)
    except ValueError:
        msg = f"Shapes {shapes} cannot be broadcasted."
        raise ValueError(msg)  # noqa: B904


class AddLinearOperator(LinearOperator):
    r"""A linear operator formed by adding two or more linear operators together.

    For linear operators :math:`A_1, A_2, \ldots, A_n`, this represents
    :math:`A_1 + A_2 + \ldots + A_n`where
    :math:`(A_1 + A_2 + \ldots + A_n)x = A_1x + A_2x + \ldots + A_nx`
    for any vector :math:`x`

    Args:
        *operator_list: Variable number of linear operators to be added
    """

    def __init__(self, *operator_list: ArithmeticType) -> None:
        self.operator_list = [
            utils.as_linop(o)
            if isinstance(op, AddLinearOperator)
            else utils.as_linop(op)
            for op in operator_list
            for o in (op.operator_list if isinstance(op, AddLinearOperator) else [op])
        ]
        shape = _broadcast_shapes([op.shape for op in self.operator_list])
        super().__init__(shape=shape, dtype=self.operator_list[0].dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return reduce(
            operator.add,
            (op @ arr for op in reversed(self.operator_list)),
        )

    def todense(self) -> jax.Array:
        return reduce(operator.add, (op.todense() for op in self.operator_list))

    def transpose(self) -> "AddLinearOperator":
        return AddLinearOperator(*(op.transpose() for op in self.operator_list))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = tuple(self.operator_list)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "AddLinearOperator":
        del aux_data
        return cls(*children)


class ProductLinearOperator(LinearOperator):
    r"""Product of linear operators.

    For linear operators :math:`A_1, A_2, \ldots, A_n`, this represents
    :math:`A_1A_2\ldots A_n` where :math:`(A_1A_2\ldots A_n)x = A_1(A_2(\ldots(A_nx)))`
    for any vector :math:`x`

    Args:
        *operator_list: Variable number of linear operators to be multiplied
    """

    def __init__(self, *operator_list: LinearOperator) -> None:
        self.operator_list = [
            utils.as_linop(o)
            if isinstance(op, ProductLinearOperator)
            else utils.as_linop(op)
            for op in operator_list
            for o in (
                op.operator_list if isinstance(op, ProductLinearOperator) else [op]
            )
        ]
        batch_shape = _broadcast_shapes([op.shape[:-2] for op in self.operator_list])
        self.__check_init__()
        shape = utils.as_shape((
            *batch_shape,
            self.operator_list[0].shape[-2],
            self.operator_list[-1].shape[-1],
        ))
        super().__init__(shape=shape, dtype=self.operator_list[0].dtype)

    def __check_init__(self) -> None:  # noqa: PLW3201
        for i, op1 in enumerate(self.operator_list[:-1]):
            op2 = self.operator_list[i + 1]
            if op1.shape[-1] != op2.shape[-2]:
                msg = (
                    f"Shape mismatch: Cannot multiply linear operators with shapes "
                    f"operator 1: ({op1.shape}) "
                    f"operator 2: ({op2.shape})"
                )
                raise ValueError(msg)
            if op1.dtype != op2.dtype:
                msg = (
                    f"Type of operator 1: {op1.dtype} and operator 2: {op2.dtype} "
                    f"mismatch."
                )
                raise TypeError(msg)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return reduce(lambda x, y: y @ x, [arr, *reversed(self.operator_list)])

    def transpose(self) -> "ProductLinearOperator":
        return ProductLinearOperator(
            *(op.transpose() for op in reversed(self.operator_list))
        )

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = tuple(self.operator_list)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, any], children: tuple[any, ...]
    ) -> "ProductLinearOperator":
        del aux_data
        return cls(*children)


# not properly tested
class CongruenceTransform(ProductLinearOperator):
    r"""Congruence transformation of linear operators.

    For linear operators :math:`A` and :math:`B`, this represents :math:`ABA^T`
    where :math:`(ABA^T)x = A(B(A^T x))` for any vector :math:`x`

    Args:
        A: First linear operator
        B: Second linear operator
    """

    def __init__(self, A: ArithmeticType, B: ArithmeticType) -> None:
        self._A = utils.as_linop(A)
        self._B = utils.as_linop(B)

        super().__init__(self._A, self._B, self._A.T)

    def transpose(self) -> LinearOperator:
        return CongruenceTransform(self._A, self._B.T)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self._A, self._B)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "CongruenceTransform":
        del aux_data
        A, B = children
        return cls(A=A, B=B)


@plum.dispatch
def congruence_transform(A: ArithmeticType, B: ArithmeticType) -> LinearOperator:
    return CongruenceTransform(A, B)


class TransposedLinearOperator(LinearOperator):
    r"""Transpose of a linear operator.

    For a linear operator :math:`A`, this represents :math:`A^T`
    where :math:`(A^T)_{ij} = A_{ji}` for all :math:`i,j`

    Args:
        operator: A linear operator to be transposed
    """

    def __init__(self, operator: LinearOperator) -> None:
        self.operator = utils.as_linop(operator)
        batch_shape = operator.shape[:-2]
        super().__init__(
            shape=(*batch_shape, operator.shape[-1], operator.shape[-2]),
            dtype=operator.dtype,
        )

    def _matmul(self, arr: jnp.array) -> jax.Array:
        return self.operator.transpose() @ arr

    def todense(self) -> jax.Array:
        return self.operator.todense().swapaxes(-1, -2)

    def transpose(self) -> LinearOperator:
        return self.operator

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.operator,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "TransposedLinearOperator":
        del aux_data
        (operator,) = children
        return cls(operator=operator)


# NOT TESTED
class InverseLinearOperator(LinearOperator):
    """Inverse of a linear operator.

    For a linear operator :math:`A`, this represents :math:`A^{-1}`
    where :math:`A^{-1}` is the unique operator such that :math:`AA^{-1} = A^{-1}A = I`
    where :math:`I` is the identity operator

    Args:
        operator: A linear operator to be inverted
    """

    def __init__(self, operator: LinearOperator) -> None:
        self.operator = operator
        super().__init__(shape=operator.shape, dtype=operator.dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return linverse(self.operator) @ arr

    def todense(self) -> jax.Array:
        return jnp.linalg.inv(self.operator.todense())

    def transpose(self) -> LinearOperator:
        return InverseLinearOperator(self.operator.transpose())

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.operator,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "InverseLinearOperator":
        del aux_data
        (operator,) = children
        return cls(operator=operator)


@ldet.dispatch
def ldet(a: InverseLinearOperator) -> jax.Array:
    """Compute the determinant of a linear operator."""
    if not is_square(a):
        msg = f"Operator {a} is not square."
        raise ValueError(msg)
    if not is_symmetric(a):
        msg = f"Operator {a} is not symmetric."
        raise ValueError(msg)

    return 1 / ldet(a.operator)


# TODO: write tests for this class
class CongruenceTransform(ProductLinearOperator):
    r""":math:`A B A^\top`."""

    def __init__(self, A: ArithmeticType, B: ArithmeticType) -> None:
        self._A = utils.as_linop(A)
        self._B = utils.as_linop(B)

        super().__init__(self._A, self._B, self._A.T)

    def transpose(self) -> LinearOperator:
        return CongruenceTransform(self._A, self._B.T)


@plum.dispatch
def congruence_transform(A: ArithmeticType, B: ArithmeticType) -> LinearOperator:
    return CongruenceTransform(A, B)


# TODO: write tests for this class
class PseudoInverseLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator, tol: float = 1e-6) -> None:
        self.operator = operator
        self.tol = tol
        super().__init__(shape=operator.shape, dtype=operator.dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return lpinverse(self.operator) @ arr

    def todense(self) -> jax.Array:
        return jnp.linalg.pinv(self.operator.todense(), rtol=self.tol)


# Register all linear operators as PyTrees
jax.tree_util.register_pytree_node_class(ScaledLinearOperator)
jax.tree_util.register_pytree_node_class(AddLinearOperator)
jax.tree_util.register_pytree_node_class(ProductLinearOperator)
jax.tree_util.register_pytree_node_class(CongruenceTransform)
jax.tree_util.register_pytree_node_class(TransposedLinearOperator)
jax.tree_util.register_pytree_node_class(InverseLinearOperator)
jax.tree_util.register_pytree_node_class(PseudoInverseLinearOperator)
