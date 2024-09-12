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


def diagonal(a: LinearOperator) -> ArithmeticType:
    return jnp.diag(a.todense())


def transpose(a: LinearOperator) -> ArithmeticType:
    return TransposedLinearOperator(a)


@plum.dispatch
def linverse(a: LinearOperator) -> ArithmeticType:
    return InverseLinearOperator(a)


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
    """Linear operator scaled with a scalar."""

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


# inverse special behavior:
# ScaledLinearOperator(inverse(operator), inverse(Scalar))
def _broadcast_shapes(shapes: Iterable[ShapeLike]) -> ShapeLike:
    try:
        return jnp.broadcast_shapes(*shapes)
    except ValueError:
        msg = f"Shapes {shapes} cannot be broadcasted."
        raise ValueError(msg)  # noqa: B904


class AddLinearOperator(LinearOperator):
    """A linear operator formed by adding two other linear operators together."""

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


class ProductLinearOperator(LinearOperator):
    """(Operator) Product of linear operators."""

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


# not properly tested
class CongruenceTransform(ProductLinearOperator):
    r""":math:`A B A^\top`."""

    def __init__(self, A: ArithmeticType, B: ArithmeticType) -> None:
        self._A = utils.as_linop(A)
        self._B = utils.as_linop(B)

        super().__init__(self._A, self._B, self._A.T)

        self.is_symmetric = self._B.is_symmetric
        self.is_positive_definite = self._B.is_positive_definite

    def transpose(self) -> LinearOperator:
        return CongruenceTransform(self._A, self._B.T)


@plum.dispatch
def congruence_transform(A: ArithmeticType, B: ArithmeticType) -> LinearOperator:
    return CongruenceTransform(A, B)


class TransposedLinearOperator(LinearOperator):
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


# NOT TESTED
class InverseLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator) -> None:
        self.operator = operator
        super().__init__(shape=operator.shape, dtype=operator.dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return linverse(self.operator) @ arr

    def todense(self) -> jax.Array:
        return jnp.linalg.inv(self.operator.todense())

    def transpose(self) -> LinearOperator:
        return InverseLinearOperator(self.operator.transpose())


# @inverse.dispatch
# def _(a: InverseLinearOperator):
#     return a.operatorm
