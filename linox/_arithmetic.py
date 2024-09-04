import operator
from functools import reduce

import jax
import jax.numpy as jnp
import plum

import linox._utils as utils
from linox._linear_operator import LinearOperator
from linox._matrix import Matrix
from linox._typing import ScalarLike, ScalarType

ArithmeticType = LinearOperator | jax.Array


# all arithmetic functions
# @plum.dispatch
def ladd(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, b)


# @plum.dispatch
def lsub(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, -b)


# @plum.dispatch
def lmul(a: ScalarType, b: LinearOperator) -> LinearOperator:
    return ScaledLinearOperator(scalar=a, operator=b)


@plum.dispatch
def lmatmul(a: LinearOperator, b: LinearOperator) -> ArithmeticType:
    return ProductLinearOperator(a, b)


# @plum.dispatch
def lneg(a: LinearOperator) -> LinearOperator:
    return ScaledLinearOperator(operator=a, scalar=-1)


# @lmatmul.dispatch # TODO(2bys): Check for a general rule.
# def _(a: LinearOperator, b: jax.Array) -> ArithmeticType:
#     return a.mv(b)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def diagonal(a: LinearOperator) -> ArithmeticType:
    return jnp.diag(a.todense())


def transpose(a: LinearOperator) -> ArithmeticType:
    return TransposedLinearOperator(a)


# @plum.dispatch
def inverse(a: LinearOperator) -> ArithmeticType:
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
    return jax.lax.map(a.mv, b.T).T


@lmatmul.dispatch
def _(a: jax.Array, b: LinearOperator) -> LinearOperator:
    from linox._matrix import Matrix  # noqa: PLC0415

    return Matrix(a) @ b


@lmatmul.dispatch
def _(a: jax.Array, b: Matrix) -> jax.Array:
    return a @ b.A


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Linear Operators
# --------------------------------------------------------------------------- #


class ScaledLinearOperator(LinearOperator):
    """Linear operator scaled with a scalar."""

    def __init__(self, operator: LinearOperator, scalar: ScalarLike) -> None:
        self.operator = operator
        scalar = jnp.asarray(scalar)
        dtype = jnp.result_type(operator.dtype, scalar.dtype)
        self.scalar = utils.as_scalar(scalar, dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return (self.operator @ vector) * self.scalar

    def todense(self) -> jax.Array:
        return self.operator.todense() * self.scalar

    def transpose(self) -> LinearOperator:
        return self.operator.transpose() * self.scalar


# inverse special behavior:
# ScaledLinearOperator(inverse(operator), inverse(Scalar))


class AddLinearOperator(LinearOperator):
    """A linear operator formed by adding two other linear operators together."""

    def __init__(self, *operator_list: ArithmeticType) -> None:
        self.operator_list = [
            o if isinstance(op, AddLinearOperator) else op
            for op in operator_list
            for o in (op.operator_list if isinstance(op, AddLinearOperator) else [op])
        ]
        self.__check_init__()
        super().__init__(
            shape=self.operator_list[0].shape, dtype=self.operator_list[0].dtype
        )

    def __check_init__(self) -> None:  # noqa: PLW3201
        shapes = [op.shape for op in self.operator_list]
        if not all(shape == shapes[0] for shape in shapes):
            msg = f"Shapes of all operators must match, but received shapes {shapes}."
            raise ValueError(msg)

    def mv(self, vector: jax.Array) -> jax.Array:
        return reduce(
            operator.add,
            (op @ vector for op in reversed(self.operator_list)),
        )

    def todense(self) -> jax.Array:
        return reduce(operator.add, (op.todense() for op in self.operator_list))

    def transpose(self) -> "AddLinearOperator":
        return AddLinearOperator(*(op.transpose() for op in self.operator_list))


class ProductLinearOperator(LinearOperator):
    """(Operator) Product of linear operators."""

    def __init__(self, *operator_list: LinearOperator) -> None:
        self.operator_list = [
            o if isinstance(op, ProductLinearOperator) else op
            for op in operator_list
            for o in (
                op.operator_list if isinstance(op, ProductLinearOperator) else [op]
            )
        ]
        self.__check_init__()
        shape = utils.as_shape((
            self.operator_list[0].shape[0],
            self.operator_list[-1].shape[1],
        ))
        super().__init__(shape=shape, dtype=self.operator_list[0].dtype)

    def __check_init__(self) -> None:  # noqa: PLW3201
        for i, op1 in enumerate(self.operator_list[:-1]):
            op2 = self.operator_list[i + 1]
            if op1.shape[1] != op2.shape[0]:
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

    def mv(self, vector: jax.Array) -> jax.Array:
        return reduce(lambda x, y: y @ x, [vector, *reversed(self.operator_list)])

    def transpose(self) -> "ProductLinearOperator":
        return ProductLinearOperator(
            self.operator2.transpose(), self.operator1.transpose()
        )


class TransposedLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator) -> None:
        self.operator = operator
        super().__init__(shape=operator.shape[::-1], dtype=operator.dtype)

    def mv(self, vector: jnp.array) -> jax.Array:
        return self.operator.transpose() @ vector

    def todense(self) -> jax.Array:
        return self.operator.todense().T

    def transpose(self) -> LinearOperator:
        return self.operator


# NOT TESTED
class InverseLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator) -> None:
        self.operator = operator
        super().__init__(shape=operator.shape, dtype=operator.dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return inverse(self.operator) @ vector

    def todense(self) -> jax.Array:
        return jnp.linalg.inv(self.operator.todense())

    def transpose(self) -> LinearOperator:
        return InverseLinearOperator(self.operator.transpose())


# @inverse.dispatch
# def _(a: InverseLinearOperator):
#     return a.operatorm
