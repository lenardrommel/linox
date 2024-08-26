from typing import Any, Callable
from nola.linox._linear_operator import LinearOperator
from nola.linox._typing import ScalarType, ScalarLike
import nola.linox._utils as utils
import jax
import jax.numpy as jnp
import plum


# all arithmetic functions
# @plum.dispatch
def ladd(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, b)


# @plum.dispatch
def lsub(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return AddLinearOperator(a, -b)


# @plum.dispatch
def lmul(a: ScalarType, b: LinearOperator) -> LinearOperator:
    return ScaledLinearOperator(a, b)


@plum.dispatch
def lmatmul(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    return ProductLinearOperator(a, b)


# @plum.dispatch
def lneg(a: LinearOperator):
    return ScaledLinearOperator(a, -1)


@lmatmul.dispatch
def _(a: LinearOperator, b: jax.Array) -> jax.Array:
    return a.mv(b)


class ScaledLinearOperator(LinearOperator):
    """Linear operator scaled with a scalar."""

    def __init__(self, operator: LinearOperator, scalar: ScalarLike):
        self.operator = operator
        dtype = jnp.result_type(operator.dtype, scalar)
        self.scalar = utils.as_scalar(scalar, dtype)

    def mv(self, vector):
        return self.operator @ vector * self.scalar

    def todense(self, vector):
        return self.operator.todense() * self.scalar

    def transpose(self):
        return self.operator.transpose() * self.scalar


# inverse special behavior:
# ScaledLinearOperator(inverse(operator), inverse(Scalar))


#
class AddLinearOperator(LinearOperator):
    """A linear operator formed by adding two other linear operators together."""

    def __init__(self, operator1, operator2):
        self.operator1 = operator1
        self.operator2 = operator2
        self.__check_init__()

    def __check_init__(self):
        if self.operator1.shape != self.operator2.shape:
            raise ValueError("The shapes of the two operators must match.")

    def mv(self, vector):
        mv1 = self.operator1.mv(vector)
        mv2 = self.operator2.mv(vector)
        return mv1 + mv2

    # TODO: Do we want to make these functions?
    def todense(self):
        return self.operator1.todense() + self.operator2.todense()

    def transpose(self):
        return self.operator1.transpose() + self.operator2.transpose()


class ProductLinearOperator(LinearOperator):
    """(Operator) Product of linear operators."""

    def __init__(self, operator1, operator2):
        self.operator1 = operator1
        self.operator2 = operator2
        self.shape = utils.as_shape(operator1.shape[0], operator2.shape[1])
        self.dtype = self.operator1.dtype

    def __check_init__(self):
        if self.operator1.shape[1] != self.operator2.shape[0]:
            raise ValueError(
                f"Shape mismatch: Cannot multiply linear operators with shapes "
                f"operator 1: ({self.operator1.shape}) "
                f"operator 2: ({self.operator2.shape})"
            )

        if self.operator1.dtype != self.operator2.dtype:
            raise TypeError(
                f"Type of operator 1: {self.operator1.dtype} and operator 2: {self.operator2.dtype} "
                f"mismatch."
            )

    def mv(self, vector) -> jax.Array:
        return self.operator1 @ (self.operator2 @ vector)

    def transpose(self) -> "ProductLinearOperator":
        return ProductLinearOperator(
            self.operator2.transpose(), self.operator1.transpose()
        )
