"""Classic matrix operators as linear operator classes."""

import jax
import jax.numpy as jnp

from linox import _utils as utils
from linox._arithmetic import (
    ScaledLinearOperator,
    congruence_transform,
    ladd,
    linverse,
    lmatmul,
    lmul,
    lsqrt,
    lsub,
)
from linox._linear_operator import LinearOperator
from linox._typing import ArrayLike, DTypeLike, ScalarLike, ScalarType, ShapeLike

# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #


class Matrix(LinearOperator):
    """A linear operator defined via a matrix.

    Parameters
    ----------
    A : ArrayLike
    """

    def __init__(self, A: ArrayLike) -> None:  # type: ignore  # noqa: PGH003
        self.A = jnp.asarray(A)
        super().__init__(self.A.shape, self.A.dtype)

    def matmat(self, vector: jax.Array) -> jax.Array:
        return self.A @ vector

    def todense(self) -> jax.Array:
        return self.A

    def transpose(self) -> "Matrix":
        return Matrix(self.A.swapaxes(-1, -2))


# register matrix special behavior
@ladd.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A + b.A)


@lsub.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A - b.A)


@lmul.dispatch
def _(a: ScalarType, b: Matrix) -> Matrix:
    return Matrix(a * b.A)


@lmatmul.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A @ b.A)


@lmatmul.dispatch
def _(a: jax.Array, b: Matrix) -> jax.Array:
    return a @ b.A


@lsqrt.dispatch
def _(a: Matrix) -> Matrix:
    return Matrix(jnp.sqrt(a.A))


@linverse.dispatch
def _(a: Matrix) -> Matrix:
    return Matrix(jnp.linalg.inv(a.A))


@congruence_transform.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A @ b.A @ a.A.swapaxes(-1, -2))


@lsqrt.register
def _(A: LinearOperator) -> LinearOperator:  # noqa: ARG001
    msg = "The square root of a general linear operator is not defined."
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def _fill_ones(ndim: int) -> jnp.ndarray:
    return [1 for _ in range(ndim)]


class Identity(LinearOperator):
    """The identity operator.

    Parameters
    ----------
    shape :
        The shape of the identity operator.
    dtype :
        The data type of the identity operator.
    """

    def __init__(self, shape: ShapeLike, *, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__((*shape[:-1], shape[-1], shape[-1]), dtype)

    def matmat(self, arr: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            arr,
            #            jnp.tile(arr, _fill_ones(jnp.max(arr.ndim, self.ndim))),
            shape=(
                *jnp.broadcast_shapes(arr.shape[:-2], self.shape[:-2]),
                self.shape[-2],
                arr.shape[-1],
            ),
        )

    # def mv(self, vector: jax.Array) -> jax.Array:
    #     ndim_diff = self.ndim - vector.ndim
    #     return jnp.tile(
    #         vector, (*self.shape[: -(ndim_diff + 2)], *_fill_ones(vector.ndim))
    #     )

    def todense(self) -> jax.Array:
        return jnp.broadcast_to(jnp.eye(self.shape[-1], dtype=self.dtype), self.shape)

    def transpose(self) -> "Identity":
        return self


@ladd.dispatch
def _(a: Identity, b: Identity) -> ScaledLinearOperator:
    new_shape = jnp.broadcast_shapes(a.shape, b.shape)
    a.__shape = (*new_shape[:-2], a.shape[-1])  # noqa: SLF001
    return 2 * a


@lsub.dispatch
def _(a: Identity, b: Identity) -> LinearOperator:
    new_shape = jnp.broadcast_shapes(a.shape, b.shape)
    return Zero(new_shape, dtype=jnp.result_type(a.dtype, b.dtype))


@lmatmul.dispatch(precedence=5)
def _(a: Identity, b: LinearOperator | Identity) -> LinearOperator:
    new_shape = _matmul_broadcast(a, b)  # Set new broadcasted shape
    b.__shape = (*new_shape[:-2], *b.shape[-2:])  # noqa: SLF001
    return b


@lmatmul.dispatch
def _(a: LinearOperator, b: Identity) -> LinearOperator:
    new_shape = _matmul_broadcast(a, b)
    a.__shape = (*new_shape[:-2], *a.shape[-2:])  # noqa: SLF001
    return a


@lsqrt.dispatch
@linverse.dispatch
def _(a: Identity) -> Identity:
    return a


@congruence_transform.dispatch
def _(a: LinearOperator | Identity, b: Identity) -> LinearOperator:
    _ = b
    return a


@congruence_transform.dispatch
def _(a: Identity, b: LinearOperator) -> LinearOperator:
    _ = a
    return b


# --------------------------------------------------------------------------- #
# Diagonal
# --------------------------------------------------------------------------- #

# special functions for diagonal.
_batch_jnp_diag = jnp.vectorize(jnp.diag, signature="(n)->(n,n)")


class Diagonal(LinearOperator):
    """A linear operator defined via a diagonal matrix.

    Parameters
    ----------
    diag :
        The diagonal of the matrix.
    """

    def __init__(self, diag: ArrayLike) -> None:  # type: ignore  # noqa: PGH003
        self.diag = jnp.asarray(diag)
        super().__init__(
            shape=(*diag.shape[:-1], diag.shape[-1], diag.shape[-1]),
            dtype=self.diag.dtype,
        )

    def matmat(self, vector: jax.Array) -> jax.Array:
        return self.diag[..., None] * vector

    def todense(self) -> jax.Array:
        return _batch_jnp_diag(self.diag)

    def transpose(self) -> "Diagonal":
        return self


@ladd.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag + b.diag)


@lsub.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag - b.diag)


@lmul.dispatch
def _(a: ScalarType, b: Diagonal) -> Diagonal:
    return Diagonal(a * b.diag)


@lmatmul.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag * b.diag)


@lsqrt.dispatch
def _(a: Diagonal) -> Diagonal:
    return Diagonal(jnp.sqrt(a.diag))


@linverse.dispatch
def _(a: Diagonal) -> Diagonal:
    return Diagonal(1 / a.diag)


@congruence_transform.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag * b.diag * a.diag)


# --------------------------------------------------------------------------- #
# Scalar
# --------------------------------------------------------------------------- #


# Special behavior for the diagonal, i.e. reutrn jnp.diag(self.diag)
class Scalar(LinearOperator):
    """A linear operator defined via a scalar.

    Parameters
    ----------
    scalar :
        The scalar.
    """

    def __init__(self, scalar: ScalarLike) -> None:
        self.scalar = jnp.asarray(scalar)

        super().__init__(shape=(), dtype=self.scalar.dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return self.scalar * vector

    def todense(self) -> jax.Array:
        return self

    def transpose(self) -> "Scalar":
        return self


@ladd.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    return Scalar(a.scalar + b.scalar)


@lsub.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    return Scalar(a.scalar - b.scalar)


@lmatmul.dispatch
@lmul.dispatch
def _(a: ScalarType | Scalar, b: Scalar) -> Scalar:
    return Scalar(utils.as_scalar(a) * b.scalar)


@lsqrt.dispatch
def _(a: Scalar) -> Scalar:
    return Scalar(jnp.sqrt(a.scalar))


@linverse.dispatch
def _(a: Scalar) -> Scalar:
    return Scalar(1 / a.scalar)


@congruence_transform.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    return Scalar(a.scalar * b.scalar * a.scalar)


@lsqrt.register
def _(A: Scalar) -> Scalar:
    return Scalar(jnp.sqrt(A.scalar))


# --------------------------------------------------------------------------- #
# Zero
# --------------------------------------------------------------------------- #


class Zero(LinearOperator):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def matmat(self, arr: jax.Array) -> jax.Array:
        return jnp.zeros(
            (
                *jnp.broadcast_shapes(arr.shape[:-2], self.shape[:-2]),
                self.shape[-2],
                arr.shape[-1],
            ),
            dtype=self.dtype,
        )

    def todense(self) -> jax.Array:
        return jnp.zeros(self.shape, dtype=self.dtype)

    def transpose(self) -> "Zero":
        return Zero(
            shape=(*self.shape[:-2], self.shape[-1], self.shape[-2]), dtype=self.dtype
        )


@ladd.dispatch(precedence=1)
def _(a: Zero, b: LinearOperator | Zero) -> LinearOperator:
    _ = a
    return b


@ladd.dispatch
def _(a: LinearOperator, b: Zero) -> LinearOperator:
    _ = b
    return a


@lsub.dispatch
def _(a: Zero, b: LinearOperator) -> LinearOperator:
    _ = a
    return -b


@lsub.dispatch(precedence=1)
def _(a: LinearOperator | Zero, b: Zero) -> LinearOperator:
    _ = b
    return a


@lmul.dispatch
def _(a: ScalarType, b: Zero) -> Zero:
    _ = a
    return b


def _matmul_broadcast(a: LinearOperator, b: LinearOperator):  # noqa: ANN202
    batch_shape = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return (*batch_shape, a.shape[-2], b.shape[-1])


@lmatmul.dispatch(precedence=5)
def _(a: Zero, b: LinearOperator | Zero) -> Zero:
    return Zero(shape=_matmul_broadcast(a, b), dtype=jnp.result_type(a.dtype, b.dtype))


@lmatmul.dispatch
def _(a: LinearOperator, b: Zero) -> Zero:
    return Zero(shape=_matmul_broadcast(a, b), dtype=jnp.result_type(a.dtype, b.dtype))


@lsqrt.dispatch
def _(a: Zero) -> Zero:
    return a


@linverse.dispatch
def _(a: Zero) -> Zero:
    _ = a
    msg = "The inverse of the zero operator is not defined."
    raise ValueError(msg)


@congruence_transform.dispatch
def _(a: Zero, b: LinearOperator | Zero) -> Zero:
    _ = b
    return a


# --------------------------------------------------------------------------- #
# Ones
# --------------------------------------------------------------------------- #


class Ones(LinearOperator):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def matmat(self, arr: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            arr.sum(axis=-2, keepdims=True),
            shape=(
                *jnp.broadcast_shapes(arr.shape[:-2], self.shape[:-2]),
                self.shape[-2],
                arr.shape[-1],
            ),
        )

    def todense(self) -> jax.Array:
        return jnp.ones(self.shape, dtype=self.dtype)

    def transpose(self) -> "Ones":
        return Ones(
            shape=(*self.shape[:-2], self.shape[-1], self.shape[-2]), dtype=self.dtype
        )


@ladd.dispatch
def _(a: Ones, b: Matrix) -> LinearOperator:
    return Matrix(b.A + jnp.ones(a.shape, dtype=b.dtype))


@ladd.dispatch(precedence=2)
def _(a: Ones, b: Ones) -> LinearOperator:
    from linox._arithmetic import ScaledLinearOperator  # noqa: PLC0415

    _ = b
    return ScaledLinearOperator(operator=a, scalar=2)


@lsub.dispatch
def _(a: Matrix, b: Ones) -> Matrix:
    _ = a
    return Matrix(a.A - jnp.ones(b.shape, dtype=a.dtype))


@lsub.dispatch(precedence=1)
def _(a: Ones, b: Ones) -> Zero:
    return Zero(
        jnp.broadcast_shapes(a.shape, b.shape), dtype=jnp.result_type(a.dtype, b.dtype)
    )


# TODO(2bys): Implement more special behaviors, e.g. summation via lmatmul.
