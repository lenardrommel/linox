r"""Classic matrix operators as linear operator classes.

This module implements various classic matrix operators as linear operators, including:

- :class:`Matrix`: Represents a general matrix :math:`A`
- :class:`Identity`: Represents the identity matrix :math:`I`
- :class:`Diagonal`: Represents a diagonal matrix :math:`\text{diag}(d)`
- :class:`Scalar`: Represents a scalar multiple of the identity :math:`\alpha I`
- :class:`Zero`: Represents the zero matrix :math:`0`
- :class:`Ones`: Represents a matrix of ones :math:`\mathbf{1}\mathbf{1}^T`
"""

import jax
import jax.numpy as jnp

from linox import utils
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
from linox.types import ArrayLike, DTypeLike, ScalarLike, ScalarType, ShapeLike
from linox.utils import as_shape

# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #


class Matrix(LinearOperator):
    r"""A linear operator defined via a matrix.

    For a matrix :math:`A`, this represents the linear operator :math:`x \mapsto Ax`.
    The action on a vector :math:`x` is given by matrix multiplication :math:`Ax`.

    Args:
        A: The matrix defining the linear operator
    """

    def __init__(self, A: ArrayLike) -> None:  # type: ignore  # noqa: PGH003
        self.A = jnp.asarray(A)
        super().__init__(self.A.shape, self.A.dtype)

    def _matmul(self, vector: jax.Array) -> jax.Array:
        return self.A @ vector

    def todense(self) -> jax.Array:
        return self.A

    def transpose(self) -> "Matrix":
        return Matrix(self.A.swapaxes(-1, -2))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.A,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, any], children: tuple[any, ...]
    ) -> "Matrix":
        del aux_data
        (A,) = children
        return cls(A=A)


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
def _(A: LinearOperator) -> LinearOperator:
    msg = "The square root of a general linear operator is not defined."
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def _fill_ones(ndim: int) -> jnp.ndarray:
    return [1 for _ in range(ndim)]


class Identity(LinearOperator):
    r"""The identity operator.

    This represents the identity matrix :math:`I`. The action on a vector :math:`x` is
    given by :math:`Ix = x`, i.e., the identity operator leaves vectors unchanged.

    Args:
        shape: The shape of the identity operator
        dtype: The data type of the identity operator (default: float32)
    """

    def __init__(self, shape: ShapeLike, *, dtype: DTypeLike = jnp.float32) -> None:
        shape = as_shape(shape)
        super().__init__((*shape, shape[-1]), dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            arr,
            shape=(
                *jnp.broadcast_shapes(arr.shape[:-2], self.shape[:-2]),
                self.shape[-2],
                arr.shape[-1],
            ),
        )

    def todense(self) -> jax.Array:
        return jnp.broadcast_to(jnp.eye(self.shape[-1], dtype=self.dtype), self.shape)

    def transpose(self) -> "Identity":
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = ()
        aux_data = {"shape": self.shape[:-1], "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Identity":
        del children
        return cls(shape=aux_data["shape"], dtype=aux_data["dtype"])


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
def _(a: Identity, b: LinearOperator | Identity) -> LinearOperator:
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
    r"""A linear operator defined via a diagonal matrix.

    For a vector :math:`d`, this represents the diagonal matrix :math:`\text{diag}(d)`.
    The action on a vector :math:`x` is given by element-wise multiplication
    :math:`\text{diag}(d)x = d \odot x` where :math:`\odot` denotes element-wise
    multiplication.

    Args:
        diag: The diagonal elements of the matrix
    """

    def __init__(self, diag: ArrayLike) -> None:  # type: ignore  # noqa: PGH003
        self.diag = jnp.asarray(diag)
        super().__init__(
            shape=(*diag.shape[:-1], diag.shape[-1], diag.shape[-1]),
            dtype=self.diag.dtype,
        )

    def _matmul(self, vector: jax.Array) -> jax.Array:
        return self.diag[..., None] * vector

    def todense(self) -> jax.Array:
        return _batch_jnp_diag(self.diag)

    def transpose(self) -> "Diagonal":
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.diag,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Diagonal":
        del aux_data
        (diag,) = children
        return cls(diag=diag)


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
    r"""A linear operator defined via a scalar.

    For a scalar :math:`\alpha`, this represents :math:`\alpha I` where :math:`I`
    is the identity matrix. The action on a vector :math:`x` is given by scalar
    multiplication :math:`(\alpha I)x = \alpha x`.

    Args:
        scalar: The scalar value defining the operator
    """

    def __init__(self, scalar: ScalarLike) -> None:
        self.scalar = jnp.asarray(scalar)

        super().__init__(shape=(), dtype=self.scalar.dtype)

    def _matmul(self, vector: jax.Array) -> jax.Array:
        return self.scalar * vector

    def todense(self) -> jax.Array:
        return self

    def transpose(self) -> "Scalar":
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.scalar,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Scalar":
        del aux_data
        (scalar,) = children
        return cls(scalar=scalar)


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
    r"""The zero operator.

    This represents the zero matrix :math:`0`. The action on a vector :math:`x` is
    given by :math:`0x = 0`, i.e., the zero operator maps all vectors to zero.

    Args:
        shape: The shape of the zero operator
        dtype: The data type of the zero operator (default: float32)
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = ()
        aux_data = {"shape": self.shape, "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Zero":
        del children
        return cls(shape=aux_data["shape"], dtype=aux_data["dtype"])


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
    r"""The ones operator.

    This represents the matrix :math:`\mathbf{1}\mathbf{1}^T` where :math:`\mathbf{1}`
    is a vector of ones. The action on a vector :math:`x` is given by
    :math:`(\mathbf{1}\mathbf{1}^T)x = \mathbf{1}(\mathbf{1}^T x)`, i.e., it sums the
    elements of :math:`x` and returns a vector of that sum.

    Args:
        shape: The shape of the ones operator
        dtype: The data type of the ones operator (default: float32)
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = ()
        aux_data = {"shape": self.shape, "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Ones":
        del children
        return cls(shape=aux_data["shape"], dtype=aux_data["dtype"])


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


# Register all matrix operators as PyTrees
jax.tree_util.register_pytree_node_class(Matrix)
jax.tree_util.register_pytree_node_class(Identity)
jax.tree_util.register_pytree_node_class(Diagonal)
jax.tree_util.register_pytree_node_class(Scalar)
jax.tree_util.register_pytree_node_class(Zero)
jax.tree_util.register_pytree_node_class(Ones)
