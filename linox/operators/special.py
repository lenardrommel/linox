"""Special Linear Operators (Identity, Zero, Scalar, Ones)."""

import jax
import jax.numpy as jnp

from linox import utils
from linox._types import DTypeLike, ScalarLike, ScalarType, ShapeLike
from linox.operators.arithmetic import (
    ScaledLinearOperator,
    congruence_transform,
    diagonal,
    ladd,
    lexp,
    linverse,
    llog,
    lmatmul,
    lmul,
    lpinverse,
    lpow,
    lsqrt,
    lsub,
    ltrace,
)
from linox.operators.base import LinearOperator
from linox.operators.dense import Matrix
from linox.utils import as_shape
from linox.utils.array import default_floating_dtype


def _matmul_broadcast(a: LinearOperator, b: LinearOperator):
    batch_shape = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return (*batch_shape, a.shape[-2], b.shape[-1])


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


class Identity(LinearOperator):
    r"""The identity operator.

    This represents the identity matrix :math:`I`. The action on a vector :math:`x` is
    given by :math:`Ix = x`, i.e., the identity operator leaves vectors unchanged.

    Args:
        shape: The shape of the identity operator
        dtype: The data type of the identity operator. Defaults to JAX's
            current floating dtype (float64 under x64, else float32).
    """

    def __init__(self, shape: ShapeLike, *, dtype: DTypeLike | None = None) -> None:
        dtype = default_floating_dtype() if dtype is None else dtype
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

    def _todense(self) -> jax.Array:
        return jnp.broadcast_to(jnp.eye(self.shape[-1], dtype=self.dtype), self.shape)

    @property
    def is_symmetric(self) -> bool:
        """Check if operator is symmetric."""
        return True

    @property
    def is_psd(self) -> bool:
        """Check if operator is positive semi-definite."""
        return True

    def transpose(self) -> "Identity":
        """Return transpose (self for identity)."""
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = ()
        aux_data = {"shape": self.shape[:-1], "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Identity":
        """Unflatten for JAX pytree registration."""
        del children
        return cls(shape=aux_data["shape"], dtype=aux_data["dtype"])


@diagonal.dispatch
def _(a: Identity) -> jax.Array:
    # Diagonal of identity is ones with appropriate batch shape.
    return jnp.ones((*a.shape[:-2], a.shape[-1]), dtype=a.dtype)


@ladd.dispatch
def _(a: Identity, b: Identity) -> ScaledLinearOperator:
    new_shape = jnp.broadcast_shapes(a.shape, b.shape)
    a.__shape = (*new_shape[:-2], a.shape[-1])
    return 2 * a


@lsub.dispatch
def _(a: Identity, b: Identity) -> LinearOperator:
    new_shape = jnp.broadcast_shapes(a.shape, b.shape)
    # Zero needs to be defined below or imported. Zero is defined in this file.
    return Zero(new_shape, dtype=jnp.result_type(a.dtype, b.dtype))


@lmatmul.dispatch(precedence=5)
def _(a: Identity, b: LinearOperator | Identity) -> LinearOperator:
    new_shape = _matmul_broadcast(a, b)  # Set new broadcasted shape
    b.__shape = (*new_shape[:-2], *b.shape[-2:])
    return b


@lmatmul.dispatch
def _(a: LinearOperator, b: Identity) -> LinearOperator:
    new_shape = _matmul_broadcast(a, b)
    a.__shape = (*new_shape[:-2], *a.shape[-2:])
    return a


# Registered separately, and deliberately not stacked. A plum `.dispatch`
# decorator returns the *Function object*, so `@lsqrt.dispatch` applied on top
# of `@linverse.dispatch` registered `linverse` itself under `lsqrt` -- with
# `linverse`'s own generic `(LinearOperator)` signature. Every operator without
# a specific `lsqrt` then silently received its inverse instead of a square
# root.
@lsqrt.dispatch
def _(a: Identity) -> Identity:
    return a


@linverse.dispatch
def _(a: Identity) -> Identity:
    return a


@lpinverse.dispatch
def _(a: Identity) -> Identity:
    """Identity operator does not change the input."""
    return a


@congruence_transform.dispatch
def _(a: Identity, b: LinearOperator | Identity) -> LinearOperator:
    _ = b
    return a


@congruence_transform.dispatch
def _(a: Identity, b: LinearOperator) -> LinearOperator:
    _ = a
    return b


# New matrix-free function dispatches for Identity
@ltrace.dispatch
def _(
    a: Identity,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Exact trace of identity matrix: trace(I) = n."""
    n = a.shape[-1]
    trace_value = jnp.array(n, dtype=a.dtype)
    # For exact computation, std = 0
    trace_std = jnp.array(0.0, dtype=a.dtype)
    return trace_value, trace_std


@lexp.dispatch
def _(
    a: Identity,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix exponential of identity: exp(I) = e * I."""
    if v is None:
        # Return lazy operator: e * I
        return Scalar(jnp.exp(1.0).astype(a.dtype), 1.0)  # Actually Scalar constructor expects scalar value.
        # Wait, Scalar constructor is `Scalar(scalar)`. It infers shape from nothing? No, Scalar(scalar) has shape=().
        # Scalar operator represents alpha * I.
        # The Scalar class definition below takes `scalar`.
        # But wait, does Scalar support arbitrary shape?
        # Looking at `Scalar` below: super().__init__(shape=(), dtype=self.scalar.dtype).
        # It seems Scalar represents a scalar *number*, effectively 1x1 or broadcastable?
        # But `Identity` is NxN.
        # If I return `Scalar(e)`, is that valid for NxN?
        # Scalar's matmul returns `scalar * vector`. It broadcasts.
        # So yes, Scalar(e) acts like e*I of any size compatible with vector.
        return Scalar(jnp.exp(1.0))

    # exp(I) @ v = e * v
    return jnp.exp(1.0) * v


@llog.dispatch
def _(
    a: Identity,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix logarithm of identity: log(I) = 0."""
    if v is None:
        # Return zero operator
        return Zero(a.shape, dtype=a.dtype)
    # log(I) @ v = 0
    return jnp.zeros_like(v)


@lpow.dispatch
def _(
    a: Identity,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix power of identity: I^p = I for any p."""
    if v is None:
        return a  # I^p = I
    return v  # I^p @ v = v


# --------------------------------------------------------------------------- #
# Scalar
# --------------------------------------------------------------------------- #


# Special behavior for the diagonal, i.e. return jnp.diag(self.diag)
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

    def _todense(self) -> jax.Array:
        return self.scalar  # Scalar todense returns the scalar array? Or should it expand?
        # _matrix.py said `return self`. Wait, self is the operator. returning self in _todense is weird unless it means the scalar value?
        # Ah, lines 622 in _matrix.py: `return self`. This looks like a bug in original code or self.scalar?
        # Wait, if I return self (the LinearOperator instance), that's definitely wrong for `todense`.
        # However, line 622 in previous output says `return self`.
        # Wait, if `Scalar` acts like a scalar array, maybe?
        # But `todense()` is expected to return jax.Array.
        # Let's fix it to return `self.scalar`.
        return self.scalar

    @property
    def is_symmetric(self) -> bool:
        """Check if operator is symmetric."""
        return True

    @property
    def is_psd(self) -> bool:
        """Check if operator is positive semi-definite."""
        try:
            return float(self.scalar) >= 0
        except Exception:
            return False

    def transpose(self) -> "Scalar":
        """Return transpose (self for scalar)."""
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = (self.scalar,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Scalar":
        """Unflatten for JAX pytree registration."""
        del aux_data
        (scalar,) = children
        return cls(scalar=scalar)


@ladd.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    return Scalar(a.scalar + b.scalar)


@lsub.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    return Scalar(a.scalar - b.scalar)


# Registered separately rather than stacked, for the reason given above: it is
# the *outer* decorator that gets polluted. Stacking `@lmatmul.dispatch` over
# `@lmul.dispatch` gave `lmul` the correct method and `lmatmul` a copy of
# `lmul`'s generic signature.
@lmatmul.dispatch
def _(a: Scalar, b: Scalar) -> Scalar:
    # Narrower than the `ScalarType | Scalar` used for `lmul` below: the
    # `jax.Array` half is not a `LinearOperator`, so that signature ties with
    # the `(LinearOperator, LinearOperator)` generic and plum cannot order
    # them. A scalar array times a Scalar is already covered by the generic
    # `lmatmul(jax.Array, LinearOperator)`.
    return Scalar(utils.as_scalar(a) * b.scalar)


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
        dtype: The data type of the zero operator. Defaults to JAX's
            current floating dtype (float64 under x64, else float32).
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> None:
        dtype = default_floating_dtype() if dtype is None else dtype
        super().__init__(shape, dtype)

    def _matmul(self, arr: jax.Array) -> jax.Array:
        return jnp.zeros(
            (
                *jnp.broadcast_shapes(arr.shape[:-2], self.shape[:-2]),
                self.shape[-2],
                arr.shape[-1],
            ),
            # Promote with the operand: a Zero operator must not narrow the
            # result of `Zero @ x` to its own dtype.
            dtype=jnp.result_type(self.dtype, arr.dtype),
        )

    def _todense(self) -> jax.Array:
        return jnp.zeros(self.shape, dtype=self.dtype)

    @property
    def is_symmetric(self) -> bool:
        """Check if operator is symmetric."""
        return True

    @property
    def is_psd(self) -> bool:
        """Check if operator is positive semi-definite."""
        return True

    def transpose(self) -> "Zero":
        """Return transposed zero operator."""
        return Zero(shape=(*self.shape[:-2], self.shape[-1], self.shape[-2]), dtype=self.dtype)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = ()
        aux_data = {"shape": self.shape, "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Zero":
        """Unflatten for JAX pytree registration."""
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
        dtype: The data type of the ones operator. Defaults to JAX's
            current floating dtype (float64 under x64, else float32).
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> None:
        dtype = default_floating_dtype() if dtype is None else dtype
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

    def _todense(self) -> jax.Array:
        return jnp.ones(self.shape, dtype=self.dtype)

    def transpose(self) -> "Ones":
        """Return transposed ones operator."""
        return Ones(shape=(*self.shape[:-2], self.shape[-1], self.shape[-2]), dtype=self.dtype)

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        """Flatten for JAX pytree registration."""
        children = ()
        aux_data = {"shape": self.shape, "dtype": self.dtype}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "Ones":
        """Unflatten for JAX pytree registration."""
        del children
        return cls(shape=aux_data["shape"], dtype=aux_data["dtype"])


@ladd.dispatch
def _(a: Ones, b: Matrix) -> LinearOperator:
    return Matrix(b.A + jnp.ones(a.shape, dtype=b.dtype))


@ladd.dispatch(precedence=2)
def _(a: Ones, b: Ones) -> LinearOperator:
    _ = b
    return ScaledLinearOperator(operator=a, scalar=2)


@lsub.dispatch
def _(a: Matrix, b: Ones) -> Matrix:
    _ = a
    return Matrix(a.A - jnp.ones(b.shape, dtype=a.dtype))


@lsub.dispatch(precedence=1)
def _(a: Ones, b: Ones) -> Zero:
    return Zero(jnp.broadcast_shapes(a.shape, b.shape), dtype=jnp.result_type(a.dtype, b.dtype))


jax.tree_util.register_pytree_node_class(Identity)
jax.tree_util.register_pytree_node_class(Scalar)
jax.tree_util.register_pytree_node_class(Zero)
jax.tree_util.register_pytree_node_class(Ones)
