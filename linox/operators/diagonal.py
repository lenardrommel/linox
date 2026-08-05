"""Diagonal Linear Operators."""

import jax
import jax.numpy as jnp

from linox._types import ArrayLike, ScalarType
from linox.operators.arithmetic import (
    congruence_transform,
    diagonal,
    ladd,
    ldiv,
    lexp,
    linverse,
    llog,
    lmatmul,
    lmul,
    lpow,
    lsqrt,
    lsub,
    ltrace,
)
from linox.operators.base import LinearOperator

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

    def __init__(self, diag: ArrayLike) -> None:
        if isinstance(diag, Diagonal):
            diag = diag.diag
        self.diag = jnp.asarray(diag)
        super().__init__(
            shape=(*diag.shape[:-1], diag.shape[-1], diag.shape[-1]),
            dtype=self.diag.dtype,
        )

    def _matmul(self, vector: jax.Array) -> jax.Array:
        return self.diag[..., None] * vector

    def _todense(self) -> jax.Array:
        return _batch_jnp_diag(self.diag)

    @property
    def is_symmetric(self) -> bool:
        # Real diagonal matrices are symmetric.
        # If complex, check imaginary part?
        # Generally yes for simplicity in real case.
        return True

    @property
    def is_psd(self) -> bool:
        # Check if all elements non-negative
        # Since we want boolean return without tracing values if possible...
        # If diag is concrete array, we can check.
        # If tracer, we fail.
        try:
            # Fast check if all >= 0?
            # For now return False to be safe unless we are sure.
            # Or if user wants to use it in introspection (non-jit), access array.
            return bool(jnp.all(self.diag >= 0))
        except:
            return False

    def transpose(self) -> "Diagonal":
        return self

    def diagonal(self) -> jax.Array:
        return self.diag

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


@diagonal.dispatch
def _(a: Diagonal) -> jax.Array:
    return a.diag


@ladd.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag + b.diag)


@ladd.dispatch
def _(a: Diagonal, b: jax.Array) -> Diagonal:
    if b.shape == () or a.shape[-1] == b.shape[-1]:
        return Diagonal(a.diag + b)

    msg = f"Shapes not aligned for addition: {a.shape} and {b.shape}"
    raise ValueError(msg)


@lsub.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag - b.diag)


@lmul.dispatch
def _(a: ScalarType, b: Diagonal) -> Diagonal:
    return Diagonal(a * b.diag)


@ldiv.dispatch
def _(a: Diagonal, b: Diagonal) -> Diagonal:
    return Diagonal(a.diag / b.diag)


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


# New matrix-free function dispatches for Diagonal
@ltrace.dispatch
def _(
    a: Diagonal,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Exact trace of diagonal matrix: trace(diag(d)) = sum(d)."""
    trace_value = jnp.sum(a.diag)
    # For exact computation, std = 0
    trace_std = jnp.array(0.0, dtype=a.dtype)
    return trace_value, trace_std


@lexp.dispatch
def _(
    a: Diagonal,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix exponential of diagonal: exp(diag(d)) = diag(exp(d))."""
    if v is None:
        # Return lazy operator: diag(exp(d))
        return Diagonal(jnp.exp(a.diag))
    # exp(diag(d)) @ v = exp(d) * v (element-wise)
    return jnp.exp(a.diag) * v


@llog.dispatch
def _(
    a: Diagonal,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix logarithm of diagonal: log(diag(d)) = diag(log(d))."""
    if v is None:
        # Return lazy operator: diag(log(d))
        return Diagonal(jnp.log(a.diag))
    # log(diag(d)) @ v = log(d) * v (element-wise)
    return jnp.log(a.diag) * v


@lpow.dispatch
def _(
    a: Diagonal,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix power of diagonal: diag(d)^p = diag(d^p)."""
    if v is None:
        # Return lazy operator: diag(d^p)
        return Diagonal(a.diag**power)
    # diag(d)^p @ v = d^p * v (element-wise)
    return (a.diag**power) * v


class CircularlySymmetricDiagonal(Diagonal):
    def __init__(self, R_real: ArrayLike, W: ArrayLike, b: ArrayLike | None) -> None:
        self._R_real = jnp.asarray(R_real)
        self._W = jnp.asarray(W)
        self._b = jnp.asarray(b) if b is not None else None

        diag = (
            self._R_real.reshape(-1, order="C"),
            self._R_real.reshape(-1, order="C"),
            self._W.reshape(-1, order="C"),
        )

        if self._b is not None:
            diag += (self._b.reshape(-1, order="C"),)

        super().__init__(jnp.concatenate(diag, axis=0))

    @property
    def R_real(self) -> jax.Array:
        return self._R_real

    @property
    def W(self) -> jax.Array:
        return self._W

    @property
    def b(self) -> jax.Array | None:
        return self._b


@linverse.dispatch
def _(d: CircularlySymmetricDiagonal) -> CircularlySymmetricDiagonal:
    return CircularlySymmetricDiagonal(
        1 / d.R_real,
        1 / d.W,
        None if d.b is None else 1 / d.b,
    )


@lsqrt.dispatch
def _(d: CircularlySymmetricDiagonal) -> CircularlySymmetricDiagonal:
    return CircularlySymmetricDiagonal(
        jnp.sqrt(d.R_real),
        jnp.sqrt(d.W),
        None if d.b is None else jnp.sqrt(d.b),
    )


jax.tree_util.register_pytree_node_class(Diagonal)
