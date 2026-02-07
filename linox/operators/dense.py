"""Dense Matrix operator."""

import jax
import jax.numpy as jnp

from linox import config
from linox.config import warn as _warn
from linox.operators.arithmetic import (
    ScaledLinearOperator,
    congruence_transform,
    diagonal,
    ladd,
    ldiv,
    linverse,
    lmatmul,
    lmul,
    lsolve,
    lsqrt,
    lsub,
)
from linox.operators.base import LinearOperator
from linox.typing import ArrayLike


class Matrix(LinearOperator):
    r"""A linear operator defined via a matrix.

    For a matrix :math:`A`, this represents the linear operator :math:`x \mapsto Ax`.
    The action on a vector :math:`x` is given by matrix multiplication :math:`Ax`.

    Args:
        A: The matrix defining the linear operator
    """

    def __init__(self, A: ArrayLike) -> None:  # type: ignore  # noqa: PGH003
        self.A = jnp.asarray(A)
        config.emit(
            config.DebugEvent(
                kind="init",
                msg=f"Matrix initialized with shape {self.A.shape} and dtype {self.A.dtype}",
                op_type=type(self).__name__,
                op_id=id(self),
                shape=self.A.shape,
                dtype=self.A.dtype,
            )
        )
        super().__init__(self.A.shape, self.A.dtype)

    def _matmul(self, vector: jax.Array) -> jax.Array:
        return self.A @ vector

    def _todense(self) -> jax.Array:
        _warn(f"Converting Matrix of shape {self.shape} to dense array.")
        return self.A

    def transpose(self) -> "Matrix":
        return Matrix(self.A.swapaxes(-1, -2))

    def __T__(self) -> "Matrix":
        """Alias for transpose."""
        return self.transpose()

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


# Provide a specialized, no-warning diagonal for Matrix.
@diagonal.dispatch
def _(a: Matrix) -> jax.Array:
    if a.A.ndim <= 2:
        return jnp.diag(a.A)
    n = a.A.shape[-1]
    idx = jnp.arange(n)
    return a.A[..., idx, idx]


# register matrix special behavior
@ladd.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A + b.A)


@lsub.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A - b.A)


@lmul.dispatch
def _(a: float, b: Matrix) -> Matrix: # Fixed type hint from ScalarType to float/complex potentially, but keeping simple for now
    return Matrix(a * b.A)


@lmatmul.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A @ b.A)


@lmatmul.dispatch
def _(a: jax.Array, b: Matrix) -> jax.Array:
    return a @ b.A


@lsqrt.dispatch
def _(a: Matrix) -> Matrix:
    if a.shape[-1] != a.shape[-2]:
        msg = f"Square root only defined for square matrices, got shape {a.shape}"
        raise ValueError(msg)
    jitter = 1e-10 if a.dtype == jnp.float64 else 1e-6
    identity = jnp.eye(a.shape[-1], dtype=a.dtype)
    try:
        chol = jnp.linalg.cholesky(a.A + jitter * identity)
    except Exception as err:
        msg = "Matrix square root requires a symmetric positive-definite matrix."
        raise ValueError(msg) from err
    return Matrix(chol)


@linverse.dispatch
def _(a: Matrix) -> Matrix:
    # Solve via LU decomposition and Inverse Linear Operator
    return Matrix(jnp.linalg.inv(a.A))


@lsolve.dispatch
def _(a: Matrix, b: jax.Array) -> jax.Array:
    return jnp.linalg.solve(a.A, b)


@congruence_transform.dispatch
def _(a: Matrix, b: Matrix) -> Matrix:
    return Matrix(a.A @ b.A @ a.A.swapaxes(-1, -2))


@lsqrt.register
def _(A: LinearOperator) -> LinearOperator:
    msg = "The square root of a general linear operator is not defined."
    raise NotImplementedError(msg)
