import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator
from linox._typing import ArrayLike, DTypeLike, ScalarLike, ShapeLike


class Matrix(LinearOperator):
    """A linear operator defined via a matrix.

    Parameters
    ----------
    A : ArrayLike
    """

    def __init__(self, A: ArrayLike) -> None:
        self.A = jnp.asarray(A)
        super().__init__(self.A.shape, self.A.dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return self.A @ vector

    def todense(self) -> jax.Array:
        return self.A

    def transpose(self) -> "Matrix":
        return Matrix(self.A.T)


# Special behavior for diagonal
# diagonal with jnp.diag(self.A)


class Identity(LinearOperator):
    """The identity operator.

    Parameters
    ----------
    shape :
        The shape of the identity operator.
    dtype :
        The data type of the identity operator.
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return vector

    def todense(self) -> jax.Array:
        return jnp.eye(*self.shape, dtype=self.dtype)

    def transpose(self) -> "Identity":
        return self


# Special behavior for diagonal function - return just ones.


class Diagonal(LinearOperator):
    """A linear operator defined via a diagonal matrix.

    Parameters
    ----------
    diag :
        The diagonal of the matrix.
    """

    def __init__(self, diag: ArrayLike) -> None:
        self.diag = jnp.asarray(diag)

    def mv(self, vector: jax.Array) -> jax.Array:
        return self.diag * vector

    def todense(self) -> jax.Array:
        return jnp.diag(self.diag)

    def transpose(self) -> "Diagonal":
        return self


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

        super().__init__(shape=(), dtupe=self.scalar.dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return self.scalar * vector

    def todense(self) -> jax.Array:
        return self

    def transpose(self) -> "Scalar":
        return self


# Special behavior for the diagonal, i.e. return self
class Zero(LinearOperator):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        _ = vector
        return jnp.zeros(self.shape[0], dtype=self.dtype)


class Ones(LinearOperator):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(shape, dtype)

    def mv(self, vector: jax.Array) -> jax.Array:
        return jnp.full(self.shape[0], jnp.sum(vector))
