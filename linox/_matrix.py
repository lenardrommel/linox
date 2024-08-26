import jax.numpy as jnp
from nola.linox._linear_operator import LinearOperator
from nola.linox._typing import ArrayLike, ShapeLike, DTypeLike, ScalarLike


class Matrix(LinearOperator):
    """A linear operator defined via a matrix.

    Parameters
    ----------
    A :
        The explicit matrix.
    """

    def __init__(self, A: ArrayLike):
        self.A = jnp.asarray(A)
        super().__init__(self.A.shape, self.A.dtype)

    def mv(self, vector):
        return self.A @ vector

    def todense(self):
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

    def __init__(self, shape: ShapeLike, dtype: DTypeLike = jnp.float64):
        super().__init__(self.shape, self.dtype)

    def mv(self, vector):
        return vector

    def todense(self):
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

    def __init__(self, diag: ArrayLike):
        self.diag = jnp.asarray(diag)

    def mv(self, vector):
        return self.diag * vector

    def todense(self):
        return jnp.diag(self.diag)

    def transpose(self):
        return self


# Special behavior for the diagonal, i.e. reutrn jnp.diag(self.diag)
class Scalar(LinearOperator):
    """A linear operator defined via a scalar.

    Parameters
    ----------
    scalar :
        The scalar.
    """

    def __init__(self, scalar: ScalarLike):
        self.scalar = jnp.asarray(scalar)

        super().__init__(shape=(), dtupe=self.scalar.dtype)

    def mv(self, vector) -> "LinearOperator":
        return self.scalar * vector

    def todense(self):
        return self

    def transpose(self) -> "Scalar":
        return self


# Special behavior for the diagonal, i.e. return self
