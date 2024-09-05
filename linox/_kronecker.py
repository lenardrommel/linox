import jax
import jax.numpy as jnp

from linox._linear_operator import LinearOperator
from linox._utils import as_linop


class Kronecker(LinearOperator):
    """A Kronecker product of two linear operators.

    Example usage:

    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    B = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)
    op = Kronecker(A, B)
    vec = jnp.ones((4,))
    result = op @ vec
    result_true = jnp.kron(A, B) @ vec
    jnp.allclose(result, result_true)
    """

    def __init__(
        self, A: LinearOperator | jax.Array, B: LinearOperator | jax.Array
    ) -> None:
        self.A = as_linop(A)
        self.B = as_linop(B)
        shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        dtype = A.dtype
        super().__init__(shape, dtype)

    def mv(self, vec: jax.Array) -> jax.Array:
        flatten = False
        if len(vec.shape) == 1:
            vec = vec[:, None]
            flatten = True
        elif vec.shape[-1] != 1:
            msg = "The input vector must have a single column."
            raise ValueError(msg)

        _, mA = self.A.shape
        _, mB = self.B.shape

        # vec(X) -> X, i.e., reshape into stack of matrices
        y = jnp.swapaxes(vec, -2, -1)
        y = y.reshape(y.shape[:-1] + (mA, mB))

        # (X @ B.T).T = B @ X.T
        y = self.B @ jnp.swapaxes(y, -1, -2)

        # A @ X @ B.T = A @ (B @ X.T).T
        y = self.A @ jnp.swapaxes(y, -1, -2)

        # vec(A @ X @ B.T), i.e., revert to stack of vectorized matrices
        y = y.reshape(y.shape[:-2] + (-1,))
        y = jnp.swapaxes(y, -1, -2)

        return (
            y[..., 0] if flatten else y
        )  # TODO(any): Find out if we need this extra dimension.

    def inv(self) -> "Kronecker":
        return Kronecker(self.A.inv(), self.B.inv())

    def sqrt(self) -> "Kronecker":
        return Kronecker(self.A.sqrt(), self.B.sqrt())
