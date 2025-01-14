import jax
import jax.numpy as jnp

from linox import LinearOperator


class SymmetricCroppingOperator(LinearOperator):
    """An operator that crops a symmetric matrix.

    This operator is used to crop a operator with equal in and output dimension.
    """

    def __init__(self, A: LinearOperator, padding: int) -> None:
        # Check that the operator has equal in and output dimension
        if A.shape[0] != A.shape[1]:
            msg = "The operator must have equal in and output dimension."
            raise ValueError(msg)

        # If padding is 0, return the original operator
        if padding == 0:
            msg = "Padding is 0, operator is not cropped."
            raise ValueError(msg)

        # Initialize operator
        self.A = A
        self.padding = padding
        shape = (self.A.shape[0] - 2 * self.padding, self.A.shape[1] - 2 * self.padding)
        super().__init__(shape, self.A.dtype)

    def _matmul(self, vector: jax.Array) -> jax.Array:
        # Extend vector with padding
        vector_ext = jnp.zeros((self.A.shape[-1], vector.shape[-1]), dtype=self.dtype)

        vector_ext = vector_ext.at[self.padding : -self.padding, :].set(vector)

        # Apply operator
        return (self.A @ vector_ext)[self.padding : -self.padding]

    def todense(self) -> jax.Array:
        """Return the dense representation of the operator.

        Note: This denses the inner-operator first.
        """
        return self.A.todense()[
            self.padding : -self.padding, self.padding : -self.padding
        ]
