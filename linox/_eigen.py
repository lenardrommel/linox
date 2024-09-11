import jax

from linox._linear_operator import LinearOperator


class EigenD(LinearOperator):
    def __init__(self, U: jax.Array, S: jax.Array) -> None:
        self.U = U
        self.S = S
        super().__init__(shape=(U.shape[0], U.shape[0]), dtype=S.dtype)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self.U @ (self.S[:, None] * (self.U.T @ vec))

    # def matmul(self, vec: jax.Array) -> jax.Array:
    #     if vec.ndim == 1:
    #         return self.mv(vec[:, None])[..., 0]
    #     return self.mv(vec)
