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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.U, self.S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, any], children: tuple[any, ...]) -> "EigenD":
        del aux_data
        U, S = children
        return cls(U=U, S=S)


# Register EigenD as a PyTree
jax.tree_util.register_pytree_node_class(EigenD)
