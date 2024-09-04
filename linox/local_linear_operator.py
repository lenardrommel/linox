from functools import partial

import jax
import jax.numpy as jnp


class LinearOperator:
    def __init__(self, shape: tuple[int, int], dtype: jnp.dtype) -> None:  # noqa: D107
        self.shape = shape
        self.dtype = dtype

    @partial(jax.jit, static_argnums=0)
    def __matmul__(self, vec: jax.Array) -> jax.Array:
        """Matrix-matrix product.

        This takes the locally implemented matrix-vector product (mv) and applies it to
        a matrix. Vectors are automatically shared. It contains also the logic for
        turning a matrix-vector product into a dense matrix.
        """
        if hasattr(self, "matmul"):  # This for the cases where mv supports matmul.
            return self.matmul(vec)

        if len(vec.shape) == 1:
            vec = vec[:, None]
            flatten = True
        else:
            flatten = False

        if vec.shape[-2] != self.shape[-1]:
            msg = f"expected vec.shape[-2] to be {self.shape[-1]}, got {vec.shape[-2]} instead."  # noqa: E501
            raise ValueError(msg)

        if len(vec.shape) > 2:
            msg = "Only 2D arrays are supported."
            raise ValueError(msg)

        if vec.dtype != self.dtype:
            msg = f"expected vec.dtype to be {self.dtype}, got {vec.dtype} instead."
            raise ValueError(msg)

        # # This text is for cnn - last_layer - full - hidden dim.: 12
        # # Alternatively to below, we can use jax.vmap for last_layer.
        # # During calibration this is around 0.28 sec.
        # res = jax.vmap(self.mv)(vec.T).T
        # # However, it might break the memory.

        # # This works for higher dimensions, but it is slower.
        # # During calibration it is around 4.8 sec.
        res = jax.lax.map(
            self.mv,
            vec.T,
            batch_size=1,
        ).T  # jax.lax.map shares over first axes.
        # # # Increasing batch_size brings it closer, but for higher dimension
        # # Batch size 10 is already difficult.
        # # Hidden dim: 32: 10 sec of calibration round.
        # # Hidden dim: 64: 30 sec of calibration round.
        return res if not flatten else res[..., :, 0]

    def __call__(self, vec: jax.Array) -> jax.Array:
        return self @ vec

    def todense(self) -> jax.Array:
        return self @ jnp.eye(self.shape[0])

    def to_dense(self) -> jax.Array:
        return self.todense()

    def mv(self, vec: jax.Array) -> jax.Array:
        raise NotImplementedError
