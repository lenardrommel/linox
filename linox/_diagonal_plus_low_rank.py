import jax
import jax.numpy as jnp

from linox import LinearOperator
from linox._arithmetic import lmatmul


class IsotropicScalingPlusLowRank(LinearOperator):
    def __init__(
        self,
        scalar: jax.Array,
        U: jax.Array,
        S: jax.Array,
    ) -> None:
        self._scalar = scalar
        self._U = U
        self._S = S

        # Move to an abstract base class instead
        super().__init__(shape=2 * self._U.shape[:1], dtype=self._U.dtype)

        # Add general tagging
        self.is_symmetric = True
        self._lr_eigh = (self._U, self._S)

    @property
    def scalar(
        self,
    ) -> float:
        return self._scalar

    @property
    def U(
        self,
    ) -> jax.Array:
        return self._U

    @property
    def S(
        self,
    ) -> jax.Array:
        return self._S

    @property
    def lr_eigh(
        self,
    ) -> jax.Array:
        # TODO(2bys): introduce case where S is not given
        return self._lr_eigh

    def mv(self, x: jnp.array) -> jnp.array:
        return self._scalar * x + self._U @ (self._S[:, None] * (self._U.T @ x))

    # def matmul(self, x: jnp.array) -> jnp.array:
    #     if x.ndim == 1:
    #         return self.mv(x[:, None])[..., 0]
    #     return self.mv(x)

    # This could be used for speeding some computations. However, above needs to
    # be implemented with self._S[:, None] ensuring that x is array.

    def inv(self) -> "IsotropicScalingPlusLowRank":
        scalar_inv = 1 / self._scalar
        Q, lr_eigvals = self.lr_eigh
        return IsotropicScalingPlusLowRank(
            scalar_inv,
            Q,
            # (1 / (lr_eigvals + self._scalar)) - scalar_inv,
            -lr_eigvals / (self._scalar * (lr_eigvals + self._scalar)),
            # The latter seems numerically more stable.
        )

    def sqrt(self) -> "IsotropicScalingPlusLowRank":
        scalar_sqrt = jnp.sqrt(self._scalar)
        Q, lr_eigvals = self.lr_eigh
        return IsotropicScalingPlusLowRank(
            scalar_sqrt, Q, scalar_sqrt * (jnp.sqrt((lr_eigvals / self.scalar) + 1) - 1)
        )


# isotropic mv is implemented for arrays, not for scalars
@lmatmul.dispatch
def _(a: IsotropicScalingPlusLowRank, b: jax.Array) -> jax.Array:
    return a.mv(b)
