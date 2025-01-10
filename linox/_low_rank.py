"""Low rank (plus isotropic scaling) operator."""

import functools

import jax
import jax.numpy as jnp

from linox._arithmetic import AddLinearOperator, linverse, lsqrt
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Identity


class LowRank(LinearOperator):
    """Low rank operator.

    Low rank operator implements the following matrix A = U @ diag(S) @ V.T.
    """

    def __init__(
        self, U: jax.Array, S: jax.Array | None = None, V: jax.Array | None = None
    ) -> None:
        # Check shapes
        if S is not None:
            assert U.shape[-1] == S.shape[-1]  # noqa: S101
        if V is not None:
            assert U.shape[-1] == V.shape[-1]  # noqa: B015

        self._U = U
        self._S = S if S is not None else jnp.ones(U.shape[-1])
        self._V = V

        # Move to an abstract base class instead
        super().__init__(shape=(*U.shape[:-2], U.shape[-2], U.shape[-2]), dtype=U.dtype)

    @property
    def U(self) -> jax.Array:
        return self._U

    @property
    def S(self) -> jax.Array:
        return self._S

    @property
    def V(self) -> jax.Array:
        return self._V if self._V is not None else self._U

    def _matmul(self, arr: jnp.array) -> jnp.array:
        return self.U @ (self.S[:, None] * (self.V.T @ arr))

    def todense(self) -> jnp.array:
        return self.U @ jnp.diag(self.S) @ self.V.T

    def transpose(self) -> "LowRank":
        return LowRank(self.V, self.S, self.U)


class SymmetricLowRank(LowRank):
    def __init__(self, U: jax.Array, S: jax.Array | None = None) -> None:
        super().__init__(U, S)

    def transpose(self) -> "SymmetricLowRank":
        return self


class IsotropicScalingPlusSymmetricLowRank(AddLinearOperator):
    def __init__(self, scalar: jax.Array, U: jax.Array, S: jax.Array) -> None:
        # assert (
        #     scalar > 0
        # ), "Scalar must be positive in current implementation."  # noqa: S101
        self._scalar = scalar

        self._U = U
        self._S = S
        # TODO: Check if dtype are not equal.

        # Move to an abstract base class instead
        super().__init__(
            self._scalar * Identity(self._U.shape[-2], dtype=self._U.dtype),
            SymmetricLowRank(self._U, self._S),
        )

        # Add general tagging
        # self.is_symmetric = True
        self._lr_eigh = (self._U, self._S)

    @property
    def scalar(self) -> float:
        return self._scalar

    @property
    def U(self) -> jax.Array:
        return self._U

    @property
    def S(self) -> jax.Array:
        return self._S

    @property
    def lr_eigh(self) -> jax.Array:
        return self._lr_eigh

    def transpose(self) -> "IsotropicScalingPlusSymmetricLowRank":
        return self


@linverse.dispatch
def _(a: IsotropicScalingPlusSymmetricLowRank) -> IsotropicScalingPlusSymmetricLowRank:
    scalar_inv = 1 / a.scalar
    Q, lr_eigvals = a.lr_eigh
    return IsotropicScalingPlusSymmetricLowRank(
        scalar_inv,
        Q,
        -lr_eigvals / (a.scalar * (lr_eigvals + a.scalar)),
        # This is numerically more stable than:
        # (1 / (lr_eigvals + a.scalar)) - scalar_inv
    )


@lsqrt.dispatch
def _(a: IsotropicScalingPlusSymmetricLowRank) -> IsotropicScalingPlusSymmetricLowRank:
    scalar_sqrt = jnp.sqrt(a.scalar)
    Q, lr_eigvals = a.lr_eigh
    return IsotropicScalingPlusSymmetricLowRank(
        scalar_sqrt,
        Q,
        scalar_sqrt * (jnp.sqrt((lr_eigvals / a.scalar) + 1) - 1),
    )


class PositiveDiagonalPlusSymmetricLowRank(AddLinearOperator):
    """A = D + a U S U^T."""

    def __init__(
        self,
        diagonal: Diagonal,  # D
        low_rank: SymmetricLowRank,  # U S U^T
        low_rank_scale: float = 1.0,  # a
    ) -> None:
        self._diagonal = diagonal
        self._low_rank = low_rank
        self._low_rank_scale = low_rank_scale

        super().__init__(self._diagonal, self._low_rank_scale * self._low_rank)

    @property
    def diagonal(self) -> jax.Array:
        return self._diagonal

    @property
    def low_rank(self) -> SymmetricLowRank:
        return self._low_rank

    @property
    def low_rank_scale(self) -> float:
        return self._low_rank_scale

    @functools.cached_property
    def _id_plus_low_rank(self) -> IsotropicScalingPlusSymmetricLowRank:
        """1 + a (D^{-1/2} U) S (D^{-1/2} U)^T = D^{-1/2} (D + a U S U^T) D^{-1/2}"""
        U, sqrt_S, _ = jnp.linalg.svd(
            (
                (self.low_rank.U * jnp.sqrt(self.low_rank.S))
                / jnp.sqrt(self._diagonal.diag[:, None])
            ),
            full_matrices=False,
            compute_uv=True,
        )

        return IsotropicScalingPlusSymmetricLowRank(
            1.0,
            U,
            self._low_rank_scale * sqrt_S**2,
        )

    def transpose(self) -> "PositiveDiagonalPlusSymmetricLowRank":
        return self


@linverse.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank) -> PositiveDiagonalPlusSymmetricLowRank:
    # (D + a U S U^T)^{-1} = D^{-1} - a D^{-1} U (S^{-1} + a U^T D^{-1} U)^-1 U^T D^{-1}

    D_inv = linverse(A.diagonal)

    schur = A.low_rank_scale * A.low_rank.U.T @ D_inv @ A.low_rank.U + jnp.diag(
        1 / A.low_rank.S
    )
    schur_eigvals, schur_eigvecs = jnp.linalg.eigh(schur)

    # Compute eigendecomposition of (D^{-1} U schur^{-1/2}) (D^{-1} U schur^{-1/2})^T
    U, sqrt_S, _ = jnp.linalg.svd(
        D_inv @ A.low_rank.U @ schur_eigvecs / jnp.sqrt(schur_eigvals),
        full_matrices=False,
        compute_uv=True,
    )

    return PositiveDiagonalPlusSymmetricLowRank(
        linverse(A.diagonal),
        SymmetricLowRank(U, sqrt_S**2),
        low_rank_scale=-A.low_rank_scale,
    )


@lsqrt.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank):
    return lsqrt(A.diagonal) @ lsqrt(A._id_plus_low_rank)
