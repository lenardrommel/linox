r"""Low rank representations as linear operators.

This module implements various low rank representations as linear operators, including:

- :class:`LowRank`: Represents a low rank matrix :math:`A = U \text{diag}(S) V^T`
- :class:`SymmetricLowRank`: Represents a symmetric low rank matrix
    :math:`A = U \text{diag}(S) U^T`
- :class:`IsotropicScalingPlusSymmetricLowRank`: Represents
    :math:`\sigma I + U \text{diag}(S) U^T`
- :class:`PositiveDiagonalPlusSymmetricLowRank`: Represents
    :math:`D + \alpha U \text{diag}(S) U^T`
    where :math:`D` is a positive diagonal matrix
"""

import functools

import jax
import jax.numpy as jnp

from linox._arithmetic import AddLinearOperator, ProductLinearOperator, linverse, lsqrt
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Identity


class LowRank(LinearOperator):
    r"""Low rank operator.

    For matrices :math:`U`, :math:`S`, and :math:`V`, this represents the low rank
    matrix :math:`A = U \text{diag}(S) V^T`. The action on a vector :math:`x` is given
    by :math:`Ax = U(S \odot (V^T x))` where :math:`\odot` denotes element-wise
    multiplication.

    Args:
        U: Left factor matrix
        S: Vector of singular values (optional, defaults to ones)
        V: Right factor matrix (optional, defaults to U)
    """

    def __init__(
        self, U: jax.Array, S: jax.Array | None = None, V: jax.Array | None = None
    ) -> None:
        # Check shapes
        if S is not None:
            assert U.shape[-1] == S.shape[-1]  # noqa: S101
        if V is not None:
            assert U.shape[-1] == V.shape[-1]  # noqa: S101

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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self._U, self._S, self._V)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "LowRank":
        del aux_data
        U, S, V = children
        return cls(U=U, S=S, V=V)


class SymmetricLowRank(LowRank):
    r"""Symmetric low rank operator.

    For matrices :math:`U` and :math:`S`, this represents the symmetric low rank matrix
    :math:`A = U \text{diag}(S) U^T`. The action on a vector :math:`x` is given by
    :math:`Ax = U(S \odot (U^T x))` where :math:`\odot` denotes element-wise
    multiplication.

    Args:
        U: Factor matrix
        S: Vector of singular values (optional, defaults to ones)
    """

    def __init__(self, U: jax.Array, S: jax.Array | None = None) -> None:
        super().__init__(U, S)

    def transpose(self) -> "SymmetricLowRank":
        return self

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self._U, self._S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "SymmetricLowRank":
        del aux_data
        U, S = children
        return cls(U=U, S=S)


class IsotropicScalingPlusSymmetricLowRank(AddLinearOperator):
    r"""Isotropic scaling plus symmetric low rank operator.

    For scalar :math:`\sigma`, matrix :math:`U`, and vector :math:`S`, this represents
    :math:`A = \sigma I + U \text{diag}(S) U^T`. The action on a vector :math:`x` is
    given by :math:`Ax = \sigma x + U(S \odot (U^T x))` where :math:`\odot` denotes
    element-wise multiplication.

    Args:
        scalar: Isotropic scaling factor :math:`\sigma`
        U: Factor matrix
        S: Vector of singular values
    """

    def __init__(self, scalar: jax.Array, U: jax.Array, S: jax.Array) -> None:
        self._scalar = scalar

        self._U = U
        self._S = S

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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        # We need to override the AddLinearOperator's tree_flatten
        children = (self._scalar, self._U, self._S)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "IsotropicScalingPlusSymmetricLowRank":
        del aux_data
        scalar, U, S = children
        return cls(scalar=scalar, U=U, S=S)


@linverse.dispatch
def _(a: IsotropicScalingPlusSymmetricLowRank) -> IsotropicScalingPlusSymmetricLowRank:
    r"""Inverse of an isotropic scaling plus symmetric low rank operator.

    For :math:`A = \sigma I + U \text{diag}(S) U^T`, this represents
    :math:`A^{-1} = \frac{1}{\sigma}I - \frac{1}{\sigma}U \text{diag}
    (\frac{S}{\sigma(S + \sigma)}) U^T`
    """
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
    r"""Square root of an isotropic scaling plus symmetric low rank operator.

    For :math:`A = \sigma I + U \text{diag}(S) U^T`, this represents
    :math:`A^{1/2} = \sqrt{\sigma}I + U \text{diag}
    (\sqrt{\sigma}(\sqrt{\frac{S}{\sigma} + 1} - 1)) U^T`
    """
    scalar_sqrt = jnp.sqrt(a.scalar)
    Q, lr_eigvals = a.lr_eigh
    return IsotropicScalingPlusSymmetricLowRank(
        scalar_sqrt,
        Q,
        scalar_sqrt * (jnp.sqrt((lr_eigvals / a.scalar) + 1) - 1),
    )


class PositiveDiagonalPlusSymmetricLowRank(AddLinearOperator):
    r"""Positive diagonal plus symmetric low rank operator.

    For positive diagonal matrix :math:`D`, matrix :math:`U`, vector :math:`S`, and
    scalar :math:`\alpha`, this represents :math:`A = D + \alpha U \text{diag}(S) U^T`.
    The action on a vector :math:`x` is given by
    :math:`Ax = Dx + \alpha U(S \odot (U^T x))` where :math:`\odot` denotes
    element-wise multiplication.

    Args:
        diagonal: Positive diagonal matrix :math:`D`
        low_rank: Symmetric low rank component :math:`U \text{diag}(S) U^T`
        low_rank_scale: Scaling factor :math:`\alpha` (default: 1.0)
    """

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
        """1 + a (D^{-1/2} U) S (D^{-1/2} U)^T = D^{-1/2} (D + a U S U^T) D^{-1/2}."""
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

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        # We need to override the AddLinearOperator's tree_flatten
        children = (self._diagonal, self._low_rank, self._low_rank_scale)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "PositiveDiagonalPlusSymmetricLowRank":
        del aux_data
        diagonal, low_rank, low_rank_scale = children
        return cls(diagonal=diagonal, low_rank=low_rank, low_rank_scale=low_rank_scale)


@linverse.dispatch
def _(A: PositiveDiagonalPlusSymmetricLowRank) -> PositiveDiagonalPlusSymmetricLowRank:
    r"""Inverse of a positive diagonal plus symmetric low rank operator.

    For :math:`A = D + \alpha U \text{diag}(S) U^T`, this represents
    :math:`A^{-1} = D^{-1} - \alpha D^{-1}U(S^{-1} + \alpha U^TD^{-1}U)^{-1}U^TD^{-1}`
    """
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
def _(
    A: PositiveDiagonalPlusSymmetricLowRank,
) -> ProductLinearOperator:
    r"""Square root of a positive diagonal plus symmetric low rank operator.

    For :math:`A = D + \alpha U \text{diag}(S) U^T`, this represents
    :math:`A^{1/2} = D^{1/2}(I + \alpha D^{-1/2}U \text{diag}(S) U^TD^{-1/2})^{1/2}`
    """
    return lsqrt(A.diagonal) @ lsqrt(A._id_plus_low_rank)  # noqa: SLF001


# Register all low rank operators as PyTrees
jax.tree_util.register_pytree_node_class(LowRank)
jax.tree_util.register_pytree_node_class(SymmetricLowRank)
jax.tree_util.register_pytree_node_class(IsotropicScalingPlusSymmetricLowRank)
jax.tree_util.register_pytree_node_class(PositiveDiagonalPlusSymmetricLowRank)
