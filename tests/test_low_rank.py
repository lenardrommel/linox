"""Test low-rank operator."""

import jax
import jax.numpy as jnp
import pytest

from linox import (
    Diagonal,
    IsotropicScalingPlusSymmetricLowRank,
    LinearOperator,
    LowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    linverse,
    lsqrt,
)
from linox.typing import ShapeLike

CaseType = tuple[LinearOperator, jnp.ndarray]


def draw_random_onb(shape: ShapeLike) -> jnp.ndarray:
    random_key1 = jax.random.PRNGKey(42)
    random_key2 = jax.random.PRNGKey(52)
    A = jax.random.uniform(random_key1, 2 * shape[:1])
    A = 0.5 * (A + A.T)
    A += jnp.eye(shape[0])
    U, _ = jnp.linalg.qr(A)
    S = jax.random.uniform(random_key2, (shape[1],))
    return U[: shape[1], :].T, S


def draw_positive_diagonal(shape: ShapeLike) -> jnp.ndarray:
    random_key = jax.random.PRNGKey(48987)
    return jax.random.uniform(random_key, (shape[0],))


def get_low_rank_operator(shape: ShapeLike) -> CaseType:
    U, S = draw_random_onb(shape)
    return LowRank(U, S), U @ jnp.diag(S) @ U.T


def get_isotropic_scaling_plus_low_rank(shape: ShapeLike) -> CaseType:
    scalar = 10 * jax.random.uniform(jax.random.PRNGKey(0), ())
    U, S = draw_random_onb(shape)
    return IsotropicScalingPlusSymmetricLowRank(scalar, U, S), scalar * jnp.eye(
        U.shape[0]
    ) + U @ jnp.diag(S) @ U.T


def get_positive_diagonal_symmetric_low_rank(shape: ShapeLike) -> CaseType:
    U, S = draw_random_onb(shape)
    diag = Diagonal(draw_positive_diagonal(shape))
    low_rank = LowRank(U, S)
    return PositiveDiagonalPlusSymmetricLowRank(
        diag, low_rank
    ), diag.todense() + low_rank.todense()


def test_low_rank() -> None:
    shape = (20, 3)
    op, arr = get_low_rank_operator(shape)
    x = jax.random.normal(jax.random.PRNGKey(0), (shape[-2], shape[-2]))
    assert op.shape == 2 * shape[:1]
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ x, arr @ x, atol=1e-7)
    assert jnp.allclose(op @ x[:, 0], arr @ x[:, 0], atol=1e-7)
    assert jnp.allclose(op.T @ x.T, arr.swapaxes(-1, -2) @ x.T, atol=1e-7)


def test_isotropic_scaling_plus_low_rank() -> None:
    shape = (20, 3)
    op, arr = get_isotropic_scaling_plus_low_rank(shape)
    x = jax.random.normal(jax.random.PRNGKey(0), (shape[-2], shape[-2]))
    assert op.shape == 2 * shape[:1]
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ x, arr @ x, atol=1e-7)
    assert jnp.allclose(op @ x[:, 0], arr @ x[:, 0], atol=1e-7)
    assert jnp.allclose(op.T @ x.T, arr.swapaxes(-1, -2) @ x.T, atol=1e-7)


def test_isotropic_scaling_plus_low_rank_lsqrt() -> None:
    shape = (20, 3)
    op, arr = get_isotropic_scaling_plus_low_rank(shape)
    sqrt_op = lsqrt(op)
    # sqrt_arr = jnp.linalg.cholesky(arr)
    assert jnp.allclose((sqrt_op @ sqrt_op.T).todense(), arr, atol=1e-6)


def test_isotropic_scaling_plus_low_rank_linverse() -> None:
    shape = (20, 8)
    op, _ = get_isotropic_scaling_plus_low_rank(shape)
    inv_op = linverse(op)
    assert jnp.allclose((inv_op @ op).todense(), jnp.eye(op.shape[0]), atol=1e-7)


@pytest.fixture(params=[(5, 5), (4, 2), (20, 3), (20, 8)])
def shape(request) -> ShapeLike:
    return request.param


def test_positive_diagonal_symmetric_low_rank(shape: ShapeLike) -> None:
    # shape = (20, 3)
    op, arr = get_positive_diagonal_symmetric_low_rank(shape)
    x = jax.random.normal(jax.random.PRNGKey(0), (shape[-2], shape[-2]))
    assert op.shape == 2 * shape[:1]
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ x, arr @ x, atol=1e-7)
    assert jnp.allclose(op @ x[:, 0], arr @ x[:, 0], atol=1e-7)
    assert jnp.allclose(op.T @ x.T, arr.swapaxes(-1, -2) @ x.T, atol=1e-7)


def test_positive_diagonal_symmetric_low_rank_lsqrt(shape: ShapeLike) -> None:
    shape = (20, 3)
    op, arr = get_positive_diagonal_symmetric_low_rank(shape)
    sqrt_op = lsqrt(op)
    assert jnp.allclose((sqrt_op @ sqrt_op.T).todense(), arr, atol=1e-6)


def test_positive_diagonal_symmetric_low_rank_linverse(shape: ShapeLike) -> None:
    shape = (20, 8)
    op, _ = get_isotropic_scaling_plus_low_rank(shape)
    inv_op = linverse(op)
    assert jnp.allclose((inv_op @ op).todense(), jnp.eye(op.shape[0]), atol=1e-7)
