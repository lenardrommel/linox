"""Basic test for the Kronecker operator."""

import jax
import jax.numpy as jnp

from linox import Kronecker, LinearOperator
from linox.types import ShapeLike

CaseType = tuple[LinearOperator, Kronecker]


def sample_kronecker(shapeA: ShapeLike, shapeB: ShapeLike) -> CaseType:
    key = jax.random.key(5)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, shapeA)
    B = jax.random.normal(key2, shapeB)
    op = Kronecker(A, B)
    return op, jnp.kron(A, B)


def test_kronecker() -> None:
    op, arr = sample_kronecker((2, 2), (3, 2))
    key = jax.random.key(1)
    vec = jax.random.normal(key, op.shape[::-1])
    assert jnp.allclose(op.todense(), arr)
    assert jnp.allclose(op @ vec, arr @ vec)
    assert jnp.allclose(op.T @ vec.T, arr.T @ vec.T)
    assert jnp.allclose(op @ vec[..., 0], arr @ vec[..., 0])
