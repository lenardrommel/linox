"""Test block operators in linox."""

import jax
import jax.numpy as jnp

from linox import LinearOperator
from linox._block import BlockDiagonal, BlockMatrix, BlockMatrix2x2  # noqa: PLC2701

CaseType = tuple[LinearOperator, jnp.ndarray]


def sample_block() -> CaseType:  # noqa: D103
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, (4, 2)).reshape((4, 2, 2))
    blocks = [[jax.random.normal(key, (4, 3)) for key in keys_row] for keys_row in keys]
    return BlockMatrix(blocks), jnp.block(blocks)


def sample_block2x2() -> CaseType:  # noqa: D103
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    a = jax.random.normal(keys[0], (4, 3))
    b = jax.random.normal(keys[1], (4, 4))
    c = jax.random.normal(keys[2], (5, 3))
    d = jax.random.normal(keys[3], (5, 4))
    return BlockMatrix2x2(a, b, c, d), jnp.block([[a, b], [c, d]])


def sample_block_diagonal() -> CaseType:  # noqa: D103
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    a = jax.random.normal(keys[0], (3, 3))
    b = jax.random.normal(keys[1], (2, 2))
    c = jax.random.normal(keys[2], (5, 5))
    d = jax.random.normal(keys[3], (4, 4))
    return BlockDiagonal(a, b, c, d), jax.scipy.linalg.block_diag(a, b, c, d)


def test_block_matrix() -> None:  # noqa: D103
    block, block_arr = sample_block()
    x = jax.random.normal(jax.random.PRNGKey(0), (block.shape[-1], 16))

    assert jnp.allclose(block.todense(), block_arr)
    assert jnp.allclose(block @ x, block_arr @ x)
    assert jnp.allclose(block @ x[:, 0], block_arr @ x[:, 0])
    assert jnp.allclose(block.T @ x.T, block_arr.swapaxes(-1, -2) @ x.T)


def test_block_matrix2x2() -> None:  # noqa: D103
    block, block_arr = sample_block2x2()
    x = jax.random.normal(jax.random.PRNGKey(0), (block.shape[-1], block.shape[-2]))
    assert jnp.allclose(block.todense(), block_arr)
    # assert jnp.allclose(block @ x, block_arr @ x)
    assert jnp.allclose(block @ x, block_arr @ x, atol=1e-7)
    assert jnp.allclose(block @ x[:, 0], block_arr @ x[:, 0], atol=1e-7)
    assert jnp.allclose(block.T @ x.T, block_arr.swapaxes(-1, -2) @ x.T, atol=1e-7)


def test_block_diagonal() -> None:  # noqa: D103
    block, block_arr = sample_block_diagonal()
    x = jax.random.normal(jax.random.PRNGKey(0), (block.shape[-1], block.shape[-2]))

    assert jnp.allclose(block.todense(), block_arr)
    assert jnp.allclose(block @ x, block_arr @ x, atol=1e-7)
    assert jnp.allclose(block @ x[:, 0], block_arr @ x[:, 0], atol=1e-7)
    assert jnp.allclose(block.T @ x.T, block_arr.swapaxes(-1, -2) @ x.T, atol=1e-7)
