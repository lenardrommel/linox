"""Regression tests: topk_eigh must return eigenvectors that match its eigenvalues.

``topk_eigh`` used to store each factor's eigenvectors already sorted, while
``KronTopkEighInfo`` documents them in original order and every consumer
(``KroneckerSelectedEigenvectors``, ``whitened_selected_columns``) maps sorted
indices through ``sort_indices`` again. Whenever the sort permutation was not
the identity (e.g. ``largest=True``), eigenvalues were paired with the wrong
eigenvectors.
"""

import jax
import jax.numpy as jnp
import pytest

import linox
from linox.operators.kron import (
    Kronecker,
    build_kron_columns_from_factors,
    topk_eigh,
    whitened_selected_columns,
)

jax.config.update("jax_enable_x64", True)


def _spd(n: int, seed: int) -> jax.Array:
    a = jax.random.normal(jax.random.PRNGKey(seed), (n, n))
    return a @ a.T + 0.1 * jnp.eye(n)


def _factors(sizes: tuple[int, ...]) -> list[jax.Array]:
    return [_spd(n, seed) for seed, n in enumerate(sizes, start=1)]


def _dense_kron(mats: list[jax.Array]) -> jax.Array:
    out = mats[0]
    for m in mats[1:]:
        out = jnp.kron(out, m)
    return out


def _max_relative_residual(dense: jax.Array, vals: jax.Array, vecs: jax.Array) -> float:
    vecs = vecs / jnp.linalg.norm(vecs, axis=0, keepdims=True)
    res = jnp.linalg.norm(dense @ vecs - vecs * vals[None, :], axis=0)
    return float(jnp.max(res / jnp.abs(vals)))


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("sizes", [(3, 4), (3, 4, 2)])
def test_selected_eigenvector_operator_matches_eigenvalues(largest, sizes):
    mats = _factors(sizes)
    op = Kronecker(linox.Matrix(mats[0]), linox.Matrix(mats[1]))
    for m in mats[2:]:
        op = Kronecker(op, linox.Matrix(m))
    k = 5
    vals, vecs, _ = topk_eigh(op, k=k, largest=largest)
    dense = _dense_kron(mats)

    expected = jnp.sort(jnp.linalg.eigvalsh(dense))
    expected = expected[::-1][:k] if largest else expected[:k]
    assert jnp.allclose(vals, expected)
    assert _max_relative_residual(dense, vals, vecs @ jnp.eye(k)) < 1e-8


@pytest.mark.parametrize("largest", [True, False])
def test_whitened_selected_columns_are_eigenvectors(largest):
    mats = _factors((3, 4))
    vals, _, info = topk_eigh([linox.Matrix(m) for m in mats], k=4, largest=largest)
    cols, _, _, full_eigs = whitened_selected_columns(info, scale_on_factor=0)
    assert jnp.allclose(full_eigs, vals)
    vecs = build_kron_columns_from_factors(cols)
    assert _max_relative_residual(_dense_kron(mats), vals, vecs) < 1e-8


@pytest.mark.parametrize("largest", [True, False])
def test_info_factor_vecs_are_in_original_order(largest):
    mats = _factors((3, 4))
    _, _, info = topk_eigh([linox.Matrix(m) for m in mats], k=2, largest=largest)
    for A, Q, w_sorted, order in zip(mats, info.factor_vecs, info.factor_eigs, info.sort_indices, strict=True):
        # Sorted eigenvalue j belongs to original column order[j].
        Q_for_sorted = Q[:, order]
        assert jnp.allclose(A @ Q_for_sorted, Q_for_sorted * w_sorted[None, :], atol=1e-8)


def test_public_eigh_on_kronecker_returns_eigenvectors():
    mats = _factors((3, 4))
    op = Kronecker(linox.Matrix(mats[0]), linox.Matrix(mats[1]))
    vals, vecs = linox.linalg.spectral.eigh(op, k=3)
    assert _max_relative_residual(_dense_kron(mats), vals, vecs @ jnp.eye(3)) < 1e-8
