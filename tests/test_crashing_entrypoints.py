"""Regression tests for public entry points that used to raise on ordinary use.

Each test here corresponds to a call that previously crashed with a TypeError,
RecursionError, AttributeError or ValueError.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Diagonal, Identity, Kronecker, Matrix
from linox.operators.arithmetic import ldet

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def spd_pair():
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (3, 3))
    B = jax.random.normal(key, (2, 2))
    return A @ A.T + jnp.eye(3), B @ B.T + jnp.eye(2)


def test_block_diag_accepts_varargs(spd_pair) -> None:
    """`block_diag` passed a list to a varargs constructor."""
    A, B = spd_pair
    op = linox.block_diag(Matrix(A), Matrix(B))

    assert op.shape == (5, 5)
    assert jnp.allclose(linox.todense(op), jax.scipy.linalg.block_diag(A, B))


def test_pinv_transpose_terminates(spd_pair) -> None:
    """`PseudoInverseLinearOperator.transpose` recursed into itself forever."""
    A, _ = spd_pair
    op = linox.pinv(Matrix(A))

    assert jnp.allclose(linox.todense(op.transpose()), jnp.linalg.pinv(A).T)
    assert jnp.allclose(linox.todense(op.T), jnp.linalg.pinv(A).T)
    assert (op.T @ jnp.ones(3)).shape == (3,)


def test_ldet_kronecker(spd_pair) -> None:
    """det(A (x) B) is a product of scalars, not of operators."""
    A, B = spd_pair
    K = Kronecker(Matrix(A), Matrix(B))

    assert jnp.allclose(ldet(K), jnp.linalg.det(jnp.kron(A, B)))
    assert jnp.allclose(linox.det(K), jnp.linalg.det(jnp.kron(A, B)))


def test_kronecker_trace(spd_pair) -> None:
    """`Kronecker.trace` called `jnp.trace` on LinearOperators."""
    A, B = spd_pair
    K = Kronecker(Matrix(A), Matrix(B))

    assert jnp.allclose(K.trace(), jnp.trace(jnp.kron(A, B)))


def test_reverse_add_with_array() -> None:
    """`__radd__` passed two positional args to a one-arg `__add__`."""
    op = Matrix(jnp.eye(3) * 2.0)
    arr = jnp.ones((3, 3))

    forward = linox.todense(op + arr)
    reverse = linox.todense(arr + op)

    assert jnp.allclose(forward, reverse)
    assert jnp.allclose(reverse, jnp.eye(3) * 2.0 + 1.0)


def test_diagonal_from_python_list() -> None:
    """`Diagonal` read `.shape` off the raw argument instead of the array."""
    op = Diagonal([1.0, 2.0, 3.0])

    assert op.shape == (3, 3)
    assert jnp.allclose(op @ jnp.ones(3), jnp.array([1.0, 2.0, 3.0]))


def test_trace_and_slogdet_defaults_on_large_operators() -> None:
    """`method="auto"` picked a stochastic path that then demanded a PRNG key."""
    big = Diagonal(jnp.ones(3000))  # larger than the dense threshold

    assert jnp.allclose(linox.trace(big), 3000.0)

    sign, logabsdet = linox.slogdet(big)
    assert jnp.allclose(sign, 1.0)
    assert jnp.allclose(logabsdet, 0.0)

    # Explicitly requesting the stochastic path with a key still works.
    est = linox.trace(big, method="hutchinson", key=jax.random.PRNGKey(0))
    assert jnp.allclose(est, 3000.0)


def test_kronecker_matrix_functions_with_vector(spd_pair) -> None:
    """lexp/llog/lpow on a Kronecker imported a module that no longer exists."""
    from linox.operators.arithmetic import lexp, llog, lpow

    A, B = spd_pair
    K = Kronecker(Matrix(A), Matrix(B))
    v = jnp.ones(6)

    assert lexp(K, v=v).shape == (6,)
    assert llog(K, v=v).shape == (6,)
    assert lpow(K, power=0.5, v=v).shape == (6,)


def test_kronecker_llog_without_vector_computes_a_logarithm(spd_pair) -> None:
    """The v=None branch returned `matrix_exp` -- the wrong matrix function."""
    A, B = spd_pair
    K = Kronecker(Matrix(A), Matrix(B))
    dense = jnp.kron(A, B)

    w, V = jnp.linalg.eigh(dense)
    expected = V @ jnp.diag(jnp.log(w)) @ V.T

    assert jnp.allclose(linox.todense(linox.operators.arithmetic.llog(K)), expected)


def test_lexp_without_vector_uses_a_real_expm(spd_pair) -> None:
    """The dense fallback called the nonexistent `jnp.linalg.matrix_exp`."""
    from linox.operators.arithmetic import lexp

    A, _ = spd_pair
    result = lexp(Matrix(A))

    assert jnp.allclose(jnp.asarray(result), jax.scipy.linalg.expm(A))


def test_svd_with_k_does_not_import_a_removed_module() -> None:
    """`svd(a, k=...)` imported `linox._algorithms._svd`, which is gone."""
    A = jax.random.normal(jax.random.PRNGKey(0), (12, 8))
    U, S, Vt = linox.svd(Matrix(A), k=3, num_iters=8)

    assert U.shape == (12, 3)
    assert S.shape == (3,)
    assert Vt.shape == (3, 8)


def test_isotropic_add_rejects_spectral_ops_on_non_symmetric_operand() -> None:
    """s*I + A silently used eigh (lower triangle only) on non-symmetric A.

    Construction and the matvec/dense paths stay permissive -- `s*I + A` is
    computed correctly for any square A, and `smart_add` funnels every
    `Identity + op` sum through this class. It is the eigh-backed shortcuts
    that must refuse rather than return a wrong answer.
    """
    non_symmetric = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    op = Matrix(non_symmetric) + Identity(2)

    # The correct paths still work.
    assert jnp.allclose(linox.todense(op), non_symmetric + jnp.eye(2))
    assert jnp.allclose(op @ jnp.ones(2), (non_symmetric + jnp.eye(2)) @ jnp.ones(2))

    # The spectral shortcut refuses instead of silently returning the
    # symmetrised answer (which used to be [[0.667, -0.333], [-0.333, 0.667]]
    # rather than the true [[0.5, -0.5], [0.0, 0.5]]).
    with pytest.raises(ValueError, match="symmetric"):
        linox.todense(linox.inverse(op, method="exact"))


def test_isotropic_add_still_fast_paths_symmetric_operands(spd_pair) -> None:
    A, _ = spd_pair
    op = Matrix(A) + Identity(3)

    assert type(op).__name__ == "IsotropicAdditiveLinearOperator"
    assert jnp.allclose(
        linox.todense(linox.inverse(op, method="exact")),
        jnp.linalg.inv(A + jnp.eye(3)),
    )
