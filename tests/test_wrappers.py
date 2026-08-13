"""Tests for the Sym / PSD / SPD structure wrappers.

Regression coverage: these wrappers previously defined `_matmat` and
`_transpose`, neither of which the base class ever calls. The overrides were
dead code, so every matvec fell through to `LinearOperator._matmul`, which
densifies -- the opposite of what wrapping an operator is for.
"""

import jax
import jax.numpy as jnp
import linox
import linox.config as config
from linox import Diagonal, Matrix
from linox.operators.wrappers import (
    PSD,
    SPD,
    Sym,
    assume_psd,
    assume_spd,
    assume_symmetric,
)

jax.config.update("jax_enable_x64", True)

WRAPPERS = [Sym, PSD, SPD]


def _record_events(fn):
    """Run `fn` and return the list of debug event kinds it emitted."""
    events = []
    config.set_debug_hook(lambda e: events.append(e.kind))
    try:
        fn()
    finally:
        config.set_debug_hook(None)
    return events


class TestWrappersStayMatrixFree:
    def test_matvec_does_not_densify(self) -> None:
        d = jnp.arange(1.0, 6.0)
        x = jnp.ones(5)
        for wrapper in WRAPPERS:
            events = _record_events(lambda w=wrapper: w(Diagonal(d)) @ x)
            assert "densify" not in events, (
                f"{wrapper.__name__} densified its operand on matvec"
            )

    def test_matvec_is_correct(self) -> None:
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (4, 4))
        A_dense = X @ X.T + jnp.eye(4)
        x = jnp.ones(4)
        for wrapper in WRAPPERS:
            assert jnp.allclose(wrapper(Matrix(A_dense)) @ x, A_dense @ x)

    def test_matmat_is_correct(self) -> None:
        key = jax.random.PRNGKey(1)
        X = jax.random.normal(key, (4, 4))
        A_dense = X @ X.T + jnp.eye(4)
        B = jax.random.normal(key, (4, 3))
        for wrapper in WRAPPERS:
            assert jnp.allclose(wrapper(Matrix(A_dense)) @ B, A_dense @ B)


class TestWrappersPreserveStructureUnderTranspose:
    def test_transpose_returns_self(self) -> None:
        d = jnp.arange(1.0, 6.0)
        for wrapper in WRAPPERS:
            op = wrapper(Diagonal(d))
            assert op.transpose() is op

    def test_dot_T_preserves_the_wrapper(self) -> None:
        """`.T` must not discard the asserted structure."""
        d = jnp.arange(1.0, 6.0)
        for wrapper in WRAPPERS:
            op = wrapper(Diagonal(d))
            assert isinstance(op.T, wrapper)
            assert op.T.is_symmetric

    def test_transpose_is_numerically_correct(self) -> None:
        key = jax.random.PRNGKey(2)
        X = jax.random.normal(key, (4, 4))
        A_dense = X @ X.T + jnp.eye(4)
        for wrapper in WRAPPERS:
            assert jnp.allclose(linox.todense(wrapper(Matrix(A_dense)).T), A_dense)


class TestWrapperProperties:
    def test_property_flags(self) -> None:
        d = jnp.arange(1.0, 4.0)
        assert Sym(Diagonal(d)).is_symmetric
        assert PSD(Diagonal(d)).is_psd
        assert PSD(Diagonal(d)).is_symmetric
        assert SPD(Diagonal(d)).is_spd
        assert SPD(Diagonal(d)).is_psd

    def test_convenience_constructors(self) -> None:
        d = jnp.arange(1.0, 4.0)
        assert isinstance(assume_symmetric(Diagonal(d)), Sym)
        assert isinstance(assume_psd(Diagonal(d)), PSD)
        assert isinstance(assume_spd(Diagonal(d)), SPD)

    def test_rejects_non_square(self) -> None:
        rect = Matrix(jnp.ones((3, 5)))
        for wrapper in WRAPPERS:
            try:
                wrapper(rect)
            except ValueError:
                continue
            raise AssertionError(f"{wrapper.__name__} accepted a non-square operator")

    def test_roundtrips_through_pytree_flatten(self) -> None:
        d = jnp.arange(1.0, 4.0)
        for wrapper in WRAPPERS:
            op = wrapper(Diagonal(d))
            leaves, treedef = jax.tree_util.tree_flatten(op)
            assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), wrapper)
