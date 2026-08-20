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
import pytest
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


class TestWrappersDelegateCapabilities:
    """A wrapper must never be less capable than the operator it wraps.

    `Sym`/`PSD`/`SPD` are tags: they promise a property, they do not change what
    the operator computes. Before delegation was registered, the generic
    fallback saw only the wrapper type, so `sqrt` raised `NotImplementedError`
    for every wrapped operator -- including ones whose own `sqrt` is exact.
    """

    @pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
    def test_sqrt_is_available_through_the_wrapper(self, wrapper) -> None:
        key = jax.random.PRNGKey(3)
        X = jax.random.normal(key, (5, 5))
        A_dense = X @ X.T + 5 * jnp.eye(5)

        factor = linox.sqrt(wrapper(Matrix(A_dense)), method="exact")

        dense = linox.todense(factor)
        assert jnp.allclose(dense @ dense.T, A_dense, atol=1e-8), (
            f"{wrapper.__name__} did not produce a valid factor S with S @ S.T == A"
        )

    @pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
    def test_sqrt_matches_the_unwrapped_operator(self, wrapper) -> None:
        key = jax.random.PRNGKey(4)
        X = jax.random.normal(key, (5, 5))
        A = Matrix(X @ X.T + 5 * jnp.eye(5))

        assert jnp.allclose(
            linox.todense(linox.sqrt(wrapper(A), method="exact")),
            linox.todense(linox.sqrt(A, method="exact")),
        )

    @pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
    def test_sqrt_preserves_the_wrapped_structure(self, wrapper) -> None:
        # The point of delegating rather than densifying: a Diagonal's square
        # root is a Diagonal, and wrapping must not cost that.
        d = jnp.arange(1.0, 6.0)

        factor = linox.sqrt(wrapper(Diagonal(d)), method="exact")

        assert isinstance(factor, Diagonal), (
            f"{wrapper.__name__} lost the Diagonal structure, got {type(factor).__name__}"
        )
        assert jnp.allclose(linox.todense(factor), jnp.diag(jnp.sqrt(d)))

    @pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
    def test_sqrt_does_not_densify(self, wrapper) -> None:
        d = jnp.arange(1.0, 6.0)

        events = _record_events(lambda w=wrapper: linox.sqrt(w(Diagonal(d)), method="exact"))

        assert "densify" not in events, (
            f"{wrapper.__name__} densified its operand to take a square root"
        )

    @pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.__name__)
    def test_sqrt_never_returns_the_inverse(self, wrapper) -> None:
        # Regression for the stacked-decorator leak: `lsqrt`'s generic method
        # was `linverse`, so wrapped operators silently received A^-1.
        d = jnp.array([4.0, 9.0, 16.0, 25.0])

        factor = linox.sqrt(wrapper(Diagonal(d)), method="exact")

        assert jnp.allclose(linox.todense(factor), jnp.diag(jnp.array([2.0, 3.0, 4.0, 5.0])))
