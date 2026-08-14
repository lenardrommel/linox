"""Symmetry and positivity preconditions on ``s*I + A``.

Every spectral shortcut on :class:`IsotropicAdditiveLinearOperator` goes
through ``eigh``, which reads only the lower triangle -- so a non-symmetric
operand used to return a silently wrong inverse. Symmetry alone is enough for
inverse/eigh/slogdet; sqrt, Cholesky, log and fractional powers additionally
need the shifted spectrum ``s + lambda`` to be non-negative.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Identity, Matrix
from linox.operators.arithmetic import (
    lcholesky,
    leigh,
    linverse,
    llog,
    lpow,
    lsqrt,
    slogdet,
)

jax.config.update("jax_enable_x64", True)

_Q = jnp.array([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(2.0)
INDEFINITE = _Q @ jnp.diag(jnp.array([-2.0, 3.0])) @ _Q.T   # symmetric, eigs -2, +3
PSD = _Q @ jnp.diag(jnp.array([1.0, 3.0])) @ _Q.T           # symmetric, eigs 1, 3
NON_SYMMETRIC = jnp.array([[1.0, 2.0], [0.0, 1.0]])


def iso(dense, s=0.1):
    return Matrix(dense) + s * Identity(2)


class TestSymmetryIsRequired:
    def test_non_symmetric_rejected_by_spectral_ops(self) -> None:
        op = iso(NON_SYMMETRIC, s=1.0)
        with pytest.raises(ValueError, match="symmetric"):
            linox.todense(linverse(op))

    def test_non_symmetric_still_allowed_for_matvec_and_dense(self) -> None:
        """`s*I + A` is computed correctly for any square A."""
        op = iso(NON_SYMMETRIC, s=1.0)
        expected = NON_SYMMETRIC + jnp.eye(2)

        assert jnp.allclose(linox.todense(op), expected)
        assert jnp.allclose(op @ jnp.ones(2), expected @ jnp.ones(2))

    def test_symmetric_operand_is_accepted(self) -> None:
        op = iso(PSD)
        assert jnp.allclose(
            linox.todense(linverse(op)), jnp.linalg.inv(PSD + 0.1 * jnp.eye(2))
        )


class TestSymmetryIsCheckedUnderJit:
    """Regression: the check used to be skipped entirely under `jax.jit`."""

    def test_non_symmetric_raises_under_jit(self) -> None:
        fn = jax.jit(lambda A: linox.todense(linverse(iso(A, s=1.0))))
        with pytest.raises(Exception, match="symmetric"):
            jax.block_until_ready(fn(NON_SYMMETRIC))

    def test_symmetric_passes_under_jit(self) -> None:
        fn = jax.jit(lambda A: linox.todense(linverse(iso(A))))
        assert jnp.allclose(
            jax.block_until_ready(fn(PSD)), jnp.linalg.inv(PSD + 0.1 * jnp.eye(2))
        )


class TestPositivityIsPerOperation:
    """Symmetry suffices for inverse/eigh/slogdet on an indefinite operand."""

    @pytest.mark.parametrize(
        "op_fn", [linverse, slogdet, leigh], ids=["inverse", "slogdet", "eigh"]
    )
    def test_indefinite_allowed_where_well_defined(self, op_fn) -> None:
        op_fn(iso(INDEFINITE))

    @pytest.mark.parametrize(
        ("op_fn", "name"),
        [
            (lsqrt, "sqrt"),
            (lcholesky, "cholesky"),
            (llog, "log"),
            (lambda a: lpow(a, power=0.5), "pow(0.5)"),
        ],
    )
    def test_indefinite_rejected_where_undefined(self, op_fn, name) -> None:
        with pytest.raises(ValueError, match="spectrum"):
            op_fn(iso(INDEFINITE))

    def test_integer_power_allowed_on_indefinite(self) -> None:
        """A negative base is fine for an integer exponent."""
        lpow(iso(INDEFINITE), power=2)

    @pytest.mark.parametrize(
        ("op_fn", "name"),
        [
            (lsqrt, "sqrt"),
            (lcholesky, "cholesky"),
            (llog, "log"),
            (lambda a: lpow(a, power=0.5), "pow(0.5)"),
        ],
    )
    def test_psd_allowed_everywhere(self, op_fn, name) -> None:
        op_fn(iso(PSD))

    def test_sqrt_of_indefinite_raises_under_jit(self) -> None:
        fn = jax.jit(lambda A: linox.todense(lsqrt(iso(A))))
        with pytest.raises(Exception, match="spectrum"):
            jax.block_until_ready(fn(INDEFINITE))

    def test_shift_can_lift_an_indefinite_spectrum(self) -> None:
        """s + lambda is what matters, not the sign of lambda alone."""
        op = iso(INDEFINITE, s=3.0)  # eigenvalues become +1 and +6
        root = linox.todense(lsqrt(op))
        assert jnp.allclose(root @ root.T, INDEFINITE + 3.0 * jnp.eye(2), atol=1e-10)


class TestCheckStaysMatrixFree:
    """The symmetry check must not densify a structured operand.

    `leigh` has structured dispatches that never build the dense matrix -- for
    a Kronecker operator a dense symmetry check would be the only O(n^2) step
    in the whole path, which is exactly backwards for this library.
    """

    def _densify_count(self, fn):
        import linox.config as config

        events = []
        config.set_debug_hook(lambda e: events.append(e.kind))
        try:
            fn()
        finally:
            config.set_debug_hook(None)
        return events.count("densify")

    def test_kronecker_operand_is_not_densified(self) -> None:
        from linox import Kronecker

        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (4, 4))
        B = jax.random.normal(key, (4, 4))
        kron = Kronecker(Matrix(A @ A.T), Matrix(B @ B.T))

        count = self._densify_count(lambda: linverse(kron + 0.1 * Identity(16)))
        assert count == 0

    def test_non_symmetric_structured_operand_still_caught(self) -> None:
        """Matrix-free probing, not a dense comparison, does the catching."""
        from linox import Kronecker

        kron = Kronecker(Matrix(NON_SYMMETRIC), Matrix(jnp.eye(2)))
        with pytest.raises(ValueError, match="symmetric"):
            linverse(kron + 0.1 * Identity(4))
