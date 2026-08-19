"""Stacked plum decorators leak one function's signature into another.

A plum `.dispatch` decorator returns the *Function object*, not the plain
function. Stacking two of them therefore registers the inner Function under
the outer one, using the inner's own generic signature:

    @lsqrt.dispatch
    @linverse.dispatch
    def _(a: Identity) -> Identity: ...

registered `linverse` itself as `lsqrt`'s `(LinearOperator)` method -- so every
operator without a specific `lsqrt` silently received its *inverse*.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Identity, Matrix, Ones, Scalar, Toeplitz
from linox.operators.arithmetic import linverse, lmatmul, lmul, lsqrt

jax.config.update("jax_enable_x64", True)


class TestNoLeakedImplementations:
    """No dispatch table may contain a method implemented by another generic."""

    @pytest.mark.parametrize(
        ("outer", "leaked_name"),
        [(lsqrt, "linverse"), (lmatmul, "lmul")],
        ids=["lsqrt<-linverse", "lmatmul<-lmul"],
    )
    def test_table_is_clean(self, outer, leaked_name) -> None:
        leaked = [m for m in outer.methods if leaked_name in str(m.implementation)]
        assert leaked == [], (
            f"{leaked_name} leaked into {outer.__name__}'s dispatch table: {leaked}"
        )


class TestLsqrtDoesNotReturnTheInverse:
    @pytest.mark.parametrize(
        "op",
        [Toeplitz(jnp.array([4.0, 1.0])), Ones((3, 3))],
        ids=["Toeplitz", "Ones"],
    )
    def test_unsupported_operators_raise_rather_than_invert(self, op) -> None:
        """Better an honest NotImplementedError than a silent wrong answer."""
        with pytest.raises(NotImplementedError):
            lsqrt(op)

    def test_supported_operators_return_a_left_square_root(self) -> None:
        """The documented contract: S @ S.T == A."""
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (4, 4))
        spd = X @ X.T + 4 * jnp.eye(4)

        for op in [
            Matrix(spd),
            linox.Diagonal(jnp.arange(1.0, 5.0)),
            Identity(4),
            linox.Kronecker(Matrix(spd), Matrix(spd)),
            Matrix(spd) + 0.5 * Identity(4),
        ]:
            factor = linox.todense(lsqrt(op))
            dense = linox.todense(op)
            assert jnp.allclose(factor @ factor.T, dense, atol=1e-6), (
                f"{type(op).__name__} is not a left square root"
            )


class TestStackedRegistrationsStillWorkIndividually:
    def test_identity_keeps_both_dispatches(self) -> None:
        assert isinstance(lsqrt(Identity(3)), Identity)
        assert isinstance(linverse(Identity(3)), Identity)

    def test_scalar_keeps_both_dispatches(self) -> None:
        assert float(lsqrt(Scalar(4.0)).scalar) == 2.0
        # (alpha I)(beta I) == (alpha beta) I. `lmatmul` previously raised
        # IndexError here, its correct method having been displaced by the leak.
        assert float(lmatmul(Scalar(2.0), Scalar(3.0)).scalar) == 6.0
        assert float(lmul(Scalar(2.0), Scalar(3.0)).scalar) == 6.0


def test_no_stacked_plum_decorators_remain() -> None:
    """Guard the pattern itself, not just today's two instances."""
    import pathlib
    import re

    root = pathlib.Path(linox.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        lines = path.read_text().splitlines()
        for i in range(len(lines) - 1):
            first, second = lines[i].strip(), lines[i + 1].strip()
            if re.match(r"^@\w+\.(dispatch|register)", first) and re.match(
                r"^@\w+\.(dispatch|register)", second
            ):
                offenders.append(f"{path.name}:{i + 1}  {first} / {second}")

    assert offenders == [], "stacked plum decorators leak signatures: " + "; ".join(
        offenders
    )
