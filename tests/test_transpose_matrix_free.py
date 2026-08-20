"""`A.T @ x` must never materialize the dense matrix.

39 of linox's 40 operators override `transpose()` and return a structured
operator, so they were always matrix-free. The gap was the base-class
fallback, which returned `self._todense().swapaxes(-1, -2)` -- so any operator
implementing only `_matmul` and `_todense` (the minimum the base class asks
for) got a transpose that built the whole matrix.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Diagonal, Identity, Kronecker, Matrix, Toeplitz
from linox.operators.arithmetic import TransposedLinearOperator
from linox.operators.base import LinearOperator

jax.config.update("jax_enable_x64", True)


class Opaque(LinearOperator):
    """An operator with only the minimum a subclass must provide.

    Records every dense materialization so tests can assert there are none.
    """

    def __init__(self, array, log):
        self._array = array
        self._log = log
        super().__init__(array.shape, array.dtype)

    def _matmul(self, other):
        return self._array @ other

    def _todense(self):
        self._log.append(1)
        return self._array


@pytest.fixture
def rectangular():
    """A 5x3 operator, so transposed shapes cannot pass by coincidence."""
    log = []
    array = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    return Opaque(array, log), array, log


class TestTransposeIsMatrixFree:
    def test_vector_rhs(self, rectangular) -> None:
        op, array, log = rectangular
        y = jax.random.normal(jax.random.PRNGKey(1), (5,))

        log.clear()
        result = op.T @ y

        assert result.shape == (3,)
        assert jnp.allclose(result, array.T @ y)
        assert log == [], "A.T @ x materialized the dense matrix"

    def test_matrix_rhs(self, rectangular) -> None:
        op, array, log = rectangular
        Y = jax.random.normal(jax.random.PRNGKey(2), (5, 2))

        log.clear()
        result = op.T @ Y

        assert result.shape == (3, 2)
        assert jnp.allclose(result, array.T @ Y)
        assert log == []

    def test_under_jit(self, rectangular) -> None:
        op, array, log = rectangular
        y = jax.random.normal(jax.random.PRNGKey(3), (5,))

        result = jax.jit(lambda v: op.T @ v)(y)
        assert jnp.allclose(result, array.T @ y)

    def test_double_transpose_returns_the_original(self, rectangular) -> None:
        op, _array, _log = rectangular
        assert op.T.transpose() is op

    def test_todense_still_works(self, rectangular) -> None:
        """Materializing on request is fine -- doing it implicitly is not."""
        op, array, _log = rectangular
        assert jnp.allclose(linox.todense(op.T), array.T)


class TestStructuredOperatorsKeepTheirFastPath:
    """The lazy wrapper must only appear when there is no structured transpose."""

    def test_structured_transposes_are_not_wrapped(self) -> None:
        key = jax.random.PRNGKey(0)
        sym = jax.random.normal(key, (4, 4))
        sym = sym @ sym.T

        for op, expected in [
            (Matrix(sym), Matrix),
            (Diagonal(jnp.arange(1.0, 5.0)), Diagonal),
            (Identity(4), Identity),
            (Kronecker(Matrix(sym), Matrix(sym)), Kronecker),
            (Toeplitz(jnp.array([2.0, 1.0, 0.5, 0.1])), Toeplitz),
        ]:
            assert isinstance(op.T, expected), f"{type(op).__name__}.T was wrapped"
            assert not isinstance(op.T, TransposedLinearOperator)

    def test_structured_transpose_values_are_correct(self) -> None:
        key = jax.random.PRNGKey(4)
        dense = jax.random.normal(key, (4, 4))
        assert jnp.allclose(linox.todense(Matrix(dense).T), dense.T)


class TestMixedDtypes:
    """`linear_transpose` demands an exact cotangent dtype match.

    Regression: a float32 operator with a float64 right-hand side raised
    "cotangent type does not match function output". Both are promoted to a
    common dtype rather than silently narrowing the right-hand side.
    """

    @pytest.mark.parametrize(
        ("op_dtype", "rhs_dtype"),
        [
            (jnp.float32, jnp.float64),
            (jnp.float64, jnp.float32),
            (jnp.float32, jnp.float32),
            (jnp.float64, jnp.float64),
        ],
    )
    def test_dtype_combinations(self, op_dtype, rhs_dtype) -> None:
        log = []
        array = jax.random.normal(jax.random.PRNGKey(0), (5, 3)).astype(op_dtype)
        rhs = jax.random.normal(jax.random.PRNGKey(1), (5,)).astype(rhs_dtype)
        op = Opaque(array, log)

        result = op.T @ rhs

        assert jnp.allclose(result, array.T @ rhs, atol=1e-5)
        assert log == []
