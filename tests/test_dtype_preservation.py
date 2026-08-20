"""Working in float64 must yield float64 -- linox must not narrow results.

`Identity`, `Zero`, `Ones` and `Permutation` synthesise their own values, and
each hard-coded `float32`. Under `jax_enable_x64` that silently narrowed
results for anyone working in double precision.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import (
    Diagonal,
    Identity,
    Kronecker,
    Matrix,
    Ones,
    Permutation,
    Toeplitz,
    Zero,
)
from linox.utils.array import default_floating_dtype

jax.config.update("jax_enable_x64", True)

F64 = jnp.float64


@pytest.fixture
def spd64():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (4, 4), dtype=F64)
    return X @ X.T + 4 * jnp.eye(4, dtype=F64)


class TestSynthesisingOperatorsFollowTheX64Setting:
    """These build their own values, so they set the dtype rather than inherit."""

    OPERATORS = [
        ("Identity", lambda: Identity(3)),
        ("Zero", lambda: Zero((3, 3))),
        ("Ones", lambda: Ones((3, 3))),
        ("Permutation", lambda: Permutation(jnp.array([1, 0, 2]))),
    ]

    @pytest.mark.parametrize(("name", "build"), OPERATORS)
    def test_default_is_float64_under_x64(self, name, build) -> None:
        assert linox.todense(build()).dtype == F64, f"{name} narrowed to float32"

    @pytest.mark.parametrize(("name", "build"), OPERATORS)
    def test_default_follows_the_flag_when_disabled(self, name, build) -> None:
        """Not simply pinned to float64 -- it tracks the setting."""
        jax.config.update("jax_enable_x64", False)
        try:
            assert linox.todense(build()).dtype == jnp.float32
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_explicit_dtype_still_wins(self) -> None:
        assert linox.todense(Identity(3, dtype=jnp.float32)).dtype == jnp.float32
        assert linox.todense(Zero((3, 3), dtype=jnp.float32)).dtype == jnp.float32


class TestOperationsPreserveFloat64:
    def test_core_operations(self, spd64) -> None:
        from linox.operators.arithmetic import (
            diagonal,
            lcholesky,
            leigh,
            linverse,
            lsqrt,
        )

        op = Matrix(spd64)
        vec = jnp.ones(4, dtype=F64)

        assert (op @ vec).dtype == F64
        assert linox.todense(op).dtype == F64
        assert linox.solve(op, vec).dtype == F64
        assert linox.todense(lsqrt(op)).dtype == F64
        assert linox.todense(linverse(op)).dtype == F64
        assert leigh(op)[0].dtype == F64
        assert linox.todense(lcholesky(op)).dtype == F64
        assert diagonal(op).dtype == F64
        assert linox.todense(op.T).dtype == F64

    def test_structured_operators(self, spd64) -> None:
        for op in [
            Matrix(spd64),
            Diagonal(jnp.arange(1.0, 5.0, dtype=F64)),
            Kronecker(Matrix(spd64), Matrix(spd64)),
            Toeplitz(jnp.array([4.0, 1.0, 0.5, 0.1], dtype=F64)),
            Matrix(spd64) + 0.1 * Identity(4),
        ]:
            assert linox.todense(op).dtype == F64, type(op).__name__


class TestNoNarrowingWhenMixing:
    """A synthesising operator must not drag a float64 operand down."""

    @pytest.mark.parametrize(
        ("name", "build"),
        [
            ("Identity", lambda: Identity(4)),
            ("Zero", lambda: Zero((4, 4))),
            ("Ones", lambda: Ones((4, 4))),
        ],
    )
    def test_matvec_against_float64(self, name, build) -> None:
        result = build() @ jnp.ones(4, dtype=F64)
        assert result.dtype == F64, f"{name} narrowed the result"

    def test_float32_operator_with_float64_rhs_promotes(self, spd64) -> None:
        """Promotion, not truncation, when the two disagree."""
        op32 = Matrix(spd64.astype(jnp.float32))
        vec64 = jnp.ones(4, dtype=F64)

        assert (op32 @ vec64).dtype == F64
        assert linox.solve(op32, vec64).dtype == F64
        assert (op32.T @ vec64).dtype == F64


class TestMatrixFreeAlgorithms:
    def test_krylov_paths_preserve_float64(self) -> None:
        from linox.linalg.approx.arnoldi import arnoldi_iteration
        from linox.linalg.approx.lanczos import (
            lanczos_matrix_function,
            lanczos_tridiag,
        )
        from linox.linalg.approx.lsmr import lsmr_solve
        from linox.linalg.spectral import svd_partial

        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (8, 8), dtype=F64)
        op = Matrix(X @ X.T + 8 * jnp.eye(8, dtype=F64))
        vec = jnp.ones(8, dtype=F64)

        assert lanczos_tridiag(op, vec / jnp.linalg.norm(vec), 5)[0].dtype == F64
        assert lanczos_matrix_function(op, vec, jnp.sqrt, 5).dtype == F64
        assert arnoldi_iteration(op, vec, 5)[0].dtype == F64
        assert svd_partial(op, k=3, num_iters=6)[0].dtype == F64
        assert lsmr_solve(op, vec)[0].dtype == F64


def test_default_floating_dtype_tracks_the_flag() -> None:
    assert default_floating_dtype() == F64
    jax.config.update("jax_enable_x64", False)
    try:
        assert default_floating_dtype() == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", True)
