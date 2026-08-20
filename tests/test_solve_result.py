"""Tests for the solve outcome contract (Solution / RESULTS).

Regression: `solve` on a singular system used to return finite, plausible,
wildly wrong numbers -- no exception, no warning, no NaN. A rank-3 6x6 system
returned values of magnitude 1e16 with residual 3.2 and nothing to indicate
anything had gone wrong.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Matrix
from linox.linalg.solution import RESULTS, LinearSolveError, Solution

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def singular():
    """A rank-3 6x6 operator: no exact solution exists."""
    U = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
    return Matrix(U @ U.T), jnp.ones(6)


@pytest.fixture
def well_posed():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (6, 6))
    return Matrix(X @ X.T + jnp.eye(6)), jnp.ones(6)


# A singular direct solve fails in one of two ways depending on the platform's
# LAPACK: usually finite-but-enormous values (caught by the residual check), but
# sometimes outright inf/NaN. Both are correct failure reports, so assert on
# "failed", not on which flavour.
FAILURE_RESULTS = {RESULTS.singular, RESULTS.nonfinite_output}


class TestFailureIsReported:
    def test_singular_raises_by_default(self, singular) -> None:
        op, b = singular
        with pytest.raises(LinearSolveError) as excinfo:
            linox.solve(op, b)
        assert excinfo.value.result in FAILURE_RESULTS

    def test_throw_false_returns_the_array(self, singular) -> None:
        op, b = singular
        x = linox.solve(op, b, throw=False)
        assert x.shape == (6,)

    def test_return_info_reports_failure(self, singular) -> None:
        op, b = singular
        x, info = linox.solve(op, b, throw=False, return_info=True)

        assert isinstance(info, Solution)
        assert int(info.result) in FAILURE_RESULTS
        assert not info.successful

        if int(info.result) == RESULTS.singular:
            # The finite-but-enormous case: the residual is what detects it,
            # since a finiteness check alone would miss it entirely.
            assert jnp.all(jnp.isfinite(x))
            assert info.stats["residual"] > 1e-2
        else:
            # The nonfinite case: detected directly, and the residual is
            # itself NaN, so there is nothing meaningful to compare it against.
            assert not jnp.all(jnp.isfinite(x))


class TestSuccessIsUnaffected:
    def test_well_posed_solve_succeeds(self, well_posed) -> None:
        op, b = well_posed
        x, info = linox.solve(op, b, return_info=True)

        assert int(info.result) == RESULTS.successful
        assert info.successful
        assert jnp.linalg.norm(linox.todense(op) @ x - b) < 1e-10

    def test_plain_call_still_returns_a_bare_array(self, well_posed) -> None:
        op, b = well_posed
        x = linox.solve(op, b)
        assert isinstance(x, jax.Array)
        assert x.shape == (6,)

    def test_converged_lsmr_is_not_flagged(self) -> None:
        """An iterative solver stops at *its* tolerance, not machine epsilon.

        Regression: a strict residual threshold applied on top of LSMR's own
        termination code flagged a perfectly good converged solve (relative
        residual 5.2e-05) as singular.
        """
        key = jax.random.PRNGKey(1)
        n = 40
        A = jax.random.normal(key, (n, n)) + jnp.eye(n) * 3
        b = jnp.ones(n)

        x, info = linox.solve(
            Matrix(A), b, method="lsmr", maxiter=100, return_info=True
        )

        assert int(info.result) == RESULTS.successful
        rel_err = jnp.linalg.norm(x - jnp.linalg.solve(A, b)) / jnp.linalg.norm(
            jnp.linalg.solve(A, b)
        )
        assert rel_err < 1e-3


class TestTransformsStillWork:
    def test_jit_grad_vmap(self, well_posed) -> None:
        op, b = well_posed
        A = linox.todense(op)

        assert jax.jit(lambda M, v: linox.solve(Matrix(M), v))(A, b).shape == (6,)
        assert jax.grad(lambda v: linox.solve(Matrix(A), v).sum())(b).shape == (6,)
        assert jax.grad(lambda M: linox.solve(Matrix(M), b).sum())(A).shape == (6, 6)
        assert jax.vmap(lambda v: linox.solve(Matrix(A), v))(jnp.ones((4, 6))).shape == (
            4,
            6,
        )

    def test_jit_on_a_singular_system_does_not_crash_at_trace_time(
        self, singular
    ) -> None:
        """Under jit the outcome is a tracer, so it cannot be raised.

        The failure is reported by a runtime callback instead; tracing and
        execution must still complete.
        """
        op, b = singular
        A = linox.todense(op)
        x = jax.jit(lambda M, v: linox.solve(Matrix(M), v))(A, b)
        assert x.shape == (6,)


class TestResultsEnum:
    def test_every_outcome_has_a_message(self) -> None:
        for outcome in RESULTS:
            assert outcome.message
            assert isinstance(outcome.message, str)

    def test_error_message_names_the_outcome(self) -> None:
        err = LinearSolveError(RESULTS.singular)
        assert "singular" in str(err)
