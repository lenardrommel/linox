"""Preconditioned conjugate gradients.

`jax.scipy.sparse.linalg.cg` reports no convergence information, which left
`linox.solve(..., method="cg")` unable to say whether a solve had succeeded --
it fell back to a loose residual guard. This implementation reports a
termination code and accepts a preconditioner.
"""

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import Diagonal, Kronecker, Matrix
from linox.linalg.approx.cg import CG_CONVERGED, CG_NOT_CONVERGED, cg_solve
from linox.linalg.solution import RESULTS, LinearSolveError

jax.config.update("jax_enable_x64", True)

F64 = jnp.float64


def spd(n, cond=1.0, seed=0):
    """An SPD matrix with a controlled condition number."""
    key = jax.random.PRNGKey(seed)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n), dtype=F64))
    return Q @ jnp.diag(jnp.linspace(1.0, cond, n)) @ Q.T


class TestSolvesCorrectly:
    def test_well_conditioned(self) -> None:
        A = spd(60, cond=100.0)
        b = jnp.ones(60, dtype=F64)
        x, info = cg_solve(Matrix(A), b)

        assert int(info["istop"]) == CG_CONVERGED
        assert jnp.linalg.norm(x - jnp.linalg.solve(A, b)) / jnp.linalg.norm(
            jnp.linalg.solve(A, b)
        ) < 1e-5

    def test_stays_matrix_free(self) -> None:
        """Only matvecs -- a Kronecker operator is never densified."""
        import linox.config as config

        kron = Kronecker(Matrix(spd(8)), Matrix(spd(8, seed=1)))
        rhs = jnp.ones(64, dtype=F64)

        events = []
        config.set_debug_hook(lambda e: events.append(e.kind))
        try:
            x, info = cg_solve(kron, rhs)
        finally:
            config.set_debug_hook(None)

        assert int(info["istop"]) == CG_CONVERGED
        assert jnp.linalg.norm(linox.todense(kron) @ x - rhs) < 1e-8
        assert events.count("densify") == 0

    def test_zero_rhs(self) -> None:
        A = spd(20)
        x, info = cg_solve(Matrix(A), jnp.zeros(20, dtype=F64))
        assert jnp.all(x == 0)
        assert int(info["istop"]) == CG_CONVERGED

    def test_rejects_matrix_rhs_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="vector right-hand side"):
            cg_solve(Matrix(spd(4)), jnp.ones((4, 2), dtype=F64))


class TestPreconditioning:
    def test_jacobi_accelerates_a_badly_scaled_system(self) -> None:
        n = 60
        scale = jnp.diag(jnp.logspace(0, 4, n))
        A = scale @ spd(n, cond=100.0) @ scale
        b = jnp.ones(n, dtype=F64)
        expected = jnp.linalg.solve(A, b)

        _plain, plain_info = cg_solve(Matrix(A), b, maxiter=60)
        precond, precond_info = cg_solve(
            Matrix(A), b, preconditioner=Diagonal(1.0 / jnp.diag(A)), maxiter=60
        )

        # Within the same budget, only the preconditioned solve converges.
        assert int(plain_info["istop"]) == CG_NOT_CONVERGED
        assert int(precond_info["istop"]) == CG_CONVERGED
        assert jnp.linalg.norm(precond - expected) / jnp.linalg.norm(expected) < 1e-5


class TestReportsFailure:
    def test_iteration_cap(self) -> None:
        A = spd(60, cond=1e4)
        _x, info = cg_solve(Matrix(A), jnp.ones(60, dtype=F64), maxiter=2)
        assert int(info["istop"]) == CG_NOT_CONVERGED

    def test_indefinite_operator_does_not_return_nan(self) -> None:
        """CG requires positive definiteness; it must fail, not produce NaN."""
        A = spd(20)
        A = A.at[0, 0].set(-5.0)
        A = (A + A.T) / 2

        x, info = cg_solve(Matrix(A), jnp.ones(20, dtype=F64), maxiter=200)

        assert int(info["istop"]) == CG_NOT_CONVERGED
        assert jnp.all(jnp.isfinite(x))


class TestJaxTransforms:
    def test_jit(self) -> None:
        A = spd(40)
        b = jnp.ones(40, dtype=F64)
        x, _ = jax.jit(lambda M, v: cg_solve(Matrix(M), v))(A, b)
        assert jnp.linalg.norm(A @ x - b) < 1e-6

    def test_reverse_mode_grad(self) -> None:
        """`lax.while_loop` has no VJP, hence the `custom_linear_solve` wrapper."""
        A = spd(20)
        b = jnp.ones(20, dtype=F64)

        grad = jax.grad(lambda v: cg_solve(Matrix(A), v)[0].sum())(b)

        # d/db sum(A^-1 b) == A^-T 1 == A^-1 1 for symmetric A.
        assert jnp.allclose(grad, jnp.linalg.solve(A, jnp.ones(20)), atol=1e-6)

    def test_preserves_float64(self) -> None:
        A = spd(10)
        x, _ = cg_solve(Matrix(A), jnp.ones(10, dtype=F64))
        assert x.dtype == F64


class TestWiredIntoSolve:
    def test_reports_success(self) -> None:
        A = spd(40, cond=50.0)
        b = jnp.ones(40, dtype=F64)
        x, info = linox.solve(Matrix(A), b, method="cg", return_info=True)

        assert int(info.result) == RESULTS.successful
        assert jnp.linalg.norm(A @ x - b) < 1e-4
        assert "normr" in info.stats

    def test_reports_failure_instead_of_a_loose_residual_guard(self) -> None:
        A = spd(60, cond=1e4)
        with pytest.raises(LinearSolveError):
            linox.solve(Matrix(A), jnp.ones(60, dtype=F64), method="cg", maxiter=2)

    def test_accepts_a_preconditioner(self) -> None:
        A = spd(40, cond=50.0)
        b = jnp.ones(40, dtype=F64)
        x = linox.solve(
            Matrix(A), b, method="cg", preconditioner=Diagonal(1.0 / jnp.diag(A))
        )
        assert jnp.linalg.norm(A @ x - b) < 1e-4

    def test_differentiable_through_solve(self) -> None:
        A = spd(20)
        b = jnp.ones(20, dtype=F64)
        grad = jax.grad(lambda v: linox.solve(Matrix(A), v, method="cg").sum())(b)
        assert grad.shape == (20,)
        assert jnp.all(jnp.isfinite(grad))


class TestBothModes:
    """Exact iteration count and reverse-mode grad are mutually exclusive.

    `lax.while_loop` has no VJP, and its counter is only observable from inside
    it. Both modes are provided, each costing one CG run.
    """

    def _system(self, n=60):
        A = spd(n, cond=100.0)
        return A, jnp.ones(n, dtype=F64)

    def test_modes_agree_on_the_solution(self) -> None:
        A, b = self._system()
        x_diff, _ = cg_solve(Matrix(A), b)
        x_tracked, _ = cg_solve(Matrix(A), b, track_iterations=True)
        assert jnp.allclose(x_diff, x_tracked)

    def test_only_tracked_mode_reports_itn(self) -> None:
        A, b = self._system()
        _x, info_default = cg_solve(Matrix(A), b)
        _x, info_tracked = cg_solve(Matrix(A), b, track_iterations=True)

        assert "itn" not in info_default
        assert int(info_tracked["itn"]) > 0

    def test_istop_is_computed_identically_in_both_modes(self) -> None:
        """The reported outcome must not depend on which mode was chosen."""
        A, b = self._system()
        _x, a = cg_solve(Matrix(A), b, maxiter=2)
        _x, c = cg_solve(Matrix(A), b, maxiter=2, track_iterations=True)
        assert int(a["istop"]) == int(c["istop"]) == CG_NOT_CONVERGED

    def test_default_mode_is_reverse_differentiable(self) -> None:
        A, b = self._system(20)
        grad = jax.grad(lambda v: cg_solve(Matrix(A), v)[0].sum())(b)
        assert jnp.allclose(grad, jnp.linalg.solve(A, jnp.ones(20)), atol=1e-6)

    def test_tracked_mode_rejects_reverse_mode_clearly(self) -> None:
        A, b = self._system(20)
        with pytest.raises(ValueError, match="[Rr]everse-mode"):
            jax.grad(lambda v: cg_solve(Matrix(A), v, track_iterations=True)[0].sum())(b)

    def test_tracked_mode_still_works_under_jit(self) -> None:
        A, b = self._system(40)
        _x, info = jax.jit(
            lambda M, v: cg_solve(Matrix(M), v, track_iterations=True)
        )(A, b)
        assert int(info["itn"]) > 0
