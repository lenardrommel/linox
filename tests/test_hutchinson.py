
import jax
import jax.numpy as jnp
from linox import Matrix
from linox.linalg.approx.hutchinson import hutchinson_diagonal, hutchinson_trace


def test_hutchinson_trace_batched() -> None:
    """Hutchinson's estimator is unbiased for trace(A).

    A = I + 0.1*J, so z^T A z = n + 0.1*(sum z)^2 for a Rademacher probe z.
    Hence E[estimate] = n + 0.1*n = 1.1*n, and
    Var = 0.01 * Var((sum z)^2) / M = 0.01 * 2n^2 / M, i.e. an analytic
    standard error of 1.0 for n=100, M=200.

    The bound uses that *analytic* standard error rather than the estimator's
    own reported one: the reported value is itself a noisy statistic, so
    scaling the tolerance by it made this test fail on ~3-sigma seeds.
    """
    n = 100
    num_samples = 200
    A = Matrix(jnp.eye(n) + 0.1 * jnp.ones((n, n)))

    true_trace = float(1.1 * n)
    analytic_stderr = float(jnp.sqrt(0.01 * 2 * n**2 / num_samples))

    estimates = []
    for seed in range(20):
        est, _std = hutchinson_trace(
            A, jax.random.PRNGKey(seed), num_samples=num_samples,
            distribution="rademacher",
        )
        estimates.append(float(est))
        # Any individual draw should sit within 5 analytic sigma.
        assert jnp.abs(est - true_trace) < 5.0 * analytic_stderr

    # Averaging 20 independent runs shrinks the error by sqrt(20); require the
    # mean to land well inside that, which is what "unbiased" actually means.
    mean_estimate = sum(estimates) / len(estimates)
    assert jnp.abs(mean_estimate - true_trace) < 3.0 * analytic_stderr / jnp.sqrt(
        len(estimates)
    )


def test_hutchinson_diagonal_batched() -> None:
    key = jax.random.PRNGKey(42)
    n = 50
    # diag = [0, 1, 2, ...]
    d = jnp.arange(n, dtype=jnp.float32)
    A = Matrix(jnp.diag(d))

    est, std = hutchinson_diagonal(A, key, num_samples=300)

    assert jnp.allclose(est, d, atol=3.0 * jnp.max(std))
