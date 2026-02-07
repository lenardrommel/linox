
import jax
import jax.numpy as jnp
from linox import Matrix
from linox.linalg.approx.hutchinson import hutchinson_diagonal, hutchinson_trace


def test_hutchinson_trace_batched() -> None:
    key = jax.random.PRNGKey(42)
    n = 100
    # Use simple Identity + constant matrix
    # trace(I + 0.1) = n + 0.1*n = 1.1*n
    # 1.1 * 100 = 110.0
    A = Matrix(jnp.eye(n) + 0.1 * jnp.ones((n, n)))

    est, std = hutchinson_trace(A, key, num_samples=200, distribution="rademacher")

    true_trace = float(1.1 * n)
    # Allow 3 sigma deviation?
    # rademacher variance is approx ||A - tr(A)/n I||_F^2 / M
    # For A=I+0.1J, it's ...
    # Just check reasonable bounds.
    assert jnp.abs(est - true_trace) < 5.0 * std
    assert jnp.abs(est - true_trace) / true_trace < 0.1  # 10% error


def test_hutchinson_diagonal_batched() -> None:
    key = jax.random.PRNGKey(42)
    n = 50
    # diag = [0, 1, 2, ...]
    d = jnp.arange(n, dtype=jnp.float32)
    A = Matrix(jnp.diag(d))

    est, std = hutchinson_diagonal(A, key, num_samples=300)

    assert jnp.allclose(est, d, atol=3.0 * jnp.max(std))
