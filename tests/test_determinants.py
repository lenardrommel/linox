
import jax
import jax.numpy as jnp
from linox import Matrix
from linox.linalg.determinants import slogdet


def test_slq_logdet_simple() -> None:
    key = jax.random.PRNGKey(42)
    n = 100
    # A = diag(1, 2, ..., n)
    # logdet = sum(log(i))
    vals = jnp.arange(1, n + 1, dtype=jnp.float32)
    A = Matrix(jnp.diag(vals))

    true_logdet = jnp.sum(jnp.log(vals))

    # SLQ
    sign, est_logdet = slogdet(A, method="slq", key=key, num_samples=30, m=20)

    assert sign == 1.0
    # This is a stochastic test, allow margin
    # SLQ converges reasonably well for well-conditioned problems
    err = jnp.abs(est_logdet - true_logdet)
    rel_err = err / true_logdet
    assert rel_err < 0.2  # 20% error tolerance for small sample
