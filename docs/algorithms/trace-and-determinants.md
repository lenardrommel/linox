# Trace and determinants

Both have exact forms for structured operators and stochastic estimators for the
rest.

## Trace

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.arange(1.0, 5.0))
value = linox.trace(d)
assert jnp.allclose(value, 10.0)
```

For a `Diagonal` or a `Kronecker` the trace is exact and free — `tr(A ⊗ B) =
tr(A)·tr(B)`:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (4, 4))
kron = linox.Kronecker(linox.Matrix(a), linox.Matrix(b))

assert jnp.allclose(kron.trace(), jnp.trace(jnp.kron(a, b)))
```

### Hutchinson estimation

For an operator with no exact route, estimate `tr(A) ≈ (1/M) Σ zᵢᵀ A zᵢ` with random
probes. Only matvecs are needed:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.hutchinson import hutchinson_trace

n = 100
op = linox.Matrix(jnp.eye(n) + 0.1 * jnp.ones((n, n)))   # trace = 1.1 * n

estimate, stderr = hutchinson_trace(op, jax.random.PRNGKey(0), num_samples=200)

assert jnp.abs(estimate - 1.1 * n) < 5.0
assert stderr > 0
```

The estimator is unbiased; `stderr` is the standard error of the mean and shrinks as
`1/√M`. It is a *stochastic* answer — two different keys give two different numbers.

`hutchinson_diagonal` estimates the diagonal the same way, and
`hutchinson_trace_and_diagonal` computes both from shared probes.

## Determinants

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)
op = linox.Matrix(spd)

sign, logabs = linox.slogdet(op)
assert jnp.allclose(logabs, jnp.linalg.slogdet(spd)[1])
assert jnp.allclose(linox.logdet(op), logabs)
```

Prefer `slogdet` to `det`: a determinant of a large matrix overflows long before its
logarithm does.

For a Kronecker product, `det(A ⊗ B) = det(A)^{n_b} · det(B)^{n_a}`:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (3, 3))
spd_a, spd_b = a @ a.T + 3 * jnp.eye(3), b @ b.T + 3 * jnp.eye(3)
kron = linox.Kronecker(linox.Matrix(spd_a), linox.Matrix(spd_b))

_sign, logabs = linox.slogdet(kron)
assert jnp.allclose(logabs, jnp.linalg.slogdet(jnp.kron(spd_a, spd_b))[1], atol=1e-8)
```

### Stochastic Lanczos quadrature

`log det A = tr(log A)`, so a log-determinant is a trace of a matrix function — which
Hutchinson probes plus Lanczos can estimate without factorising:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.slq import slq_logdet

key = jax.random.PRNGKey(0)
n = 60
q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
spd = q @ jnp.diag(jnp.linspace(1.0, 20.0, n)) @ q.T
op = linox.Matrix(spd)

estimate, _stderr = slq_logdet(op, key, num_samples=40, m=25)
exact = jnp.linalg.slogdet(spd)[1]

assert jnp.abs(estimate - exact) / jnp.abs(exact) < 0.05
```

This is the algorithm that makes GP marginal likelihoods tractable at scale — it
needs `num_samples * m` matvecs and no factorisation.

!!! note "Estimators need a key"
    Any stochastic path requires an explicit PRNG key. With `method="auto"` and no
    key supplied, linox falls back to the exact route rather than inventing
    randomness.
