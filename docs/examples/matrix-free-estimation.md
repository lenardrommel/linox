# Matrix-free estimation

Quantities that normally need a factorisation, obtained from matvecs alone.

## The setting

An operator you can apply but not decompose — too large, or available only as a
function. Traces, diagonals and log-determinants are still reachable through
stochastic estimation.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
n = 400
q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
spectrum = jnp.linspace(1.0, 30.0, n)
op = linox.Matrix(q @ jnp.diag(spectrum) @ q.T)
```

## Trace

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.hutchinson import hutchinson_trace

key = jax.random.PRNGKey(0)
n = 400
q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
spectrum = jnp.linspace(1.0, 30.0, n)
op = linox.Matrix(q @ jnp.diag(spectrum) @ q.T)

estimate, stderr = hutchinson_trace(op, key, num_samples=300)
exact = jnp.sum(spectrum)

assert jnp.abs(estimate - exact) < 6 * stderr
```

The estimator is unbiased, so the error shrinks as `1/√M`. Report `stderr` — it is
what tells you whether the estimate is usable.

## Log-determinant

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.slq import slq_logdet

key = jax.random.PRNGKey(0)
n = 200
q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
spectrum = jnp.linspace(1.0, 20.0, n)
dense = q @ jnp.diag(spectrum) @ q.T
op = linox.Matrix(dense)

estimate, _stderr = slq_logdet(op, key, num_samples=50, m=30)
exact = jnp.sum(jnp.log(spectrum))

assert jnp.abs(estimate - exact) / jnp.abs(exact) < 0.05
```

`log det A = tr(log A)`, so this is Hutchinson probing composed with a Lanczos
approximation of `log`. Cost: `num_samples × m` matvecs, no factorisation.

## Diagonal

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.hutchinson import hutchinson_diagonal

key = jax.random.PRNGKey(0)
n = 50
diag = jnp.arange(1.0, n + 1)
op = linox.Matrix(jnp.diag(diag) + 0.01 * jnp.ones((n, n)))

estimate, _stderr = hutchinson_diagonal(op, key, num_samples=2000)
assert estimate.shape == (n,)
```

Diagonal estimation converges more slowly than trace estimation — the trace averages
`n` entries, while each diagonal entry is estimated on its own.

## Getting the budget right

The two knobs do different things, and confusing them wastes work:

| Knob | Controls | Symptom when too small |
|---|---|---|
| `num_samples` | variance | estimate jumps between keys |
| `m` (Krylov depth) | bias | estimate is consistently off |

Diagnose by varying one at a time:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.slq import slq_logdet

n = 100
q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
spectrum = jnp.linspace(1.0, 10.0, n)
op = linox.Matrix(q @ jnp.diag(spectrum) @ q.T)
exact = jnp.sum(jnp.log(spectrum))

# Different keys, same settings: spread reveals variance.
estimates = jnp.array([
    slq_logdet(op, jax.random.PRNGKey(s), num_samples=20, m=20)[0]
    for s in range(4)
])
assert jnp.all(jnp.isfinite(estimates))

# More Krylov depth reduces bias.
deep, _ = slq_logdet(op, jax.random.PRNGKey(0), num_samples=20, m=40)
assert jnp.abs(deep - exact) / jnp.abs(exact) < 0.1
```

If the spread across keys is small but every estimate sits on the same side of the
truth, that is bias — raise `m`, not `num_samples`.

## Degenerate spectra

Lanczos exhausts its Krylov space when eigenvalues repeat; for a multiple of the
identity, after a single step. The spurious modes are dropped rather than fed into
`log`, so the answer stays finite and correct:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.slq import slq_logdet

op = linox.Diagonal(3.0 * jnp.ones(10))
estimate, _ = slq_logdet(op, jax.random.PRNGKey(0), num_samples=5, m=25)

assert jnp.isfinite(estimate)
assert jnp.abs(estimate - 10 * jnp.log(3.0)) < 1e-6
```
