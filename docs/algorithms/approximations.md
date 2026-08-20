# Approximations

The matrix-free machinery, usable directly when you want control over the iteration.

## Krylov bases

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.lanczos import lanczos_tridiag

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (30, 30))
op = linox.Matrix(dense @ dense.T + 30 * jnp.eye(30))
v = jnp.ones(30) / jnp.sqrt(30.0)

q, alpha, beta = lanczos_tridiag(op, v, num_iters=10)

assert q.shape == (30, 10)          # orthonormal Krylov basis
assert alpha.shape == (10,)         # tridiagonal diagonal
assert beta.shape == (9,)           # off-diagonal
assert jnp.linalg.norm(q.T @ q - jnp.eye(10)) < 1e-8
```

`lanczos_tridiag` reduces a symmetric operator to tridiagonal form using `num_iters`
matvecs. `arnoldi_iteration` does the same for non-symmetric operators, producing a
Hessenberg matrix.

Full reorthogonalisation is on by default (`reortho=True`), which costs more per step
but keeps the basis orthogonal.

## Partial eigendecomposition

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.lanczos import lanczos_eigh

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (40, 40)))
spd = q @ jnp.diag(jnp.linspace(1.0, 40.0, 40)) @ q.T
op = linox.Matrix(spd)

values, _vectors = lanczos_eigh(op, jnp.ones(40), num_iters=30, k=3)
assert values.shape == (3,)
```

## Partial SVD

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.spectral import svd_partial

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (30, 20))

u, s, vt = svd_partial(linox.Matrix(a), k=5, num_iters=20)
assert u.shape == (30, 5) and vt.shape == (5, 20)
assert jnp.allclose(u.T @ u, jnp.eye(5), atol=1e-8)
```

Built on Lanczos bidiagonalisation; `linox.svd(op, k=...)` is the same thing behind
the public name.

## LSMR

Least squares by an iterative method, for rectangular or rank-deficient systems:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.lsmr import lsmr_solve

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (30, 15))
b = jnp.ones(30)

x, info = lsmr_solve(linox.Matrix(a), b)
assert x.shape == (15,)
assert int(info["istop"]) in (1, 2, 3, 4)
```

`info` carries `istop`, `itn`, `normr` and condition estimates.

## Choosing iteration counts

There is no universal answer, but the shape of the trade is consistent:

- **Krylov iterations (`num_iters`, `m`)** control *bias*. More iterations resolve
  more of the spectrum. Beyond the numerical rank there is nothing left to resolve.
- **Probes (`num_samples`)** control *variance*, shrinking as `1/√M`.

For a log-determinant, doubling `m` and doubling `num_samples` do different things:
the first reduces the approximation error of `log`, the second reduces the noise of
the trace estimate. If your estimate is biased, more samples will not help.
