# Matrix functions

`f(A)` for a symmetric operator, computed spectrally where the structure allows and
by Krylov methods otherwise.

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.array([1.0, 2.0, 4.0]))

assert jnp.allclose(linox.todense(linox.exp(d)), jnp.diag(jnp.exp(jnp.array([1.0, 2.0, 4.0]))))
assert jnp.allclose(linox.todense(linox.log(d)), jnp.diag(jnp.log(jnp.array([1.0, 2.0, 4.0]))))
```

For a `Diagonal` this is elementwise. For an `EigenD` or an isotropic shift it is
elementwise on the eigenvalues. For anything else it is a Krylov approximation.

## f(A)v without f(A)

The useful case is applying a matrix function to a vector, which needs only matvecs:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.lanczos import lanczos_matrix_function

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (40, 40))
spd = dense @ dense.T + 40 * jnp.eye(40)
op = linox.Matrix(spd)
v = jnp.ones(40)

approx = lanczos_matrix_function(op, v, jnp.sqrt, num_iters=25)

w, q = jnp.linalg.eigh(spd)
exact = q @ jnp.diag(jnp.sqrt(w)) @ q.T @ v
assert jnp.linalg.norm(approx - exact) / jnp.linalg.norm(exact) < 1e-6
```

Lanczos builds a small tridiagonal matrix `T` whose eigendecomposition approximates
the action of `f(A)` on the starting vector. The cost is `num_iters` matvecs.

`arnoldi_matrix_function` does the same for non-symmetric operators.

### Breakdown

Lanczos exhausts its Krylov space when the spectrum is degenerate — for the identity,
after a single step. Everything computed after that is noise, and the spurious
eigenvalues sit at or below zero, so `log` of them would be `-inf` and the result
NaN. Modes carrying no quadrature weight are dropped, so a degenerate spectrum gives
the right answer:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.slq import slq_logdet

# log det of 2I over 8 dimensions is 8 log 2.
op = linox.Diagonal(2.0 * jnp.ones(8))
estimate, _std = slq_logdet(op, jax.random.PRNGKey(0), num_samples=5, m=20)

assert jnp.isfinite(estimate)
assert jnp.abs(estimate - 8 * jnp.log(2.0)) < 1e-6
```

## Powers and exponentials

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (10, 10))
op = linox.Matrix(dense @ dense.T + 10 * jnp.eye(10))

half = linox.pow(op, 0.5)
assert linox.todense(half).shape == (10, 10)
```

The `l`-prefixed forms `lexp`, `llog`, `lpow` take an optional vector, returning
`f(A)v` directly rather than a lazy operator:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (10, 10))
op = linox.Matrix(dense @ dense.T + 10 * jnp.eye(10))

y = linox.lexp(op, v=jnp.ones(10), num_iters=15)
assert y.shape == (10,)
```
