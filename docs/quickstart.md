# Quickstart

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
```

## Build an operator

Wrap an array, or describe structure directly:

```python
import jax
import jax.numpy as jnp
import linox

dense = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)

a = linox.Matrix(spd)                              # a general matrix
d = linox.Diagonal(jnp.array([1.0, 2.0, 3.0, 4.0]))  # only the diagonal is stored
i = linox.Identity(4)                              # nothing is stored

assert a.shape == d.shape == i.shape == (4, 4)
```

## Compose

Arithmetic is lazy. Nothing is evaluated until you apply the result to something.

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.arange(1.0, 5.0))
op = 2.0 * d + linox.Identity(4)

x = op @ jnp.ones(4)
assert jnp.allclose(x, 2.0 * jnp.arange(1.0, 5.0) + 1.0)
```

## Apply, solve, decompose

```python
import jax
import jax.numpy as jnp
import linox

dense = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
op = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))
b = jnp.ones(4)

x = op @ b                    # matvec
y = linox.solve(op, b)        # solve op @ y == b
w, q = linox.eigh(op)         # eigendecomposition

assert jnp.linalg.norm(linox.todense(op) @ y - b) < 1e-10
assert w.shape == (4,)
```

## Failure is reported, not hidden

A singular system raises rather than returning a plausible-looking wrong answer:

```python
import jax
import jax.numpy as jnp
import linox
from linox import LinearSolveError

u = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
singular = linox.Matrix(u @ u.T)          # rank 3, not invertible

try:
    linox.solve(singular, jnp.ones(6))
    raise AssertionError("expected a failure")
except LinearSolveError:
    pass

# Opt out, or inspect the outcome yourself:
x = linox.solve(singular, jnp.ones(6), throw=False)
x, info = linox.solve(singular, jnp.ones(6), throw=False, return_info=True)
assert info.result != linox.RESULTS.successful
```

## It is all JAX

Operators are pytrees, so they pass through transformations unchanged:

```python
import jax
import jax.numpy as jnp
import linox

dense = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)
b = jnp.ones(4)

jitted = jax.jit(lambda m, v: linox.solve(linox.Matrix(m), v))
grad = jax.grad(lambda v: linox.solve(linox.Matrix(spd), v).sum())
batched = jax.vmap(lambda v: linox.solve(linox.Matrix(spd), v))

assert jitted(spd, b).shape == (4,)
assert grad(b).shape == (4,)
assert batched(jnp.ones((5, 4))).shape == (5, 4)
```

## Next

[Linear operators](concepts/linear-operators.md) explains the model;
[Choosing a method](guides/choosing-a-method.md) covers exact versus approximate.
