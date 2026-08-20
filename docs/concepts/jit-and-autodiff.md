# JIT and autodiff

Operators are JAX pytrees, so the transformations work without special handling.

## jit

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)

@jax.jit
def solve_it(matrix, rhs):
    return linox.solve(linox.Matrix(matrix), rhs)

assert solve_it(spd, jnp.ones(4)).shape == (4,)
```

An operator can also cross the boundary as an argument, because it flattens into its
arrays:

```python
import jax
import jax.numpy as jnp
import linox

op = linox.Diagonal(jnp.arange(1.0, 5.0))

@jax.jit
def apply(operator, vector):
    return operator @ vector

assert jnp.allclose(apply(op, jnp.ones(4)), jnp.arange(1.0, 5.0))
```

## grad and vmap

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)
b = jnp.ones(4)

wrt_rhs = jax.grad(lambda v: linox.solve(linox.Matrix(spd), v).sum())(b)
wrt_matrix = jax.grad(lambda m: linox.solve(linox.Matrix(m), b).sum())(spd)
over_batch = jax.vmap(lambda v: linox.solve(linox.Matrix(spd), v))(jnp.ones((5, 4)))

assert wrt_rhs.shape == (4,)
assert wrt_matrix.shape == (4, 4)
assert over_batch.shape == (5, 4)
```

## What changes under a trace

Two things behave differently, and both are deliberate.

**Checks become runtime errors.** A precondition cannot be raised at trace time,
because the condition is a tracer with no truth value. Rather than skip the check —
which leaves it absent exactly where mistakes are hardest to see — linox defers to a
runtime callback, and a violation surfaces when the computation executes.

**Structural decisions are made at trace time.** Which dispatch runs, and which
rewrite applies, is fixed when the function is traced. That is why they depend on
operator *types* and static shapes, never on values.

## Iterative solvers and reverse mode

`lax.while_loop` has no reverse-mode rule, so an iterative solver that early-exits
cannot be differentiated directly. CG routes its solution through
`lax.custom_linear_solve`, which supplies the adjoint — for a symmetric operator the
cotangent is itself a solve:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (20, 20)))
spd = q @ jnp.diag(jnp.linspace(1.0, 5.0, 20)) @ q.T
b = jnp.ones(20)

g = jax.grad(lambda v: linox.solve(linox.Matrix(spd), v, method="cg").sum())(b)
assert jnp.allclose(g, jnp.linalg.solve(spd, jnp.ones(20)), atol=1e-6)
```

The trade is that the iteration count is then not observable. `cg_solve` offers both:
`track_iterations=True` reports the exact count and gives up reverse mode. See
[Solving](../algorithms/solving.md).
