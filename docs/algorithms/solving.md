# Solving

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (6, 6))
op = linox.Matrix(dense @ dense.T + 6 * jnp.eye(6))

x = linox.solve(op, jnp.ones(6))
assert jnp.linalg.norm(linox.todense(op) @ x - jnp.ones(6)) < 1e-10
```

`solve` dispatches on the operator: a `Diagonal` divides, a `Kronecker` solves
through its factors, a diagonal-plus-low-rank uses Woodbury, and a general matrix
factorises.

## Failure is reported

A singular system raises rather than returning finite nonsense:

```python
import jax
import jax.numpy as jnp
import linox
from linox import LinearSolveError

u = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
singular = linox.Matrix(u @ u.T)          # rank 3

try:
    linox.solve(singular, jnp.ones(6))
    raise AssertionError("expected a failure")
except LinearSolveError as exc:
    assert exc.result in (linox.RESULTS.singular, linox.RESULTS.nonfinite_output)
```

This matters because the failure mode is not obvious: the returned values were
finite, of magnitude 10¹⁶, with no NaN to trip over.

Two opt-outs:

```python
import jax
import jax.numpy as jnp
import linox

u = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
singular = linox.Matrix(u @ u.T)

x = linox.solve(singular, jnp.ones(6), throw=False)          # accept it
x, info = linox.solve(singular, jnp.ones(6), throw=False, return_info=True)

assert info.result != linox.RESULTS.successful
assert "residual" in info.stats
```

`RESULTS` covers `successful`, `singular`, `max_steps_reached`, `breakdown`,
`stagnation`, `conlim`, `nonfinite_input` and `nonfinite_output`.

Under `jit` the outcome is a traced value and cannot be raised at trace time; the
failure is reported by a runtime callback, and `info.result` is available to branch
on inside the computation.

## Methods

| `method=` | Solver | Requires |
|---|---|---|
| `"auto"` | by operator size and structure | — |
| `"exact"` | the structured or dense factorisation | square, nonsingular |
| `"cg"` | preconditioned conjugate gradients | symmetric positive definite |
| `"lsmr"` | LSMR | anything, including rectangular |

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (40, 40)))
spd = linox.Matrix(q @ jnp.diag(jnp.linspace(1.0, 50.0, 40)) @ q.T)
b = jnp.ones(40)

exact = linox.solve(spd, b, method="exact")
iterative = linox.solve(spd, b, method="cg")

assert jnp.allclose(exact, iterative, atol=1e-4)
```

## Conjugate gradients

CG needs only matvecs, so a matrix-free operator stays matrix-free. It accepts a
preconditioner:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (60, 60)))
scale = jnp.diag(jnp.logspace(0, 4, 60))
badly_scaled = scale @ (q @ jnp.diag(jnp.linspace(1.0, 100.0, 60)) @ q.T) @ scale
op = linox.Matrix(badly_scaled)
b = jnp.ones(60)

jacobi = linox.Diagonal(1.0 / jnp.diag(badly_scaled))
x = linox.solve(op, b, method="cg", preconditioner=jacobi, maxiter=200)

assert jnp.linalg.norm(badly_scaled @ x - b) / jnp.linalg.norm(b) < 1e-4
```

On that system, plain CG does not converge within 60 iterations and the
preconditioned version does.

### Iteration count versus gradients

`lax.while_loop` has no reverse-mode rule, and its counter is only visible from
inside it. Both are available, one CG run either way:

```python
import jax
import jax.numpy as jnp
import linox
from linox.linalg.approx.cg import cg_solve

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (30, 30)))
op = linox.Matrix(q @ jnp.diag(jnp.linspace(1.0, 20.0, 30)) @ q.T)
b = jnp.ones(30)

x, info = cg_solve(op, b)                            # differentiable
assert "itn" not in info

x, info = cg_solve(op, b, track_iterations=True)     # exact count
assert int(info["itn"]) > 0
```

## Least squares

For rectangular or rank-deficient systems, use the pseudo-inverse:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (8, 4))
b = jnp.ones(8)

x = linox.pinverse(linox.Matrix(a)) @ b
assert jnp.allclose(x, jnp.linalg.lstsq(a, b, rcond=None)[0], atol=1e-8)
```

`solve` on a rectangular operator raises a shape error — it solves square systems.
