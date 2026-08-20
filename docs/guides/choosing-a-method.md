# Choosing a method

Most operations take `method=`. The default, `"auto"`, decides by operator size
against a configurable dense threshold.

```python
import linox

assert linox.config.get_max_dense_n() == 2000
```

Below the threshold, `auto` prefers exact routes; above it, approximate ones.

## The decision, in short

| Situation | Use |
|---|---|
| Operator has structure (diagonal, Kronecker, low-rank shift) | `"exact"` — the structured path *is* the fast path |
| Dense, fits comfortably in memory | `"exact"` |
| Large, symmetric positive definite, matvec is cheap | `"cg"` |
| Large, rectangular or rank-deficient | `"lsmr"` |
| You need a trace or log-determinant of something huge | Hutchinson / SLQ, with a key |

The first row is the one people miss. "Exact" does not mean "dense": for a Kronecker
product the exact solve goes through the factors and never forms the matrix.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
spd = a @ a.T + 3 * jnp.eye(3)
kron = linox.Kronecker(linox.Matrix(spd), linox.Matrix(spd))

# Exact *and* matrix-free.
x = linox.solve(kron, jnp.ones(9), method="exact")
assert jnp.linalg.norm(jnp.kron(spd, spd) @ x - jnp.ones(9)) < 1e-8
```

## Exact versus iterative

Exact factorisation costs O(n³) once and then solves cheaply for many right-hand
sides. An iterative method costs matvecs per solve, and converges at a rate set by
the condition number.

Prefer iterative when the operator is large *and* the matvec is much cheaper than
O(n²) — which is exactly when the operator is structured or matrix-free. If your
operator is a dense array in memory, a factorisation is usually better.

## Preconditioning

CG's convergence depends on the condition number. A preconditioner `M ≈ A⁻¹` that is
cheap to apply changes the rate, often dramatically:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (60, 60)))
scale = jnp.diag(jnp.logspace(0, 4, 60))
badly_scaled = scale @ (q @ jnp.diag(jnp.linspace(1.0, 100.0, 60)) @ q.T) @ scale
op, b = linox.Matrix(badly_scaled), jnp.ones(60)

jacobi = linox.Diagonal(1.0 / jnp.diag(badly_scaled))
x = linox.solve(op, b, method="cg", preconditioner=jacobi, maxiter=200)
assert jnp.linalg.norm(badly_scaled @ x - b) / jnp.linalg.norm(b) < 1e-4
```

The Jacobi preconditioner — the reciprocal diagonal — costs nothing and is a
reasonable first attempt whenever the scaling is uneven.

## Stochastic methods

Hutchinson and SLQ return an *estimate* with a standard error. Two rules:

1. **Report the uncertainty.** They return `(estimate, stderr)`; ignoring the second
   value discards the information that tells you whether to trust the first.
2. **Iterations reduce bias, samples reduce variance.** If the estimate is
   systematically off, more samples will not fix it — increase the Krylov depth.

## Unknown methods are rejected

```python
import linox

try:
    linox.solve(linox.Identity(4), __import__("jax").numpy.ones(4), method="cgg")
    raise AssertionError("expected a rejection")
except ValueError as exc:
    assert "Unknown method" in str(exc)
```

A typo will not silently select the default.

## Configuring the default

```python
import linox

original = linox.config.get_max_dense_n()
linox.config.set_max_dense_n(500)
assert linox.config.get_max_dense_n() == 500
linox.config.set_max_dense_n(original)
```

`config.set_default_method(operation, method)` pins a choice for a given operation,
overriding the size heuristic but not an explicit argument.
