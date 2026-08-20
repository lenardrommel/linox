# linox

Structured, matrix-free linear algebra in JAX.

linox represents a linear map as an **operator** rather than a matrix. An operator
knows how to apply itself to a vector, and often knows more than that — that it is
diagonal, a Kronecker product, a low-rank update — and uses that structure to avoid
ever forming the dense matrix.

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.array([1.0, 2.0, 3.0, 4.0]))
i = linox.Identity(4)

op = d + 0.5 * i          # nothing is computed yet
x = op @ jnp.ones(4)      # one elementwise multiply, no 4x4 matrix

assert x.shape == (4,)
```

## What it is for

Problems where the matrix is too large to materialise, too structured to justify it,
or both: Gaussian process covariances, Kronecker-factored operators from discretised
PDEs, diagonal-plus-low-rank posteriors, kernel matrices.

An operator that is a Kronecker product of two 1000×1000 factors represents a
10⁶×10⁶ matrix. linox solves against it without allocating one.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (3, 3))
spd_a, spd_b = a @ a.T + 3 * jnp.eye(3), b @ b.T + 3 * jnp.eye(3)

kron = linox.Kronecker(linox.Matrix(spd_a), linox.Matrix(spd_b))
assert kron.shape == (9, 9)

# Solved through the factors; the 9x9 matrix is never built.
x = linox.solve(kron, jnp.ones(9))
assert x.shape == (9,)
```

## Where to start

| If you want to | Read |
|---|---|
| Install it | [Installation](installation.md) |
| See it work in five minutes | [Quickstart](quickstart.md) |
| Understand the model | [Linear operators](concepts/linear-operators.md) |
| Know which operator to reach for | [Basic operators](operators/basic.md) |
| Solve a system | [Solving](algorithms/solving.md) |
| Pick between exact and approximate | [Choosing a method](guides/choosing-a-method.md) |
| Find out why something got slow | [Avoiding densification](guides/avoiding-densification.md) |

## Design in one paragraph

Operators are JAX pytrees, so they pass through `jit`, `grad` and `vmap` like any
other value. Operations are dispatched on operator *type* using
[plum](https://github.com/beartype/plum), so `solve(Kronecker(...), b)` reaches a
Kronecker-specific implementation while `solve(Matrix(...), b)` reaches a dense one,
without either knowing about the other. Arithmetic is lazy and rewrites itself:
`A + s * I` becomes a single operator with fast spectral methods rather than a
generic sum.

## Status

Alpha. The API is still moving — see the [roadmap](development/roadmap.md) for what
is settled and what is not.
