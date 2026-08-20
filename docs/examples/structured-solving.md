# Structured solving

A regularised least-squares problem where the operator is diagonal plus low rank —
the shape a Laplace approximation or a rank-`r` posterior update takes.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
n, r = 200, 5

prior_precision = linox.Diagonal(jnp.linspace(1.0, 4.0, n))
factor = jax.random.normal(key, (n, r)) / jnp.sqrt(n)
update = linox.SymmetricLowRank(factor, jnp.ones(r))

posterior = prior_precision + update
assert type(posterior).__name__ == "PositiveDiagonalPlusSymmetricLowRank"
```

The rewrite matters: `solve` against this uses the Woodbury identity, so the cost is
a diagonal solve plus an `r×r` solve rather than an `n×n` factorisation.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
n, r = 200, 5
prior = linox.Diagonal(jnp.linspace(1.0, 4.0, n))
factor = jax.random.normal(key, (n, r)) / jnp.sqrt(n)
posterior = prior + linox.SymmetricLowRank(factor, jnp.ones(r))

b = jnp.ones(n)
x = linox.solve(posterior, b)

residual = jnp.linalg.norm(linox.todense(posterior) @ x - b) / jnp.linalg.norm(b)
assert residual < 1e-8
```

## Why not just build the matrix

For `n = 200` you could. The point is that the code does not change as `n` grows:
the storage is `O(nr)` and the solve is `O(nr²)`, so the same three lines work at
`n = 10⁶` where the dense matrix would need eight terabytes.

## The log-determinant comes free

The matrix determinant lemma gives `det(D + USUᵀ)` from the same factors:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
n, r = 60, 3
prior = linox.Diagonal(jnp.linspace(1.0, 4.0, n))
factor = jax.random.normal(key, (n, r)) / jnp.sqrt(n)
posterior = prior + linox.SymmetricLowRank(factor, jnp.ones(r))

_sign, logabs = linox.slogdet(posterior)
reference = jnp.linalg.slogdet(linox.todense(posterior))[1]

assert jnp.allclose(logabs, reference, atol=1e-8)
```

That is the term a marginal likelihood needs, and the expensive one if computed
naively.

## Differentiating through it

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
n, r = 40, 2
factor = jax.random.normal(key, (n, r)) / jnp.sqrt(n)
b = jnp.ones(n)

def objective(scale):
    op = linox.Diagonal(scale * jnp.linspace(1.0, 4.0, n))
    posterior = op + linox.SymmetricLowRank(factor, jnp.ones(r))
    return linox.solve(posterior, b).sum()

grad = jax.grad(objective)(2.0)
assert jnp.isfinite(grad)
```

The gradient flows through the Woodbury solve; nothing special is required.
