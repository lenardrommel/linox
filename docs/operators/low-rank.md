# Low-rank operators

A rank-`r` update to an `n×n` matrix needs `O(nr)` storage, and the Woodbury identity
turns a solve against it into a solve of size `r`.

| Operator | Represents |
|---|---|
| `LowRank(U, S, V)` | `U diag(S) Vᵀ` |
| `SymmetricLowRank(U, S)` | `U diag(S) Uᵀ` |
| `IsotropicScalingPlusSymmetricLowRank(s, U, S)` | `s·I + U diag(S) Uᵀ` |
| `PositiveDiagonalPlusSymmetricLowRank(D, LR)` | `D + U diag(S) Uᵀ` |

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
u = jax.random.normal(key, (8, 2))
s = jnp.ones(2)

lr = linox.SymmetricLowRank(u, s)
assert lr.shape == (8, 8)
assert jnp.allclose(linox.todense(lr), u @ jnp.diag(s) @ u.T)
```

## The rewrites that matter

Diagonal-plus-low-rank is recognised from the arithmetic and solved by Woodbury:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
d = linox.Diagonal(jnp.arange(1.0, 9.0))
lr = linox.SymmetricLowRank(jax.random.normal(key, (8, 2)), jnp.ones(2))

combined = d + lr
assert type(combined).__name__ == "PositiveDiagonalPlusSymmetricLowRank"

x = linox.solve(combined, jnp.ones(8))
assert jnp.linalg.norm(linox.todense(combined) @ x - jnp.ones(8)) < 1e-8
```

Solving `(D + U S Uᵀ)x = b` this way costs a diagonal solve plus an `r×r` solve,
never an `n×n` factorisation.

Similarly for a scalar shift:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
lr = linox.SymmetricLowRank(jax.random.normal(key, (8, 2)), jnp.ones(2))

shifted = 2.0 * linox.Identity(8) + lr
assert type(shifted).__name__ == "IsotropicScalingPlusSymmetricLowRank"
```

## Rectangular low rank

`LowRank` takes separate left and right factors:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
u = jax.random.normal(key, (6, 2))
v = jax.random.normal(jax.random.fold_in(key, 1), (4, 2))
s = jnp.ones(2)

lr = linox.LowRank(u, s, v)
assert jnp.allclose(linox.todense(lr), u @ jnp.diag(s) @ v.T)
```

!!! warning "`LowRank.shape` is wrong for rectangular factors"
    With differently shaped `U` and `V`, `.shape` reports `(m, m)` while `todense()`
    correctly gives `(m, n)`. The matvec computes the right numbers, but anything
    trusting `.shape` — composition checks, shape-based dispatch — will misbehave.
    Tracked as a known bug; prefer square factors until it is fixed.

## Where these come from

Low-rank updates are the natural shape of a Laplace approximation, a limited-memory
quasi-Newton Hessian, or a GP posterior after a rank-`r` observation. In each case
the base is diagonal or scalar and the update is thin, which is exactly the pattern
the rewrites above match.
