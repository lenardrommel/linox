# Composition

Arithmetic produces operators. Nothing is evaluated until the result is applied.

| Expression | Result |
|---|---|
| `A + B` | `AddLinearOperator`, or a rewrite |
| `A - B` | as above, with `B` negated |
| `s * A` | `ScaledLinearOperator` |
| `A @ B` | `ProductLinearOperator` |
| `A.T` | the structured transpose, or a lazy wrapper |
| `linox.inv(A)` | `InverseLinearOperator` — solves on application |
| `linox.pinv(A)` | `PseudoInverseLinearOperator` |

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = linox.Matrix(jax.random.normal(key, (4, 4)))
b = linox.Diagonal(jnp.arange(1.0, 5.0))

expr = 2.0 * a @ b + linox.Identity(4)
assert (expr @ jnp.ones(4)).shape == (4,)
```

## Products apply right to left

`(A @ B) @ x` computes `A @ (B @ x)`. The intermediate matrix is never formed, which
is what makes deep compositions affordable:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = linox.Matrix(jax.random.normal(key, (4, 4)))
b = linox.Matrix(jax.random.normal(jax.random.fold_in(key, 1), (4, 4)))

product = a @ b
x = jnp.ones(4)

assert jnp.allclose(product @ x, linox.todense(a) @ (linox.todense(b) @ x))
```

## Lazy inverses

`inv` does not invert anything. It returns an operator that solves on application:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
op = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

inverse = linox.inv(op)
x = inverse @ jnp.ones(4)                 # a solve, not a matrix inverse

assert jnp.linalg.norm(linox.todense(op) @ x - jnp.ones(4)) < 1e-10
```

This composes: `inv(A) @ B` is a product whose left factor solves, so applying it to
a vector is one solve rather than an inversion followed by a multiply.

## Blocks

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = linox.Matrix(jax.random.normal(key, (2, 2)))
b = linox.Matrix(jax.random.normal(jax.random.fold_in(key, 1), (2, 2)))

bd = linox.block_diag(a, b)
bm = linox.bmat([[a, linox.Zero((2, 2))], [linox.Zero((2, 2)), b]])

assert bd.shape == bm.shape == (4, 4)
assert jnp.allclose(linox.todense(bd), linox.todense(bm))
```

`BlockDiagonal` splits the right-hand side, applies each block to its slice and
concatenates — the off-diagonal zeros cost nothing.

## Congruence

`A B Aᵀ` appears often enough — change of basis, covariance propagation — to have its
own operator:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = linox.Matrix(jax.random.normal(key, (3, 3)))
b = linox.Matrix(jnp.eye(3) * 2.0)

c = linox.congruence_transform(a, b)
expected = linox.todense(a) @ linox.todense(b) @ linox.todense(a).T

assert jnp.allclose(linox.todense(c), expected)
```
