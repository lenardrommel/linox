# Kernel operators

A kernel matrix `K[i,j] = k(x_i, x_j)` is defined by a function and a set of points.
`kernel_operator` builds an operator from those directly, and picks a representation
based on the structure it can detect.

```python
import jax
import jax.numpy as jnp
import linox

def rbf(a, b):
    return jnp.exp(-0.5 * jnp.sum((a - b) ** 2))

x = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
k = linox.kernel_operator(rbf, x)

assert k.shape == (32, 32)
assert (k @ jnp.ones(32)).shape == (32,)
```

The kernel is a plain callable taking two points and returning a scalar; linox
vectorises it.

## Structure detection

For a self-covariance on a uniform one-dimensional grid, a stationary kernel makes
the matrix Toeplitz — constant along diagonals — so only the first row is needed:

```python
import jax.numpy as jnp
import linox

def rbf(a, b):
    return jnp.exp(-0.5 * jnp.sum((a - b) ** 2))

grid = jnp.linspace(0.0, 1.0, 64)
k = linox.kernel_operator(rbf, grid)

assert k.shape == (64, 64)
assert (k @ jnp.ones(64)).shape == (64,)
```

Otherwise you get an `ArrayKernel`, which evaluates in chunks rather than building
the full matrix at once. `chunk_size` controls the block size.

## Cross-covariance

Pass a second point set for a rectangular `k(X₀, X₁)`:

```python
import jax
import jax.numpy as jnp
import linox

def rbf(a, b):
    return jnp.exp(-0.5 * jnp.sum((a - b) ** 2))

key = jax.random.PRNGKey(0)
train = jax.random.normal(key, (20, 2))
test = jax.random.normal(jax.random.fold_in(key, 1), (5, 2))

k = linox.kernel_operator(rbf, test, train)
assert k.shape == (5, 20)
assert (k @ jnp.ones(20)).shape == (5,)
```

## In practice: regularised solves

A GP posterior needs `(K + σ²I)⁻¹ y`. Writing it that way produces an isotropic
additive operator:

```python
import jax
import jax.numpy as jnp
import linox

def rbf(a, b):
    return jnp.exp(-0.5 * jnp.sum((a - b) ** 2))

x = jax.random.normal(jax.random.PRNGKey(0), (24, 1))
y = jnp.sin(x[:, 0] * 3.0)

k = linox.kernel_operator(rbf, x)
regularised = k + 1e-2 * linox.Identity(24)

alpha = linox.solve(regularised, y)
assert alpha.shape == (24,)
```

!!! note "Symmetry"
    A self-covariance kernel matrix is symmetric, which is what the spectral
    shortcuts require. A cross-covariance is not square and does not go down that
    path.

## JIT

Kernel operators flatten with the point sets as leaves and the callable as static
data, so they can be passed into a jitted function:

```python
import jax
import jax.numpy as jnp
import linox

def rbf(a, b):
    return jnp.exp(-0.5 * jnp.sum((a - b) ** 2))

k = linox.kernel_operator(rbf, jnp.linspace(0.0, 1.0, 16))

@jax.jit
def apply(operator, vector):
    return operator @ vector

assert apply(k, jnp.ones(16)).shape == (16,)
```
