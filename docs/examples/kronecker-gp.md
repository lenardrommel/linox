# A Kronecker-structured GP

On a tensor-product grid with a separable kernel, the covariance factorises:
`K = K_x ⊗ K_y`. That turns an `n²×n²` problem into two `n×n` ones.

```python
import jax
import jax.numpy as jnp
import linox

def rbf_matrix(points, lengthscale=0.3):
    diff = points[:, None] - points[None, :]
    return jnp.exp(-0.5 * (diff / lengthscale) ** 2)

grid = jnp.linspace(0.0, 1.0, 20)
kx = linox.Matrix(rbf_matrix(grid))
ky = linox.Matrix(rbf_matrix(grid))

prior = linox.Kronecker(kx, ky)
assert prior.shape == (400, 400)
```

Two 20×20 factors stand in for a 400×400 covariance. At a 300×300 grid it would be
90000×90000 — eight gigabytes dense, two matrices of 720KB structured.

## Posterior mean

Add observation noise and solve. The sum is recognised as an isotropic shift:

```python
import jax
import jax.numpy as jnp
import linox

def rbf_matrix(points, lengthscale=0.3):
    diff = points[:, None] - points[None, :]
    return jnp.exp(-0.5 * (diff / lengthscale) ** 2)

grid = jnp.linspace(0.0, 1.0, 20)
kx = linox.Matrix(rbf_matrix(grid) + 1e-6 * jnp.eye(20))
prior = linox.Kronecker(kx, kx)

xx, yy = jnp.meshgrid(grid, grid, indexing="ij")
observations = jnp.sin(4 * xx) * jnp.cos(4 * yy)
y = observations.reshape(-1)

noisy = prior + 1e-2 * linox.Identity(400)
assert type(noisy).__name__ == "IsotropicAdditiveLinearOperator"

alpha = linox.solve(noisy, y)
assert alpha.shape == (400,)
```

## Marginal likelihood

The log-determinant is the expensive term, and Kronecker structure makes it cheap:

```python
import jax
import jax.numpy as jnp
import linox

def rbf_matrix(points, lengthscale=0.3):
    diff = points[:, None] - points[None, :]
    return jnp.exp(-0.5 * (diff / lengthscale) ** 2)

grid = jnp.linspace(0.0, 1.0, 12)
k = linox.Matrix(rbf_matrix(grid) + 1e-4 * jnp.eye(12))
prior = linox.Kronecker(k, k)

y = jnp.sin(jnp.linspace(0.0, 6.0, 144))
_sign, logdet = linox.slogdet(prior)
quadratic = y @ linox.solve(prior, y)

log_marginal = -0.5 * (quadratic + logdet + 144 * jnp.log(2 * jnp.pi))
assert jnp.isfinite(log_marginal)
```

`det(A ⊗ B) = det(A)^{n_b} det(B)^{n_a}`, so the 144×144 determinant comes from two
12×12 ones.

## Leading eigenpairs

For a low-rank approximation of the prior, the spectrum of a Kronecker product is the
pairwise products of the factor spectra:

```python
import jax
import jax.numpy as jnp
import linox

def rbf_matrix(points, lengthscale=0.3):
    diff = points[:, None] - points[None, :]
    return jnp.exp(-0.5 * (diff / lengthscale) ** 2)

grid = jnp.linspace(0.0, 1.0, 16)
k = linox.Matrix(rbf_matrix(grid) + 1e-4 * jnp.eye(16))
prior = linox.Kronecker(k, k)

values, vectors, _info = linox.topk_eigh(prior, k=10, largest=True)
assert values.shape == (10,)
assert jnp.all(values[:-1] >= values[1:])       # descending
```

`vectors` stays matrix-free: it holds the selected factor columns rather than the
256×10 dense array.

## What generalises

The pattern — factorise the operator, let the arithmetic find the structure, solve
through the factors — is the same for any separable kernel on a product grid, and for
the Kronecker-factored curvature approximations used in second-order optimisation.
