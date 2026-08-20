# Kronecker products

`Kronecker(A, B)` represents `A ⊗ B` — a matrix of size `(mₐ·m_b, nₐ·n_b)` stored as
two small factors.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (4, 4))

kron = linox.Kronecker(linox.Matrix(a), linox.Matrix(b))
assert kron.shape == (12, 12)
assert jnp.allclose(linox.todense(kron), jnp.kron(a, b))
```

Two 1000×1000 factors describe a 10⁶×10⁶ matrix from two million numbers.

## The vec trick

`(A ⊗ B) vec(X) = vec(B X Aᵀ)`, so a matvec is two small matrix products rather than
one enormous one — O(n³ᐟ²) instead of O(n²) in the size of the full operator:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (4, 4))
kron = linox.Kronecker(linox.Matrix(a), linox.Matrix(b))

x = jnp.ones(12)
assert jnp.allclose(kron @ x, jnp.kron(a, b) @ x)
```

## What is specialised

Almost everything, through the factors:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (3, 3))
spd_a, spd_b = a @ a.T + 3 * jnp.eye(3), b @ b.T + 3 * jnp.eye(3)
kron = linox.Kronecker(linox.Matrix(spd_a), linox.Matrix(spd_b))
dense = jnp.kron(spd_a, spd_b)

x = linox.solve(kron, jnp.ones(9))
assert jnp.linalg.norm(dense @ x - jnp.ones(9)) < 1e-8

sign, logabs = linox.slogdet(kron)
assert jnp.allclose(logabs, jnp.linalg.slogdet(dense)[1])

root = linox.todense(linox.sqrt(kron))
assert jnp.allclose(root @ root.T, dense, atol=1e-8)
```

The identities used: `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`, `det(A⊗B) = det(A)^n_b · det(B)^n_a`, and
the eigenvalues of `A⊗B` are the pairwise products of those of `A` and `B`.

## Nesting

Factors are operators, so Kroneckers nest — the usual shape for a tensor-product
grid:

```python
import jax.numpy as jnp
import linox

small = linox.Matrix(jnp.eye(2) * 2.0)
nested = linox.Kronecker(small, linox.Kronecker(small, small))

assert nested.shape == (8, 8)
assert jnp.allclose(linox.todense(nested), jnp.eye(8) * 8.0)
```

## Top-k eigenpairs

The spectrum of `A ⊗ B` is every pairwise product of the factor eigenvalues, so the
largest few can be found without touching the full operator:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (4, 4))
spd_a, spd_b = a @ a.T + 0.1 * jnp.eye(3), b @ b.T + 0.1 * jnp.eye(4)
kron = linox.Kronecker(linox.Matrix(spd_a), linox.Matrix(spd_b))

values, vectors, _info = linox.topk_eigh(kron, k=3, largest=True)
dense = jnp.kron(spd_a, spd_b)

assert jnp.allclose(values, jnp.sort(jnp.linalg.eigvalsh(dense))[::-1][:3], atol=1e-8)

# The eigenvectors satisfy the eigenvalue equation.
q = linox.todense(vectors)
for i in range(3):
    assert jnp.linalg.norm(dense @ q[:, i] - values[i] * q[:, i]) < 1e-8
```

`vectors` is a `KroneckerSelectedEigenvectors` — a matrix-free operator holding the
selected factor columns, never the outer product.
