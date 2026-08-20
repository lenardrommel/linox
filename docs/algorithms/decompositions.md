# Decompositions

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (5, 5))
spd = linox.Matrix(dense @ dense.T + 5 * jnp.eye(5))

values, vectors = linox.eigh(spd)
assert values.shape == (5,)
```

## eigh

For symmetric operators. Structured operators use their structure: a `Kronecker`
combines factor eigendecompositions, an `EigenD` returns what it already holds.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
spd = a @ a.T + 3 * jnp.eye(3)
kron = linox.Kronecker(linox.Matrix(spd), linox.Matrix(spd))

values, vectors = linox.eigh(kron)

# Both come back as Kronecker operators, not arrays -- the eigenvalues are the
# pairwise products of the factors', and the eigenvectors the Kronecker product
# of the factors'. Neither is materialised.
assert isinstance(values, linox.Kronecker)
assert isinstance(vectors, linox.Kronecker)

as_array = jnp.diagonal(linox.todense(values))
expected = jnp.sort(jnp.linalg.eigvalsh(jnp.kron(spd, spd)))
assert jnp.allclose(jnp.sort(as_array), expected, atol=1e-8)
```

!!! warning "The return type is not uniform"
    `eigh` gives `(array, Matrix)` for a dense operator but `(Kronecker, Kronecker)`
    for a Kronecker product. Keeping the structured form is the right call — the
    dense eigenvector matrix may be far too large — but the inconsistency means
    generic code has to handle both. `linox.todense` normalises if you need an
    array.

For the largest few eigenpairs of a Kronecker product without forming it, see
[`topk_eigh`](../operators/kronecker.md#top-k-eigenpairs).

## SVD

Full, or a partial SVD computed matrix-free by Lanczos bidiagonalisation:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (20, 12))
op = linox.Matrix(a)

u, s, vt = linox.svd(op, k=4, num_iters=12)
assert u.shape == (20, 4) and s.shape == (4,) and vt.shape == (4, 12)

# The rank-4 reconstruction is close to the best possible.
best = jnp.linalg.svd(a, full_matrices=False)
optimal = jnp.linalg.norm(a - best[0][:, :4] @ jnp.diag(best[1][:4]) @ best[2][:4])
assert jnp.linalg.norm(a - u @ jnp.diag(s) @ vt) <= optimal * 1.01 + 1e-8
```

Passing `k` is what keeps it matrix-free. Without it the operator is densified.

## Cholesky and square roots

`sqrt` returns a **left square root**: a factor `S` with `S Sᵀ = A`. It is not
necessarily symmetric, and not necessarily the principal root.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (5, 5))
spd = dense @ dense.T + 5 * jnp.eye(5)
op = linox.Matrix(spd)

s = linox.todense(linox.sqrt(op))
assert jnp.allclose(s @ s.T, spd, atol=1e-8)
```

Which factor you get depends on the operator: a Cholesky factor for a dense matrix,
the elementwise root for a `Diagonal`, `Q√(Λ+s)` for an isotropic shift. All satisfy
`S Sᵀ = A`, which is the contract.

For the symmetric principal root `A^{1/2}`, ask for the Krylov method:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(2)
dense = jax.random.normal(key, (30, 30))
spd = dense @ dense.T + 0.1 * jnp.eye(30)
op = linox.Matrix(spd)

principal = linox.sqrt(op, method="lanczos", num_iters=25)

w, v = jnp.linalg.eigh(spd)
exact = v @ jnp.diag(jnp.sqrt(w)) @ v.T
x = jax.random.normal(key, (30,))
assert jnp.linalg.norm(principal @ x - exact @ x) / jnp.linalg.norm(exact @ x) < 0.05
```

!!! note "Not every operator has a square root"
    An operator with no `lsqrt` dispatch raises `NotImplementedError` rather than
    falling back to something that is not a square root:

    ```python
    import jax.numpy as jnp
    import linox

    try:
        linox.sqrt(linox.Toeplitz(jnp.array([4.0, 1.0])), method="exact")
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    ```

## QR and LU

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (5, 5))
op = linox.Matrix(a @ a.T + 5 * jnp.eye(5))

q, r = linox.qr(op)
assert jnp.allclose(q @ r, linox.todense(op), atol=1e-8)

lu = linox.lu_factor(op)
x = linox.lu_solve(op, jnp.ones(5))
assert jnp.linalg.norm(linox.todense(op) @ x - jnp.ones(5)) < 1e-8
```

Both densify. They exist for completeness rather than as the fast path.
