# Structured operators

Operators that carry a factorisation, and use it.

## EigenD

An operator held as `Q Λ Qᵀ`. Spectral functions become elementwise operations on
the eigenvalues:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
q, _ = jnp.linalg.qr(jax.random.normal(key, (4, 4)))
op = linox.EigenD(linox.Matrix(q), linox.Diagonal(jnp.linspace(1.0, 4.0, 4)))

assert op.shape == (4, 4)
assert jnp.allclose(op.eigenvalues, jnp.linspace(1.0, 4.0, 4))
```

Both factors are themselves operators, so the eigenvectors can be a `Kronecker` — a
full eigendecomposition of a huge structured matrix without a dense `Q`.

## IsotropicAdditiveLinearOperator

`s·I + A` for symmetric `A`. Built automatically when you write that expression:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

op = spd + 0.1 * linox.Identity(4)
assert type(op).__name__ == "IsotropicAdditiveLinearOperator"
```

`A` and `s·I + A` share eigenvectors, so one cached eigendecomposition serves
inverse, square root, log, powers, determinant and trace.

### It requires symmetry, and checks

The shortcuts use `eigh`, which reads only the lower triangle. A non-symmetric
operand would give a silently wrong inverse, so it is rejected:

```python
import jax.numpy as jnp
import linox

non_symmetric = linox.Matrix(jnp.array([[1.0, 2.0], [0.0, 1.0]]))
op = non_symmetric + linox.Identity(2)

# The matvec is correct for any square operand, so it stays allowed.
assert jnp.allclose(linox.todense(op), jnp.array([[2.0, 2.0], [0.0, 2.0]]))

# The spectral shortcut refuses rather than returning the symmetrised answer.
try:
    linox.todense(linox.inverse(op, method="exact"))
    raise AssertionError("expected a rejection")
except ValueError as exc:
    assert "symmetric" in str(exc)
```

The check is matrix-free — it probes `⟨x, Ay⟩ == ⟨Ax, y⟩` rather than comparing
matrices — and applies under `jit` too, where it becomes a runtime error.

### Positivity is required per operation

Symmetry alone is enough for inverse, `eigh` and `slogdet`. A square root, Cholesky
factor, logarithm or fractional power additionally needs the *shifted* spectrum
`s + λ` to be non-negative:

```python
import jax.numpy as jnp
import linox

# A symmetric but indefinite operand: eigenvalues -2 and +3.
q = jnp.array([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(2.0)
indefinite = linox.Matrix(q @ jnp.diag(jnp.array([-2.0, 3.0])) @ q.T)

small_shift = indefinite + 0.1 * linox.Identity(2)
linox.inverse(small_shift, method="exact")        # fine: well defined

try:
    linox.sqrt(small_shift, method="exact")
    raise AssertionError("expected a rejection")
except ValueError as exc:
    assert "spectrum" in str(exc)

# A shift large enough to lift the spectrum makes it valid again.
big_shift = indefinite + 3.0 * linox.Identity(2)
root = linox.todense(linox.sqrt(big_shift, method="exact"))
assert jnp.allclose(root @ root.T, linox.todense(big_shift), atol=1e-10)
```

## Factor operators

Where a factorisation is already known, wrap it rather than recomputing:

- `Triangular(A, lower=True)` — a triangular factor
- `CholeskyFactor(L)` — a Cholesky factor
- `PSDFromFactor(L)` — the operator `L Lᵀ`, never formed

## Property wrappers

`Sym`, `PSD` and `SPD` assert a property the type system cannot see:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
op = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

wrapped = linox.assume_psd(op)
assert wrapped.is_psd and wrapped.is_symmetric
assert isinstance(wrapped.T, linox.PSD)          # structure survives transpose
```

They are unchecked promises, and they stay matrix-free — wrapping does not copy or
densify anything.
