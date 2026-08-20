# Structure and laziness

Two ideas do most of the work in linox: arithmetic builds a description rather than a
result, and that description is rewritten into whatever structure is most useful.

## Laziness

`A + B` does not add anything:

```python
import jax.numpy as jnp
import linox

a = linox.Diagonal(jnp.arange(1.0, 5.0))
b = linox.Diagonal(jnp.ones(4))

total = a + b
assert isinstance(total, linox.Diagonal)   # rewritten, not deferred
```

Two diagonals *can* be added cheaply, so they are. Where no such rule exists, the sum
is held as a composite and evaluated only on application:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
m = linox.Matrix(jax.random.normal(key, (4, 4)))
k = linox.Kronecker(linox.Matrix(jnp.eye(2)), linox.Matrix(jnp.eye(2)))

composite = m @ k
assert type(composite).__name__ == "ProductLinearOperator"

# Applied right to left; no 4x4 product is formed.
assert (composite @ jnp.ones(4)).shape == (4,)
```

## Rewriting

Sums are inspected for patterns worth specialising. The most important is an
isotropic shift — a matrix plus a multiple of the identity, which is what
regularisation and GP jitter look like:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

shifted = spd + 0.1 * linox.Identity(4)
assert type(shifted).__name__ == "IsotropicAdditiveLinearOperator"
```

That is not cosmetic. The rewritten operator caches an eigendecomposition of `A` and
serves inverse, square root, log, determinant and powers from it, since the
eigenvectors of `A` and `s·I + A` coincide and only the eigenvalues shift.

A diagonal plus a symmetric low-rank term is recognised similarly and solved by the
Woodbury identity rather than by forming the sum.

The rewrites are meaning-preserving:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
raw = dense @ dense.T + 4 * jnp.eye(4)

rewritten = linox.Matrix(raw) + 0.5 * linox.Identity(4)
plain = linox.Matrix(raw + 0.5 * jnp.eye(4))

assert jnp.allclose(linox.todense(rewritten), linox.todense(plain))
```

## Structure is a promise about the operator

Some structure is intrinsic — a `Diagonal` is diagonal by construction. Some is
asserted by you:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)

asserted = linox.PSD(linox.Matrix(spd))
assert asserted.is_psd
assert asserted.is_symmetric
```

`Sym`, `PSD` and `SPD` attach a property without checking it. They are promises, and
wrapping does not copy or materialise anything — the wrapper delegates every matvec
to the operator underneath.

## When laziness ends

A composite collapses to numbers when you apply it, solve with it, or ask for its
dense form. Everything before that is bookkeeping — which is why building a large
expression costs nothing until it is used.
