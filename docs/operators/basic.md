# Basic operators

The building blocks. Each stores only what it needs.

| Operator | Represents | Stores |
|---|---|---|
| `Matrix(A)` | a general matrix | the array |
| `Diagonal(d)` | `diag(d)` | the diagonal |
| `Identity(n)` | `I` | nothing |
| `Scalar(a)` | `a·I` | one number |
| `Zero(shape)` | `0` | nothing |
| `Ones(shape)` | `11ᵀ` | nothing |
| `Permutation(p)` | a permutation matrix | the permutation |
| `Toeplitz(c)` | a symmetric Toeplitz matrix | the first column |

```python
import jax.numpy as jnp
import linox

m = linox.Matrix(jnp.eye(3) * 2.0)
d = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
i = linox.Identity(3)
z = linox.Zero((3, 3))
o = linox.Ones((3, 3))
p = linox.Permutation(jnp.array([2, 0, 1]))

for op in (m, d, i, z, o, p):
    assert op.shape == (3, 3)
    assert (op @ jnp.ones(3)).shape == (3,)
```

## Why the trivial ones earn their place

`Identity` and `Zero` are not conveniences. They are what the rewrite rules match
against — `A + s * Identity(n)` is the pattern that produces a fast spectral
operator, and `Zero` terms are dropped from sums:

```python
import jax.numpy as jnp
import linox

a = linox.Diagonal(jnp.arange(1.0, 4.0))
assert jnp.allclose(
    linox.todense(a + linox.Zero((3, 3))),
    linox.todense(a),
)
```

## Constructors

`linox.eye`, `zeros`, `ones` and `diag` mirror the numpy spelling:

```python
import jax.numpy as jnp
import linox

assert isinstance(linox.eye(4), linox.Identity)
assert isinstance(linox.diag(jnp.arange(1.0, 5.0)), linox.Diagonal)
assert linox.zeros(3).shape == (3, 3)
assert linox.ones(3).shape == (3, 3)
```

`as_linop` promotes an array, and is what every function calls on its input:

```python
import jax.numpy as jnp
import linox

assert isinstance(linox.as_linop(jnp.eye(3)), linox.Matrix)
assert isinstance(linox.as_linop(linox.Identity(3)), linox.Identity)
```

## Toeplitz

A symmetric Toeplitz operator is defined by its first column and applies itself by
FFT in O(n log n) rather than O(n²):

```python
import jax.numpy as jnp
import linox

t = linox.Toeplitz(jnp.array([4.0, 1.0, 0.5, 0.1]))
dense = linox.todense(t)

assert dense[0, 1] == dense[1, 2] == 1.0     # constant along diagonals
assert jnp.allclose(dense, dense.T)
```

!!! note "Not every operation is specialised"
    Structure buys you the operations someone has written for it. `Toeplitz` has a
    fast matvec and a fast solve; it has no square-root dispatch, and asking for one
    raises `NotImplementedError` rather than quietly falling back to something else.

## Dtypes

Operators that synthesise their own values follow JAX's floating dtype — float64
under x64, float32 otherwise — and an explicit `dtype=` overrides:

```python
import jax.numpy as jnp
import linox

assert linox.todense(linox.Identity(3)).dtype == jnp.float64
assert linox.todense(linox.Identity(3, dtype=jnp.float32)).dtype == jnp.float32
```
