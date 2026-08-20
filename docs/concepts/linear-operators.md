# Linear operators

A `LinearOperator` is a linear map that knows how to apply itself. It has a `shape`
and a `dtype` like an array, but it does not have to store one.

```python
import jax.numpy as jnp
import linox

op = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))
assert op.shape == (3, 3)
assert (op @ jnp.ones(3)).shape == (3,)
```

`Diagonal` stores three numbers and represents a 3×3 matrix. That gap is the point
of the library: for a Kronecker product of two 1000×1000 factors it is two million
numbers standing in for a matrix with 10¹² entries.

## The minimum an operator provides

Every operator supplies a matrix-vector product. Most also supply a dense form, used
only when something genuinely requires the matrix:

```python
import jax.numpy as jnp
import linox

op = linox.Diagonal(jnp.array([1.0, 2.0, 3.0]))

matvec = op @ jnp.ones(3)                 # what the operator is for
dense = linox.todense(op)                 # what it stands for

assert dense.shape == (3, 3)
assert jnp.allclose(dense @ jnp.ones(3), matvec)
```

`todense` is always available and almost always the wrong thing to reach for. See
[Avoiding densification](../guides/avoiding-densification.md).

## Right-hand sides

An operator accepts a vector or a block of columns, and the two agree:

```python
import jax
import jax.numpy as jnp
import linox

op = linox.Diagonal(jnp.arange(1.0, 5.0))
rhs = jax.random.normal(jax.random.PRNGKey(0), (4, 3))

block = op @ rhs
columns = jnp.stack([op @ rhs[:, j] for j in range(3)], axis=-1)

assert block.shape == (4, 3)
assert jnp.allclose(block, columns)
```

## Transpose

`.T` returns an operator, preserving structure where the operator knows its own
transpose:

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.arange(1.0, 5.0))
assert isinstance(d.T, linox.Diagonal)     # a diagonal is its own transpose

m = linox.Matrix(jnp.arange(6.0).reshape(2, 3))
assert m.T.shape == (3, 2)
```

For an operator with no structured transpose, `.T` is a lazy wrapper that derives
the adjoint from the forward matvec — it does not build the matrix.

The defining property holds for every operator:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
op = linox.Matrix(jax.random.normal(key, (5, 3)))
x = jax.random.normal(jax.random.fold_in(key, 1), (3,))
y = jax.random.normal(jax.random.fold_in(key, 2), (5,))

# <Ax, y> == <x, A^T y>
assert jnp.allclose(jnp.vdot(op @ x, y), jnp.vdot(x, op.T @ y))
```

!!! note "Transpose, not adjoint"
    `.T` is a plain transpose. For complex operators that is not the conjugate
    transpose. `is_symmetric` and `is_hermitian` distinguish the two correctly.

## Operators are pytrees

Every operator is registered as a JAX pytree, so it flattens into its arrays and can
cross a `jit` boundary as an argument:

```python
import jax
import jax.numpy as jnp
import linox

op = linox.Diagonal(jnp.arange(1.0, 5.0))
leaves, treedef = jax.tree_util.tree_flatten(op)
rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

assert jnp.allclose(rebuilt @ jnp.ones(4), op @ jnp.ones(4))
```

This is what makes `jax.jit(f)(operator, x)` work. See
[JIT and autodiff](jit-and-autodiff.md).
