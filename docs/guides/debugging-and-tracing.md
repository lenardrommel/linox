# Debugging and tracing

## Inspecting an operator's structure

A composite operator is a tree. `linop_graph` shows it:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
expr = 2.0 * linox.Matrix(jax.random.normal(key, (4, 4))) + linox.Identity(4)

graph = linox.linop_graph(expr)
text = graph.pretty()

assert isinstance(text, str)
assert len(text) > 0
```

`op.graph_str()` is the same thing as a method. This answers "what did my arithmetic
actually build?", which matters because the rewrites mean the answer is often not
what you wrote.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

# Written as a sum; built as something else entirely.
built = spd + 0.5 * linox.Identity(4)
assert type(built).__name__ == "IsotropicAdditiveLinearOperator"
```

## Recording what an operation does

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
op = linox.Matrix(jax.random.normal(key, (4, 4)))

result, report = linox.inspect_run(lambda: linox.det(op))
assert isinstance(report.summary(), str)
```

`inspect_run` executes a callable and returns the result alongside a report of the
events it emitted — densifications, matmuls, warnings.

## Debug mode

```python
import linox

original = linox.is_debug()
linox.set_debug(True)
assert linox.is_debug()
linox.set_debug(original)
```

Debug mode enables performance warnings on densification. It is off by default
because the warning path costs something on every operation.

## Validating an operator

`validate` checks that an operator does what it claims — shapes are consistent, and
asserted properties actually hold:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)

linox.validate(linox.Matrix(spd))
```

This matters most with the property wrappers, which are unchecked promises. `PSD(A)`
does not verify that `A` is positive semi-definite; `validate` will.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)

# Randomised probing, no densification.
assert linox.is_symmetric(linox.Matrix(spd))
assert linox.is_square(linox.Matrix(spd))
```

## When results look wrong

A checklist, in the order that has actually found bugs in this codebase:

1. **Compare against the dense answer on a small case.** `linox.todense(op)` and
   `jnp.linalg` give you a reference. If they disagree, the operator is lying.
2. **Check the property, not the value.** Does `A.T` satisfy `⟨Ax,y⟩ = ⟨x,Aᵀy⟩`?
   Does `S Sᵀ = A`? A wrong implementation often produces plausible numbers that
   fail the defining identity.
3. **Check what was actually built.** `type(expr).__name__` after a rewrite.
4. **Check the outcome, not just the output.** `solve(..., return_info=True)` tells
   you whether it converged.
5. **Compare eager against jit.** Divergence points at a trace-time decision made on
   a value rather than a type.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (5, 5))
spd = dense @ dense.T + 5 * jnp.eye(5)
op = linox.Matrix(spd)
b = jnp.ones(5)

eager = linox.solve(op, b)
jitted = jax.jit(lambda m, v: linox.solve(linox.Matrix(m), v))(spd, b)
reference = jnp.linalg.solve(spd, b)

assert jnp.allclose(eager, jitted, atol=1e-10)
assert jnp.allclose(eager, reference, atol=1e-10)
```
