# Avoiding densification

The point of an operator is not to become a matrix. When one does anyway it is
usually silent, and usually the reason something got slow or ran out of memory.

## Detecting it

linox emits debug events, and you can hook them:

```python
import jax
import jax.numpy as jnp
import linox
import linox.config as config

def events_during(fn):
    """Record the kinds of debug event a call emits."""
    seen = []
    config.set_debug_hook(lambda e: seen.append(e.kind))
    try:
        fn()
    finally:
        config.set_debug_hook(None)
    return seen

op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(0), (4, 4)))

assert "densify" not in events_during(lambda: op @ jnp.ones(4))
assert "densify" in events_during(op.todense)
```

!!! warning "Two different signals, and neither is complete"
    This is fiddlier than it should be, so it is worth stating precisely.

    - The **`densify`** event is emitted by the `.todense()` *method*.
    - `linox.todense(op)` — the *function* — calls the private `_todense()` and
      emits **nothing**.
    - Operations that densify internally (`det`, `qr`, `lu_factor`, full `svd`)
      emit a **`warn`** event instead.

    So a check for `densify` alone will miss most real densification. Watch for both
    kinds:

    ```python
    import jax
    import jax.numpy as jnp
    import linox
    import linox.config as config

    def densifications(fn):
        seen = []
        config.set_debug_hook(lambda e: seen.append(e.kind))
        try:
            fn()
        finally:
            config.set_debug_hook(None)
        return sum(k in ("densify", "warn") for k in seen)

    op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(0), (4, 4)))

    assert densifications(lambda: op @ jnp.ones(4)) == 0
    assert densifications(lambda: linox.det(op)) > 0
    ```

    Even then a zero count is strong evidence rather than proof. When it really
    matters, subclass the operator and instrument its `_todense` — that is how a
    densifying transpose was found that the event counter had missed entirely.

## What is matrix-free

Reliably free: matvecs on structured operators, `.T`, structured solves, CG, the
Krylov algorithms, Hutchinson and SLQ, and the `Sym`/`PSD`/`SPD` wrappers.

```python
import jax
import jax.numpy as jnp
import linox
import linox.config as config

def densifications(fn):
    seen = []
    config.set_debug_hook(lambda e: seen.append(e.kind))
    try:
        fn()
    finally:
        config.set_debug_hook(None)
    return sum(k in ("densify", "warn") for k in seen)

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (4, 4))
spd = a @ a.T + 4 * jnp.eye(4)
kron = linox.Kronecker(linox.Matrix(spd), linox.Matrix(spd))

assert densifications(lambda: kron @ jnp.ones(16)) == 0
assert densifications(lambda: kron.T @ jnp.ones(16)) == 0
assert densifications(lambda: linox.solve(kron, jnp.ones(16))) == 0
```

Densifies by design: `todense`, `det`, `lu_factor`, `lu_solve`, `qr`, full `svd`
without `k`, and `eigh` on an operator with no structured dispatch.

## The common mistakes

**Asking for a full decomposition when you want a few components.** `svd(op)`
densifies; `svd(op, k=10)` does not.

**Reaching for an exact trace or log-determinant of something enormous.** Use
Hutchinson or SLQ, with a key.

**Breaking a rewrite by reassociating.** `A + s * I` is recognised as an isotropic
shift; burying it inside another sum may hide the pattern. Build the expression so
the structure is visible at the top level.

**Leaving a debug `todense` in.** It is the quickest way to turn a matrix-free
program dense.

## Checking your own code

The most robust guard is a test, not a habit. The conformance suite asserts zero
densification for operators documented matrix-free; the same technique works for
your own pipeline:

```python
import jax
import jax.numpy as jnp
import linox
import linox.config as config

def test_my_pipeline_stays_matrix_free():
    seen = []
    config.set_debug_hook(lambda e: seen.append(e.kind))
    try:
        op = linox.Kronecker(linox.Matrix(jnp.eye(4)), linox.Matrix(jnp.eye(4)))
        linox.solve(op, jnp.ones(16))
    finally:
        config.set_debug_hook(None)
    assert sum(k in ("densify", "warn") for k in seen) == 0

test_my_pipeline_stays_matrix_free()
```
