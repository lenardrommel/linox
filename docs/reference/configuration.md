# Configuration

```python
import linox

original = linox.config.get_max_dense_n()
linox.config.set_max_dense_n(1000)
assert linox.config.get_max_dense_n() == 1000
linox.config.set_max_dense_n(original)
```

## Settings

| Setting | Default | Effect |
|---|---|---|
| `max_dense_n` | 2000 | Size above which `method="auto"` prefers approximate routes |
| `debug` | off | Enables densification performance warnings |
| `warn_on_densify` | — | Whether `todense` emits a warning |

```python
import linox

original = linox.is_debug()
linox.set_debug(True)
assert linox.is_debug()
linox.set_debug(original)
```

Debug mode also honours the `LINOX_DEBUG` environment variable.

## Method defaults

`set_default_method` pins the choice for one operation, overriding the size
heuristic but not an explicit `method=` argument:

```python title="not-executed"
linox.config.set_default_method("solve", "cg")
```

Valid method names per operation are declared in `linox.config.VALID_METHODS`, and
anything outside them is rejected:

```python
import linox

assert "cg" in linox.config.VALID_METHODS["solve"]
assert "lanczos" in linox.config.VALID_METHODS["sqrt"]
```

## Debug events

```python
import jax.numpy as jnp
import linox
import linox.config as config

seen = []
config.set_debug_hook(lambda event: seen.append(event.kind))
try:
    linox.Diagonal(jnp.arange(1.0, 5.0)).todense()
finally:
    config.set_debug_hook(None)

assert "densify" in seen
```

A `DebugEvent` carries `kind`, `msg`, `op_type`, `shape`, `dtype` and timing. Kinds
include `densify`, `matmul`, `warn`, `init` and the profiling kinds emitted around
solves and decompositions.

See [Avoiding densification](../guides/avoiding-densification.md) for the caveats —
in particular, `linox.todense()` the *function* does not emit a `densify` event.

## API reference

::: linox.config
