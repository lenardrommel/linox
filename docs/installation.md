# Installation

```bash title="not-executed"
pip install linox
```

Or from source:

```bash title="not-executed"
git clone https://github.com/lenardrommel/linox
cd linox
pip install -e .
```

Python 3.10 or newer.

## Dependencies

The core install pulls in `jax`, `plum-dispatch`, `numpy`, `scipy` and `jaxtyping`.

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `test` | pytest, hypothesis, pytest-cases | Running the test suite |
| `docs` | mkdocs-material, mkdocstrings | Building this site |
| `interop` | skerch, torch | [skerch interop](guides/skerch-interop.md) |

```bash title="not-executed"
pip install -e ".[test]"
```

The `interop` extra is deliberately separate: skerch pulls in torch and h5py, which
is a lot of install for a JAX library whose own test suite never touches them.

!!! note "pytest is capped below 9"
    `pytest-cases` does not yet support pytest 9 — it fails during plugin load,
    before collecting anything. The `test` extra pins `pytest>=8,<9` accordingly.

## Double precision

JAX defaults to float32. linox currently enables x64 at import time, so operators
default to float64:

```python
import jax
import linox
import jax.numpy as jnp

assert jax.config.jax_enable_x64
assert linox.todense(linox.Identity(3)).dtype == jnp.float64
```

Operators follow the flag rather than pinning a dtype, so if you disable x64 they
produce float32. Explicit dtypes always win:

```python
import jax.numpy as jnp
import linox

assert linox.todense(linox.Identity(3, dtype=jnp.float32)).dtype == jnp.float32
```

!!! warning "This is likely to change"
    Enabling x64 as an import side effect changes the dtype policy of your whole
    program, which a library should not do. It is tracked for removal; once it goes,
    linox will follow whatever you have configured rather than deciding for you.
