# Method dispatch

linox chooses an implementation from the *type* of the operator, using
[plum](https://github.com/beartype/plum) for multiple dispatch. `solve` is not one
function with a pile of `isinstance` checks; it is a generic with a method per
operator type.

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
d = linox.Diagonal(jnp.arange(1.0, 5.0))
dense = jax.random.normal(key, (4, 4))
m = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

# Same call, different implementations underneath.
assert linox.solve(d, jnp.ones(4)).shape == (4,)
assert linox.solve(m, jnp.ones(4)).shape == (4,)
```

Solving against a `Diagonal` divides elementwise. Solving against a `Kronecker`
solves against the factors. Neither implementation knows the other exists.

## Adding structure without touching the core

Because dispatch is external to the classes, a new operator can bring its own
implementations, and a new *operation* can be defined over existing operators. That
is the same extensibility argument lineax makes for `singledispatch`.

## The `method=` argument

Where an operation has both an exact and an approximate route, the choice is exposed:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (30, 30))
spd = linox.Matrix(dense @ dense.T + 30 * jnp.eye(30))

exact = linox.sqrt(spd, method="exact")
approx = linox.sqrt(spd, method="lanczos", num_iters=25)
```

`method="auto"` picks by operator size, using a dense threshold you can configure.
Unrecognised values raise rather than silently falling back:

```python
import jax.numpy as jnp
import linox

try:
    linox.sqrt(linox.Identity(4), method="definitely-not-a-method")
    raise AssertionError("expected a rejection")
except ValueError as exc:
    assert "Unknown method" in str(exc)
```

That validation matters more than it looks. Before it existed, a typo selected the
default silently — which is precisely how `sqrt` came to ignore `method=` entirely
without anyone noticing.

## Precedence and ambiguity

When two methods match and neither is more specific, plum raises
`AmbiguousLookupError` rather than guessing. Some dispatches carry an explicit
`precedence` to break such ties deliberately.

!!! warning "Never stack dispatch decorators"
    A plum `.dispatch` decorator returns the *Function object*, not the function it
    decorated. Stacking two registers the inner function under the outer generic
    with the inner's signature:

    ```python title="not-executed"
    @lsqrt.dispatch      # registers `linverse` as an `lsqrt` method!
    @linverse.dispatch
    def _(a: Identity) -> Identity:
        return a
    ```

    That exact code made `lsqrt` return the *inverse* for every operator without a
    specific dispatch. Register separately. A test scans the package for the pattern
    so it cannot come back.
