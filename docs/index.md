# linox

**Linear operators in JAX** — A library for structured linear algebra in iterative ML workflows.

## v0.0.3 Release

linox v0.0.3 delivers:
- Clean package structure (`operators/`, `linalg/`, `utils/`)
- Unified public API with `method="auto"|"exact"|"approx"` dispatch
- Approximation backends (Lanczos, Hutchinson, SLQ, LSMR)
- Operator introspection, canonicalization, and fingerprint-based caching
- Tracing and performance lint
- PSD/symmetry wrappers and factored operators

## Quick Start

```python
import jax
import linox as lo

# Create operators
A = lo.Matrix(jax.random.normal(jax.random.key(0), (100, 100)))
D = lo.Diagonal(jax.numpy.ones(100))

# Compose operators lazily
B = A + 0.1 * D  # No materialization

# Use operators
x = jax.numpy.ones(100)
y = B @ x  # Lazy matvec

# Solve linear systems
b = lo.solve(B, y)
```

## Documentation

- [API Reference](reference.md) — Complete API documentation
