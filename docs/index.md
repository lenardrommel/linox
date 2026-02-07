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
- [ROADMAP v0.0.3](ROADMAP_v0_0_3.md) — Implementation plan and milestones

### Architecture Decision Records (ADRs)

- [ADR-0001: Core Architecture](0001-core-architecture.md)
- [ADR-0002: JIT Strategy](0002-jit-and-staging.md)
- [ADR-0003: PSD and Symmetry](0003-psd-and-symmetry.md)
- [ADR-0004: Caching Fingerprints](0004-caching-fingerprints.md)
- [ADR-0005: Tracing and Lint](0005-tracing-and-lint.md)
- [ADR-0006: Operator Validation](0006-operator-validation.md)
- [ADR-0007: Parallelism Defaults](0007-parallelism-defaults.md)
- [ADR-0008: Kernel Toeplitz](0008-kernel-toeplitz.md)

### Design Documents

- [API Design](API.md) — Public API specification
- [Operators](OPERATORS.md) — Operator taxonomy and tags
- [Methods](METHODS.md) — Method dispatch and priority rules
- [Caching and Tracing](CACHING_AND_TRACING.md) — Caching strategy and debug tracing
- [Operator Introspection](OPERATOR_INTROSPECTION.md) — IR and canonicalization
- [Approx Backend](APPROX_BACKEND.md) — Approximation algorithms