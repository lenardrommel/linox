# ADR-0006: Operator validation

- **Status**: Accepted (v0.0.3)
- **Date**: 2026-01-26
- **Decision owners**: linox maintainers

## Context

linox operators are composed heavily (e.g. `Kronecker(Add(...), IsotropicAdd(...))`), and
a lot of failures are avoidable *earlier* than in a downstream `solve` or `eigh`.
Examples:

- mismatching shapes (`Add(A, B)` with different shapes)
- inconsistent dtypes (mixing `float32` and `float64` without an explicit cast policy)
- invalid scalars (negative scale on PSD operator where ordering/PSD assumptions are used)
- invalid structural promises (`Symmetric(op)` wrapper around a non-symmetric matvec)

We want **reliable defaults** while keeping the performance characteristics that make
linox useful (lazy ops, no densification).

## Decision

### D1. Validate all operators by default (cheap, structural validation)

Every `LinearOperator` constructor validates *structural invariants* that are:

- **O(1)** in operator size (no `todense`, no factorization)
- **JIT-safe** (runs in Python, outside of `jit`)
- **deterministic** and **side-effect free**

This includes, depending on operator type:

- shape/rank checks (square where required, broadcast rules, etc.)
- dtype normalization checks (optional casting policy handled elsewhere)
- argument sanity checks (finite scalars, non-negative jitter, etc.)
- structural wrapper contracts (e.g. `PSD(op)` requires the wrapped operator to be square)

Validation must not perform large computations, allocate large buffers, or call
JAX operations that would stage significant work.

### D2. Separate *debug* validation from default validation

Numerical/semantic checks that can be expensive or require probing the operator
(e.g. “is this actually symmetric/PSD?”) are **not** run by default.

They are available behind a debug flag, controlled by `linox.config` and/or an
environment variable (e.g. `LINOX_DEBUG=1`), and may include:

- symmetry checks via random probes: `xᵀ(Ay) ≈ yᵀ(Ax)` for random `x, y`
- PSD checks via `xᵀAx ≥ -tol` for random probes
- finite output checks on a few random matvecs

These checks are best-effort and are intended for development and debugging.

### D3. Expose the validation helper

We expose a small public helper so users can validate entire composite graphs:

```python
import linox as lo

lo.validate(op)                 # runs cheap validation recursively
lo.validate(op, mode="debug")   # also runs expensive probe checks (if enabled)
```

This is particularly useful after building large operator graphs programmatically,
or when interoping with external operator libraries.

## Consequences

**Pros**
- Earlier, clearer errors: failures are caught at construction time with a helpful message.
- Safer composition: structural promises are checked consistently everywhere.
- Debugging becomes easier: debug mode can catch “silent” structural lies.

**Cons**
- A small constant overhead at operator construction time.
- A stricter stance may surface issues that previously “worked by accident”.

## Notes / guidance

- Default validation should never densify or require a materialized matrix.
- “PSD” and “Symmetric” wrappers should be treated as **promises** with optional
  debug verification; we don’t attempt to “prove” them by default.
- Validation errors should include:
  - operator type
  - expected vs actual shapes/dtypes
  - a short hint on how to fix (cast, wrap, transpose, etc.)
