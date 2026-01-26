# ADR-0002: JIT Strategy and Staging

## Status
Accepted (v0.0.3)

## Context
Users want:
- `matvec` to be jittable ("the more jit the better")
- not to lose lazy evaluation (no forced densification)
- predictable behavior with caching and tracing

## Decision A: `matvec/matmat` must be jittable for all operators
Each operator provides a pure JAX `matvec` (and ideally `matmat`) path.

### Rationale
- enables XLA fusion and device parallelism
- supports large structured operators with MVP-only algorithms

### Consequences
- operator data must be stored as arrays / PyTree children
- no Python object manipulation inside `matvec` paths
- `__matmul__` should route to matvec/matmat and remain jittable

## Decision B: Staging APIs instead of `@linox.jit`
We do not introduce a magic `@linox.jit` decorator that tries to "jit everything".
Instead, we provide explicit compilation helpers:

- `compile_matvec(op) -> (x -> op@x)` (mostly for convenience)
- `compile_solve(op, *, method=..., **kw) -> (b -> solve(op,b))`
- `compile_slogdet(op, *, method=..., **kw) -> (() -> slogdet(op))`
- `compile_slq(op, *, probes, lanczos_k, block_size, ...) -> callable`

### Rationale
- avoids confusion: caches/traces do not work inside JIT
- keeps user control over compilation boundaries
- matches how JAX is typically used in training loops (compile once, call many times)

### Consequences
- compilation functions may accept a cache object to precompute decompositions
- staged callables are pure JAX and can run inside `jax.jit`, `vmap`, `pmap`, etc.

## Decision C: Caching stays outside JIT
Caches are Python-side and keyed by fingerprints. Compiled kernels can consume cached artifacts as explicit inputs or closed-over constants.

### Rationale
- JAX forbids Python side-effects during tracing
- cache lookup should not occur inside staged computations

### Consequences
- patterns:
  1) precompute decomposition via cache, pass into jitted kernel
  2) compile a function that *assumes* decomposition already provided
- tracing can wrap compilation and outer-loop calls, but not inner compiled kernels

## Decision D: Preserve laziness while compiling
"Lowering to JAX" means: lowering the *operator structure and parameters* into a PyTree, not lowering to a dense matrix.

### Rationale
- keep memory efficiency
- maintain structure-aware dispatch (Kronecker, shift, etc.)
- allow MVP-based approximate algorithms to stay scalable

### Consequences
- any densification is explicitly marked and traced (and ideally linted)
- compilation boundaries must not force `.todense()` implicitly

