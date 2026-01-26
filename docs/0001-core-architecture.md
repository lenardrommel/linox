# ADR-0001: Core Architecture for linox v0.0.3

## Status
Accepted (v0.0.3)

## Context
linox is a JAX-native LinearOperator library aiming to support structured linear algebra in iterative ML workflows (GPs, operator learning, kernel methods). The v0.0.3 release focuses on:

- compositional structure-aware algorithms (Kronecker, shifts, low-rank, etc.)
- caching expensive decompositions across repeated calls
- execution tracing with performance guidance ("lint")

## Decision A: Operators are PyTrees
All operator instances are JAX PyTrees.

### Rationale
- JIT-compile `matvec` / `matmat` and operator pipelines
- pass operators as arguments to jitted functions
- stable staging of kernels in training loops

### Consequences
- each concrete operator must define `tree_flatten` / `tree_unflatten`
- children contain arrays + child-operators; aux contains static metadata (shape/dtype/flags)
- caching and tracing must not rely on Python object identity

## Decision B: PSD and Symmetry are explicit wrappers
We represent algebraic properties using wrappers:
- `Sym(op)` indicates self-adjoint (symmetric/Hermitian)
- `PSD(op)` indicates positive semidefinite
- optional `SPD(op)` indicates positive definite

### Rationale
- mirrors gpytorch/linear_operator practice
- enables specialized stable algorithms (eigendecomposition-based solves, kron-topk, stable logdet)
- avoids unsafe implicit inference

### Consequences
- wrappers are semantic promises; may be unchecked by default for speed
- optional debug checks can validate (approx) symmetry/PSD assumptions
- tracing records when assumptions are used

## Decision C: Two-tier execution model (lazy + compiled kernels)
- Always: lazy `LinearOperator` evaluation via `matvec/matmat`
- Optional: staged compilation of kernels (solve/logdet/trace/topk) into pure-jittable callables

### Rationale
- keep operator graph lazy (no densification)
- enable highly optimized compiled inner loops in training

### Consequences
- caching lives outside JIT (Python-side)
- tracing is opt-in and not expected inside JIT (can be done around compilation or outside)

## Decision D: Cache is fingerprint-based and outside JIT
Expensive decompositions (eigh/cholesky/factor extraction/analysis) are cached in a session cache keyed by operator fingerprint.

### Rationale
- huge speedups for repeated GP workloads
- fingerprint changes when content/structure changes → no explicit invalidation necessary

### Consequences
- fingerprint policy must be well-defined (auto content hashing for small leaves)
- cache must be bounded (LRU)
- approximate methods: no keyed caching on (rank/maxiter) in v0.0.3 unless trivially safe

## Decision E: Tracing includes performance lint
Tracing records dispatch, cache hits/misses, algorithm choices, and emits "lint" findings for common performance pitfalls.

### Rationale
- makes performance predictable
- guides users toward batched MVP patterns and avoids accidental densify

### Consequences
- tracing is opt-in with near-zero overhead when disabled
- lint rules must be aionable and stable across releases

