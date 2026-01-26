# ADR-0003: PSD and Symmetry Wrappers

## Status
Accepted (v0.0.3)

## Context
GP kernels are PSD; many efficient algorithms rely on PSD/Sym assumptions:
- stable eigendecomposition usage
- `sI + K` solves and logdets via shifted eigenvalues
- kronecker eigen-composition and kron top-k selection

## Decision
Introduce explicit wrappers:

- `Sym(op)` declares self-adjointness (symmetric/Hermitian)
- `PSD(op)` declares positive semidefinite
- optional `SPD(op)` declares positive definite

Wrappers are compositional and may preserve/propagate under arithmetic operations.

## Rationale
- makes assumptions explicit and traceable
- enables safer defaults (prefer stable exact methods when PSD)
- aligns with gpytorch's operator design

## Consequences
- default behavior: trust wrappers (fast path)
- debug option: validate symmetry numerically on random probes:
  - `||Ax - A^T x|| / ||Ax||` small → likely symmetric
  - PSD validation via `x^T A x >= -tol` on probes (heuristic)
- tracing records when PSD/Sym assumptions are used for algorithm selection

