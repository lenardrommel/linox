# ADR-0007: Parallelism and batched execution defaults

- **Status**: Accepted (v0.0.3)
- **Date**: 2026-01-26
- **Decision owners**: linox maintainers

## Context

JAX/XLA exploits parallelism well **inside** a single compiled program, but many
linear-algebra workloads contain *embarrassingly parallel* structure that can be
missed if expressed as Python loops:

- Hutchinson trace / SLQ probes: `zᵀAz` for many random `z`
- multiple RHS solves: `solve(A, b_i)` for many `b_i`
- block Krylov methods (Lanczos/Arnoldi) can expose more SIMD/GPU parallelism than
  single-vector versions

We want linox to “do the right thing by default”, especially on GPU/TPU.

## Decision

### P1. Prefer batched probes by default (multi-RHS MVPs)

For trace estimators, SLQ, and similar Monte-Carlo routines:

- generate probe matrix `Z` with shape `(n, p)`
- compute `AZ = A @ Z` **once** (batched MVP)
- reduce: `trace ≈ sum(Z * AZ) / p`

This is default-on and controlled by config (e.g. `linox.config.parallel.probes_batch=True`).

### P2. Prefer block Krylov when the backend supports it

For Lanczos/Arnoldi-based approximations (e.g. SLQ), we support:

- **block Lanczos** with `block_size >= 1`
- default `block_size` tuned for typical GPU use (e.g. 8), but configurable

Block methods typically improve hardware utilization and can reduce variance for
some estimators.

### P3. Multi-device parallelism is optional and utility-driven

If multiple devices are available, we allow users to opt into distributing work
(e.g. across probes) via a small utility:

- detect device mesh / available GPUs
- use `jax.shard_map` or `pmap`/`vmap` patterns to shard probe batches

This is not required for correctness; it is an optimization path and can come later.

### P4. Tracing emits *warnings only* for missed parallelism

The tracing/lint subsystem may flag patterns that likely hurt performance:

- sequential Python loop over probes
- repeated single-RHS solves where a multi-RHS solve is possible
- densification in the middle of a batched routine

These are **warnings** (not errors) by default.

## Consequences

**Pros**
- Large speedups on GPU by turning “many small matvecs” into “one big matvec”.
- Cleaner APIs: multi-RHS becomes a first-class concept (`(n, r)` inputs).
- Tracing becomes a teaching tool: shows users how to structure code for XLA.

**Cons**
- Batched code can increase peak memory; linox should estimate bytes and pick
  safe defaults (see config and lint).
- Some approximate methods may have different numerical behavior in block form;
  we keep conservative defaults and allow opt-out.

## Practical guidance

- Expose `matvec`/`matmat` consistently; “matmat” (`A @ B`) should be the primary path.
- Prefer `vmap`/batching over Python loops inside core algorithms.
- When memory is tight, reduce `p` (probes) or `block_size` before falling back to loops.
