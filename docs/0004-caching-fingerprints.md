# ADR-0004: Fingerprint-Based Caching

## Status
Accepted (v0.0.3)

## Context
Repeated solve/logdet/trace on the same operator structure is common. We want caching that:
- reuses decompositions across operations
- does not require manual invalidation
- works across separately constructed but equivalent operators

## Decision
Use hierarchical fingerprints (content-addressable identity) as cache keys.

### Fingerprint contents
- operator type
- shape/dtype
- structural children fingerprints
- scalar params (shifts/scales)
- leaf array identity:
  - auto policy: content hash for small arrays; metadata-only for large arrays

### Cache scope
- session cache (LRU) is the primary cache in v0.0.3

### What is cached
- exact decompositions: `eigh`, `cholesky`, `sqrt` (via eigh), factor extractions
- analyses: canonical form, IR summaries for dispatch/lint
- approximate methods: only cache probe vectors (if configured), not keyed Krylov bases

## Rationale
- avoids explicit invalidation: operator change => fingerprint change
- high reuse in GP workloads

## Consequences
- caching is outside JIT
- densification is never cached unless explicitly requested (avoid memory blowup)
- cache policies must be configurable (max entries / max bytes / eviction)

