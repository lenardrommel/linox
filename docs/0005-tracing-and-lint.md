# ADR-0005: Tracing and Performance Lint

## Status
Accepted (v0.0.3)

## Context
Users need observability into:
- which algorithm was selected
- where time/memory goes
- whether caching helped
- whether code missed easy parallelism (e.g., unbatched probes)

## Decision
Provide opt-in tracing and a performance lint report.

### Trace records
- dispatch decision: rule / method chosen
- cache hit/miss: key and artifact type
- execution: elapsed time, sizes, approximate/exact
- densify events: any materialization to dense must be traced

### Lint detects
- sequential probe loops (Hutchinson/SLQ) instead of batched probes
- repeated single-RHS solves where multi-RHS would help
- accidental densification in hot paths
- missed structure (e.g., nested kron not flattened)

### Rationale
- improves trust and debuggability
- helps users tune performance without guesswork

### Consequences
- tracing must be near-zero overhead when disabled
- lint is best-effort and should not block execution
- trace output should be stable across minor versions

