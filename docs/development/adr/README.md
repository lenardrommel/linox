# Architecture decision records

Short documents recording decisions that were not obvious, and why the alternative
was rejected. Kept because the reasoning is otherwise lost, and someone reasonably
proposes the alternative again a year later.

Two records are referenced from the source (`wrappers.py` cites ADR-0003,
`validation.py` cites ADR-0006) but were never written. The decisions they describe
are documented in the pages below until they are:

| Decision | Where it is written up |
|---|---|
| Property wrappers as unchecked promises | [Structure and laziness](../../concepts/structure-and-laziness.md) |
| Validation as an explicit call rather than a constructor check | [Debugging](../../guides/debugging-and-tracing.md) |
| `sqrt` returns a left square root | [Decompositions](../../algorithms/decompositions.md) |
| Symmetry enforced per operation, not at construction | [Structured operators](../../operators/structured.md) |
| Both CG modes rather than choosing one | [Solving](../../algorithms/solving.md) |

## Format

```markdown title="not-executed"
# ADR-NNNN: Title

## Status
Accepted | Superseded by ADR-MMMM

## Context
What forced a decision.

## Decision
What was decided.

## Consequences
What this makes easy, and what it makes hard.
```
