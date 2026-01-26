# JIT and "Lowering" in linox (Without Losing Laziness)

linox is built around a single constraint:

> **Operators stay lazy.** We never lower to dense unless the user explicitly requests it.

At the same time, we want aggressive JIT compilation for performance.

This document explains how both are achieved.

---

## 1) What "JIT-compatible" means in linox

### Requirement: matvec is pure JAX
Every `LinearOperator` must be representable as a **JAX PyTree** and must implement a matvec/matmat path using only JAX primitives.

That ensures:

- `jax.jit(lambda x: op @ x)` works
- batched MVP (`op @ X` with `X.shape=(N,R)`) works
- `vmap/pmap/shard_map` can work on operator pipelines

### PyTree rule of thumb
- **Children**: JAX arrays and child operators
- **Aux**: shape, dtype, tags, structural metadata

This lets JAX treat structure as static while still transforming the numeric parts.

---

## 2) "Lowering to JAX" does NOT mean densification

In linox:
- lowering = representing the *operator graph + parameters* as a PyTree
- NOT: producing `op.todense()`

We preserve lazy evaluation by compiling *the matvec program*, not by materializing matrices.

---

## 3) Two-tier compilation strategy

### Tier A: Always-jittable matvec/matmat
This is the baseline. It is required.

Example:
```python
y = jax.jit(lambda x: K @ x)(x)

