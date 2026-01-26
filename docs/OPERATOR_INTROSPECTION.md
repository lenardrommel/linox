# Operator Introspection & Tracking (linox)

This document defines a **uniform way** to inspect and reason about a `LinearOperator`
without densifying it. The key motivation is to enable **global** optimizations
(e.g. Kronecker top-k eigenpairs) that would be missed by naive recursion over binary trees.

---

## 1. Problem statement

Many operators are built as binary trees:
- `Kronecker(A, B)`
- `AddLinearOperator(A, B)`
- `ProductLinearOperator(A, B)`
- `ScaledLinearOperator(A, s)`

A recursive algorithm that only sees the local node will often do suboptimal work.
Example: `topk_eigh(Kronecker(Kronecker(A, B), C))` should operate on the **flattened**
factor list `[A, B, C]`, not compute top-k for the inner kron first.

Therefore, we need:
1. **Structure extraction** (flatten, pull out scalars, detect special forms)
2. **Property tracking** (symmetric/PSD/diagonal/unitary/lowrank…)
3. **Canonicalization** (normalize representation so algorithms can rely on it)

---

## 2. Minimal Introspection API

### Required on every operator
- `children() -> tuple[LinearOperator, ...]` (empty for leaf operators)
- `op_type: str` or `__class__.__name__`
- `tags: set[str]` (optional lightweight properties)

### Recommended properties
- `is_square`
- `is_symmetric`
- `is_psd` (or stronger: `is_spd`)
- `is_diagonal`
- `is_lowrank`
- `rank` (if lowrank)
- `supports_exact_eigh`, `supports_exact_slogdet`, ... (capabilities)

These can be implemented as:
- explicit attributes on special operators
- or derived by rules (e.g. Kronecker is PSD if all factors are PSD and scalar>=0)

---

## 3. OperatorIR (Intermediate Representation)

Introduce a small IR used by algorithm selection.

Examples:

### A) Kronecker IR
```python
KroneckerIR(
  scalar: float | None,
  factors: list[LinearOperator],
  tags: {"symmetric", "psd"}
)
```

### B) Isotropic shift IR (sI + A)
```python
IsotropicShiftIR(
  shift: float,
  base: LinearOperator,
  tags: {"symmetric"}
)
```

### C) Lowrank-plus-diagonal IR
```python
DiagPlusLowRankIR(
  diag: Diagonal,
  U: jax.Array,
  S: jax.Array,
  scale: float
)
```

**Why IR?**
- Lets `eigh`, `slogdet`, `trace`, `solve` choose methods globally.
- Lets you cache analysis results (see caching doc).

---

## 4. Canonicalization pass

`canonicalize(op)` rewrites operator trees to a standard form.

### Canonicalization rules (examples)
- Flatten associative ops:
  - `Kronecker(Kronecker(A,B),C) -> Kronecker(A,B,C)` (internally factor list)
  - same for Add/Product
- Pull out scalars:
  - `Scaled(Kronecker(...), s) -> Kronecker(...), scalar=s`
  - `Scaled(Scaled(A,a),b) -> Scaled(A, a*b)`
- Rewrite special cases:
  - `Scaled(Identity(n), s) + A` with symmetric A -> `IsotropicAdditiveLinearOperator(s, A)`
- Simplify:
  - `A + Zero -> A`, `A @ Identity -> A`
- Normalize ordering (optional but helpful for caching):
  - sort factors by size or stable signature

---

## 5. How algorithms should use introspection

Public entry points (e.g. `eigh`, `sqrt`, `slogdet`, `trace`, `solve`) should:
1. `op = canonicalize(op)` (cheap if cached)
2. `ir = analyze(op)`
3. dispatch based on IR (exact/structured/approx)

### Example: top-k eigensolver
- if `ir` is KroneckerIR + PSD:
  - run heap search on product grid (your `topk_eigh_info` approach)
  - compute factor eigenpairs once
- else:
  - fallback to Lanczos/Lobpcg for general symmetric op

---

## 6. Relationship to `linop_graph` and debugging

Your `graph.py` is a natural place to:
- render operator trees
- expose canonicalized vs raw graphs
- annotate nodes with tags (symmetric/psd/diagonal)
- show cached items (optional)

Recommended additions:
- `linop_graph(op, canonical=False)`
- `linop_graph(op, canonical=True)`
- `linop_graph(op, show_tags=True, show_cache=True)` (debug mode only)

---

## 7. Checklist for v0.0.3

- [ ] Define `OperatorIR` types (dataclasses)
- [ ] Implement `analyze(op)` returning IR or a generic `OpInfo`
- [ ] Implement `canonicalize(op)` with core rewrite rules
- [ ] Ensure `extract_kronecker_factors` is used by `eigh(k=...)` and `topk_eigh_info`
- [ ] Add property propagation rules (Kronecker/Add/Product/Scale)
- [ ] Extend `linop_graph` to visualize canonical form and tags
