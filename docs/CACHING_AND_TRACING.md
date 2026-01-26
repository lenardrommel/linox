# Caching, Tracing & Structure-Aware Execution (linox)

This document describes **how linox should cache**, **how it should trace**, and how both interact with
**JAX/JIT** and **structure-aware algorithms** (e.g. Kronecker top-k eigensolvers, Woodbury, isotropic shifts).

> Goals
> - Speed: avoid recomputing expensive decompositions / probes repeatedly.
> - Correctness: caches must not silently become stale.
> - JAX compatibility: keep **JIT/pytree** behavior predictable.
> - Observability: users can **see** when densification happens, when caches are hit/missed, etc.

---

## 1. Two kinds of caching: A) operator-local, B) analysis/session caches

### A) Operator-local caches (lazy, stable, structural)
These caches are stored **inside operator instances** and are typically safe when:
- The operator is **pure/immutable** (no in-place mutation).
- The cached result depends only on the operator’s children & parameters.

Examples of good operator-local caches:
- `IsotropicAdditiveLinearOperator`: cache `Q, S` from `leigh(A)`
- `ToeplitzKernel`: cache FFT plans / precomputed spectrum (if stable)
- `Kronecker`: cache extracted factor list `factors` and extracted `scalar`
- `BlockDiagonal`: cache block offsets / index maps

**Rules:**
- Caches are created **lazily** on first use.
- Any constructor or rewrite that changes children should invalidate derived caches.
- Caches should never store **huge dense arrays** by default (unless explicitly requested).

**Implementation approach (recommended):**
- Use `functools.cached_property` for small derived objects (lists, shapes, indices).
- Use an explicit `_cache: dict[str, Any]` for larger items with manual invalidation.
- Add `clear_cache(self)` to every operator that stores caches.

**JAX note:** operator-local caches are Python-side; they are not part of the pytree.
That’s fine if caches only affect performance, not numerical results.

---

### B) Analysis/session caches (LRU, weakref, memoization)
This class of caching lives **outside** operators, and it is useful when:
- The same operator is recreated often (e.g. inside a training loop), so operator-local cache is lost.
- You want to cache results of expensive global analysis like `analyze(op)` / `canonicalize(op)`.

Examples:
- `analyze(op) -> OperatorIR`
- `canonicalize(op) -> op'`
- repeated `is_symmetric(op)` probes
- repeated `topk_eigh_info(op)` factor eigenpairs

**Hard constraint:** JAX transformations may rebuild Python objects,
so a naive `id(op)` cache is not reliable across traces.

**Recommended strategy:**
- Provide an explicit `cache_key(op)` used by the analysis cache.
- Default `cache_key` should be stable for pure operators:
  - based on (op type, shapes, dtype, scalars, and children cache keys)
  - for dense arrays: hash of metadata + optionally content hash (expensive) or user-supplied tag.

**Memory safety:**
- Use `weakref.WeakKeyDictionary` when keys are operator objects.
- Use `functools.lru_cache` only with stable, hashable keys (strings/tuples).

**API proposal:**
- `linox.cache.get_analysis_cache()` returns a cache object
- `linox.cache.with_cache(cache): ...` context manager
- `linox.cache.clear()`

---

## 2. What should be cached (priorities)

### High priority (big wins)
1. **Decompositions**
   - eigendecomposition: `leigh`, partial eigenpairs (`eigh(k=...)`)
   - Cholesky-like factors for structured operators (`lsqrt_exact` for PSD, etc.)
2. **Structure analysis**
   - `extract_kronecker_factors`, `extract_add_terms`, `pull_out_scalars`
   - canonicalization results
3. **Expensive stochastic probes**
   - Hutchinson trace probes (optionally cache probe vectors for determinism)
   - SLQ (stochastic Lanczos quadrature) runs (cache tridiagonal outputs)

### Medium priority
- Toeplitz FFT spectra
- Block structures (index maps)
- Permutation index arrays

### Low priority / avoid by default
- Dense materializations (`todense`) unless user explicitly requests caching.

---

## 3. Cache invalidation & immutability

### Preferred: immutability by design
linox operators should be treated as immutable.
That makes caching simple: caches never become stale.

If mutation exists (should be avoided):
- Provide `_invalidate_cache()` and call it on any mutation.
- Add a version counter `_version` and include it in `cache_key`.

---

## 4. Interaction with JIT and PyTrees

### Principle: "Numerics in JAX, caching in Python"
- Numerical operations should remain JAX-traceable.
- Caches and tracing are Python-side instrumentation.

### What breaks JIT
- Reading/writing Python dict caches *inside* a jitted function will not behave as expected.
- Creating new Python objects in a jitted path may cause recompiles / tracing overhead.

### Practical guideline
- Build LinearOperators and caches **outside** jitted steps when possible.
- If you need structure-aware behavior inside JIT, the operator must carry enough static structure
  and you must avoid Python-side branching on traced values.

**Good pattern:**
- `ir = analyze(op)` outside JIT
- JIT uses `ir`’s static decisions (method selection, factor lists, etc.)

---

## 5. Tracing: what we want to observe

### Trace events (core)
- `todense` called (who called it, size, reason)
- algorithm selection: exact vs approximate, which method chosen
- cache hit/miss for:
  - operator-local caches (e.g. `IsotropicAdditiveLinearOperator._ensure_eigh()`)
  - analysis cache (e.g. `extract_kronecker_factors`)
- linear solves: which solver, iterations (if iterative), convergence metrics
- stochastic estimators: number of probes, RNG seeds (optional)

### Trace levels
- `WARN`: densification warnings, fallbacks to dense, potential O(n^3)
- `INFO`: algorithm selection, summary stats
- `DEBUG`: cache hits/misses, call stacks

### Storage & output
- Provide `debug.inspect_run(fn, ...) -> TraceReport` (you already have a version)
- Optionally add:
  - `TraceReport.to_markdown()`
  - `TraceReport.to_json()`
  - pretty printer for terminal

---

## 6. Proposed "trace context" design

### A) Context manager
```python
with linox.debug.trace() as tr:
    y = linox.solve(A, b)
print(tr.summary())
```

### B) Event hooks
- central `emit(event_name, **payload)` used across modules
- events contain:
  - operator types / shapes
  - chosen method
  - timing (optional, host side)
  - cache status

### C) Minimal overhead when disabled
- tracing checks must be cheap (`if not config.emit: return`)
- no string formatting unless tracing enabled

---

## 7. Concrete recommendations for your top-k Kronecker eigensolver

### Cacheable pieces
- For each factor `A_i`:
  - `leigh(A_i)` results `(w_i, Q_i)`
  - sorting indices used by `largest/which`

### Where to cache
- Operator-local: in `ArrayKernel` / factor operator if stable
- Analysis cache: keyed by `cache_key(A_i)` for reuse across training steps

### Determinism vs speed
- If you use approximate `leigh` (Lanczos/Lobpcg), cache must include:
  - method name
  - RNG seed/key (if randomized)
  - tolerance, max iters
Otherwise you’ll cache the wrong thing.

---

## 8. Suggested "public" debug docs for users

You likely want a lightweight user-facing explanation in README:

- “linox avoids densification by default”
- “enable debug mode: `LINOX_DEBUG=1` or `lo.config.set_debug(True)`”
- “inspect densifications: `debug.inspect_run(...)`”

---

## 9. Checklist for v0.0.3

- [ ] Add `clear_cache()` to operators that cache decompositions / structure
- [ ] Add `linox.structure.analyze(op) -> OperatorIR` (cacheable)
- [ ] Add `linox.canonicalize(op)` pass and cache it
- [ ] Add a session cache (weakref/LRU)
- [ ] Extend tracing events: densify, solve, eig, sqrt, slogdet, trace
- [ ] Document how caches interact with JIT
