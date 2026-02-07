# linox v0.0.3 — Unified Plan

> **One document to rule them all.**
> This consolidates all ADRs, design notes, API specs, and roadmap items into a single actionable plan for the v0.0.3 release.

---

## 0. What v0.0.3 Is

linox is a JAX-native LinearOperator library for structured linear algebra in iterative ML workflows (GPs, operator learning, kernel methods). v0.0.2 works but has a flat module structure, inconsistent API naming (`l*` prefixes), no unified `method=` dispatch, and no approximation backend integration.

**v0.0.3 delivers:**
1. Clean package structure (`operators/`, `linalg/`, `utils/`)
2. Unified public API with `method="auto"|"exact"|"approx"` on every heavy function
3. Approximation backends wired in (Lanczos, Hutchinson, SLQ, LSMR)
4. Operator introspection, canonicalization, and fingerprint-based caching
5. Tracing and performance lint
6. PSD/symmetry wrappers and new factored operators

**Non-goals for this release:** full SciPy parity, perfect `method="auto"` heuristics, GPU-tuned approximation defaults from day one.

---

## 1. Architecture Decisions (locked in)

These are final. Each maps to an ADR document; the key commitments are summarized here.

### A1. Operators are PyTrees (ADR-0001)
- Every operator defines `tree_flatten` / `tree_unflatten`.
- Children = arrays + child operators. Aux = static metadata (shape, dtype, flags).
- Caching and tracing must not rely on Python object identity (JAX may rebuild objects across traces).

### A2. PSD and Symmetry are explicit wrappers (ADR-0001, ADR-0003)
- `Sym(op)` — self-adjoint promise.
- `PSD(op)` — positive semidefinite promise.
- Optional `SPD(op)` — positive definite.
- These are semantic promises, unchecked by default. Debug mode validates via random probes (`x^T(Ay) ≈ y^T(Ax)`, `x^TAx ≥ -tol`).
- Wrappers propagate under arithmetic (e.g., Kronecker of PSD factors is PSD).

### A3. Two-tier execution: lazy operators + staged compiled kernels (ADR-0001, ADR-0002)
- **Always:** lazy `LinearOperator` evaluation via `matvec`/`matmat`.
- **Optional:** staged compilation of kernels (`compile_matvec`, `compile_solve`, `compile_slogdet`, `compile_slq`) into pure-jittable callables.
- "Lowering to JAX" means lowering the *operator graph + parameters* as a PyTree, **not** densifying.

### A4. matvec/matmat must be jittable for all operators (ADR-0002)
- Operator data stored as arrays / PyTree children — no Python object manipulation inside matvec paths.
- `__matmul__` routes to matvec/matmat and remains jittable.
- `jax.jit(lambda x: op @ x)` must work. So must `vmap`/`pmap`/`shard_map`.

### A5. Staging APIs instead of `@linox.jit` (ADR-0002)
- No magic decorator. Instead, explicit compilation helpers:
  - `compile_matvec(op) -> (x -> op@x)`
  - `compile_solve(op, *, method=...) -> (b -> solve(op, b))`
  - `compile_slogdet(op, *, method=...) -> (() -> slogdet(op))`
  - `compile_slq(op, *, probes, lanczos_k, block_size, ...) -> callable`
- Caches are Python-side and keyed by fingerprints. Compiled kernels consume cached artifacts as closed-over constants.

### A6. Cache is fingerprint-based and outside JIT (ADR-0001, ADR-0004)
- Fingerprint = `(op_type, shape, dtype, scalars, children_fingerprints)`.
- For leaf arrays: content hash for small arrays, metadata-only for large arrays (configurable).
- Session cache (LRU) is the primary cache. Bounded by max entries / max bytes.
- Approximate methods: no keyed caching on `(rank, maxiter)` in v0.0.3 unless trivially safe.
- Operators are treated as **immutable** — caches never become stale.

### A7. Tracing includes performance lint (ADR-0001, ADR-0005)
- Opt-in. Near-zero overhead when disabled.
- Records: dispatch decisions, cache hits/misses, densification events, solver iterations, algorithm choices.
- Lint detects: sequential probe loops, repeated single-RHS solves, accidental densification, missed structure (nested kron not flattened).
- Levels: `WARN` (densify, fallbacks), `INFO` (algorithm selection), `DEBUG` (cache details, call stacks).

### A8. Validation: cheap by default, expensive in debug (ADR-0006)
- Every constructor validates structural invariants: O(1), JIT-safe, deterministic.
  - Shape/rank checks, dtype normalization, finite scalars, structural wrapper contracts.
- Numerical/semantic checks (symmetry probes, PSD probes) only in `LINOX_DEBUG=1` mode.
- `lo.validate(op)` for cheap recursive check; `lo.validate(op, mode="debug")` for expensive probes.

### A9. Prefer batched execution by default (ADR-0007)
- Hutchinson/SLQ: generate probe matrix `Z (n, p)`, compute `AZ` once (batched MVP).
- Block Lanczos with configurable `block_size` (default ≈ 8 for GPU).
- Multi-device parallelism is optional, utility-driven (later).
- `matmat` (`A @ B`) is the primary path; `matvec` is a special case.

---

## 2. Package Structure

```
linox/
  __init__.py
  api.py                 # stable public re-exports (the "front door")
  config.py              # max_dense_n, default_methods, warn_on_densify, random_seed, parallel settings
  typing.py
  utils/
    __init__.py
    array.py             # as_linop, todense, as_scalar, as_shape, allclose, eye
    debug.py             # inspect_run, densify warnings, trace context
    tags.py              # operator tag helpers
  operators/
    __init__.py          # centralized PyTree registration
    base.py              # LinearOperator (abstract base)
    dense.py             # Matrix
    special.py           # Identity, Zero, Scalar, Ones
    diagonal.py          # Diagonal
    arithmetic.py        # ScaledLinearOperator, AddLinearOperator, ProductLinearOperator,
                         #   TransposedLinearOperator, InverseLinearOperator,
                         #   PseudoInverseLinearOperator, CongruenceTransform
    block.py             # BlockMatrix, BlockDiagonal, BlockMatrix2x2
    kron.py              # Kronecker, KroneckerSelectedEigenvectors
    kernel.py            # KernelOperator, ArrayKernel, ToeplitzKernel
    toeplitz.py          # Toeplitz
    eigen.py             # EigenD
    isotropic.py         # IsotropicAdditiveLinearOperator
    lowrank.py           # LowRank, SymmetricLowRank,
                         #   IsotropicScalingPlusSymmetricLowRank,
                         #   PositiveDiagonalPlusSymmetricLowRank
    factor.py            # Triangular, CholeskyFactor, PSDFromFactor (NEW)
    permutation.py       # Permutation
  linalg/
    __init__.py
    solve.py             # solve, psolve + iterative dispatch
    spectral.py          # eigh, svd, topk/partial methods
    functions.py         # sqrt, exp, log, pow, cholesky
    determinants.py      # det, slogdet, logdet
    trace.py             # trace (exact + Hutchinson)
    woodbury.py          # Woodbury + determinant lemma helpers
    approx/
      __init__.py
      lanczos.py         # lanczos_tridiag, lanczos_eigh, lanczos_matrix_function, SLQ
      arnoldi.py         # arnoldi_iteration, arnoldi_matrix_function
      hutchinson.py      # hutchinson_trace, hutchinson_diagonal, joint
      slq.py             # stochastic Lanczos quadrature (trace(f(A)))
      lsmr.py            # lsmr_solve (+ damping/regularization)
  structure/
    __init__.py
    analyze.py           # analyze(op) -> OperatorIR
    canonicalize.py      # canonicalize(op) -> op'
    ir.py                # KroneckerIR, IsotropicShiftIR, DiagPlusLowRankIR, ...
    fingerprint.py       # cache_key(op) -> hashable
  cache/
    __init__.py          # get_analysis_cache(), with_cache(), clear()
    session.py           # LRU / weakref session cache
  graph.py               # linop_graph visualization (raw, canonical, tags, cache)
```

### Module mapping from v0.0.2

| v0.0.2 file | v0.0.3 location |
|---|---|
| `_linear_operator.py` | `operators/base.py` |
| `_arithmetic.py` | `operators/arithmetic.py` |
| `_block.py` | `operators/block.py` |
| `_kronecker.py` | `operators/kron.py` |
| `_toeplitz.py` | `operators/toeplitz.py` |
| `_eigen.py` | `operators/eigen.py` |
| `_isotropicadd.py` | `operators/isotropic.py` |
| `_low_rank.py` | `operators/lowrank.py` |
| `_trace.py` | `linalg/trace.py` + `linalg/approx/hutchinson.py` |
| `_matrix_functions.py` | `linalg/functions.py` + `linalg/approx/lanczos.py` |
| `_lanczos_arnoldi.py` | `linalg/approx/lanczos.py` + `linalg/approx/arnoldi.py` |
| `_lsmr.py` | `linalg/approx/lsmr.py` |
| `_svd.py` | `linalg/spectral.py` |

---

## 3. Operator Taxonomy

### Core
`LinearOperator` (base), `Matrix` (dense), `Identity`, `Zero`, `Scalar`, `Ones`, `Diagonal`

### Arithmetic / composition
`ScaledLinearOperator`, `AddLinearOperator`, `ProductLinearOperator`, `TransposedLinearOperator`, `InverseLinearOperator`, `PseudoInverseLinearOperator`, `CongruenceTransform`

### Structured
`BlockMatrix`, `BlockDiagonal`, `BlockMatrix2x2`, `Toeplitz`, `Kronecker`, `KroneckerSelectedEigenvectors`, `EigenD`, `Permutation`

### Low rank
`LowRank` (`U diag(S) V^T`), `SymmetricLowRank` (`U diag(S) U^T`), `IsotropicScalingPlusSymmetricLowRank` (`σI + U diag(S) U^T`), `PositiveDiagonalPlusSymmetricLowRank` (`D + α U diag(S) U^T`)

### Isotropic shifts
`IsotropicAdditiveLinearOperator` (`sI + A` for symmetric `A`)

### PSD / factored (NEW in v0.0.3)
- `Triangular(matrix, lower=True)` — fast triangular solve
- `CholeskyFactor(L, lower=True)` — triangular + logdet
- `PSDFromFactor(L, lower=True)` — represents `A = L L^T`

### Tags (instead of subclass explosion)
Every operator may carry: `symmetric: bool`, `psd: bool`, `unitary: bool`, `triangular: "lower"|"upper"|None`. Convenience wrappers: `assume_symmetric(A)`, `assume_psd(A)`, `assume_unitary(A)`, `triangular(A, lower=True)`.

### Auto-rewrite rules
- `Scalar(s) + A` or `ScaledLinearOperator(Identity(n), s) + A` where A is symmetric → `IsotropicAdditiveLinearOperator(s, A)`
- If A is `SymmetricLowRank` → `IsotropicScalingPlusSymmetricLowRank(s, U, S)`
- `op + 0` / `op * 1` / `A @ Identity` simplify
- `Scaled(Scaled(A, a), b)` → `Scaled(A, a*b)`

---

## 4. Public API

**Core principle:** functions are the canonical entry points. Operator overloading is syntactic sugar.

All functions accept `jax.Array | LinearOperator`. Internally normalize via `utils.as_linop(x)`.

### 4.1 Construction & utilities

| Function | Signature | Returns |
|---|---|---|
| `as_linop(A)` | `A -> LinearOperator` | Wraps arrays as `Matrix` |
| `todense(A)` | `A -> jax.Array` | Materializes (warns in debug) |
| `allclose(A, B, **kw)` | `-> bool` | Approximate equality |
| `diagonal(A)` | `-> jax.Array` | Diagonal extraction |
| `transpose(A)` | `-> LinearOperator` | Lazy transpose |
| `symmetrize(A)` | `-> LinearOperator` | `0.5(A + A^T)` |
| `kron(A, B)` | `-> LinearOperator` | Kronecker product |
| `eye(n, dtype=None)` | `-> Identity` | Structure-preserving identity |

### 4.2 Solvers

```python
solve(A, b, *, method="auto", **kw) -> jax.Array
psolve(A, b, *, method="auto", **kw) -> jax.Array
```

`method` values: `"auto"`, `"exact"`, `"cg"`, `"gmres"`, `"lsmr"`

### 4.3 Operator transforms

```python
inverse(A, *, method="auto", **kw)  -> LinearOperator
pinverse(A, *, method="auto", **kw) -> LinearOperator
sqrt(A, *, method="auto", **kw)     -> LinearOperator
cholesky(A, *, method="auto", **kw) -> LinearOperator
exp(A, *, method="auto", **kw)      -> LinearOperator   # optional v0.0.3
log(A, *, method="auto", **kw)      -> LinearOperator   # optional v0.0.3
```

Note: `sqrt`/`cholesky`/`log`/`exp` may return **function-backed operators** (apply-mode via Krylov) for large problems.

### 4.4 Spectral decompositions

```python
eigh(A, k=None, *, which="LM", method="auto", **kw)
  -> (evals: jax.Array, evecs: jax.Array | LinearOperator)
```

- `k=None`: full decomposition (structured exact → dense fallback)
- `k=int`: partial decomposition (Kronecker top-k → Lanczos/LOBPCG → dense)
- `which`: `"LM"` (largest magnitude), `"SM"`, `"LA"` (largest algebraic), `"SA"`
- `method`: `"auto"`, `"exact"`, `"lanczos"`, `"lobpcg"`, `"kron"`
- Previous `topk_eigh(A, k)` becomes an internal backend for `eigh(A, k=k, method="kron")`

```python
svd(A, k=None, *, method="auto", **kw) -> (U, s, Vt)   # optional v0.0.3
```

### 4.5 Scalars / traces / determinants

```python
trace(A, *, method="auto", **kw)   -> jax.Array
det(A, *, method="auto", **kw)     -> jax.Array
slogdet(A, *, method="auto", **kw) -> (sign, logabsdet)
```

`method` values: `"exact"`, `"hutchinson"` (trace), `"slq"` / `"stochastic"` (logdet)

### 4.6 Operator overloading

`scalar * A`, `A * scalar`, `A + B`, `A - B`, `A @ B`, `A @ v`, `A / B` (diagonal-like), `A.T`, `-A`

### 4.7 Deprecations

All `l*` names (`lsqrt`, `leigh`, `linverse`, `lsolve`, ...) kept as wrappers emitting `DeprecationWarning`, forwarding to canonical functions. Removal planned for v0.0.4.

---

## 5. Method Dispatch

### 5.1 Pattern

Each public function is a thin wrapper around internal exact/approx multimethods:

```python
def sqrt(A, *, method="auto", **kw):
    A = utils.as_linop(A)
    m = config.resolve_method("sqrt", A, method)
    if m in ("auto", "exact"):
        try:
            return _sqrt_exact(A, **kw)
        except NotImplementedError:
            if m == "exact":
                raise
    return _sqrt_approx(A, **kw)
```

Internal dispatch uses `plum.dispatch` for operator-specific overloads (already in v0.0.2).

### 5.2 Priority rules for `method="auto"`

#### `solve(A, b)`
1. Diagonal / Identity / Scalar / Zero
2. Triangular / CholeskyFactor / PSDFromFactor
3. Toeplitz (if JAX-native solver available)
4. Kronecker
5. Isotropic-add (`sI + Symmetric`)
6. Diag + LowRank (Woodbury)
7. Dense exact (if `n ≤ config.max_dense_n`)
8. Iterative (`cg`/`gmres`/`lsmr`) — default for large

#### `inverse(A)`
1. Anything with specialized `solve` → lazy inverse backed by `solve`
2. Structural inverses (Diagonal, Scalar, Permutation, EigenD, Kronecker, Isotropic-add, Diag+LowRank)
3. Dense exact (threshold)
4. Approx inverse operator (iterative apply)

#### `sqrt(A)`
1. Diagonal / Scalar / Identity
2. PSDFromFactor → return factor (best exact sqrt)
3. EigenD / Isotropic-add / Kronecker spectral
4. Diag+LowRank (whitening + low-rank sqrt)
5. Dense exact (threshold)
6. Approx: Lanczos `f(A)v` wrapped as operator

#### `eigh(A, k=None)`
- `k=None`: structured exact → dense exact (threshold) → approx fallback
- `k=int`: Kronecker top-k → LowRank exact modes → Lanczos/LOBPCG

#### `trace(A)`
1. Exact: diagonal-summable (Diagonal, Scalar*I, Identity, Zero, PSDFromFactor)
2. Exact from decomposition (EigenD)
3. Hutchinson (default large)

#### `slogdet(A)`
1. Exact: diagonal / triangular / PSDFromFactor
2. Exact from decomposition (EigenD, Kronecker, Isotropic-add)
3. Matrix determinant lemma (diag+lowrank)
4. Dense exact (threshold)
5. SLQ: `trace(log(A))`

### 5.3 Config knobs

```python
config.max_dense_n = 2000          # densification threshold
config.default_methods = {}        # e.g. {"slogdet": "slq"} for large default
config.warn_on_densify = True      # emit warnings in debug mode
config.parallel.probes_batch = True
config.parallel.block_size = 8     # block Lanczos default
```

---

## 6. Approximation Backends

All live in `linalg/approx/`, surfaced via `method=...` on public functions.

### Lanczos (symmetric)
- `lanczos_tridiag`: Krylov basis + tridiagonal reduction
- `lanczos_eigh`: top-k eigenpairs (`k << n`)
- `lanczos_matrix_function`: `f(A)v` for symmetric A
- SLQ (stochastic Lanczos quadrature): `trace(f(A))`

### Arnoldi (general)
- `arnoldi_iteration`: Hessenberg reduction
- `arnoldi_matrix_function`: `f(A)v` for non-symmetric A

### Hutchinson
- `hutchinson_trace`: unbiased trace estimator (Rademacher vectors preferred)
- `hutchinson_diagonal`: diagonal estimator
- `hutchinson_trace_and_diagonal`: joint estimation

### LSMR
- `lsmr_solve`: least squares solver (+ damping/regularization)

### Numerical notes
- Reorthogonalization in Lanczos: expose `reortho=True|False`.
- SLQ requires PSD/symmetric assumptions; enforce via tags or `assume_psd`.
- `which` in Krylov: support `"LA"` and `"SA"` robustly; treat `"LM"`/`"SM"` as aliases when PSD.
- Batched probes are default (probe matrix `Z (n, p)`, compute `AZ` once).

---

## 7. Operator Introspection & Canonicalization

### 7.1 Introspection API (required on every operator)
- `children() -> tuple[LinearOperator, ...]`
- `op_type: str` (or `__class__.__name__`)
- `tags: set[str]` (lightweight properties)
- Recommended properties: `is_square`, `is_symmetric`, `is_psd`, `is_diagonal`, `is_lowrank`, `rank`, `supports_exact_eigh`, `supports_exact_slogdet`, ...

### 7.2 Operator IR (intermediate representation)

Small dataclasses used by algorithm selection:

```python
@dataclass
class KroneckerIR:
    scalar: float | None
    factors: list[LinearOperator]
    tags: set[str]  # e.g. {"symmetric", "psd"}

@dataclass
class IsotropicShiftIR:
    shift: float
    base: LinearOperator
    tags: set[str]

@dataclass
class DiagPlusLowRankIR:
    diag: Diagonal
    U: jax.Array
    S: jax.Array
    scale: float
```

### 7.3 Canonicalization pass

`canonicalize(op)` rewrites operator trees to standard form:

- **Flatten associative ops:** `Kronecker(Kronecker(A,B),C)` → `Kronecker(A,B,C)` (factor list). Same for Add/Product.
- **Pull out scalars:** `Scaled(Kronecker(...), s)` → `Kronecker(...), scalar=s`. `Scaled(Scaled(A,a),b)` → `Scaled(A, a*b)`.
- **Rewrite special cases:** `Scaled(Identity(n), s) + A` → `IsotropicAdditiveLinearOperator(s, A)` for symmetric A.
- **Simplify:** `A + Zero → A`, `A @ Identity → A`.
- Optionally normalize ordering (sort factors by size for caching).

### 7.4 Algorithm selection flow

```
1. op = canonicalize(op)      # cheap if cached
2. ir = analyze(op)           # returns OperatorIR
3. dispatch based on IR       # exact / structured / approx
```

Example: `eigh(Kronecker(Kronecker(A,B),C), k=50)` → canonicalize flattens to `[A,B,C]` → `KroneckerIR` → heap search on product eigenvalue grid.

---

## 8. Caching

### 8.1 Two kinds

**A) Operator-local caches** (lazy, inside operator instances):
- `IsotropicAdditiveLinearOperator`: cached eigendecomposition `(Q, S)`
- `Kronecker`: cached extracted factor list + scalar
- `BlockDiagonal`: cached block offsets / index maps
- Use `functools.cached_property` for small items; explicit `_cache: dict` for large items with `clear_cache()`.
- Python-side only — not part of PyTree.

**B) Session cache** (LRU, outside operators):
- Keys: operator fingerprints (hierarchical, content-addressable).
- Stores: exact decompositions, canonical forms, IR summaries, factor extractions.
- Memory safety: `weakref.WeakKeyDictionary` for operator keys, `functools.lru_cache` for hashable keys.
- API: `linox.cache.get_analysis_cache()`, `linox.cache.with_cache(cache)`, `linox.cache.clear()`.

### 8.2 What to cache (priority)

| Priority | Item |
|---|---|
| **High** | Eigendecompositions, Cholesky factors, structure analysis (`extract_kronecker_factors`, canonicalization) |
| **Medium** | Toeplitz FFT spectra, block index maps, permutation indices |
| **Low / avoid** | Dense materializations (`todense`) — only if explicitly requested |

### 8.3 Interaction with JIT
- "Numerics in JAX, caching in Python."
- Build operators and caches **outside** jitted steps.
- `ir = analyze(op)` outside JIT; JIT uses IR's static decisions.
- Compiled kernels can consume cached artifacts as closed-over constants.

---

## 9. Tracing & Debug

### 9.1 Trace events
- `todense` called (who, size, reason)
- Algorithm selection (exact vs approx, which method)
- Cache hit/miss (operator-local and session)
- Linear solves (solver, iterations, convergence)
- Stochastic estimators (probes, seeds)

### 9.2 Usage

```python
# Context manager
with linox.debug.trace() as tr:
    y = linox.solve(A, b)
print(tr.summary())

# Environment variable
LINOX_DEBUG=1 python script.py

# Programmatic
linox.config.set_debug(True)
```

### 9.3 Lint findings (warnings, not errors)
- Sequential probe loops instead of batched
- Repeated single-RHS solves where multi-RHS possible
- Accidental densification in hot paths
- Missed structure (nested kron not flattened)

### 9.4 Output formats
- `TraceReport.to_markdown()`
- `TraceReport.to_json()`
- Pretty printer for terminal
- `linop_graph(op, canonical=True, show_tags=True, show_cache=True)` for debug visualization

---

## 10. Concrete Code-Level Fixes

These are specific items from the v0.0.2 code review that should be addressed:

1. **`LinearOperator` base class:**
   - Remove `from this import d` (unused).
   - Ensure `tree_flatten` is an instance method (not `@classmethod`).
   - Add `__array_priority__` (high value) so operator ops win against ndarray.
   - Do **not** add `__jax_array__` (prevents accidental densification).

2. **Deprecation consolidation:**
   - Non-`l*` names are canonical exports in `api.py`.
   - `l*` names are wrappers in `api.py` that warn + forward.

3. **Kronecker `topk_eigh` → `eigh(k=...)`:**
   - Move logic to `linalg/spectral.py` as implementation for `eigh(A, k=..., method="kron")`.
   - Internally detect `isinstance(A, Kronecker)` and route to heap merge.

4. **Isotropic additive operators:**
   - Ensure `sqrt`, `inverse`, `pinverse`, `cholesky` dispatch through unified entry points.

5. **Toeplitz solver decision (OPEN):**
   - Option A: SciPy-based exact (fast, not purely JAX).
   - Option B: JAX-native FFT-based matvec + iterative solve (pure, scalable).
   - **Recommendation:** default to JAX-native; allow SciPy as opt-in for small problems.

6. **Operator arithmetic edge cases:**
   - `op + 0` / `op - 0` → return op (use `Zero` internally).
   - `0.1 * op` and `op * 0.1` both work.
   - `op + scalar` → either warn or add element-wise (decision: warn by default, allow via explicit `op + scalar * Ones(n)`).
   - `linox.eye(n)` creates universal Identity.
   - Operators should support lazy row/column/element access.

---

## 11. Open Design Decisions ✅ RESOLVED

All design decisions have been locked in for v0.0.3:

| # | Question | Options | **Decision** |
|---|---|---|---|
| 1 | Toeplitz solver default | (A) SciPy exact (B) JAX-native iterative | **A** — SciPy hybrid with pure-JAX backward pass (see ADR-0008) |
| 2 | `eigh` evecs return type | (A) always `jax.Array` (B) `LinearOperator` when structured | **A** — Always return `jax.Array` for simplicity |
| 3 | Approx `sqrt` semantics | (A) operator approximating `A^{1/2}` (B) factor `L` s.t. `LL^T ≈ A` | **A** — Apply-mode operator (Lanczos); factor returned via `cholesky()` |
| 4 | `op + scalar` behavior | (A) error (B) warn + broadcast (C) silent broadcast | **A** — Raises error; use `op + scalar * lo.eye(n)` explicitly |

---

## 12. Milestones & Checklist

### Milestone A — Structure refactor (no behavior changes) ✅ COMPLETE
- [x] Create `operators/`, `linalg/`, `utils/`, `structure/`, `cache/` directories
- [x] Move modules to new locations per mapping table (§2)
- [x] Add `api.py` re-export layer (keep old imports working via shims)
- [x] Centralize PyTree registration in `operators/__init__.py`
- [x] Add `Triangular`, `CholeskyFactor`, `PSDFromFactor` to `operators/factor.py`
- [x] All existing tests pass

### Milestone B — Unified entry points + method dispatch ✅ COMPLETE
- [x] Implement `sqrt(A, method=...)` wrapper + internal `_sqrt_exact` / `_sqrt_approx`
- [x] Implement `eigh(A, k=None, method=...)`, integrating Kronecker top-k as `method="kron"`
- [x] Implement `inverse(A, method=...)` + align with `solve(A, b, method=...)`
- [x] Move determinant/trace logic into `linalg/determinants.py` + `linalg/trace.py`
- [x] Implement `config.resolve_method()` with priority rules from §5.2
- [x] Add `config.max_dense_n`, `config.default_methods`, `config.warn_on_densify`

### Milestone C — Auto-rewrites + special cases ✅ COMPLETE
- [x] `eye(n)` helper
- [x] Isotropic-add rewrite in `AddLinearOperator.__init__` (detect `sI + Sym`)
- [x] Prefer `IsotropicScalingPlusSymmetricLowRank` when base is `SymmetricLowRank`
- [x] Wire Woodbury + determinant lemma for `PositiveDiagonalPlusSymmetricLowRank`
- [x] PSD-from-factor dispatch: `sqrt(PSDFromFactor) → CholeskyFactor`, `slogdet → 2*sum(log(abs(diag(L))))`
- [x] Simplification rules: `A + Zero → A`, `A @ Identity → A`, `Scaled(Scaled(A,a),b) → Scaled(A,a*b)`

### Milestone D — Introspection + caching ✅ COMPLETE
- [x] Define `OperatorIR` dataclasses in `structure/ir.py`
- [x] Implement `analyze(op) → OperatorIR` in `structure/analyze.py`
- [x] Implement `canonicalize(op)` with core rewrite rules in `structure/canonicalize.py`
- [x] Implement `cache_key(op)` fingerprinting in `structure/fingerprint.py`
- [x] Implement session cache (LRU) in `cache/session.py`
- [x] Add `clear_cache()` to operators that cache decompositions
- [x] Ensure `extract_kronecker_factors` is used by `eigh(k=...)` and `topk_eigh_info`
- [x] Property propagation rules (Kronecker/Add/Product/Scale PSD/Sym inference)

### Milestone E — Approx backend integration ✅ COMPLETE
- [x] `trace(A, method="hutchinson", num_samples=..., key=...)`
- [x] `slogdet(A, method="slq", num_samples=..., num_iters=..., key=...)`
- [x] `sqrt(A, method="lanczos", num_iters=...)` → apply-mode operator
- [x] `eigh(A, k=..., method="lanczos", num_iters=..., which=...)`
- [x] `solve(A, b, method="lsmr")` for least squares
- [x] Batched probes as default in Hutchinson/SLQ
- [x] Block Lanczos with configurable `block_size`

### Milestone F — Tracing + lint ✅ COMPLETE
- [x] Implement trace context manager: `with linox.debug.trace() as tr:`
- [x] Central `emit(event_name, **payload)` used across modules
- [x] Trace events: densify, solve, eig, sqrt, slogdet, trace, cache hit/miss
- [x] Lint rules: unbatched probes, repeated single-RHS, accidental densify, unflattened kron
- [x] `TraceReport` with `.summary()`, `.to_markdown()`, `.to_json()`
- [x] Near-zero overhead when disabled (`if not config.emit: return`)

### Milestone G — Tests + docs ✅ COMPLETE
- [x] Unit tests per operator (matmul vs dense baseline)
- [x] Algorithm tests: approx vs exact for small n
- [x] JIT tests: `jax.jit(lambda v: (A @ v).sum())` for all operator types
- [x] Property tests: `A.T.T == A`, `symmetrize(A)` symmetry, PSD wrapper propagation
- [x] Validation tests: constructor rejects bad shapes/dtypes
- [x] `README.md` quickstart update
- [x] `CHANGELOG.md` for v0.0.3
- [x] User-facing docs: "linox avoids densification by default", debug mode, inspect_run

---

## 13. Dependency & Compatibility Notes

- **Hard dependencies:** JAX, plum (multiple dispatch). No changes.
- **Optional:** SciPy (for Toeplitz Levinson solver, if kept as opt-in path).
- **Python:** 3.10+ (for `match`/`case` if used in dispatch, otherwise 3.9+).
- **Deprecation policy:** `l*` names warn in v0.0.3, removed in v0.0.4.
- **Public surface:** everything imported from `linox.api` (or top-level `linox`) is stable. Internal paths (`operators/arithmetic.py`, etc.) may move.

---

## 14. Summary of What's New in v0.0.3

| Category | What |
|---|---|
| **Structure** | `operators/`, `linalg/`, `utils/` package layout |
| **API** | Unified `method=` on all heavy functions; deprecate `l*` prefixes |
| **Operators** | `Triangular`, `CholeskyFactor`, `PSDFromFactor`; PSD/Sym wrappers |
| **Dispatch** | `method="auto"` with priority rules; `config.resolve_method()` |
| **Approx** | Hutchinson trace, SLQ logdet, Lanczos sqrt/eigh, LSMR solve — all via `method=` |
| **Introspection** | `OperatorIR`, `analyze()`, `canonicalize()` |
| **Caching** | Fingerprint-based session cache; operator-local lazy caches |
| **Tracing** | Opt-in tracing + performance lint |
| **Validation** | Cheap structural validation by default; debug probes |
| **Parallelism** | Batched probes default; block Lanczos; multi-RHS as first-class |