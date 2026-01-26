# Linox v0.0.3 – Refactor & API Unification Plan

> Goal: **clean public API + clearer package structure + “one entrance function”** per linear-algebra op (exact **or** approximate), while preserving matrix-free behavior and keeping *structured* operators fast.

---

## 0. High-level goals

### What v0.0.3 should feel like
- **One obvious way** to do common ops: `sqrt`, `eigh`, `inverse`, `solve`, `cholesky`, `slogdet`, `det`, `diagonal`, `symmetrize`, `kron`, …
- Same call works for **arrays or operators** (`jax.Array | LinearOperator`).
- Every op has a consistent `method=` story:
  - `method="auto"` (default): pick the best available implementation.
  - `method="exact"`: closed-form / structured / dense fallback (with warnings on densification).
  - `method="approx"` (or specific names like `"lanczos"`, `"hutchinson"`): scalable approximations.

### Non-goals (for this release)
- Full SciPy feature parity.
- Perfect heuristics for `method="auto"` (start simple + configurable).
- GPU-perfect performance of every approximation from day 1.

---

## 1. Proposed package structure

### Target tree
```
linox/
  __init__.py
  api.py                # public re-exports, stable surface
  config.py
  typing.py
  utils/
    __init__.py
    array.py            # as_scalar, as_shape, as_linop, todense, allclose, …
    debug.py            # inspect_run, densify-tracing utilities
  operators/
    __init__.py
    base.py             # LinearOperator base
    dense.py            # Matrix
    special.py          # Identity, Zero, Scalar, Ones
    diagonal.py         # Diagonal (+ diag utilities)
    arithmetic.py       # Scaled/Add/Product/Transpose/Inverse/PseudoInverse/Congruence
    block.py            # BlockMatrix, BlockDiagonal, BlockMatrix2x2
    kron.py             # Kronecker, KroneckerSelectedEigenvectors
    toeplitz.py         # Toeplitz operator(s) + (optional) algorithms
    eigen.py            # EigenD
    kernels.py          # KernelOperator, ArrayKernel, ToeplitzKernel
  linalg/
    __init__.py
    solve.py            # solve, psolve (+ iterative methods, precond hooks)
    spectral.py         # eigh, svd, topk/partial methods
    functions.py        # sqrt, exp, log, pow, cholesky-like factors
    determinants.py     # det, slogdet, logdet
    trace.py            # trace estimators (Hutchinson, etc.)
  approx/               # optional submodule, can also live inside linalg/*
    lanczos.py
    hutchinson.py
    randomized.py
```

### Rationale
- `operators/*` contains **representations** (data + matmul).
- `linalg/*` contains **algorithms** (exact + approximate).
- `api.py` is the **single stable “front door”**.

---

## 2. Public API proposal (v0.0.3)

### Core principle
Keep operator overloading (`A+B`, `A@x`, `A.T`, …) but make **functions** the canonical entry points, with consistent signatures.

### “One entrance function” pattern

#### `sqrt`
```python
sqrt(A, *, method="auto", **kw) -> LinearOperator
```
- `method="exact"`:
  - Use specialized closed-form where available (Diagonal, EigenD, IsotropicAdditive, Kronecker spectral, …)
  - Else dense fallback: `jnp.linalg.cholesky` (PSD) or `eigh` + spectral sqrt (Hermitian).
- `method="lanczos"` / `"krylov"` / `"approx"`:
  - Use Lanczos/Krylov for `A^{1/2} @ v` or a low-rank factor approximation (depending on your existing implementation).
  - Prefer returning an operator that supports `@` efficiently (e.g. low-rank / factor form).

#### `eigh`
```python
eigh(A, k=None, *, which="LM", method="auto", **kw)
  -> (evals, evecs_operator_or_array)
```
- `k=None` ⇒ exact (structured if possible, else dense)
- `k=int` ⇒ partial:
  - `method="kron"` for Kronecker (your existing `topk_eigh`)
  - `method="lanczos"` / `"subspace"` for general symmetric operators
- `which`: `"LM"`, `"SM"` (largest/smallest magnitude) or `"LA"/"SA"` (algebraic), depending on what you support.

#### `inverse`
```python
inverse(A, *, method="auto", **kw) -> LinearOperator
```
- `method="exact"`: return `InverseLinearOperator(A)` (lazy) + prefer specialized `solve` hooks.
- `method="approx"`: return an operator backed by iterative solves / preconditioner (nice for very large operators).

#### `solve`
```python
solve(A, b, *, method="auto", **kw) -> jax.Array
```
- `method="exact"`: structured solver (Diagonal, Triangular, Toeplitz, Kronecker, …) else dense.
- `method="cg" / "gmres"`: iterative, preconditioner hook.

#### Determinants / trace
```python
trace(A, *, method="auto"|"exact"|"hutchinson", **kw) -> jax.Array
slogdet(A, *, method="auto"|"exact"|"stochastic", **kw) -> (sign, logabsdet)
```

---

## 3. Dispatch + method selection strategy

### Keep what you already have
You’re already using `plum.dispatch` heavily for operator-specific overloads. Keep that.

### Add a **thin method-selection wrapper**
Pattern:
- Internal multimethods:
  - `_sqrt_exact(A)`
  - `_sqrt_approx(A, **kw)`
- Public wrapper:
  - `sqrt(A, method="auto", **kw)` chooses between them.

Example skeleton:
```python
def sqrt(A, *, method="auto", **kw):
    A = utils.as_linop(A)
    method = config.resolve_method("sqrt", A, method)
    if method in ("auto", "exact"):
        try:
            return _sqrt_exact(A, **kw)
        except NotImplementedError:
            if method == "exact":
                raise
    return _sqrt_approx(A, **kw)
```

### `method="auto"` heuristics (start simple)
- Prefer structured implementations if:
  - operator type advertises one (e.g. implements `_sqrt` / `_eigh` / `_solve`)
- Else if `config.max_dense_n` threshold allows ⇒ densify
- Else ⇒ approximate (Lanczos / Hutchinson / iterative solve)

Make thresholds explicit:
- `config.set_max_dense_n(2_000)` (example)
- `config.set_default_method("sqrt", "lanczos")` etc.

---

## 4. Implementation notes from current code review

### A. Fix a couple of sharp edges in `LinearOperator`
From `_linear_operator.py`:
- Remove `from this import d` (unused).
- `tree_flatten` should be an **instance method**, not a `@classmethod` (jax expects `obj.tree_flatten()`).
- Consider adding:
  - `__array_priority__` high value (ensure `LinearOperator` ops win against ndarray)
  - `__jax_array__` (optional) if you want implicit conversion; many libraries avoid this to prevent accidental densification.

### B. Consolidate deprecations
In `_arithmetic.py`, you already have `_deprecated_l_prefix`. In v0.0.3:
- Make **non-`l*` names** the canonical exports.
- Keep `l*` as wrappers that warn + forward.

### C. Kronecker `topk_eigh` → `eigh(k=...)`
Your `Kronecker` module already exposes `topk_eigh`.
- Move its logic under `linalg/spectral.py` as an implementation option for `eigh(A, k=...)`.
- Internally detect `isinstance(A, Kronecker)` and route to the closed-form heap merge.

### D. Isotropic additive operators
`IsotropicAdditiveLinearOperator` is a great example of “structured exact”.
- Ensure `sqrt`, `inverse`, `pinverse`, `cholesky` dispatch to it from the unified entry points.

### E. Toeplitz algorithms
You currently have a Levinson solver and a SciPy `solve_toeplitz` path.
- Decide explicitly:
  - If SciPy is acceptable as a dependency and you’re okay with “not fully JIT/pure” paths.
  - Or provide a JAX-native FFT-based Toeplitz matvec + iterative solve as default.
Add to plan as a decision gate.

---

## 5. Compatibility layer & versioning

### v0.0.3 deprecations
- `ladd, lsub, lmul, lmatmul, lsqrt, leigh, linverse, ...`:
  - Keep as wrappers with `DeprecationWarning`.
  - Ensure docs + changelog mention removal in v0.0.4 (or later, your choice).

### Public surface freeze
- Everything imported from `linox.api` is considered stable.
- Internals may move freely (`_arithmetic.py` → `operators/arithmetic.py`, etc.)

---

## 6. Work plan (checklist)

### Milestone 1 — Structure refactor (no behavior changes)
- [ ] Create new folders (`operators/`, `linalg/`, `utils/`) and move modules.
- [ ] Add `api.py` re-export layer (keep old imports working via shims).
- [ ] Centralize PyTree registration in `operators/__init__.py`.

### Milestone 2 — Unified entry points
- [ ] Implement `sqrt(A, method=...)` wrapper + internal exact/approx.
- [ ] Implement `eigh(A, k=None, method=...)`, integrating Kronecker topk.
- [ ] Implement `inverse(A, method=...)` + `solve(A, b, method=...)` alignment.
- [ ] Move determinant/trace logic into `linalg/`.

### Milestone 3 — Approx algorithms integration
- [ ] Add Hutchinson trace API: `trace(A, method="hutchinson", n_samples=...)`.
- [ ] Add stochastic logdet / slogdet option.
- [ ] Add Lanczos/Krylov sqrt option (existing branch).

### Milestone 4 — Tests + docs
- [ ] Unit tests for each operator (shape, dtype, `@` correctness vs dense baseline).
- [ ] Property tests: `A.T.T == A`, `symmetrize` symmetry, etc.
- [ ] JIT sanity checks (`jax.jit(lambda v: (A@v).sum())` works).
- [ ] Add `README.md` quickstart + “avoid densification” guidelines.
- [ ] Add `CHANGELOG.md` for v0.0.3.

---

## 7. Open design decisions (write down now)

1. **Toeplitz solver default**
   - (A) SciPy-based exact (fast, but not purely JAX)  
   - (B) JAX-native iterative solve (pure, scalable)

2. **Return types for spectral ops**
   - `eigh` returns `(evals, evecs)` where `evecs` is:
     - (A) `jax.Array` always
     - (B) `LinearOperator` when structured (recommended)

3. **Approx sqrt semantics**
   - Return:
     - (A) an operator approximating `A^{1/2}`
     - (B) a factor `L` such that `L@L.T ≈ A` (often more useful)

---

## 8. Suggested “planning docs” to add to the repo

- `docs/ROADMAP.md` (this file)
- `docs/API_v0.0.3.md` (the signatures + examples)
- `docs/STRUCTURED_OPERATORS.md` (Kronecker, IsotropicAdd, Toeplitz, LowRank…)
- `docs/APPROX_ALGORITHMS.md` (Hutchinson, Lanczos, stochastic logdet, …)
