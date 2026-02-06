# ROADMAP v0.0.3 (directory-by-directory)

Goal: refactor structure + unify API + integrate approximation backends + exploit special cases.

---

## 1. Target directory layout

```
linox/
  __init__.py
  api.py                 # stable public re-exports
  config.py
  typing.py
  utils/
    __init__.py
    array.py             # as_linop, todense, as_scalar, as_shape, allclose, eye
    debug.py             # inspect_run, densify warnings
    tags.py              # optional: operator tags helpers
  operators/
    __init__.py
    base.py              # LinearOperator
    dense.py             # Matrix
    special.py           # Identity, Zero, Scalar, Ones
    diagonal.py          # Diagonal
    arithmetic.py        # Add/Scaled/Product/Transpose/Inverse/PseudoInverse/Congruence
    block.py             # BlockMatrix, BlockDiagonal, 2x2
    kron.py              # Kronecker (+ selected eigenvectors)
    kernel.py            # Kernel operators
    toeplitz.py          # Toeplitz operator (and structure helpers)
    eigen.py             # EigenD
    isotropic.py         # IsotropicAdditiveLinearOperator
    lowrank.py           # LowRank family
    factor.py            # Triangular, CholeskyFactor, PSDFromFactor (new)
  linalg/
    __init__.py
    solve.py             # solve/psolve + iterative dispatch
    spectral.py          # eigh/topk/lanczos hooks
    functions.py         # sqrt/log/exp/pow + apply-mode operators
    determinants.py      # det/slogdet/logdet + SLQ hooks
    trace.py             # trace + Hutchinson
    woodbury.py          # Woodbury + determinant lemma helpers
    approx/
      __init__.py
      lanczos.py
      arnoldi.py
      hutchinson.py
      slq.py
      lsmr.py
```

---

## 2. Where your current modules map to

From current codebase / branch uploads:
- `_linear_operator.py` → `operators/base.py`
- `_arithmetic.py` → `operators/arithmetic.py` (+ deprecation wrappers in `api.py`)
- `_block.py` → `operators/block.py`
- `_kronecker.py` → `operators/kron.py`
- `_toeplitz.py` → `operators/toeplitz.py`
- `_eigen.py` → `operators/eigen.py`
- `_isotropicadd.py` → `operators/isotropic.py`
- `_trace.py` → `linalg/trace.py` (and/or `linalg/approx/hutchinson.py`)
- `_matrix_functions.py` → `linalg/functions.py` + `linalg/approx/lanczos.py` / `arnoldi.py`
- `_lanczos_arnoldi.py` + `_lanzcos.py` → `linalg/approx/lanczos.py` & `linalg/approx/arnoldi.py`
- `_lsmr.py` → `linalg/approx/lsmr.py`
- `_svd.py` → `linalg/spectral.py` (or `linalg/svd.py`)

Low-rank module (your branch `_low_rank.py`) → `operators/lowrank.py` + `linalg/woodbury.py`

---

## 3. Milestones

### Milestone A — Refactor without behavior change
- [ ] Move files into `operators/`, `linalg/`, `utils/`
- [ ] Add `api.py` that re-exports canonical functions + operator classes
- [ ] Keep old names (`lsqrt`, `leigh`, …) as deprecating wrappers

### Milestone B — Canonical functions + method wrappers
- [ ] Implement `sqrt/eigh/solve/inverse/slogdet/trace` wrappers
- [ ] Introduce internal multimethods `_xxx_exact` and `_xxx_approx`
- [ ] Add config knobs: `max_dense_n`, per-op default methods, densify warnings

### Milestone C — Special cases / rewrites
- [ ] Add `eye(n)` helper and isotropic-add rewrite in `add`
- [ ] Prefer `IsotropicScalingPlusSymmetricLowRank` when base is symmetric low rank
- [ ] Wire Woodbury + determinant lemma for `PositiveDiagonalPlusSymmetricLowRank`
- [ ] Add PSD-from-factor operators (`CholeskyFactor`, `PSDFromFactor`)

### Milestone D — Approx backend integration
- [ ] `trace(method="hutchinson")`
- [ ] `slogdet(method="slq")`
- [ ] `sqrt(method="lanczos")` apply-mode operator
- [ ] `eigh(k=..., method="lanczos")`
- [ ] `solve(method="lsmr")` for least squares

### Milestone E — Tests + docs
- [ ] Unit tests per operator (matmul vs dense)
- [ ] Algorithm tests: compare approx vs exact for small n
- [ ] JIT tests for key paths
- [ ] Docs in `docs/` (this set)

---

## 4. Design decisions to lock in

1) `eigh` returns eigenvalues as arrays; eigenvectors as array or operator.
2) `sqrt/cholesky` may return apply-mode operators for large problems.
3) `method="auto"` prefers structure, then approximate, then dense (configurable).


# Other design stuff:
  * operator + 0/1 should use matrix Zeros or matrix Ones and add them
  * linox.eye(dim) makes universal identiy
  * left/right add multiplication like op * 0.1 and 0.1 * op should both work
  * op + 0.1 and 0.1 + op should either trigger a warning or be allowed and then add element wise the scalar
  * operators should have a function that gets either the row/column or element of course lazily.