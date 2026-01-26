# Method selection & dispatch (v0.0.3)

This document describes how `method="auto"` chooses between **structured exact**, **dense exact**, and **approx** backends.

---

## Core idea

Each public function is a thin wrapper:

- internal exact multimethod: `_op_exact.dispatch(...)`
- internal approx multimethod: `_op_approx.dispatch(...)`
- public wrapper selects based on `method=...` and availability.

Example:
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

---

## Priority rules (what to exploit first)

Below: recommended **fast path order** for `method="auto"`.

### `solve(A, b)`
1. **Diagonal / Identity / Scalar / Zero**
2. **Triangular / CholeskyFactor / PSD-from-factor**
3. **Toeplitz (if JAX-native solver available)**
4. **Kronecker**
5. **Isotropic-add** (`sI + Symmetric`)
6. **Diag + LowRank** (Woodbury)
7. **Dense exact** (if `n <= config.max_dense_n`)
8. **Iterative** (`cg/gmres/lsmr`) *(default for large)*

### `inverse(A)`
1. Anything with specialized `solve` ⇒ return lazy inverse backed by `solve`
2. Structural inverses (Diagonal, Scalar, Permutation, EigenD, Kronecker, Isotropic-add, Diag+LowRank)
3. Dense exact (threshold)
4. Approx inverse operator (iterative apply)

### `sqrt(A)`
1. Diagonal / Scalar / Identity
2. PSD-from-factor: return factor (best exact “sqrt”)
3. EigenD / Isotropic-add / Kronecker spectral
4. Diag+LowRank (via whitening + low-rank sqrt)
5. Dense exact (threshold)
6. Approx: Lanczos f(A)v wrapped as operator

### `eigh(A, k=None)`
- if `k is None`:
  1. Structured exact (EigenD, Diagonal, Kronecker, Isotropic-add, Toeplitz-special if available)
  2. Dense exact (threshold)
  3. Approx fallback (if user forces)
- if `k is int`:
  1. Kronecker top-k
  2. LowRank / SymmetricLowRank exact modes
  3. Lanczos / LOBPCG

### `trace(A)`
1. Exact: diagonal-summable (Diagonal, Scalar*I, Identity, Zero, PSD-from-factor if cheap)
2. Exact from decomposition (EigenD)
3. Hutchinson trace (default large)

### `slogdet(A)`
1. Exact: diagonal / triangular / PSD-from-factor
2. Exact from decomposition (EigenD, Kronecker, Isotropic-add if eigens available)
3. Matrix determinant lemma for diag+lowrank
4. Dense exact (threshold)
5. SLQ: stochastic Lanczos quadrature `trace(log(A))`

---

## Approx backends (what plugs in)

- Hutchinson: `trace(A)` and `diag(A)`
- Lanczos (symmetric): eigenpairs, f(A)v, SLQ
- Arnoldi (general): f(A)v, GMRES-ish
- LSMR: least squares solves (and regularized variants)

These live in `linalg/approx/*` (or `linalg/*`), re-exported via the public functions with `method=...`.

---

## Config knobs (recommended)

- `config.max_dense_n`: densification threshold for exact fallbacks
- `config.default_methods`: per-op defaults (e.g. `{"slogdet": "slq"}` for large)
- `config.warn_on_densify`: emit warnings in debug mode
- `config.random_seed` or pass PRNG keys explicitly to stochastic methods
