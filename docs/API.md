# Linox API (target v0.0.3)

This document defines the **public, stable** API surface.  
Design goal: **functions are the canonical entry points**. Operators remain the building blocks.

---

## Core conventions

### Inputs
All functions accept either:
- `jax.Array` (dense), or
- `LinearOperator` (preferred for large / structured problems)

Use `utils.as_linop(x)` internally for normalization.

### Outputs
- Functions that conceptually return an **operator** return a `LinearOperator`.
- Functions that conceptually return a **vector / scalar / decomposition** return arrays/scalars (with operator-friendly options).

### Methods
Most heavy functions accept:
- `method="auto"` (default): choose best available implementation
- `method="exact"`: structured exact if available, otherwise dense fallback (may densify)
- `method="approx"`: scalable approximation (often Krylov / Hutchinson / SLQ)
- plus named methods (e.g. `"lanczos"`, `"slq"`, `"cg"`, …)

See `docs/METHODS.md`.

---

## Public functions

### Basic construction & utilities
- `as_linop(A) -> LinearOperator`
- `todense(A) -> jax.Array`
- `allclose(A, B, **kw) -> bool`
- `diagonal(A) -> jax.Array`
- `transpose(A) -> LinearOperator`
- `symmetrize(A) -> LinearOperator`
- `kron(A, B) -> LinearOperator`
- `eye(n, dtype=None) -> Identity`  *(recommended for structure-preserving code)*

### Linear algebra: solvers
- `solve(A, b, *, method="auto", **kw) -> jax.Array`
- `psolve(A, b, *, method="auto", **kw) -> jax.Array`

Typical `method` values:
- `"exact"`: diagonal / triangular / structured solvers, else dense
- `"cg"`, `"gmres"`, `"lsmr"`: iterative methods
- `"auto"` prefers structured exact, then iterative, then dense (configurable)

### Linear algebra: operator transforms
- `inverse(A, *, method="auto", **kw) -> LinearOperator`
- `pinverse(A, *, method="auto", **kw) -> LinearOperator`
- `sqrt(A, *, method="auto", **kw) -> LinearOperator`
- `cholesky(A, *, method="auto", **kw) -> LinearOperator`
- `exp(A, *, method="auto", **kw) -> LinearOperator` *(optional v0.0.3)*
- `log(A, *, method="auto", **kw) -> LinearOperator` *(optional v0.0.3)*

> Note: `sqrt/cholesky/log/exp` may return **function-backed operators** for large problems (apply-mode via Krylov).

### Spectral decompositions

#### `eigh`
```python
eigh(A, k=None, *, which="LM", method="auto", **kw)
  -> (evals: jax.Array, evecs: jax.Array | LinearOperator)
```

Computes eigenvalues/eigenvectors for **Hermitian / symmetric** operators (the `eigh` in the name is intentional:
it assumes `A = A.T` (or `A = A.H` for complex). If you need general (non-symmetric) eigenvalues, that is a different API.

**Parameters**
- `k: int | None`
  - `k=None` (default): **full** decomposition (all eigenpairs).
  - `k=int`: **partial** decomposition (top-*k* eigenpairs), using structured shortcuts or Krylov methods.
- `which: str`
  Selects **which** eigenvalues are returned when `k` is not `None`.
  Common values (inspired by ARPACK/Scipy conventions):

  - `"LM"`: **Largest Magnitude** (default). Returns the eigenvalues with largest `abs(λ)`.
  - `"SM"`: **Smallest Magnitude**. Returns eigenvalues with smallest `abs(λ)`.
  - `"LA"`: **Largest Algebraic**. Returns largest `λ` (most positive).
  - `"SA"`: **Smallest Algebraic**. Returns smallest `λ` (most negative).

  For PSD matrices (`λ ≥ 0`), `"LM"` and `"LA"` coincide; similarly `"SM"` and `"SA"` coincide.

- `method: str`
  - `"auto"`: prefer structured exact/partial routines, then Krylov (`lanczos`/`lobpcg`), then dense.
  - `"exact"`: force full exact decomposition (structured if available; else dense).
  - `"lanczos"`: Krylov method for symmetric problems (good default for large `n`).
  - `"lobpcg"`: block method for symmetric problems (often good when you want multiple modes).
  - `"kron"`: use Kronecker-specialized partial eigensolver (if `A` is a Kronecker operator).
  - Additional options may exist depending on installed backends.

**Return types**
- `evals`: a rank-1 `jax.Array` of eigenvalues (length `n` if `k=None`, else length `k`).
- `evecs`: either
  - a dense `jax.Array` of eigenvectors (shape `(n, n)` or `(n, k)`), or
  - a `LinearOperator` representing the eigenvector matrix `Q` (recommended for large problems).

  When returned as an operator, you can apply it matrix-free:
  - `Q @ x` computes `Qx`
  - `Q.T @ x` computes `Qᵀx`

**Behavior**
- `k=None` (full):
  1. structured exact when available (e.g. `Diagonal`, `EigenD`, `Kronecker`, `IsotropicAdd`, …)
  2. else dense fallback (may densify, subject to `config.max_dense_n`)
- `k=int` (partial):
  1. structured partial when available (e.g. Kronecker top-k)
  2. else Krylov (Lanczos) or block method (LOBPCG)
  3. dense fallback (only if allowed; typically avoid for large)

**Examples**
```python
# full eigen-decomposition (small n or structured)
w, Q = eigh(A)

# top-20 largest eigenvalues (largest magnitude)
w, Q = eigh(A, k=20, which="LM", method="lanczos")

# smallest eigenvalues (useful for preconditioner quality checks)
w, Q = eigh(A, k=10, which="SM", method="lanczos")

# Kronecker-specialized top-k
w, Q = eigh(kron(A, B), k=50, method="kron")
```

#### Relationship to `topk_eigh`
Previous versions exposed a separate `topk_eigh(A, k)` helper, especially for Kronecker operators.
In v0.0.3, **`topk_eigh` is treated as an internal backend** for:
```python
eigh(A, k=k, method="kron")
```
and may remain as a deprecated alias for compatibility.

- `eigh(A, k=None, *, which="LM", method="auto", **kw) -> (evals: jax.Array, evecs: jax.Array | LinearOperator)`
  - `k=None`: full decomposition (structured exact if available; else dense)
  - `k=int`: partial decomposition (structured special cases; else Lanczos/Lobpcg)
- `svd(A, k=None, *, method="auto", **kw) -> (s, U, Vt)` *(optional v0.0.3)*

Optional convenience:
- `eigendecomp(A, k=None, *, method="auto", **kw) -> EigenD`
  - returns a decomposition operator (recommended when you want an operator output)
- `eigh(A, k=None, *, which="LM", method="auto", **kw) -> (evals: jax.Array, evecs: jax.Array | LinearOperator)`
  - `k=None`: full decomposition (structured exact if available; else dense)
  - `k=int`: partial decomposition (structured special cases; else Lanczos/Lobpcg)
- `svd(A, k=None, *, method="auto", **kw) -> (s, U, Vt)` *(optional v0.0.3)*



### Scalars / traces / determinants
- `trace(A, *, method="auto", **kw) -> jax.Array`
- `det(A, *, method="auto", **kw) -> jax.Array`
- `slogdet(A, *, method="auto", **kw) -> (sign, logabsdet)`

Typical `method` values:
- `"exact"`: diagonal / triangular / PSD-from-factor / structured formulas
- `"hutchinson"`: trace estimation
- `"slq"` / `"stochastic"`: logdet via stochastic Lanczos quadrature (trace(log(A)))

---

## Operator overloading (supported)
- `scalar * A`, `A * scalar`
- `A + B`, `A - B`
- `A @ B`, `A @ v`
- `A / B` *(restricted; mostly diagonal-like)*
- `A.T`
- `-A`

Functions remain canonical; operator overloading is syntactic sugar.

---

## Deprecations
The legacy `l*` names (`lsqrt`, `leigh`, `linverse`, …) are kept as wrappers that warn and forward to the canonical functions.


---

## PSD-from-factor (recommended v0.0.3 addition)

Two new operator types make PSD workflows fast and unambiguous:

- `CholeskyFactor(L, lower=True)`: represents a triangular factor `L` and supports fast triangular solves + `logdet`.
- `PSDFromFactor(L, lower=True)`: represents the PSD operator `A = L @ L.T` (or `U.T @ U` if `lower=False`).

### Key API behaviors
- `sqrt(PSDFromFactor(...), method="exact") -> CholeskyFactor`
  - exact square root is the factor itself (best possible, avoids spectral work)
- `cholesky(PSDFromFactor(...), method="exact") -> CholeskyFactor`
  - returns the stored factor
- `solve(PSDFromFactor(...), b, method="exact")`
  - dense triangular factor: `cho_solve`
  - matrix-free factor: fall back to `cg` on `A @ x`
- `slogdet(PSDFromFactor(...), method="exact")`
  - dense triangular factor: `2 * sum(log(abs(diag(L))))`
  - otherwise: `method="slq"` (stochastic Lanczos quadrature)
