# Approximation backend (Lanczos/Hutchinson/SLQ/LSMR)

This document summarizes the approximation algorithms integrated via `method=...`.

---

## Primary building blocks

### Lanczos (symmetric)
- `lanczos_tridiag`: Krylov basis + tridiagonal reduction
- `lanczos_eigh`: top-k eigenpairs (k << n)
- `lanczos_matrix_function`: computes `f(A) v` for symmetric `A`
- **SLQ** (stochastic Lanczos quadrature): estimates `trace(f(A))`  
  Used for `slogdet` via `trace(log(A))`.

### Arnoldi (general)
- `arnoldi_iteration`: Hessenberg reduction
- `arnoldi_matrix_function`: `f(A) v` for non-symmetric `A`

### Hutchinson
- `hutchinson_trace`: unbiased trace estimator
- `hutchinson_diagonal`: diagonal estimator
- `hutchinson_trace_and_diagonal`: joint estimation

### LSMR
- `lsmr_solve`: least squares solver (also supports damping / regularization)

---

## How they surface in the public API

- `trace(A, method="hutchinson", num_samples=..., key=...)`
- `slogdet(A, method="slq", num_samples=..., num_iters=..., key=...)`
- `eigh(A, k=..., method="lanczos", num_iters=..., which=...)`
- `sqrt(A, method="lanczos", num_iters=...)`  
  Returns an operator that applies `sqrt(A)` via `f(A)v` internally.
- `solve(A, b, method="lsmr" | "cg" | "gmres", ...)`

---

## Numerical notes
- Reorthogonalization in Lanczos improves stability; expose `reortho=True|False`.
- SLQ requires (approximate) PSD/symmetric assumptions; enforce via tags or `assume_psd`.
- Hutchinson: Rademacher vectors usually lower variance for trace than Gaussian.


---

## `which` in Krylov eigensolvers (Lanczos / LOBPCG)

For symmetric problems, Krylov methods typically compute eigenpairs at the spectrum extremes.

- To target **largest** eigenvalues: use `"LA"` (or `"LM"` if PSD / magnitude is intended).
- To target **smallest** eigenvalues: use `"SA"` / `"SM"`; often implemented via shift-invert or by running on `-A` and flipping signs.

In v0.0.3, it is recommended to:
- support `"LA"` and `"SA"` robustly for Lanczos/LOBPCG
- treat `"LM"`/`"SM"` as aliases when PSD is known (`psd=True`) or when users accept magnitude-based sorting of computed modes.
