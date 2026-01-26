# Operators (v0.0.3)

This document describes the operator taxonomy, tags, and which algorithms they enable.

---

## Operator families

### Core
- `LinearOperator` (base)
- `Matrix` (dense)
- `Identity`, `Zero`, `Scalar`, `Ones`
- `Diagonal`

### Arithmetic / composition
- `ScaledLinearOperator`
- `AddLinearOperator`
- `ProductLinearOperator`
- `TransposedLinearOperator`

### Structure
- `BlockMatrix`, `BlockDiagonal`, `BlockMatrix2x2`
- `Toeplitz`
- `Kronecker`
- `EigenD`

### Low rank
- `LowRank`: `U diag(S) Vᵀ`
- `SymmetricLowRank`: `U diag(S) Uᵀ`
- `IsotropicScalingPlusSymmetricLowRank`: `σI + U diag(S) Uᵀ`
- `PositiveDiagonalPlusSymmetricLowRank`: `D + α U diag(S) Uᵀ`

### Isotropic shifts
- `IsotropicAdditiveLinearOperator`: `sI + A` for symmetric `A`

### PSD / triangular (recommended additions)
- `Triangular(Matrix, lower=True)` (fast triangular solve)
- `CholeskyFactor(L, lower=True)` (triangular + logdet)
- `PSDFromFactor(L, lower=True)`: represents `A = L Lᵀ`

---

## Tags / subgroup properties

To avoid subclass explosion, operators may carry tags:
- `symmetric: bool`
- `psd: bool`
- `unitary: bool`
- `triangular: "lower"|"upper"|None`

Algorithms use tags to choose safe/fast methods (especially for `method="auto"`).

Wrappers:
- `assume_symmetric(A)`
- `assume_psd(A)`
- `assume_unitary(A)`
- `triangular(A, lower=True)`

---

## Auto-rewrite: `sI + A` → Isotropic-add

When adding a **structural isotropic shift** to a symmetric operator:
- `Scalar(s) + A`
- `ScaledLinearOperator(Identity(n), s) + A`
the addition dispatch should return:
- `IsotropicAdditiveLinearOperator(s, A)`

For the special case where `A` is `SymmetricLowRank`, prefer:
- `IsotropicScalingPlusSymmetricLowRank(s, U, S)`

This keeps GP-style code efficient and stable.
