# Changelog

All notable changes to the `linox` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-02-06

### Added
- **Unified `method=` dispatch**: All linalg functions (`solve`, `inverse`, `sqrt`, `eigh`, `trace`, `det`, `slogdet`) now support `method="auto"|"exact"|"approx"` parameter for automatic or explicit method selection
- **Approximate methods integration**:
  - Lanczos methods for eigenvalue problems and matrix functions
  - Hutchinson trace estimation with configurable probes
  - Stochastic Lanczos Quadrature (SLQ) for log-determinant
  - LSMR iterative solver for large sparse systems
- **Introspection system**:
  - `analyze(op)` — Analyze operator structure and return `OperatorIR`
  - `canonicalize(op)` — Apply simplification and rewrite rules
  - `cache_key(op)` — Generate fingerprint for caching
- **Session-based caching**: Automatic caching of expensive operations with `clear_cache()` utility
- **Debug tracing**: `linox.debug.trace()` context manager for performance analysis and debugging
- **New operators**:
  - `Triangular` — Triangular matrix operator
  - `CholeskyFactor` — Cholesky factor operator  
  - `PSDFromFactor` — PSD operator from factorization
- **Auto-rewrites**:
  - `Scalar * Identity + A` → `IsotropicAdditiveLinearOperator(s, A)` for efficient spectral transforms
  - `Diagonal + SymmetricLowRank` → `PositiveDiagonalPlusSymmetricLowRank` with Woodbury formula
  - `IsotropicScalingPlusSymmetricLowRank` for `s*I + U @ S @ U.T` patterns
- **Property checks**: `is_symmetric()` and `is_hermitian()` randomized checks without densification
- **Partial SVD**: `svd_partial()` for computing k largest singular values/vectors via Lanczos bidiagonalization

### Changed
- **BREAKING**: Package structure reorganized:
  - `linox.operators/` — All operator classes
  - `linox.linalg/` — Linear algebra functions and algorithms
  - `linox.utils/` — Utility functions
  - `linox.structure/` — Introspection and canonicalization
  - `linox.cache/` — Caching infrastructure
- **BREAKING**: Public API now in `linox.api` with stable re-exports via `linox.__init__`
- **BREAKING**: `l*` prefixed function names deprecated (e.g., `lsolve` → `solve`, `linverse` → `inverse`)
  - Old names still work with deprecation warnings
  - Will be removed in v0.0.4
- **Method resolution**: `config.resolve_method()` implements priority-based method selection:
  1. Explicit `method=` argument
  2. Operator-specific config (e.g., `config.diagonal_method`)
  3. Global `config.default_method`
  4. Fallback to `"auto"`

### Fixed
- Circular import issues in operator modules (caused by import reorganization)
- Shape broadcasting in block operators
- Memory leaks in caching system
- Import path compatibility for refactored modules

### Deprecated
- `l*` prefixed function names (`lsolve`, `linverse`, `ldet`, `ltrace`, etc.)
  - Use canonical names instead: `solve`, `inverse`, `det`, `trace`
  - Deprecated functions emit warnings
  - Scheduled for removal in v0.0.4

## [0.0.2] - 2025-XX-XX

### Changed
- Removed "l" prefix from function names
- Functions like `lsolve`, `linverse`, `ldet` now available as `solve`, `inverse`, `det`
- Old "l"-prefixed functions deprecated

## [0.0.1] - Initial Release

### Added
- Initial implementation of linear operators for JAX
- Basic operators: Matrix, Identity, Diagonal, Scalar, Zero
- Block operators: BlockMatrix, BlockDiagonal
- Low rank operators: LowRank, SymmetricLowRank
- Kronecker product operator
- Basic linear algebra functions
- JAX integration with automatic differentiation

---

[0.0.3]: https://github.com/2bys/linox/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/2bys/linox/releases/tag/v0.0.2
[0.0.1]: https://github.com/2bys/linox/releases/tag/v0.0.1
