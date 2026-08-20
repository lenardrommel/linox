# API reference

Generated from the source. See [Operators](operators.md) for the classes and
[Configuration](configuration.md) for settings.

## Creation

::: linox.api
    options:
      members:
        - as_linop
        - todense
        - eye
        - zeros
        - ones
        - diag
        - kron
        - block_diag
        - bmat
        - toeplitz

## Solving

::: linox.api
    options:
      members:
        - solve
        - psolve
        - inverse
        - pinverse
        - lu_factor
        - lu_solve

::: linox.linalg.solution

## Decompositions

::: linox.api
    options:
      members:
        - eigh
        - svd
        - qr
        - cholesky
        - sqrt

## Traces and determinants

::: linox.api
    options:
      members:
        - trace
        - det
        - slogdet
        - logdet
        - diagonal

## Matrix functions

::: linox.api
    options:
      members:
        - exp
        - log
        - pow

## Properties

::: linox.api
    options:
      members:
        - is_square
        - is_symmetric
        - is_hermitian
        - symmetrize
        - congruence_transform
        - allclose
        - validate

## Matrix-free algorithms

::: linox.linalg.approx.cg

::: linox.linalg.approx.lanczos

::: linox.linalg.approx.arnoldi

::: linox.linalg.approx.hutchinson

::: linox.linalg.approx.slq

::: linox.linalg.approx.lsmr
    options:
      members:
        - lsmr_solve

::: linox.linalg.spectral
    options:
      members:
        - svd_partial
        - lanczos_bidiag
