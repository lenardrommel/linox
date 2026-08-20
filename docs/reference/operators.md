# Operator reference

## Base

::: linox.operators.base

## Basic

::: linox.operators.dense

::: linox.operators.diagonal
    options:
      members:
        - Diagonal

::: linox.operators.special

## Composition

::: linox.operators.arithmetic
    options:
      members:
        - ScaledLinearOperator
        - AddLinearOperator
        - ProductLinearOperator
        - TransposedLinearOperator
        - InverseLinearOperator
        - PseudoInverseLinearOperator
        - CongruenceTransform

## Structured

::: linox.operators.kron
    options:
      members:
        - Kronecker
        - KroneckerSelectedEigenvectors
        - topk_eigh

::: linox.operators.isotropic
    options:
      members:
        - IsotropicAdditiveLinearOperator

::: linox.operators.lowrank

::: linox.operators.eigen

::: linox.operators.block

::: linox.operators.toeplitz
    options:
      members:
        - Toeplitz

::: linox.operators.permutation
    options:
      members:
        - Permutation

::: linox.operators.factor

## Kernels

::: linox.operators.kernel
    options:
      members:
        - kernel_operator
        - KernelOperator
        - ArrayKernel
        - ToeplitzKernel

## Property wrappers

::: linox.operators.wrappers
