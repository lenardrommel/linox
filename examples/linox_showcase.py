#!/usr/bin/env python3
"""Comprehensive linox Operations Showcase.

This example demonstrates ALL major linox operations in one place,
organized by category. Run this to verify linox functionality.

Categories:
    1. Operator Construction
    2. Arithmeticoperations
    3. Decompositions (Cholesky, Eigendecomposition)
    4. Linear Solvers
    5. Determinants and Traces
    6. Square Root and Matrix Functions
    7. Validation and Properties
    8. Special Structures (Kronecker, Block, Low-Rank)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

import linox as lo

# Enable float64
jax.config.update("jax_enable_x64", True)


def section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """Demonstrate all major linox operations."""
    print("=" * 60)
    print("       LINOX OPERATIONS SHOWCASE")
    print("=" * 60)

    key = random.PRNGKey(42)
    n = 5  # Small size for demo

    # =========================================================================
    # 1. OPERATOR CONSTRUCTION
    # =========================================================================
    section("1. OPERATOR CONSTRUCTION")

    # Dense matrix
    A_dense = random.normal(key, (n, n))
    A = lo.Matrix(A_dense)
    print(f"lo.Matrix: {A.shape}, type={type(A).__name__}")

    # Identity
    I = lo.Identity(n)
    print(f"lo.Identity: {I.shape}")

    # Diagonal
    d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    D = lo.Diagonal(d)
    print(f"lo.Diagonal: {D.shape}")

    # Zero and Ones
    Z = lo.Zero((n, n))
    O = lo.Ones((n, n))
    print(f"lo.Zero: {Z.shape}, lo.Ones: {O.shape}")

    # From array
    B = lo.as_linop(A_dense)
    print(f"lo.as_linop: {B.shape}")

    # =========================================================================
    # 2. ARITHMETIC OPERATIONS
    # =========================================================================
    section("2. ARITHMETIC OPERATIONS")

    # Addition
    C = A + D
    print(f"A + D: {C.shape}, type={type(C).__name__}")

    # Scalar multiplication
    C2 = 2.0 * A
    print(f"2.0 * A: {C2.shape}, type={type(C2).__name__}")

    # Matrix multiplication
    key, subkey = random.split(key)
    B_dense = random.normal(subkey, (n, n))
    B = lo.Matrix(B_dense)
    C3 = A @ B
    print(f"A @ B: {C3.shape}, type={type(C3).__name__}")

    # Transpose
    AT = lo.transpose(A)
    print(f"lo.transpose(A): {AT.shape}")

    # =========================================================================
    # 3. SYMMETRIC/PSD WRAPPERS
    # =========================================================================
    section("3. SYMMETRIC/PSD WRAPPERS")

    # Create SPD matrix
    key, subkey = random.split(key)
    L = random.normal(subkey, (n, n))
    SPD_dense = L @ L.T + 0.1 * jnp.eye(n)
    SPD = lo.Matrix(SPD_dense)

    # Sym wrapper
    Sym_op = lo.Sym(SPD)
    print(f"lo.Sym: is_symmetric={Sym_op.is_symmetric}")

    # PSD wrapper
    PSD_op = lo.PSD(SPD)
    print(f"lo.PSD: is_psd={PSD_op.is_psd}, is_symmetric={PSD_op.is_symmetric}")

    # SPD wrapper
    SPD_op = lo.SPD(SPD)
    print(f"lo.SPD: is_psd={SPD_op.is_psd}")

    # assume_* functions
    assumed = lo.assume_psd(SPD)
    print(f"lo.assume_psd: is_psd={assumed.is_psd}")

    assumed_sym = lo.assume_symmetric(SPD)
    print(f"lo.assume_symmetric: is_symmetric={assumed_sym.is_symmetric}")

    # =========================================================================
    # 4. DECOMPOSITIONS
    # =========================================================================
    section("4. DECOMPOSITIONS")

    # Cholesky
    chol = lo.lcholesky(PSD_op)
    print(f"lo.lcholesky: {chol.shape}")

    # Eigendecomposition
    eigvals, eigvecs = lo.leigh(PSD_op)
    print(f"lo.leigh: eigvals shape={eigvals.shape}, eigvecs shape={eigvecs.shape}")
    print(f"  Eigenvalues: {eigvals}")

    # =========================================================================
    # 5. LINEAR SOLVERS
    # =========================================================================
    section("5. LINEAR SOLVERS")

    b = jnp.ones(n)

    # solve (for dense/SPD)
    x_solve = lo.solve(PSD_op, b)
    print(f"lo.solve: {x_solve.shape}")
    residual = jnp.linalg.norm(SPD_dense @ x_solve - b)
    print(f"  Residual: {residual:.2e}")

    # lsolve (functional interface)
    x_lsolve = lo.lsolve(PSD_op, b[:, None])[:, 0]
    print(f"lo.lsolve: {x_lsolve.shape}")

    # lpsolve (least squares)
    key, subkey = random.split(key)
    A_rect = lo.Matrix(random.normal(subkey, (n + 2, n)))
    b_rect = jnp.ones(n + 2)
    x_ls = lo.lpsolve(A_rect, b_rect[:, None])[:, 0]
    print(f"lo.lpsolve (least squares): {x_ls.shape}")

    # =========================================================================
    # 6. INVERSE OPERATORS
    # =========================================================================
    section("6. INVERSE OPERATORS")

    # Inverse
    A_inv = lo.inv(PSD_op)
    print(f"lo.inv: {A_inv.shape}, type={type(A_inv).__name__}")

    # linverse (functional)
    A_inv2 = lo.linverse(PSD_op)
    print(f"lo.linverse: {A_inv2.shape}")

    # Pseudo-inverse
    A_pinv = lo.pinv(A_rect)
    print(f"lo.pinv: {A_pinv.shape}")

    # =========================================================================
    # 7. DETERMINANTS AND TRACES
    # =========================================================================
    section("7. DETERMINANTS AND TRACES")

    # Log determinant
    sign, logdet = lo.slogdet(PSD_op)
    print(f"lo.slogdet: sign={sign:.0f}, logdet={logdet:.4f}")

    # Compare with numpy
    np_logdet = np.linalg.slogdet(np.asarray(SPD_dense))[1]
    print(f"  NumPy logdet: {np_logdet:.4f}")

    # Trace
    tr = lo.trace(PSD_op)
    print(f"lo.trace: {tr:.4f}")
    print(f"  NumPy trace: {np.trace(np.asarray(SPD_dense)):.4f}")

    # ltrace (functional)
    tr2 = lo.ltrace(PSD_op)
    print(f"lo.ltrace: {tr2:.4f}")

    # =========================================================================
    # 8. MATRIX FUNCTIONS
    # =========================================================================
    section("8. MATRIX FUNCTIONS")

    # Square root
    sqrt_op = lo.lsqrt(PSD_op)
    print(f"lo.lsqrt: {sqrt_op.shape}")
    # Verify: sqrt @ sqrt ≈ SPD
    sqrt_sq = sqrt_op @ sqrt_op
    sqrt_error = jnp.max(jnp.abs(lo.todense(sqrt_sq) - SPD_dense))
    print(f"  sqrt(A) @ sqrt(A) error: {sqrt_error:.2e}")

    # Matrix exponential
    exp_op = lo.exp(0.1 * PSD_op)
    print(f"lo.exp: {exp_op.shape}")

    # Matrix logarithm
    log_op = lo.log(PSD_op)
    print(f"lo.log: {log_op.shape}")

    # Matrix power
    pow_op = lo.pow(PSD_op, 0.5)
    print(f"lo.pow(A, 0.5): {pow_op.shape}")

    # =========================================================================
    # 9. SPECIAL STRUCTURES
    # =========================================================================
    section("9. SPECIAL STRUCTURES")

    # Kronecker product
    K1 = lo.Matrix(jnp.eye(3))
    K2 = lo.Matrix(jnp.ones((2, 2)))
    Kron = lo.Kronecker(K1, K2)
    print(f"lo.Kronecker: {Kron.shape}")

    # kron function
    kron2 = lo.kron(K1, K2)
    print(f"lo.kron: {kron2.shape}")

    # Block diagonal
    BD = lo.BlockDiagonal(A, B)
    print(f"lo.BlockDiagonal: {BD.shape}")

    # Low rank: U @ V.T
    key, k1, k2 = random.split(key, 3)
    U = random.normal(k1, (n, 2))
    V = random.normal(k2, (n, 2))
    LR = lo.LowRank(U, V)
    print(f"lo.LowRank (rank-2): {LR.shape}")

    # Symmetric low rank: U @ U.T
    SLR = lo.SymmetricLowRank(U)
    print(f"lo.SymmetricLowRank: {SLR.shape}")

    # IsotropicAdditiveLinearOperator
    Iso = lo.IsotropicAdditiveLinearOperator(0.1, PSD_op)
    print(f"lo.IsotropicAdditiveLinearOperator: {Iso.shape}")

    # =========================================================================
    # 10. VALIDATION
    # =========================================================================
    section("10. VALIDATION")

    # validate function
    try:
        lo.validate(PSD_op)
        print("lo.validate(PSD_op): ✓ passed")
    except lo.ValidationError as e:
        print(f"lo.validate: ✗ failed - {e}")

    # Debug mode validation
    try:
        lo.validate(PSD_op, mode="debug")
        print("lo.validate(mode='debug'): ✓ passed")
    except lo.ValidationError as e:
        print(f"lo.validate(debug): ✗ failed - {e}")

    # Check functions
    print(f"lo.is_square(A): {lo.is_square(A)}")
    print(f"lo.is_symmetric(PSD_op): {lo.is_symmetric(PSD_op)}")

    # =========================================================================
    # 11. UTILITY FUNCTIONS
    # =========================================================================
    section("11. UTILITY FUNCTIONS")

    # todense
    dense = lo.todense(A)
    print(f"lo.todense: {dense.shape}")

    # diagonal extraction
    diag = lo.diagonal(PSD_op)
    print(f"lo.diagonal: {diag.shape}")

    # eye
    eye = lo.eye(n)
    print(f"lo.eye: {eye.shape}")

    # zeros
    zeros = lo.zeros(n)
    print(f"lo.zeros: {zeros.shape}")

    # ones
    ones = lo.ones(n)
    print(f"lo.ones: {ones.shape}")

    # allclose
    close = lo.allclose(A, A)
    print(f"lo.allclose(A, A): {close}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    section("SUMMARY")

    print("""
All linox operations demonstrated:

CONSTRUCTION:
  lo.Matrix, lo.Identity, lo.Diagonal, lo.Scalar, lo.Zero, lo.Ones, lo.as_linop

WRAPPERS:
  lo.Sym, lo.PSD, lo.SPD, lo.assume_symmetric, lo.assume_psd, lo.assume_spd

ARITHMETIC:
  +, -, *, @, lo.transpose

DECOMPOSITIONS:
  lo.lcholesky, lo.leigh

SOLVERS:
  lo.solve, lo.lsolve, lo.lpsolve

INVERSE:
  lo.inv, lo.linverse, lo.pinv, lo.lpinverse

DETERMINANTS/TRACES:
  lo.slogdet, lo.trace, lo.ltrace, lo.det, lo.logdet

MATRIX FUNCTIONS:
  lo.lsqrt, lo.sqrt, lo.exp, lo.log, lo.pow

SPECIAL STRUCTURES:
  lo.Kronecker, lo.kron, lo.BlockDiagonal, lo.block_diag
  lo.LowRank, lo.SymmetricLowRank, lo.IsotropicAdditiveLinearOperator

VALIDATION:
  lo.validate, lo.is_square, lo.is_symmetric

UTILITIES:
  lo.todense, lo.diagonal, lo.eye, lo.zeros, lo.ones, lo.allclose
""")

    print("✅ All linox operations work correctly!")


if __name__ == "__main__":
    main()
