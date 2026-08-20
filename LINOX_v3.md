# Current Structure

## Linear Operators (Subclasses of `LinearOperator`)

- `LinearOperator`: Abstract base class for matrix-free finite-dimensional linear operators. Follows `jax.numpy.ndarray` behavior (shape, dtype, matmul, etc.).
- `Matrix`: Represents a general matrix $A$.
- `Identity`: Represents the identity matrix $I$.
- `Diagonal`: Represents a diagonal matrix $\text{diag}(d)$.
- `Scalar`: Represents a scalar multiple of the identity $\alpha I$.
- `Zero`: Represents the zero matrix (all elements are zero).
- `Ones`: Represents a matrix of ones $\mathbf{1}\mathbf{1}^T$.
- `ScaledLinearOperator`: Represents a linear operator scaled by a scalar $\alpha A$.
- `AddLinearOperator`: Represents the sum of multiple linear operators $A_1 + A_2 + \dots$.
- `ProductLinearOperator`: Represents the product (composition) of multiple linear operators $A_1 A_2 \dots$.
- `TransposedLinearOperator`: Represents the transpose of a linear operator $A^T$.
- `InverseLinearOperator`: Represents the inverse of a linear operator $A^{-1}$. Uses `solve` for matmul.
- `PseudoInverseLinearOperator`: Represents the Moore-Penrose pseudo-inverse $A^\dagger$.
- `CongruenceTransform`: Represents the congruence transformation $ABA^T$.
- `BlockMatrix`: Represents a general block matrix $\begin{bmatrix} A_{11} & \dots \\ \dots & A_{mn} \end{bmatrix}$.
- `BlockMatrix2x2`: Specialized 2x2 block matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$.
- `BlockDiagonal`: Represents a block diagonal matrix.
- `EigenD`: Represents a symmetric operator via its eigenvalue decomposition $Q \Lambda Q^T$.
- `IsotropicAdditiveLinearOperator`: Represents an isotropic shift $sI + A$ where $A$ is symmetric. Optimized for spectral transforms.
- `KernelOperator`: Base class for operators derived from a kernel function.
- `ArrayKernel`: Represents a kernel matrix computed as a dense array.
- `ToeplitzKernel`: Represents a kernel matrix with Toeplitz structure (e.g., for stationary kernels on a grid).
- `Kronecker`: Represents the Kronecker product $A \otimes B$.
- `KroneckerSelectedEigenvectors`: Matrix-free representation of a subset of eigenvectors of a Kronecker product.
- `LowRank`: Represents a low-rank matrix $U \text{diag}(S) V^T$.
- `SymmetricLowRank`: Represents a symmetric low-rank matrix $U \text{diag}(S) U^T$.
- `IsotropicScalingPlusSymmetricLowRank`: Represents $\sigma I + U \text{diag}(S) U^T$.
- `PositiveDiagonalPlusSymmetricLowRank`: Represents $D + \alpha U \text{diag}(S) U^T$ for a positive diagonal $D$.
- `Permutation`: Represents a permutation matrix $P$.
- `Toeplitz`: Represents a general Toeplitz matrix defined by a vector.

## API Functions

### Arithmetic & Basics
- `add(A, B)` / `ladd`: Addition of operators or arrays.
- `sub(A, B)` / `lsub`: Subtraction of operators or arrays.
- `mul(α, A)` / `lmul`: Scalar multiplication.
- `div(A, B)` / `ldiv`: Element-wise division (mostly for diagonal-like operators).
- `matmul(A, B)` / `lmatmul`: Matrix multiplication (composition).
- `neg(A)` / `lneg`: Negation.
- `sqrt(A)` / `lsqrt`: Matrix square root (often Cholesky or spectral).
- `transpose(A)`: Returns the transpose $A^T$.
- `diagonal(A)`: Extracts the diagonal of the operator.
- `symmetrize(A)`: Returns $0.5(A + A^T)$.

### Solvers & Decompositions
- `inverse(A)` / `linverse`: Returns $A^{-1}$ as a lazy operator.
- `pinverse(A)` / `lpinverse`: Returns $A^\dagger$ as a lazy operator.
- `solve(A, b)` / `lsolve`: Solves the linear system $Ax = b$.
- `psolve(A, b)` / `lpsolve`: Solves $Ax = b$ using the pseudo-inverse.
- `cholesky(A)` / `lcholesky`: Returns a Cholesky-like factor $L$.
- `lu_factor(A)`: LU factorization.
- `lu_solve(A, b)`: LU-based solver.
- `eigh(A)` / `leigh`: Eigenvalue decomposition for Hermitian operators.
- `svd(A)`: Singular Value Decomposition.
- `qr(A)` / `lqr`: QR decomposition.
- `topk_eigh(A, k)`: Efficiently finds the top-k eigenvalues/vectors (esp. for Kronecker).

### Properties & Checks
- `det(A)` / `ldet`: Determinant.
- `slogdet(A)`: Sign and log-determinant.
- `is_square(A)`: Checks if the operator is square.
- `is_symmetric(A)`: Checks if the operator is symmetric (often via randomized probing).
- `is_hermitian(A)`: Checks if the operator is Hermitian.

### Utilities
- `as_linop(A)`: Converts an array or operator into a `LinearOperator`.
- `todense(A)` / `as_dense(A)`: Converts a `LinearOperator` to a dense `jax.Array`.
- `allclose(A, B)`: Checks if two operators are numerically close.
- `as_shape(shape)`: Utility for shape normalization.
- `as_scalar(s)`: Utility for scalar normalization.

### Debug & Visualization
- `linop_graph(A)`: Returns a tree representation of the operator's structure.
- `inspect_run(fn, ...)`: Traces an execution to record events like `todense` calls.
- `config.set_debug(bool)`: Enables/disables debug mode (printing warnings on densification).

# API (for v 0.0.3)
- `scalar * A, A * scalar`
- `A + B, A - B`
- `A @ B`
- `A @ v`
- `A / B`
- `A.T`
- `diagonal(A)`
- `inverse(A, method="exact" | "approximate")`: Unified entry point for matrix inversion.
- `solve(A, b)`
- `psolve(A, b)`
- `inv(A)`: Alias for `inverse`.
- `pinv(A)`: Alias for `pinverse`.
- `lsqrt(A, method="exact" | "approximate")`: Unified entry point for matrix square roots (e.g., Cholesky/spectral vs. Lanczos/Krylov).
- `eigh(A, k=None)`: Unified entry point for eigendecomposition (exact if `k=None`, partial/approximate if `k` is specified).
- `svd(A)`
- `qr(A)`
- `lu_factor(A)`
- `lu_solve(A, b)`
- `cholesky(A)`
- `det(A)`
- `slogdet(A)`
- `kron(A, B)`
- `is_square(A)`
- `is_symmetric(A)`
- `is_hermitian(A)`
- `symmetrize(A)`

## utils
- `utils.as_linop`
- `utils.todense`
- `utils.allclose`
- `utils.as_shape`
- `utils.as_scalar`


## debug
- `debug.inspect_run`

## config
- `config.warn`
- `config.emit`
- `config.set_debug`
- `config.set_dtype`


## types
### API Types
``
### Argument Types
### Output Types

## Scripts
- `api.py` (TODO)
- `operators/`
- `linalg/`
- `utils/`

-------------------------------------------------------

# Target Structure
`api.py` (TODO)
`operators/`
`|_`
`linalg/`
`|_`
