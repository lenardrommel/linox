# `linox`: Linear Operators in JAX

`linox` is a Python package that provides a collection of linear operators for JAX, enabling efficient and flexible linear algebra operations with lazy evaluation. This package is designed as an JAX alternative to [`probnum.linops`](https://probnum.readthedocs.io/en/latest/api/linops.html), but it is currently still under development having less and more instable features. It has no dependencies other than [JAX](https://github.com/jax-ml/jax) and [`plum`](https://github.com/beartype/plum) for multiple dispatch.

## Features

- **Lazy Evaluation**: All operators support lazy evaluation, allowing for efficient computation of complex linear transformations
- **JAX Integration**: Built on top of JAX, providing automatic differentiation, parallelization, JIT compilation, and GPU/TPU support
- **Composable Operators**: Operators can be combined to form complex linear transformations

## Linear Operators

### Basic Operators
- `Matrix`: General matrix operator
- `Identity`: Identity matrix operator
- `Diagonal`: Diagonal matrix operator
- `Scalar`: Scalar multiple of identity
- `Zero`: Zero matrix operator
- `Ones`: Matrix of ones operator

### Block Operators
- `BlockMatrix`: General block matrix operator
- `BlockMatrix2x2`: 2x2 block matrix operator
- `BlockDiagonal`: Block diagonal matrix operator

### Low Rank Operators
- `LowRank`: General low rank operator
- `SymmetricLowRank`: Symmetric low rank operator
- `IsotropicScalingPlusSymmetricLowRank`: Isotropic scaling plus symmetric low rank
- `PositiveDiagonalPlusSymmetricLowRank`: Positive diagonal plus symmetric low rank

### Special Operators
- `Kronecker`: Kronecker product operator
- `Permutation`: Permutation matrix operator
- `EigenD`: Eigenvalue decomposition operator

## Benefits of JAX Integration

- **Automatic Differentiation**: Compute gradients automatically through operator compositions
- **JIT Compilation**: Speed up computations with just-in-time compilation
- **Vectorization**: Efficient batch processing of linear operations via e.g. `jax.vmap`
- **GPU/TPU Support**: Run computations on accelerators without code changes
- **Functional Programming**: Pure functions enable better optimization and parallelization

## Quick Example

```python
import jax
import jax.numpy as jnp
from linox import Matrix, Diagonal, BlockMatrix

# Create operators
A = Matrix(jnp.array([[1, 2], [3, 4]], dtype=jnp.float32))
D = Diagonal(jnp.array([1, 2], dtype=jnp.float32))

# Compose operators
B = BlockMatrix([[A, D], [D, A]])

# Apply to vector
x = jnp.ones((4,), dtype=jnp.float32)
y = B @ x  # Lazy evaluation

# Parallelize over batch of vectors
x_batched = jnp.ones((10, 4), dtype=jnp.float32)
y_batched = jax.vmap(B)(x_batched)
```

## Densification Warnings and Debug Mode

Some operations fall back to dense computations when a lazy, structure‑preserving
path is not available (e.g., diagonal of a general product of non‑diagonal factors,
explicit inverse materialization). To help diagnose performance, `linox` can emit
warnings whenever an operation densifies.

By default, these warnings are suppressed. Enable them via the API or an
environment variable:

```python
from linox import set_debug

# Turn on debug warnings
set_debug(True)

# Turn them off again
set_debug(False)
```

Or set an environment variable before running Python:

```bash
export LINOX_DEBUG=1   # enables densification warnings
python your_script.py
```

Examples of operations that may warn when debug is enabled:
- `diagonal(op)` when it must convert an operator to dense to compute the diagonal.
- Decompositions like `leigh`, `svd`, `lqr` falling back to dense.
- `InverseLinearOperator.todense()` and pseudo‑inverse matmul paths that need dense.
- `Matrix.todense()` when explicitly materializing the dense array.

Note: Many structure‑aware paths remain lazy (e.g., diagonals of Kronecker
products and of diagonal‑like products). The warnings help ensure large operators
aren’t accidentally densified.
