<div align="center">
  <img src="linox_logo.png" alt="linox logo" width="180" />
</div>

# `linox`: Linear Operators in JAX

**Version:** 0.0.2

`linox` is a Python package that provides a collection of linear operators for JAX, enabling efficient and flexible linear algebra operations with lazy evaluation. This package is designed as a JAX alternative to [`probnum.linops`](https://probnum.readthedocs.io/en/latest/api/linops.html), but it is currently still under development having less and more instable features. It has no dependencies other than [JAX](https://github.com/jax-ml/jax) and [`plum`](https://github.com/beartype/plum) for multiple dispatch.

**Note (v0.0.2):** The API has been updated to remove the "l" prefix from function names. Functions like `lsolve`, `linverse`, `ldet`, etc. are now available as `solve`, `inverse`, `det`, etc. The old "l"-prefixed functions are deprecated and will be removed in version 0.0.3.

<div align="center">
  <picture>
    <img alt="Kronecker GP predictions (2D)" src="examples/osci_gp_operator_walkthrough_predictions.png" width="45%" />
  </picture>
  <picture>
    <img alt="Kronecker GP posterior uncertainty (2D)" src="examples/osci_gp_operator_walkthrough_uncertainty.png" width="45%" />
  </picture>
  <br/>
  <sub>Matrix‑free Gaussian Process predictions and posterior uncertainty on a 2D heat‑equation task using Kronecker‑structured kernels.</sub>
  <br/>
</div>

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
- `Toeplitz`: Toeplitz matrix operator
- `IsotropicAdditiveLinearOperator`: Efficient operator for `s*I + A` with spectral transforms

### Composite Operators
- `ScaledLinearOperator`: Scalar multiple of an operator
- `AddLinearOperator`: Sum of multiple operators
- `ProductLinearOperator`: Product of multiple operators
- `TransposedLinearOperator`: Transpose of an operator
- `InverseLinearOperator`: Inverse of an operator
- `PseudoInverseLinearOperator`: Pseudo-inverse of an operator
- `CongruenceTransform`: Congruence transformation `A B A^T`

## Arithmetic Operations

### Linear System Solvers
- `solve(A, b)`: Solve the linear system `Ax = b`
- `psolve(A, b)`: Solve using pseudo-inverse for singular/rectangular systems
- `lu_factor(A)`: LU factorization
- `lu_solve(A, b)`: Solve using LU factorization

### Matrix Decompositions
- `eigh(A)`: Eigendecomposition for Hermitian matrices
- `svd(A)`: Singular Value Decomposition
- `qr(A)`: QR decomposition
- `cholesky(A)`: Cholesky decomposition

### Matrix Functions
- `inverse(A)`: Compute inverse `A^{-1}`
- `pinverse(A)`: Compute pseudo-inverse `A^†`
- `sqrt(A)`: Compute matrix square root
- `transpose(A)`: Transpose operator
- `det(A)`: Compute determinant
- `slogdet(A)`: Compute sign and log-determinant

### Element-wise & Structural Operations
- `diagonal(A)`: Extract diagonal elements
- `symmetrize(A)`: Symmetrize operator `(A + A^T)/2`
- `congruence_transform(A, B)`: Compute `A B A^T`
- `kron(A, B)`: Kronecker product
- `iso(s, A)`: Create isotropic additive operator `s*I + A`

### Arithmetic Operators
- `add(A, B)`: Add two operators
- `sub(A, B)`: Subtract operators
- `mul(scalar, A)`: Scalar multiplication
- `matmul(A, B)`: Matrix multiplication
- `neg(A)`: Negate operator
- `div(A, B)`: Division (for diagonal operators)

### Property Checks
- `is_square(A)`: Check if operator is square
- `is_symmetric(A)`: Check symmetry without densification (randomized)
- `is_hermitian(A)`: Check Hermitian property without densification (randomized)

### Utilities
- `todense(A)`: Convert to dense array
- `allclose(A, B)`: Compare operators
- `set_debug(enabled)`: Enable/disable densification warnings
- `is_debug()`: Check debug mode status

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
from linox import Matrix, Diagonal, BlockMatrix, inverse, solve, det

# Create operators
A = Matrix(jnp.array([[1, 2], [3, 4]], dtype=jnp.float32))
D = Diagonal(jnp.array([1, 2], dtype=jnp.float32))

# Compose operators
B = BlockMatrix([[A, D], [D, A]])

# Apply to vector
x = jnp.ones((4,), dtype=jnp.float32)
y = B @ x  # Lazy evaluation

# Solve linear system
b = jnp.ones((4,), dtype=jnp.float32)
x_solved = solve(B, b)

# Compute inverse and determinant
B_inv = inverse(B)
det_B = det(B)

# Parallelize over batch of vectors
x_batched = jnp.ones((10, 4), dtype=jnp.float32)
y_batched = jax.vmap(B)(x_batched)
```

## Gaussian Process Operator (Matrix‑Free, Kronecker Structured)

Linox makes it easy to build Gaussian Process (GP) operators that factorize across
function and spatial dimensions. This leverages Kronecker structure and preserves
matrix‑free behavior, so you can compose large kernels without materializing
massive dense arrays.

Example: a modular GP prior with a function kernel ⊗ spatial kernel

```python
import jax
import jax.numpy as jnp
from helper.new_gp import (
    CombinationConfig,
    DimensionSpec,
    ModularGPPrior,
    StructureConfig,
    params_from_structure,
)
from helper.gp import KernelType, CombinationStrategy

# Enable double precision for numerical stability (optional)
jax.config.update("jax_enable_x64", True)

# 2D setup (one function dim u, two spatial dims x,y)
structure = StructureConfig(
    spatial_dims=[
        DimensionSpec(name="x", kernel_type=KernelType.RBF),
        DimensionSpec(name="y", kernel_type=KernelType.RBF),
    ],
    function_dims=[DimensionSpec(name="u", kernel_type=KernelType.L2)],
)
combo = CombinationConfig(strategy=CombinationStrategy.ADDITIVE, output_scale=1.0)
prior = ModularGPPrior(structure, combo)
params = params_from_structure(structure)

# Training data (N_train functions, evaluated on an (nx, ny) grid)
N_train, N_test = 25, 3
nx, ny = 15, 15
nx_plot, ny_plot = 25, 25

# See helper.plotting.generate_preprocess_data_2d for data creation
from helper.plotting import generate_preprocess_data_2d
(
    operator_inputs,         # (N_train, nx, ny)
    spatial_inputs,          # (nx, ny, 2)
    outputs,                 # (N_train * nx * ny,)
    operator_inputs_test,    # (N_test, nx, ny)
    spatial_inputs_test,     # (nx, ny, 2)
    outputs_test,            # (N_test * nx * ny,)
    spatial_inputs_plot,     # (nx_plot, ny_plot, 2)
) = generate_preprocess_data_2d(
    x_range=(0.0, jnp.pi), y_range=(0.0, jnp.pi),
    nx=nx, ny=ny, T=0.1, alpha=0.5,
    N_train=N_train, N_test=N_test,
    nx_plot=nx_plot, ny_plot=ny_plot,
)

# Build the Kronecker‑structured kernel and run predictions
pred_mean_flat, pred_cov = prior.predict(
    operator_inputs,
    outputs,
    spatial_inputs,
    operator_inputs_test,
    spatial_inputs_plot,
    params,
)

# pred_mean_flat has shape (N_test * nx_plot * ny_plot,)
# pred_cov is a LinearOperator (matrix‑free) you can densify only for plotting
```

Why this is fast and memory‑efficient
- Kronecker structure: The prior kernel is built as `K_function ⊗ K_spatial`, using
  `linox.Kronecker`, so large grids are handled as compositions rather than dense
  matrices.
- Matrix‑free algebra: Solves and products are done via LinearOperators (e.g.,
  `IsotropicAdditiveLinearOperator`, `linverse`, `lsolve`) without forming dense blocks.
- Lazy properties: Many operations (like `diagonal`) propagate into factors and avoid
  densification unless explicitly required (see “Densification Warnings”).

Illustrative outputs (2D heat‑equation demo)

![GP Predictions (2D)](examples/osci_gp_operator_walkthrough_predictions.png)
![Posterior Uncertainty (2D)](examples/osci_gp_operator_walkthrough_uncertainty.png)

See the example notebook for a walkthrough: `examples/gp_operator_walkthrough.ipynb`.

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
aren't accidentally densified.

## Related Work & Citations

### matfree
`linox` draws inspiration from and complements [`matfree`](https://github.com/pnkraemer/matfree) by Nicholas Krämer, which provides matrix-free linear algebra methods in JAX including randomized and deterministic methods for trace estimation, functions of matrices, and matrix factorizations.

If you use matrix-free methods or differentiable linear algebra iterations in your work, consider citing the matfree library:

**For differentiable Lanczos or Arnoldi iterations:**
```bibtex
@article{kraemer2024gradients,
  title={Gradients of functions of large matrices},
  author={Krämer, Nicholas and Moreno-Muñoz, Pablo and Roy, Hrittik and Hauberg, Søren},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={49484--49518},
  year={2024}
}
```

**For differentiable LSMR implementation:**
```bibtex
@article{roy2025matrix,
  title={Matrix-Free Least Squares Solvers: Values, Gradients, and What to Do With Them},
  author={Roy, Hrittik and Hauberg, Søren and Krämer, Nicholas},
  journal={arXiv preprint arXiv:2510.19634},
  year={2025}
}
```

### Other JAX Linear Algebra Libraries
- [`probnum.linops`](https://probnum.readthedocs.io/en/latest/api/linops.html): The original inspiration for linox, providing linear operators in Python/NumPy
- [`matfree`](https://pnkraemer.github.io/matfree/): Specialized matrix-free methods for large-scale problems

## Installation

```bash
pip install linox
```

Or install from source:
```bash
git clone https://github.com/2bys/linox.git
cd linox
pip install -e .
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues on the [GitHub repository](https://github.com/2bys/linox).

## License

This project is licensed under the MIT License.
