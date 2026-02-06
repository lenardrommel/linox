# Linox Algorithms Module

Matrix-free algorithms for large-scale linear algebra operations in JAX.

This module provides a comprehensive collection of iterative and stochastic algorithms inspired by the [matfree](https://github.com/pnkraemer/matfree) library by Nicholas Krämer et al., adapted for linox's LinearOperator abstraction.

## Overview

The algorithms in this module enable efficient computation with large linear operators without explicit matrix construction or densification. They are particularly useful for:

- **Gaussian Process inference**: Log-determinant and trace estimation for GP marginal likelihood
- **Large-scale eigenvalue problems**: Computing a few eigenvalues/eigenvectors without full decomposition
- **Matrix function applications**: Computing f(A)v for functions like exp, log, sqrt
- **Iterative linear solvers**: Solving large sparse or structured linear systems
- **Stochastic estimation**: Monte Carlo methods for traces and diagonals

## Implemented Algorithms

### Phase 1: Core Iterative Methods (High Priority) ✅

#### 1. Lanczos and Arnoldi Iterations

**Lanczos Tridiagonalization** (`lanczos_tridiag`)
- Reduces symmetric operators to tridiagonal form using Krylov subspaces
- Foundation for efficient eigenvalue computation and matrix functions
- Optional full reorthogonalization for numerical stability

```python
from linox._algorithms import lanczos_tridiag
import jax.numpy as jnp
from linox import Matrix

A = Matrix(symmetric_matrix)
v0 = jnp.ones(n)
Q, alpha, beta = lanczos_tridiag(A, v0, num_iters=50, reortho=True)
# Q: orthonormal Lanczos vectors
# alpha, beta: tridiagonal matrix elements
```

**Arnoldi Iteration** (`arnoldi_iteration`)
- Generalizes Lanczos to non-symmetric operators
- Produces Hessenberg reduction
- Used in GMRES and general eigenvalue problems

```python
from linox._algorithms import arnoldi_iteration

A = Matrix(general_matrix)
Q, H = arnoldi_iteration(A, v0, num_iters=30)
# Q: orthonormal Arnoldi vectors
# H: upper Hessenberg matrix
```

**Lanczos Eigenvalue Solver** (`lanczos_eigh`)
- Computes k largest/smallest eigenvalues using Lanczos method
- Much faster than full eigendecomposition when k << n
- Integrated with linox dispatch system via `leigh`

```python
from linox._algorithms import lanczos_eigh

eigs, vecs = lanczos_eigh(A, v0, num_iters=100, k=10, which='LA')
# Returns top 10 eigenvalues and eigenvectors
```

#### 2. Hutchinson Trace Estimation

**Stochastic Trace Estimation** (`hutchinson_trace`)
- Monte Carlo estimation of trace(A) using random test vectors
- Rademacher (±1) or Gaussian sampling
- Unbiased estimator with O(1/√num_samples) convergence

```python
from linox._algorithms import hutchinson_trace
import jax

key = jax.random.PRNGKey(0)
trace_est, trace_std = hutchinson_trace(A, key, num_samples=100)
```

**Diagonal Estimation** (`hutchinson_diagonal`)
- Stochastic estimation of diag(A) without densification
- Useful when diagonal is not directly accessible

**Joint Estimation** (`hutchinson_trace_and_diagonal`)
- Efficiently computes both estimates using same samples
- ~2x faster than separate calls

#### 3. LSMR Solver

**Least Squares Minimal Residual** (`lsmr_solve`)
- Iterative solver for least-squares problems: min ||Ax - b||₂
- Handles over-determined, under-determined, and rank-deficient systems
- Matrix-free: only requires A @ v and A.T @ v operations
- Tikhonov regularization via damping parameter

```python
from linox._algorithms import lsmr_solve

x, info = lsmr_solve(A, b, atol=1e-6, btol=1e-6, maxiter=1000, damp=0.01)
print(f"Converged in {info['itn']} iterations")
print(f"Residual norm: {info['normr']}")
```

### Phase 2: Matrix Functions (Medium Priority) ✅

#### 4. Matrix Function Approximations

**Lanczos-based Matrix Functions** (`lanczos_matrix_function`)
- Computes f(A)v for symmetric operators using Lanczos approximation
- Works with any scalar function: exp, log, sqrt, custom functions
- Avoids forming full matrix function f(A)

```python
from linox._algorithms import lanczos_matrix_function
import jax.numpy as jnp

# Matrix exponential
exp_Av = lanczos_matrix_function(A, v, jnp.exp, num_iters=20)

# Matrix logarithm
log_Av = lanczos_matrix_function(A, v, jnp.log, num_iters=20)

# Custom function
custom_func = lambda M: jnp.linalg.matrix_power(M, 0.5)
sqrt_Av = lanczos_matrix_function(A, v, custom_func, num_iters=20)
```

**Arnoldi-based Matrix Functions** (`arnoldi_matrix_function`)
- Generalizes matrix functions to non-symmetric operators
- Uses Hessenberg reduction instead of tridiagonalization

**Chebyshev Polynomial Approximation** (`chebyshev_matrix_function`)
- Alternative method using Chebyshev expansion
- Efficient for smooth functions on known spectral intervals
- Particularly good when same function applied to multiple vectors

```python
from linox._algorithms import chebyshev_matrix_function

# Spectrum of A in [-1, 1]
result = chebyshev_matrix_function(A, v, jnp.exp, num_terms=30, bounds=(-1, 1))
```

#### 5. Stochastic Lanczos Quadrature

**Trace of Matrix Functions** (`stochastic_lanczos_quadrature`)
- Estimates trace(f(A)) using combined Hutchinson + Lanczos
- **Critical for GP inference**: log|K| estimation without forming K
- Combines stochastic sampling with deterministic Krylov approximation

```python
from linox._algorithms import stochastic_lanczos_quadrature
import jax

# Estimate log-determinant: log|A| = trace(log(A))
key = jax.random.PRNGKey(0)
logdet_est, logdet_std = stochastic_lanczos_quadrature(
    A, jnp.log, key, num_samples=100, num_iters=30
)

# Estimate trace(exp(A))
trace_exp, std_exp = stochastic_lanczos_quadrature(
    A, jnp.exp, key, num_samples=50, num_iters=20
)
```

### Phase 3: Advanced Decompositions ✅

Tridiagonalization and bidiagonalization are implemented as core components of Lanczos and Arnoldi iterations.

## Integration with Linox Arithmetic

New dispatched functions have been added to `linox._arithmetic`:

### `ltrace` - Trace Estimation

```python
import linox
import jax

A = linox.Matrix(large_matrix)
key = jax.random.PRNGKey(0)
trace_est, trace_std = linox.ltrace(A, key=key, num_samples=200)
```

### `lexp`, `llog`, `lpow` - Matrix Functions

```python
import linox
import jax.numpy as jnp

# Matrix exponential
result = linox.lexp(A, v=v, num_iters=20, method='lanczos')

# Matrix logarithm
result = linox.llog(A, v=v, num_iters=20, method='lanczos')

# Matrix power
result = linox.lpow(A, power=0.5, v=v, num_iters=20, method='lanczos')
```

## Usage Examples

### Example 1: GP Log-Determinant Estimation

```python
import jax
import jax.numpy as jnp
import linox
from linox._algorithms import stochastic_lanczos_quadrature

# Large GP covariance matrix (structured, don't densify!)
K = linox.some_gp_kernel_operator(X, lengthscale=1.0)

# Estimate log|K| for GP marginal likelihood
key = jax.random.PRNGKey(42)
logdet_K, logdet_std = stochastic_lanczos_quadrature(
    K, jnp.log, key, num_samples=100, num_iters=50
)

print(f"log|K| ≈ {logdet_K:.2f} ± {logdet_std:.2f}")
```

### Example 2: Few Eigenvalues of Large Matrix

```python
from linox._algorithms import lanczos_eigh
import jax.numpy as jnp

# Large symmetric operator
A = linox.SomeStructuredOperator(params)

# Compute top 10 eigenvalues
v0 = jax.random.normal(jax.random.PRNGKey(0), (n,))
eigs, vecs = lanczos_eigh(A, v0, num_iters=100, k=10, which='LA')

print(f"Top eigenvalue: {eigs[0]:.4f}")
```

### Example 3: Matrix Exponential for ODEs

```python
from linox._algorithms import lanczos_matrix_function
import jax.numpy as jnp

# Solve du/dt = -A u with u(0) = u0
# Solution: u(t) = exp(-tA) u0
A = linox.Matrix(laplacian_operator)
u0 = initial_condition
t = 0.1

# Compute u(t) = exp(-tA) u0 without forming exp(-tA)
A_scaled = linox.Matrix(-t * A.todense())  # Scale by -t
u_t = lanczos_matrix_function(A_scaled, u0, jnp.exp, num_iters=30)
```

### Example 4: Large Least-Squares Problem

```python
from linox._algorithms import lsmr_solve

# Overdetermined system: min ||Ax - b||
A = linox.SomeLargeOperator(m=10000, n=5000)
b = observations

# Iterative solve with regularization
x, info = lsmr_solve(A, b, atol=1e-6, btol=1e-6, damp=0.01, maxiter=500)

print(f"Converged: {info['istop'] in [1,2,3]}")
print(f"Iterations: {info['itn']}")
print(f"Residual: {info['normr']:.6f}")
```

## Algorithm Comparison

| Algorithm | Use Case | When to Use | Complexity per Iteration |
|-----------|----------|-------------|-------------------------|
| **Lanczos** | Symmetric eigenvalues | k << n eigenvalues needed | O(n) per iter + matvec |
| **Arnoldi** | General eigenvalues | Non-symmetric operators | O(n·k) per iter + matvec |
| **Hutchinson** | Trace estimation | Large operators, can't compute trace directly | O(n) per sample |
| **LSMR** | Least-squares | Large sparse/structured systems | O(n) per iter + matvec + A^T matvec |
| **Lanczos f(A)v** | Matrix functions | Symmetric A, single vector | O(n) per iter + matvec |
| **SLQ** | trace(f(A)) | GP log-determinant, large operators | O(n·k·s) for k iters, s samples |

## Performance Tips

1. **Hutchinson sampling**: Use Rademacher (±1) distribution for minimal variance
2. **Lanczos iterations**: Increase `num_iters` for better accuracy (typically 20-100)
3. **Reorthogonalization**: Enable for `num_iters > 50` to maintain numerical stability
4. **LSMR**: Start with generous `maxiter`, tune tolerances based on problem
5. **SLQ**: Balance `num_samples` (reduces stochastic error) vs `num_iters` (reduces bias)

## JAX Compatibility

All algorithms are:
- **JIT-compilable**: Use `@jax.jit` for performance
- **Differentiable**: Support reverse-mode autodiff (where applicable)
- **Vectorizable**: Use `jax.vmap` for batching
- **PyTree-compatible**: Work with JAX transformations

```python
import jax

# JIT compilation
@jax.jit
def traced_logdet(K, key):
    return stochastic_lanczos_quadrature(K, jnp.log, key, num_samples=50, num_iters=20)

# Vectorization
keys = jax.random.split(jax.random.PRNGKey(0), 10)
logdets = jax.vmap(lambda k: traced_logdet(K, k))(keys)
```

## References and Citations

This module is inspired by and draws heavily from the [matfree library](https://github.com/pnkraemer/matfree):

**Please cite matfree when using these algorithms:**

```bibtex
@software{matfree2024,
  author = {Krämer, Nicholas and contributors},
  title = {matfree: Matrix-free linear algebra in JAX},
  year = {2024},
  url = {https://github.com/pnkraemer/matfree}
}
```

**Key papers for differentiable Lanczos/Arnoldi:**

```bibtex
@article{kramer2024gradients,
  title={Gradients of functions of large matrices},
  author={Krämer, Nicholas and Schober, Michael and Hennig, Philipp},
  journal={arXiv preprint arXiv:2405.17277},
  year={2024}
}
```

**Key papers for LSMR:**

```bibtex
@article{fong2011lsmr,
  title={LSMR: An iterative algorithm for sparse least-squares problems},
  author={Fong, David C-L and Saunders, Michael A},
  journal={SIAM Journal on Scientific Computing},
  volume={33},
  number={5},
  pages={2950--2971},
  year={2011}
}

@article{roy2025gradients,
  title={Gradients of Stochastic Trace Estimators via Differentiable Matrix-Free Linear Solvers},
  author={Roy, A and Krämer, N and De Bortoli, V and Doucet, A},
  journal={arXiv preprint},
  year={2025}
}
```

**Classic references:**

```bibtex
@article{hutchinson1990stochastic,
  title={A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines},
  author={Hutchinson, Michael F},
  journal={Communications in Statistics-Simulation and Computation},
  volume={19},
  number={2},
  pages={433--450},
  year={1990}
}

@book{saad2003iterative,
  title={Iterative methods for sparse linear systems},
  author={Saad, Yousef},
  year={2003},
  publisher={SIAM}
}

@book{higham2008functions,
  title={Functions of matrices: theory and computation},
  author={Higham, Nicholas J},
  year={2008},
  publisher={SIAM}
}
```

## Existing Algorithms

The following algorithms were already present in linox:

### `lanczos_solve_sqrt` (from `_lanzcos.py`)
- Builds low-rank inverse factor for PSD operators using CG/Lanczos
- Returns D where D @ D.T ≈ A^{-1}
- Useful for preconditioning and Kronecker products

### `solve_toeplitz_jax` (from `_toeplitz.py`)
- Hybrid SciPy+JAX solver for Toeplitz systems
- Uses Levinson recursion with custom VJP for differentiation
- Efficient O(n²) complexity instead of O(n³)

## Future Extensions

Potential additions for future releases:

1. **Conjugate Gradient (CG)**: For SPD systems
2. **GMRES**: For general nonsymmetric systems
3. **Preconditioners**: Incomplete Cholesky, multigrid
4. **Hessenberg factorization**: For general eigenvalue problems
5. **Adaptive methods**: Automatic iteration count selection
6. **Block methods**: Block Lanczos/Arnoldi for multiple eigenvalues

## Testing

Comprehensive tests are provided in `tests/test_algorithms.py` covering:
- Correctness against known solutions
- Orthogonality of basis vectors
- Convergence behavior
- Edge cases (identity, diagonal matrices)
- JAX transformation compatibility (JIT, vmap)

Run tests with:
```bash
pytest tests/test_algorithms.py -v
```

## Contributing

When adding new algorithms:

1. Follow the existing module structure
2. Add comprehensive docstrings with examples
3. Include references to source papers
4. Add tests in `tests/test_algorithms.py`
5. Update this README
6. Ensure JAX compatibility (JIT, autodiff, vmap)

## License

This implementation is part of linox and follows its license. The algorithms are inspired by matfree (Apache 2.0 license) with adaptations for linox's LinearOperator interface.
