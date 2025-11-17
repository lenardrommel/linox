"""Matrix-free algorithms for large-scale linear algebra.

This module provides a collection of iterative and stochastic algorithms for
working with large linear operators without explicit matrix construction.

The implementations are inspired by the matfree library
(https://github.com/pnkraemer/matfree) by Nicholas Kraemer et al., with
adaptations for linox's LinearOperator abstraction.

Submodules
----------
_lanczos_arnoldi
    Lanczos tridiagonalization and Arnoldi iteration for eigenvalue problems
_trace
    Stochastic trace estimation using Hutchinson's method
_matrix_functions
    Matrix function approximations using Krylov methods
_toeplitz
    Toeplitz system solvers (existing)
_lanzcos
    Low-rank inverse factors via CG/Lanczos (existing)

Exported Functions
------------------

Eigenvalue & Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~
lanczos_tridiag : Lanczos tridiagonalization for symmetric operators
arnoldi_iteration : Arnoldi iteration for general operators
lanczos_eigh : Compute few eigenvalues using Lanczos

Trace Estimation
~~~~~~~~~~~~~~~~
hutchinson_trace : Stochastic trace estimation
hutchinson_diagonal : Stochastic diagonal estimation
hutchinson_trace_and_diagonal : Joint trace and diagonal estimation

Linear Solvers
~~~~~~~~~~~~~~
lanczos_solve_sqrt : Low-rank inverse factors (existing)

Matrix Functions
~~~~~~~~~~~~~~~~
lanczos_matrix_function : f(A)v using Lanczos for symmetric A
arnoldi_matrix_function : f(A)v using Arnoldi for general A
stochastic_lanczos_quadrature : Estimate trace(f(A)) using SLQ

Toeplitz Solvers
~~~~~~~~~~~~~~~~
solve_toeplitz_jax : JAX-compatible Toeplitz solver (existing)
levinson : Levinson recursion for Toeplitz systems (existing)

References
----------
.. [1] N. Kraemer, M. Schober, and P. Hennig, "Gradients of functions of large matrices,"
       arXiv preprint arXiv:2405.17277, 2024.

.. [2] A. Roy, N. Kraemer, V. De Bortoli, and A. Doucet, "Gradients of Stochastic
       Trace Estimators via Differentiable Matrix-Free Linear Solvers," arXiv, 2025.

.. [3] matfree: Matrix-free linear algebra in JAX.
       https://github.com/pnkraemer/matfree
       Citation: Please cite the matfree library and relevant papers when using
       algorithms inspired by their work.

Examples
--------
Eigenvalue computation with Lanczos:

>>> from linox import Matrix
>>> import jax.numpy as jnp
>>> from linox._algorithms import lanczos_eigh
>>> A = Matrix(jnp.diag(jnp.arange(1.0, 101.0)))
>>> v0 = jnp.ones(100)
>>> eigs, vecs = lanczos_eigh(A, v0, num_iters=20, k=5)

Trace estimation with Hutchinson:

>>> import jax
>>> from linox._algorithms import hutchinson_trace
>>> key = jax.random.PRNGKey(0)
>>> trace_est, trace_std = hutchinson_trace(A, key, num_samples=100)

Matrix exponential:

>>> from linox._algorithms import lanczos_matrix_function
>>> v = jnp.ones(100)
>>> exp_Av = lanczos_matrix_function(A, v, jnp.exp, num_iters=20)

Log-determinant estimation:

>>> from linox._algorithms import stochastic_lanczos_quadrature
>>> logdet_est, logdet_std = stochastic_lanczos_quadrature(
...     A, jnp.log, key, num_samples=50, num_iters=20
... )
"""

# Lanczos and Arnoldi
from linox._algorithms._lanczos_arnoldi import (
    arnoldi_iteration,
    lanczos_eigh,
    lanczos_tridiag,
)

# Trace estimation
from linox._algorithms._trace import (
    hutchinson_diagonal,
    hutchinson_trace,
    hutchinson_trace_and_diagonal,
)

# Matrix functions
from linox._algorithms._matrix_functions import (
    arnoldi_matrix_function,
    lanczos_matrix_function,
    stochastic_lanczos_quadrature,
)

# Existing algorithms
from linox._algorithms._lanzcos import lanczos_solve_sqrt
from linox._algorithms._toeplitz import levinson, solve_toeplitz_jax

__all__ = [
    # Eigenvalue & decomposition
    "lanczos_tridiag",
    "arnoldi_iteration",
    "lanczos_eigh",
    # Trace estimation
    "hutchinson_trace",
    "hutchinson_diagonal",
    "hutchinson_trace_and_diagonal",
    # Linear solvers
    "lanczos_solve_sqrt",
    # Matrix functions
    "lanczos_matrix_function",
    "arnoldi_matrix_function",
    "stochastic_lanczos_quadrature",
    # Toeplitz
    "solve_toeplitz_jax",
    "levinson",
]
