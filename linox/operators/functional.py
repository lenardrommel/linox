
from collections.abc import Callable

import jax
import jax.numpy as jnp

from linox.linalg.approx.arnoldi import arnoldi_matrix_function
from linox.linalg.approx.lanczos import lanczos_matrix_function
from linox.operators.base import LinearOperator


class MatrixFunctionLinearOperator(LinearOperator):
    """Lazy operator representing f(A).

    Computes f(A)v using Krylov subspace methods (Lanczos or Arnoldi)
    without forming f(A) explicitly.
    """

    def __init__(
        self,
        operator: LinearOperator,
        func: Callable[[jax.Array], jax.Array],
        method: str = "auto",
        num_iters: int = 20,
        dtype=None,
    ) -> None:
        self.operator = operator
        self.func = func
        self.method = method
        self.num_iters = num_iters

        if dtype is None:
            dtype = operator.dtype

        super().__init__(shape=operator.shape, dtype=dtype)

    def _matmul(self, x: jax.Array) -> jax.Array:
        # Determine method if auto
        method = self.method
        if method == "auto":
            # Use lanczos if symmetric, else arnoldi
            # Assuming is_symmetric property exists and is populated
            method = "lanczos" if getattr(self.operator, "is_symmetric", False) else "arnoldi"

        if method == "lanczos":
            # Batched handling? lanczos_matrix_function expects vector v.
            # If x is matrix (n, k), map over columns.
            if x.ndim > 1 and x.shape[1] > 1:
                return jax.vmap(lambda col: lanczos_matrix_function(
                    self.operator, col, self.func, self.num_iters
                ), in_axes=1, out_axes=1)(x)

            return lanczos_matrix_function(
                self.operator, x.squeeze(), self.func, self.num_iters
            ).reshape(x.shape)

        if method == "arnoldi":
            if x.ndim > 1 and x.shape[1] > 1:
                return jax.vmap(lambda col: arnoldi_matrix_function(
                    self.operator, col, self.func, self.num_iters
                ), in_axes=1, out_axes=1)(x)

            return arnoldi_matrix_function(
                self.operator, x.squeeze(), self.func, self.num_iters
            ).reshape(x.shape)

        msg = f"Unknown MatrixFunction method: {method}"
        raise ValueError(msg)

    def _todense(self) -> jax.Array:
        # Fallback to dense evaluation
        A_dense = self.operator.todense()
        # For symmetric A, use eigh
        if getattr(self.operator, "is_symmetric", False):
             w, V = jax.scipy.linalg.eigh(A_dense)
             return V @ jnp.diag(self.func(w)) @ V.T

        # General case (requires scipy on CPU usually, or approx)
        # JAX doesn't have partial funm generically for dense except via eig/eigh
        # jax.scipy.linalg.expm exists, but not logm/sqrtm in JAX core cleanly on all backends?
        # We'll use diagonalization assumption for now.
        w, V = jax.linalg.eig(A_dense)
        return V @ jnp.diag(self.func(w)) @ jnp.linalg.inv(V)

    def _transpose(self) -> LinearOperator:
        # f(A)^T = f(A^T) ?
        # If A is symmetric, A=A^T, f(A) symmetric.
        # If A normal, yes.
        # Approximation: return MatrixFunction(A.T, func)
        return MatrixFunctionLinearOperator(
            self.operator.T, self.func, self.method, self.num_iters, self.dtype
        )
