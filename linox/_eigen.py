# _eigen.py

r"""Linear operator for representing an eigenvalue decomposition.

This module includes a linear operator that represents an eigenvalue decomposition
of an array.

- :class:`EigenD`: Represents a linear operator :math:`A` in its eigenvalue
    decomposition form :math:`A = U \Lambda U^T` where :math:`U` is orthogonal
    and :math:`\Lambda` is diagonal
"""

import jax
from jax import numpy as jnp

import linox as lo
from linox._arithmetic import leigh, lexp, llog, lpow, ltrace
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Matrix
from linox.typing import ArrayLike, DTypeLike, ScalarLike, ScalarType, ShapeLike
from linox.utils import as_linop, as_shape


class EigenD(LinearOperator):
    r"""A linear operator representing an eigenvalue decomposition.
    Represents a linear operator :math:`A` in its eigenvalue decomposition form
    :math:`A = U \Lambda U^T` where :math:`U` is orthogonal and :math:`\Lambda`
    is diagonal.
    Args:
        A: The square matrix to decompose.
    """  # noqa: D205

    def __init__(self, A: ArrayLike) -> None:
        self.A = as_linop(A)
        self._S = None
        self._Q = None
        super().__init__(shape=(A.shape[0], A.shape[0]), dtype=A.dtype)

    def _ensure_eigh(self) -> None:
        if (self._S is None) or (self._Q is None):
            self._S, self._Q = leigh(self.A)

    @property
    def U(self) -> LinearOperator:
        self._ensure_eigh()
        return self._Q

    @property
    def S(self) -> Diagonal:
        self._ensure_eigh()
        return self._S

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self.U @ (self.S[:, None] * (self.U.T @ vec))

    def tree_flatten(self) -> tuple[tuple[any, ...], dict[str, any]]:
        children = (self.U, self.S)
        aux_data = {}
        return children, aux_data

    def todense(self) -> jax.Array:
        """Convert the linear operator to a dense matrix."""
        U = self.U.todense()
        S = self.S.todense()
        eigvals, eigvecs = jnp.linalg.eigh(U @ jnp.diag(S) @ U.T)
        return eigvals, eigvecs

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, any],
        children: tuple[any, ...],
    ) -> "EigenD":
        del aux_data
        U, S = children
        return cls(U=U, S=S)


@leigh.dispatch
def _(a: EigenD) -> tuple[LinearOperator, LinearOperator]:
    return a.eigvals, a.eigvecs


# New matrix-free function dispatches for EigenD
@ltrace.dispatch
def _(a: EigenD, key: jax.Array | None = None, num_samples: int = 100, distribution: str = "rademacher") -> tuple[jax.Array, jax.Array]:
    """Exact trace from eigenvalues: trace(A) = sum(λ)."""
    trace_value = jnp.sum(a.S)
    trace_std = jnp.array(0.0, dtype=a.dtype)  # Exact computation
    return trace_value, trace_std


@lexp.dispatch
def _(a: EigenD, v: jax.Array | None = None, num_iters: int = 20, method: str = "lanczos") -> jax.Array | LinearOperator:
    """Matrix exponential from eigendecomposition: exp(UΛU^T) = U exp(Λ) U^T."""
    if v is None:
        # Return lazy operator: U @ Diagonal(exp(λ)) @ U^T
        exp_S = Diagonal(jnp.exp(a.S))
        # Construct via congruence transform would be ideal, but for simplicity:
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.U, exp_S)
    else:
        # exp(A) @ v = U @ exp(Λ) @ U^T @ v
        return a.U @ (jnp.exp(a.S) * (a.U.T @ v))


@llog.dispatch
def _(a: EigenD, v: jax.Array | None = None, num_iters: int = 20, method: str = "lanczos") -> jax.Array | LinearOperator:
    """Matrix logarithm from eigendecomposition: log(UΛU^T) = U log(Λ) U^T."""
    if v is None:
        # Return lazy operator: U @ Diagonal(log(λ)) @ U^T
        log_S = Diagonal(jnp.log(a.S))
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.U, log_S)
    else:
        # log(A) @ v = U @ log(Λ) @ U^T @ v
        return a.U @ (jnp.log(a.S) * (a.U.T @ v))


@lpow.dispatch
def _(a: EigenD, *, power: float, v: jax.Array | None = None, num_iters: int = 20, method: str = "lanczos") -> jax.Array | LinearOperator:
    """Matrix power from eigendecomposition: (UΛU^T)^p = U Λ^p U^T."""
    if v is None:
        # Return lazy operator: U @ Diagonal(λ^p) @ U^T
        pow_S = Diagonal(a.S ** power)
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.U, pow_S)
    else:
        # A^p @ v = U @ Λ^p @ U^T @ v
        return a.U @ ((a.S ** power) * (a.U.T @ v))


# Register EigenD as a PyTree
jax.tree_util.register_pytree_node_class(EigenD)


# A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
# A = A @ A.T + jnp.eye(3) * 1e-6
# print("Original matrix:\n", A)
# op = EigenD(A)
# print("Dense matrix:\n", op.todense())
# print("Shape:", op.shape)
