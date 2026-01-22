# _eigen.py

r"""Linear operator for representing an eigenvalue decomposition.

This module provides:

- :class:`EigenD`: Represents a symmetric linear operator in its eigenvalue
    decomposition form :math:`A = Q \Lambda Q^T` where :math:`Q` is orthogonal
    and :math:`\Lambda` is diagonal. Both Q and Lambda are LinearOperators.
"""

import jax
import jax.numpy as jnp

from linox._arithmetic import diagonal, lcholesky, leigh, linverse, lpinverse, lsqrt
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal
from linox.utils import as_linop


class EigenD(LinearOperator):
    r"""A linear operator representing an eigenvalue decomposition.

    Represents :math:`A = Q \Lambda Q^T` where :math:`Q` is orthogonal and
    :math:`\Lambda` is diagonal. Both Q and Lambda are stored as LinearOperators,
    enabling structured representations (e.g., Kronecker products of eigenvectors).

    Args:
        Q: Orthogonal matrix of eigenvectors (LinearOperator or array)
        Lambda: Diagonal matrix of eigenvalues (LinearOperator or array)

    Example:
        >>> import jax.numpy as jnp
        >>> from linox import Matrix, Diagonal, leigh
        >>> A = Matrix(jnp.diag(jnp.array([1.0, 2.0, 3.0])))
        >>> lam, Q = leigh(A)
        >>> eigend = EigenD(Q, Diagonal(lam))
        >>> # eigend represents A = Q @ diag(lam) @ Q.T
    """

    def __init__(
        self,
        Q: LinearOperator | jax.Array,
        Lambda: LinearOperator | jax.Array,
    ) -> None:
        self._Q = as_linop(Q)
        if isinstance(Lambda, jax.Array) and Lambda.ndim == 1:
            self._Lambda = Diagonal(Lambda)
        else:
            self._Lambda = as_linop(Lambda)

        n = self._Q.shape[0]
        super().__init__(shape=(n, n), dtype=self._Q.dtype)

    @property
    def Q(self) -> LinearOperator:
        return self._Q

    @property
    def Lambda(self) -> LinearOperator:
        return self._Lambda

    @property
    def eigenvalues(self) -> jax.Array:
        return diagonal(self._Lambda)

    def _matmul(self, vec: jax.Array) -> jax.Array:
        return self._Q @ (self._Lambda @ (self._Q.T @ vec))

    def _todense(self) -> jax.Array:
        Q_dense = self._Q._todense()
        lam = self.eigenvalues
        return Q_dense @ (lam[:, None] * Q_dense.T)

    def transpose(self) -> "EigenD":
        return self

    def tree_flatten(self) -> tuple[tuple, dict]:
        children = (self._Q, self._Lambda)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict,
        children: tuple,
    ) -> "EigenD":
        Q, Lambda = children
        return cls(Q=Q, Lambda=Lambda)


@leigh.dispatch
def _(a: EigenD) -> tuple[jax.Array, LinearOperator]:
    return a.eigenvalues, a.Q


@linverse.dispatch
def _(a: EigenD) -> EigenD:
    inv_lam = 1.0 / a.eigenvalues
    return EigenD(a.Q, Diagonal(inv_lam))


@lpinverse.dispatch
def _(a: EigenD, tol: float = 1e-12) -> EigenD:
    lam = a.eigenvalues
    inv_lam = jnp.where(jnp.abs(lam) > tol, 1.0 / lam, 0.0)
    return EigenD(a.Q, Diagonal(inv_lam))


@lsqrt.dispatch
def _(a: EigenD) -> EigenD:
    sqrt_lam = jnp.sqrt(a.eigenvalues)
    return EigenD(a.Q, Diagonal(sqrt_lam))


@lcholesky.dispatch
def _(a: EigenD) -> LinearOperator:
    sqrt_lam = jnp.sqrt(a.eigenvalues)
    return a.Q @ Diagonal(sqrt_lam)


@diagonal.dispatch
def _(a: EigenD) -> jax.Array:
    Q_dense = a.Q._todense()
    lam = a.eigenvalues
    return jnp.sum(Q_dense**2 * lam[None, :], axis=1)


jax.tree_util.register_pytree_node_class(EigenD)
