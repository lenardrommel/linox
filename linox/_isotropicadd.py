# _isotropicadd.py

import jax
import jax.numpy as jnp

from linox._arithmetic import (
    AddLinearOperator,
    ScaledLinearOperator,
    lcholesky,
    leigh,
    linverse,
    lpinverse,
)
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Identity

jax.config.update("jax_enable_x64", True)


class IsotropicAdditiveLinearOperator(AddLinearOperator):
    """Isotropic additive linear operator.

    Represents a linear operator of the form A = s * I + A where A is a
    linear operator and I is the identity matrix.

    Args:
        s: Scalar value to be added to the diagonal.
        A: Linear operator to be added.
    """

    def __init__(self, s: jax.Array, A: LinearOperator) -> None:
        self._A = A
        if A.shape[-1] != A.shape[-2]:
            msg = "A must be a square matrix."
            raise ValueError(msg)
        self._s = ScaledLinearOperator(
            Identity(self._A.shape[0], dtype=self._A.dtype), s
        )
        self._Q = None
        self._S = None
        self._projector = None
        self._complement = None
        super().__init__(self._s, self._A)

    def _ensure_eigh(self) -> None:
        if (self._S is None) or (self._Q is None):
            self._S, self._Q = leigh(self._A)
            # invalidate derived caches
            self._projector = None
            self._complement = None

    def _invalidate_cache(self) -> None:
        self._Q = self._S = self._projector = self._complement = None

    @property
    def s(self) -> jax.Array:
        return self._s

    @property
    def shape(self) -> tuple[int, int]:
        return self._A.shape

    @property
    def operator(self) -> LinearOperator:
        return self._A

    @property
    def Q(self) -> LinearOperator:
        self._ensure_eigh()
        return self._Q

    @property
    def S(self) -> LinearOperator:
        self._ensure_eigh()
        return self._S

    @property
    def projector(self) -> LinearOperator:
        self._ensure_eigh()
        if self._projector is None:
            self._projector = self._Q @ self._Q.T
        return self._projector

    @property
    def complement(self) -> LinearOperator:
        self._ensure_eigh()
        if self._complement is None:
            self._complement = (
                Identity(self.shape[0], dtype=self._A.dtype) - self.projector
            )
        return self._complement

    def _matmul(self, arr: jax.Array):  # noqa: ANN202
        return self._s @ arr + self._A @ arr

    def todense(self) -> jax.Array:
        return self._s.todense() + self._A.todense()


@lcholesky.dispatch
def _(a: IsotropicAdditiveLinearOperator, jitter: float = 1e-10) -> jax.Array:
    return jnp.linalg.cholesky(a.todense() + jitter * jnp.eye(a.shape[0]))


# we need a log-determinant function here
# TODO: Implement lsolve for IsotropicAdditiveLinearOperator via eigendecomposition
# A \kron B + s I = (Q_A \kron Q_B) (Lambda_A \kron Lambda_B + s I) (Q_A \kron Q_B)^T


@linverse.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    lam = Diagonal(S)
    inv_iso = linverse(a.s)

    D = Diagonal(lam.diag / (s * (lam.diag + s)))

    return inv_iso - (Q @ D @ Q.T)


@lpinverse.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    lam = Diagonal(S)
    inv_iso = lpinverse(a.s)

    D = Diagonal(lam.diag / (s * (lam.diag + s)))

    return inv_iso - (Q @ D @ Q.T)
