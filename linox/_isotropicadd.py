# _isotropicadd.py

import jax
import jax.numpy as jnp

from linox import utils
from linox._arithmetic import (
    AddLinearOperator,
    ScaledLinearOperator,
    diagonal,
    lcholesky,
    leigh,
    lexp,
    linverse,
    llog,
    lpinverse,
    lpow,
    lsqrt,
    ltrace,
)
from linox._linear_operator import LinearOperator
from linox._matrix import Diagonal, Identity

jax.config.update("jax_enable_x64", True)


class IsotropicAdditiveLinearOperator(AddLinearOperator):
    r"""Isotropic additive linear operator for matrices of the form.

        A_iso := s I + A,

    where ``s`` is a scalar (or a 0-arg scalar LinearOperator) and ``A`` is a
    symmetric LinearOperator. This class exposes fast, matrix-free implementations
    of common spectral transforms (inverse, pseudo-inverse, square root, log,
    powers, exp, Cholesky-like factor) by working in the eigenbasis of ``A``.

    ----------
    Core idea
    ----------
    If ``A = Q Λ Qᵀ`` is an eigendecomposition of ``A`` (with Λ diagonal and
    ``Qᵀ Q = I``), then

        s I + A = Q (Λ + s I) Qᵀ,

    so any spectral function ``f`` (e.g. inverse, sqrt, log, power, exp) satisfies

        f(s I + A) = Q f(Λ + s I) Qᵀ,

    which reduces the linear-algebra to elementwise operations on the eigenvalues.

    This class computes/caches an (optionally truncated) eigendecomposition via
    ``leigh(A)`` and then dispatches the following:

    * ``linverse``:       (s I + A)⁻¹ = (1/s) I − Q diag(λ / (s (λ + s))) Qᵀ
                          (Woodbury / projector–complement split)
    * ``lpinverse``:      pseudo-inverse using the same spectral formula with
                          safe handling of zero/near-zero modes.
    * ``lsqrt``:          (s I + A)^{1/2} = Q diag(√(λ + s)) Qᵀ
    * ``lcholesky``:      returns a factor L with L Lᵀ = s I + A, namely
                          L = Q diag(√(λ + s))   (orthonormal “spectral” factor)
    * ``llog``:           log(s I + A) = Q diag(log(λ + s)) Qᵀ
    * ``lpow``:           (s I + A)^p = Q diag((λ + s)^p) Qᵀ
    * ``diagonal``:       diag(s I + A) = s · 1 + diag(A)
    * ``ltrace``:         tr(s I + A) = s·n + tr(A)  (with Hutchinson if needed)
    * ``lexp``:           exp(s I + A) = Q diag(exp(λ + s)) Qᵀ

    -------------------------------
    Projector / anti-projector view
    -------------------------------
    When ``leigh`` returns a **truncated** eigenspace ``Q ∈ ℝ^{n×k}`` (k ≤ n),
    let P := Q Qᵀ be the projector onto the retained subspace and
    P⊥ := I − P the orthogonal complement. Then

        (s I + A)⁻¹
        = Q (Λ + s I)⁻¹ Qᵀ  +  (1/s) P⊥,

    i.e. the inverse acts as ``(Λ + s I)⁻¹`` on span(Q) and as ``(1/s) I`` on
    its orthogonal complement. The implementation of ``linverse`` uses the
    equivalent Woodbury form

        (s I + A)⁻¹ = (1/s) [ I − Q diag(λ / (λ + s)) Qᵀ ].

    If ``leigh`` is **full-rank**, then P = I and P⊥ = 0, which recovers the
    usual full spectral formulas.

    -------------
    Caching notes
    -------------
    * ``Q`` and ``S`` (eigenvectors/eigenvalues) are cached lazily by
      ``_ensure_eigh()``. Any operation that changes the operator should call
      ``_invalidate_cache()``.
    * ``projector`` (Q Qᵀ) and ``complement`` (I − Q Qᵀ) are also cached on demand.

    ----------

    Arguments:
    ----------
    s : jax.Array
        Scalar added to the diagonal (isotropic shift). May be wrapped into a
        scalar ``ScaledLinearOperator(Identity, s)``.
    A : LinearOperator
        Symmetric linear operator (square). Non-symmetric inputs raise ``ValueError``.

    -------

    Returns:
    -------
    A LinearOperator supporting matrix-free application and spectral transforms
    of ``s I + A`` via the multipledispatch functions listed above.

    -------

    Example:
    -------
    >>> n = 100
    >>> s = jnp.array(0.1)
    >>> A = utils.as_linop(jnp.diag(jnp.linspace(0.0, 5.0, n)))  # symmetric
    >>> L = IsotropicAdditiveLinearOperator(s, A)
    >>> x = jnp.ones((n,))
    >>> y = (linverse(L) @ x)          # apply (s I + A)^{-1} to a vector
    >>> d = diagonal(L)                 # exact diagonal
    >>> z = (lsqrt(L) @ x)              # apply (s I + A)^{1/2} to a vector

    """

    def __init__(self, s: jax.Array, A: LinearOperator) -> None:
        self._A = utils.as_linop(A)
        if self._A.shape[-1] != self._A.shape[-2]:
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
    def scalar(self) -> jax.Array:
        return self._s.scalar

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
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    # Cholesky of A + sI = Q * sqrt(Λ + sI) where A = Q Λ Q^T
    new_lam = utils.as_linop(jnp.diag(jnp.sqrt(S + s)))
    return Q @ new_lam


@lsqrt.dispatch(precedence=1)
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    new_lam = utils.as_linop(jnp.diag(jnp.sqrt(S + s)))
    return Q @ new_lam @ Q.T


# we need a log-determinant function here
# TODO: Implement lsolve for IsotropicAdditiveLinearOperator via eigendecomposition
# A \kron B + s I = (Q_A \kron Q_B) (Lambda_A \kron Lambda_B + s I) (Q_A \kron Q_B)^T


@linverse.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar

    inv_iso = linverse(a.s)

    D = Diagonal(S / (s * (S + s)))

    return inv_iso - (Q @ D @ Q.T)


@lpinverse.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> LinearOperator:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar

    inv_iso = lpinverse(a.s)

    D = Diagonal(S / (s * (S + s)))

    return inv_iso - (Q @ D @ Q.T)


@leigh.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> tuple[LinearOperator, LinearOperator]:
    a._ensure_eigh()  # noqa: SLF001
    Q, S = a.Q, a.S  # cached
    s = a.s.scalar
    new_lam = utils.as_linop(S + s)
    return new_lam, Q


@diagonal.dispatch
def _(a: IsotropicAdditiveLinearOperator) -> jax.Array:
    # Sum of diagonals: diag(A) + s * 1
    return jnp.asarray(diagonal(a.operator)) + jnp.asarray(diagonal(a.s))


# New matrix-free function dispatches for IsotropicAdditive
@ltrace.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    key: jax.Array | None = None,
    num_samples: int = 100,
    distribution: str = "rademacher",
) -> tuple[jax.Array, jax.Array]:
    """Trace of sI + A: trace(sI + A) = s*n + trace(A)."""
    n = a.shape[-1]
    s = a.s.scalar

    # Recursively compute trace of A
    trace_A, std_A = ltrace(
        a.operator, key=key, num_samples=num_samples, distribution=distribution
    )

    trace_value = s * n + trace_A
    trace_std = std_A  # std of constant + random variable = std of random variable

    return trace_value, trace_std


@lexp.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix exponential of sI + A using eigendecomposition.

    exp(sI + A) = exp(s) * exp(A) since sI and A commute... NO, this is wrong!
    Actually: exp(sI + A) = U exp(s + λ) U^T where A = U λ U^T
    """
    a._ensure_eigh()  # noqa: SLF001
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    eigvals = a.S + s

    if v is None:
        # Return lazy operator: U @ Diagonal(exp(s + λ)) @ U^T
        exp_eigvals = Diagonal(jnp.exp(eigvals))
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.Q, exp_eigvals)
    # exp(sI + A) @ v = U @ exp(s + λ) @ U^T @ v
    return a.Q @ (jnp.exp(eigvals) * (a.Q.T @ v))


@llog.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix logarithm of sI + A using eigendecomposition.

    log(sI + A) = U log(s + λ) U^T where A = U λ U^T
    """
    a._ensure_eigh()  # noqa: SLF001
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    eigvals = a.S + s

    if v is None:
        # Return lazy operator: U @ Diagonal(log(s + λ)) @ U^T
        log_eigvals = Diagonal(jnp.log(eigvals))
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.Q, log_eigvals)
    # log(sI + A) @ v = U @ log(s + λ) @ U^T @ v
    return a.Q @ (jnp.log(eigvals) * (a.Q.T @ v))


@lpow.dispatch
def _(
    a: IsotropicAdditiveLinearOperator,
    *,
    power: float,
    v: jax.Array | None = None,
    num_iters: int = 20,
    method: str = "lanczos",
) -> jax.Array | LinearOperator:
    """Matrix power of sI + A using eigendecomposition.

    (sI + A)^p = U (s + λ)^p U^T where A = U λ U^T
    """
    a._ensure_eigh()  # noqa: SLF001
    s = a.s.scalar

    # Eigenvalues of sI + A are s + λ(A)
    eigvals = a.S + s

    if v is None:
        # Return lazy operator: U @ Diagonal((s + λ)^p) @ U^T
        pow_eigvals = Diagonal(eigvals**power)
        from linox._arithmetic import congruence_transform  # noqa: PLC0415

        return congruence_transform(a.Q, pow_eigvals)
    # (sI + A)^p @ v = U @ (s + λ)^p @ U^T @ v
    return a.Q @ ((eigvals**power) * (a.Q.T @ v))
