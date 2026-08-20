"""Use linox operators with `skerch <https://github.com/andres-fr/skerch>`_.

skerch implements randomized sketching algorithms -- sketched SVD and Hermitian
eigendecomposition, operator and Frobenius norm estimation, Girard-Hutchinson
trace and diagonal -- against a duck-typed linear operator. Its contract is
small: a ``.shape`` attribute and the two matmul protocols::

    U, S, Vh = ssvd(to_skerch(A), "cpu", torch.float64, outer_dims=8)

skerch is built on PyTorch and has no JAX support of its own, so the operands it
hands over and the results it expects back are both torch tensors.
:class:`SkerchLinOp` is the bridge. skerch's own ``TorchLinOpWrapper`` does not
serve here: it is hardcoded to ``torch.from_numpy``, which does not accept a JAX
array.

Limitations:

* Real dtypes only. skerch supports complex operators; linox has no exercised
  complex path, and the adjoint identity used below assumes a real transpose.
  Complex operators are rejected rather than silently given wrong answers.
* Not differentiable. Operands are detached at the boundary.
* Batched operators (``ndim > 2``) are not supported -- skerch's contract is a
  single ``(height, width)`` pair.
* Every application round-trips through host memory. skerch sketches in a few
  large blocks rather than many small matvecs, so this is amortized well, but it
  is not suited to a tight inner loop.
* Importing linox enables JAX's x64 mode process-wide, which can surprise a
  torch-first program sharing the interpreter. That is pre-existing linox
  behaviour, noted here because this module is where the two meet.
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp

from linox.operators.base import LinearOperator

from ._torch import to_jax, to_torch

if TYPE_CHECKING:
    import torch

__all__ = ["SkerchLinOp", "to_skerch"]


class SkerchLinOp:
    """A linox operator wearing skerch's linear-operator interface.

    Wraps ``operator`` so that it accepts and returns torch tensors,
    exposing the ``.shape`` / ``__matmul__`` / ``__rmatmul__`` trio that skerch's
    algorithms consume. Operands may be vectors or matrices; skerch sketches with
    matrices by default.

    Parameters
    ----------
    operator
        The operator to adapt. Must be two-dimensional and real-valued.

    Raises
    ------
    ValueError
        If ``operator`` is batched or has a complex dtype.

    Examples
    --------
    >>> import torch  # doctest: +SKIP
    >>> from skerch.algorithms import snorm  # doctest: +SKIP
    >>> norms, _ = snorm(  # doctest: +SKIP
    ...     to_skerch(A), "cpu", torch.float64, num_meas=20, seed=0
    ... )
    """

    def __init__(self, operator: LinearOperator) -> None:
        if not isinstance(operator, LinearOperator):
            msg = f"Expected a linox LinearOperator, got {type(operator).__name__}."
            raise TypeError(msg)
        if operator.ndim != 2:
            msg = f"skerch operators are two-dimensional, but this one has shape {operator.shape}. Batched operators are not supported."
            raise ValueError(msg)
        if jnp.issubdtype(operator.dtype, jnp.complexfloating):
            msg = (
                f"Complex operators are not supported (dtype {operator.dtype}). "
                "The adjoint is computed as a plain transpose, which is wrong "
                "for complex operators."
            )
            raise ValueError(msg)

        self._operator = operator
        # skerch unpacks this as `h, w = lop.shape` and feeds the entries to
        # `torch.empty`, so hand it plain ints rather than whatever integer type
        # the operator's shape carries.
        self.shape = (int(operator.shape[0]), int(operator.shape[1]))

    @property
    def operator(self) -> LinearOperator:
        """The wrapped linox operator."""
        return self._operator

    def __matmul__(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply the operator, ``self @ x``, for a vector or matrix ``x``."""
        return to_torch(self._operator @ to_jax(x), like=x)

    def __rmatmul__(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply the adjoint, ``x @ self``, for a vector or matrix ``x``.

        Routed through ``operator.T`` rather than linox's own ``__rmatmul__``,
        which returns a lazy :class:`LinearOperator` for two-dimensional left
        operands -- skerch expects an array it can transpose and slice-assign.
        Going via the transpose also keeps the adjoint matrix-free, since
        ``TransposedLinearOperator`` derives it from the forward matvec.
        """
        xj = to_jax(x)
        if xj.ndim == 1:
            return to_torch(self._operator.T @ xj, like=x)
        # (j, m) @ (m, n) == ((n, m) @ (m, j)).T, valid for real operators.
        return to_torch((self._operator.T @ xj.T).T, like=x)

    def __repr__(self) -> str:
        return f"SkerchLinOp({self._operator!r})"


def to_skerch(operator: LinearOperator) -> SkerchLinOp:
    """Adapt a linox operator for use with skerch.

    Convenience wrapper around :class:`SkerchLinOp`; see that class for the
    supported operators and the limitations of the bridge.
    """
    return SkerchLinOp(operator)
