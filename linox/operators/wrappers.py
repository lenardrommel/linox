"""Property wrapper operators for symmetry and PSD assumptions.

This module provides lightweight wrapper operators that tag operands with
semantic properties like symmetry (Sym) and positive semidefiniteness (PSD).

These wrappers enable specialized algorithm dispatch without modifying
the underlying operator. They are compositional and may propagate under
arithmetic operations.

See Also
--------
ADR-0003 : Architecture decision record for PSD and symmetry wrappers.

Examples
--------
>>> import linox as lo
>>> K = lo.Diagonal(jnp.array([1.0, 2.0, 3.0]))
>>> K_psd = lo.PSD(K)  # Promise that K is PSD
>>> K_psd.is_psd
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from linox.operators.base import LinearOperator

if TYPE_CHECKING:
    pass


class Sym(LinearOperator):
    """Wrapper declaring an operator as symmetric (self-adjoint).

    This is a semantic promise that A = A^T (or A = A^H for complex).
    The wrapper does not verify this numerically unless debug validation
    is performed.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Attributes
    ----------
    is_symmetric : bool
        Always True for Sym-wrapped operators.
    wrapped : LinearOperator
        The underlying wrapped operator.

    Raises
    ------
    ValueError
        If the operator is not square.

    See Also
    --------
    PSD : Wrapper for positive semidefinite operators.
    assume_symmetric : Convenience function to create Sym wrappers.

    Examples
    --------
    >>> A = lo.Matrix(jnp.array([[1, 2], [2, 1]]))
    >>> A_sym = lo.Sym(A)
    >>> A_sym.is_symmetric
    True
    """

    wrapped: LinearOperator

    def __init__(self, op: LinearOperator) -> None:
        self.wrapped = op

        if op.shape[-2] != op.shape[-1]:
            msg = f"Sym wrapper requires square operator, got shape {op.shape}"
            raise ValueError(msg)

        super().__init__(shape=op.shape, dtype=op.dtype)

    @property
    def is_symmetric(self) -> bool:
        """Return True, indicating this operator is declared symmetric."""
        return True

    @property
    def is_psd(self) -> bool:
        """Return the PSD status of the wrapped operator."""
        return getattr(self.wrapped, "is_psd", False)

    def _matmat(self, x: jax.Array) -> jax.Array:
        """Compute matrix-matrix product by delegating to wrapped operator."""
        return self.wrapped @ x

    def _transpose(self) -> LinearOperator:
        """Return self, since symmetric operators equal their transpose."""
        return self

    def _todense(self) -> jax.Array:
        """Return dense representation of the wrapped operator."""
        return self.wrapped.todense()

    def children(self) -> tuple[LinearOperator, ...]:
        """Return child operators for tree traversal."""
        return (self.wrapped,)

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten for JAX PyTree compatibility."""
        return (self.wrapped,), {}

    @classmethod
    def tree_unflatten(cls, aux: dict, children: tuple) -> Sym:
        """Unflatten from JAX PyTree representation."""
        (wrapped,) = children
        return cls(wrapped)


class PSD(LinearOperator):
    """Wrapper declaring an operator as positive semidefinite.

    This is a semantic promise that x^T A x >= 0 for all x. The wrapper
    does not verify this numerically unless debug validation is performed.

    PSD operators are also implicitly symmetric.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Attributes
    ----------
    is_psd : bool
        Always True for PSD-wrapped operators.
    is_symmetric : bool
        Always True (PSD implies symmetric).
    wrapped : LinearOperator
        The underlying wrapped operator.

    Raises
    ------
    ValueError
        If the operator is not square.

    See Also
    --------
    Sym : Wrapper for symmetric operators.
    assume_psd : Convenience function to create PSD wrappers.

    Examples
    --------
    >>> K = lo.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    >>> K_psd = lo.PSD(K)
    >>> K_psd.is_psd
    True
    >>> K_psd.is_symmetric
    True
    """

    wrapped: LinearOperator

    def __init__(self, op: LinearOperator) -> None:
        self.wrapped = op

        if op.shape[-2] != op.shape[-1]:
            msg = f"PSD wrapper requires square operator, got shape {op.shape}"
            raise ValueError(msg)

        super().__init__(shape=op.shape, dtype=op.dtype)

    @property
    def is_symmetric(self) -> bool:
        """Return True, since PSD operators are symmetric."""
        return True

    @property
    def is_psd(self) -> bool:
        """Return True, indicating this operator is declared PSD."""
        return True

    def _matmat(self, x: jax.Array) -> jax.Array:
        """Compute matrix-matrix product by delegating to wrapped operator."""
        return self.wrapped @ x

    def _transpose(self) -> LinearOperator:
        """Return self, since PSD operators are symmetric."""
        return self

    def _todense(self) -> jax.Array:
        """Return dense representation of the wrapped operator."""
        return self.wrapped.todense()

    def children(self) -> tuple[LinearOperator, ...]:
        """Return child operators for tree traversal."""
        return (self.wrapped,)

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten for JAX PyTree compatibility."""
        return (self.wrapped,), {}

    @classmethod
    def tree_unflatten(cls, aux: dict, children: tuple) -> PSD:
        """Unflatten from JAX PyTree representation."""
        (wrapped,) = children
        return cls(wrapped)


class SPD(PSD):
    """Wrapper declaring an operator as symmetric positive definite.

    This is a semantic promise that x^T A x > 0 for all x != 0 (strictly
    positive definite). SPD operators are a subset of PSD operators.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Attributes
    ----------
    is_spd : bool
        Always True for SPD-wrapped operators.
    is_psd : bool
        Always True (SPD implies PSD).
    is_symmetric : bool
        Always True (SPD implies symmetric).

    See Also
    --------
    PSD : Wrapper for positive semidefinite operators.
    Sym : Wrapper for symmetric operators.

    Examples
    --------
    >>> K = lo.Diagonal(jnp.array([1.0, 2.0, 3.0]))  # All positive
    >>> K_spd = lo.SPD(K)
    >>> K_spd.is_spd
    True
    """

    @property
    def is_spd(self) -> bool:
        """Return True, indicating this operator is declared SPD."""
        return True

    @classmethod
    def tree_unflatten(cls, aux: dict, children: tuple) -> SPD:
        """Unflatten from JAX PyTree representation."""
        (wrapped,) = children
        return cls(wrapped)


# Convenience functions


def assume_symmetric(op: LinearOperator) -> Sym:
    """Wrap an operator to declare it as symmetric.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Returns
    -------
    Sym
        The wrapped operator with is_symmetric=True.

    See Also
    --------
    Sym : The wrapper class.
    assume_psd : For positive semidefinite operators.

    Examples
    --------
    >>> A = lo.Matrix(jnp.array([[1, 2], [2, 1]]))
    >>> A_sym = lo.assume_symmetric(A)
    >>> A_sym.is_symmetric
    True
    """
    if isinstance(op, Sym):
        return op
    return Sym(op)


def assume_psd(op: LinearOperator) -> PSD:
    """Wrap an operator to declare it as positive semidefinite.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Returns
    -------
    PSD
        The wrapped operator with is_psd=True and is_symmetric=True.

    See Also
    --------
    PSD : The wrapper class.
    assume_symmetric : For symmetric operators without PSD guarantee.

    Examples
    --------
    >>> K = lo.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    >>> K_psd = lo.assume_psd(K)
    >>> K_psd.is_psd
    True
    """
    if isinstance(op, PSD):
        return op
    return PSD(op)


def assume_spd(op: LinearOperator) -> SPD:
    """Wrap an operator to declare it as symmetric positive definite.

    Parameters
    ----------
    op : LinearOperator
        The operator to wrap. Must be square.

    Returns
    -------
    SPD
        The wrapped operator with is_spd=True, is_psd=True, is_symmetric=True.

    See Also
    --------
    SPD : The wrapper class.
    assume_psd : For positive semidefinite operators.

    Examples
    --------
    >>> K = lo.Diagonal(jnp.array([1.0, 2.0, 3.0]))
    >>> K_spd = lo.assume_spd(K)
    >>> K_spd.is_spd
    True
    """
    if isinstance(op, SPD):
        return op
    return SPD(op)


# Register PyTrees
jax.tree_util.register_pytree_node_class(Sym)
jax.tree_util.register_pytree_node_class(PSD)
jax.tree_util.register_pytree_node_class(SPD)
