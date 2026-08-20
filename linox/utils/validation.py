"""Operator validation utilities.

This module provides validation helpers for LinearOperator instances,
implementing both cheap structural validation (default) and more expensive
numerical probe-based validation (debug mode).

See Also
--------
ADR-0006 : Architecture decision record describing the validation strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from linox.operators.base import LinearOperator


class ValidationError(ValueError):
    """Raised when operator validation fails.

    Parameters
    ----------
    message : str
        Description of the validation failure.
    operator : LinearOperator
        The operator that failed validation.
    hint : str, optional
        Suggestion for how to fix the issue.

    Examples
    --------
    >>> raise ValidationError("Shape mismatch", op, hint="Use transpose")
    """

    def __init__(
        self,
        message: str,
        operator: LinearOperator,
        hint: str | None = None,
    ) -> None:
        self.operator = operator
        self.hint = hint

        full_message = f"Validation failed for {type(operator).__name__} (shape={operator.shape}, dtype={operator.dtype}): {message}"
        if hint:
            full_message += f"\nHint: {hint}"

        super().__init__(full_message)


def validate(
    op: LinearOperator,
    *,
    mode: Literal["default", "debug"] = "default",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    num_probes: int = 5,
    key: jax.Array | None = None,
) -> bool:
    """Validate a LinearOperator and its children recursively.

    Performs structural validation by default. In debug mode, also performs
    numerical probe-based validation to check symmetry and PSD promises.

    Parameters
    ----------
    op : LinearOperator
        The operator to validate.
    mode : {"default", "debug"}, optional
        Validation mode. "default" performs only cheap structural checks.
        "debug" also performs expensive numerical probes.
    rtol : float, optional
        Relative tolerance for numerical checks. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for numerical checks. Default is 1e-8.
    num_probes : int, optional
        Number of random probes for numerical validation. Default is 5.
    key : jax.Array, optional
        JAX random key for probe generation. If None, uses a default key.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValidationError
        If validation fails, with details about the failure and hints.

    See Also
    --------
    linox.config : For setting LINOX_DEBUG=1 to enable debug validation globally.

    Examples
    --------
    >>> import linox as lo
    >>> A = lo.Matrix(jnp.eye(3))
    >>> lo.validate(A)  # Cheap structural validation
    True

    >>> # Debug mode includes probe-based checks
    >>> lo.validate(A, mode="debug")
    True
    """
    if key is None:
        key = jax.random.key(42)

    # Structural validation (always performed)
    _validate_structural(op)

    # Numerical validation (debug mode only)
    if mode == "debug":
        _validate_numerical(op, rtol=rtol, atol=atol, num_probes=num_probes, key=key)

    return True


def _validate_structural(op: LinearOperator) -> None:
    """Perform cheap O(1) structural validation.

    Checks:
    - Shape validity (non-negative dimensions)
    - Dtype validity
    - Children validity (recursive)
    - Structural invariants specific to operator type

    Parameters
    ----------
    op : LinearOperator
        Operator to validate.

    Raises
    ------
    ValidationError
        If structural validation fails.
    """
    # Check shape
    if len(op.shape) < 2:
        raise ValidationError(
            f"Operator must have at least 2 dimensions, got {len(op.shape)}",
            op,
            hint="Wrap scalars in Scalar() or ensure shape is (n, m)",
        )

    for i, dim in enumerate(op.shape):
        if dim < 0:
            raise ValidationError(
                f"Dimension {i} is negative: {dim}",
                op,
                hint="All dimensions must be non-negative",
            )

    # Check dtype
    if op.dtype is None:
        raise ValidationError(
            "Operator dtype is None",
            op,
            hint="Ensure dtype is set during construction",
        )

    # Validate children recursively
    if hasattr(op, "children") and callable(op.children):
        for child in op.children():
            _validate_structural(child)


def _validate_numerical(
    op: LinearOperator,
    *,
    rtol: float,
    atol: float,
    num_probes: int,
    key: jax.Array,
) -> None:
    """Perform expensive numerical probe-based validation.

    Checks:
    - Symmetry verification via random probes (if symmetric flag is set)
    - PSD verification via random probes (if psd flag is set)
    - Finite output check

    Parameters
    ----------
    op : LinearOperator
        Operator to validate.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    num_probes : int
        Number of random probes to use.
    key : jax.Array
        JAX random key.

    Raises
    ------
    ValidationError
        If numerical validation fails.
    """
    is_square = op.shape[-2] == op.shape[-1]
    if not is_square:
        # Skip numerical checks for non-square operators
        return

    n = op.shape[-1]

    # Generate random probes
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, (n, num_probes), dtype=op.dtype)
    jax.random.normal(key2, (n, num_probes), dtype=op.dtype)

    # Check finite output
    AX = op @ X
    if not jnp.all(jnp.isfinite(AX)):
        raise ValidationError(
            "Operator produces non-finite outputs on random probes",
            op,
            hint="Check for NaN/Inf in operator data or numerical instability",
        )

    # Check symmetry if claimed
    is_symmetric = getattr(op, "is_symmetric", False)
    if is_symmetric:
        AT_X = op.T @ X
        sym_error = jnp.max(jnp.abs(AX - AT_X))
        norm_AX = jnp.max(jnp.abs(AX)) + atol

        if sym_error / norm_AX > rtol:
            raise ValidationError(
                f"Operator claims symmetry but ||Ax - A^Tx|| / ||Ax|| = {sym_error / norm_AX:.2e} > rtol={rtol}",
                op,
                hint="Remove Sym() wrapper or fix underlying operator",
            )

    # Check PSD if claimed
    is_psd = getattr(op, "is_psd", False)
    if is_psd:
        # Check x^T A x >= -tol for random probes
        for i in range(num_probes):
            x = X[:, i]
            Ax = AX[:, i]
            xTAx = jnp.real(jnp.dot(jnp.conj(x), Ax))

            if xTAx < -atol:
                raise ValidationError(
                    f"Operator claims PSD but x^T A x = {xTAx:.2e} < -atol={atol}",
                    op,
                    hint="Remove PSD() wrapper or ensure operator is actually PSD",
                )

    # Validate children recursively in debug mode
    if hasattr(op, "children") and callable(op.children):
        for i, child in enumerate(op.children()):
            child_key = jax.random.fold_in(key, i)
            _validate_numerical(child, rtol=rtol, atol=atol, num_probes=num_probes, key=child_key)
