"""Canonicalization of linear operators."""

from linox.operators import LinearOperator
from linox.operators.arithmetic import smart_add, smart_matmul


def canonicalize(op: LinearOperator) -> LinearOperator:
    """Canonicalize a linear operator.

    Attempts to simplify the operator structure by applying rewrite rules
    and flattening nested structures. Since most simplifications are applied
    during construction (via smart_add/smart_matmul), this function determines
    if further simplification is possible.

    Args:
        op: The linear operator to canonicalize.

    Returns:
        A potentially simplified linear operator.
    """
    # For now, we assume construction-time simplification is sufficient.
    # Future valid canonicalization might involve deep traversal or graph rewriting.
    return op
