
"""Hypothesis strategies for generating linear operators."""

import jax.numpy as jnp
import linox
from hypothesis import strategies as st


def _valid_shapes(min_dim=1, max_dim=5):
    """Strategy for square shapes."""
    return st.integers(min_value=min_dim, max_value=max_dim).map(lambda n: (n, n))


@st.composite
def linear_operators(draw, shape=None, depth=1):
    """Recursive strategy to generate random LinearOperators.
    
    Args:
        draw: Hypothesis draw function.
        shape: Tuple[int, int], optional shape for the operator.
        depth: Recursion depth.
    """
    if shape is None:
        shape = draw(_valid_shapes())
    
    n, m = shape
    if n != m:
        # Simplified: force square for now to easier handle combinations
        pass

    # Base case: Matrix
    if depth <= 1:
        # Create a dense matrix
        A = draw(st.lists(
            st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=n, max_size=n),
            min_size=n, max_size=n
        ).map(lambda l: jnp.array(l)))
        return linox.Matrix(A)

    # Recursive cases
    op_type = draw(st.sampled_from(["scaled", "add", "product", "kron", "matrix"]))

    if op_type == "matrix":
         return draw(linear_operators(shape=shape, depth=1)) # Fallback to base

    if op_type == "scaled":
        scalar = draw(st.floats(min_value=-5, max_value=5, allow_nan=False))
        op = draw(linear_operators(shape=shape, depth=depth-1))
        return linox.ScaledLinearOperator(op, scalar)

    if op_type == "add":
        op1 = draw(linear_operators(shape=shape, depth=depth-1))
        op2 = draw(linear_operators(shape=shape, depth=depth-1))
        return linox.AddLinearOperator(op1, op2)

    if op_type == "product":
        op1 = draw(linear_operators(shape=shape, depth=depth-1))
        op2 = draw(linear_operators(shape=shape, depth=depth-1))
        return linox.ProductLinearOperator(op1, op2)

    if op_type == "kron":
        factors = []
        for i in range(1, int(n**0.5) + 1):
             if n % i == 0:
                 factors.append((i, n // i))
        
        if not factors or (len(factors) == 1 and factors[0] == (1, n)): # Prime or 1
             return draw(linear_operators(shape=shape, depth=1)) # Fallback

        rA, rB = draw(st.sampled_from(factors))
        
        opA = draw(linear_operators(shape=(rA, rA), depth=depth-1))
        opB = draw(linear_operators(shape=(rB, rB), depth=depth-1))
        return linox.Kronecker(opA, opB)

    return draw(linear_operators(shape=shape, depth=1)) # Fallback
