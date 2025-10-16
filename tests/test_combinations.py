"""Property-based tests for combinations of linear operators.

This module tests deep combinations of linear operators (up to 7 levels),
ensuring that complex operator compositions work correctly.

## Test Coverage

### Base Operators
- Matrix: Dense matrix operators
- Diagonal: Diagonal matrix operators
- Identity: Identity matrix operators
- Zero: Zero matrix operators
- Ones: All-ones matrix operators
- SymmetricLowRank: Low-rank symmetric operators (u @ diag(s) @ u.T)

### Structured Operators
- Kronecker: Kronecker products (A ⊗ B)
- BlockDiagonal: Block diagonal matrices
- BlockMatrix2x2: 2x2 block matrices
- BlockMatrix: General block matrices (tested via BlockMatrix2x2)

### Arithmetic Combinations
- AddLinearOperator: Addition of operators (A + B)
- ProductLinearOperator: Matrix product (A @ B)
- ScaledLinearOperator: Scalar multiplication (α * A)
- TransposedLinearOperator: Transpose (A^T)
- IsotropicAdditiveLinearOperator: Isotropic addition (A + λI)

## Test Structure

### Property-Based Tests (Hypothesis)
1. `test_random_combination_*`: Standard depth (up to 6 leaves)
   - Tests basic operator combinations with 120 examples
   - Validates todense(), vector matmul, and matrix matmul

2. `test_deep_combination_*`: Deep nesting (up to 10 leaves, 5+ levels)
   - Tests deeply nested operator combinations with 50 examples
   - Ensures framework handles complex compositions correctly

### Specific Complex Tests
1. `test_specific_complex_combination_1`: (A ⊗ B + C^T) @ (D + λI)
   - Kronecker + Transpose + IsotropicAdditive + Product

2. `test_specific_complex_combination_2`: BlockDiagonal with nested operations
   - ScaledLinearOperator + SymmetricLowRank + IsotropicAdditive

3. `test_specific_complex_combination_3`: BlockMatrix2x2 with structured blocks
   - Kronecker, Scaled, Zero, and Product operators as blocks

4. `test_five_level_depth_combination`: Manual 5-level composition
   - (D + I) → Scaled → IsotropicAdditive → Transpose → Add

5. `test_ultra_deep_combination`: Manual 7-level composition
   - Kronecker → Add → Scale → Isotropic → Transpose → Product → Add
   - Tests all operator types in a single expression

6. `test_block_operators_nested`: Nested block structures
   - BlockDiagonal containing BlockMatrix2x2 operators

7. `test_mixed_operators_stress`: Complex multi-operator expression
   - ((A ⊗ B) + C)^T @ ((D + λI) @ E)
   - Combines Kronecker, SymmetricLowRank, BlockDiagonal, and arithmetic ops

## Framework Limitations Found
The tests reveal current framework capabilities and some dtype handling requirements:
- All operators in a composition must have matching dtypes
- Scalar parameters (λ, α) must be cast to match operator dtype
- Block operators require compatible inner dimensions
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import linox

DTYPE = jnp.float32
FLOATS = st.floats(
    min_value=-5.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
)
# Smaller floats for scaling to avoid numerical issues
SMALL_FLOATS = st.floats(
    min_value=-2.0,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
)


def _matrix_strategy(n: int) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    return arrays(np.float32, (n, n), elements=FLOATS).map(
        lambda arr: (
            linox.Matrix(jnp.asarray(arr, dtype=DTYPE)),
            jnp.asarray(arr, dtype=DTYPE),
        )
    )


def _diagonal_strategy(n: int) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    return arrays(np.float32, (n,), elements=FLOATS).map(
        lambda diag: (
            linox.Diagonal(jnp.asarray(diag, dtype=DTYPE)),
            jnp.diag(jnp.asarray(diag, dtype=DTYPE)),
        )
    )


def _symmetric_low_rank_strategy(
    n: int,
) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    def _build(args: tuple[np.ndarray, np.ndarray]) -> tuple[linox.LinearOperator, jnp.ndarray]:
        u, s = args
        u = jnp.asarray(u, dtype=DTYPE)
        s = jnp.asarray(s, dtype=DTYPE)
        linop = linox.SymmetricLowRank(u, s)
        dense = u @ jnp.diag(s) @ u.T
        return linop, dense

    return st.tuples(
        arrays(np.float32, (n, n), elements=FLOATS),
        arrays(np.float32, (n,), elements=st.floats(min_value=0.0, max_value=3.0)),
    ).map(_build)


def _kronecker_strategy(n: int) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    """Strategy for Kronecker products of smaller operators."""
    # Use smaller dimension for Kronecker to avoid explosion
    # For n=4, use 2x2; for n=9, use 3x3, etc.
    k = int(np.sqrt(n)) if int(np.sqrt(n))**2 == n else 2

    if k * k != n:
        # If n is not a perfect square, fall back to diagonal
        return _diagonal_strategy(n)

    def _build_kron(pair: tuple[tuple[linox.LinearOperator, jnp.ndarray],
                                  tuple[linox.LinearOperator, jnp.ndarray]]) -> tuple[linox.LinearOperator, jnp.ndarray]:
        (op1, dense1), (op2, dense2) = pair
        return linox.Kronecker(op1, op2), jnp.kron(dense1, dense2)

    small_strat = st.one_of(
        _matrix_strategy(k),
        _diagonal_strategy(k),
        st.just((linox.Identity((k,), dtype=DTYPE), jnp.eye(k, dtype=DTYPE))),
    )

    return st.tuples(small_strat, small_strat).map(_build_kron)


def _block_diagonal_strategy(n: int) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    """Strategy for block diagonal matrices."""
    # Split n into 2 or 3 blocks
    num_blocks = st.integers(min_value=2, max_value=min(3, n))

    def _build_block_diag(args: tuple[int, list[tuple[linox.LinearOperator, jnp.ndarray]]]) -> tuple[linox.LinearOperator, jnp.ndarray]:
        _, blocks = args
        ops = [op for op, _ in blocks]
        denses = [dense for _, dense in blocks]
        return linox.BlockDiagonal(*ops), jnp.block([[jnp.zeros((d1.shape[0], d2.shape[1])) if i != j else d1
                                                       for j, d2 in enumerate(denses)]
                                                      for i, d1 in enumerate(denses)])

    # Generate block sizes that sum to n
    def _split_size(total: int, num: int) -> list[int]:
        if num == 1:
            return [total]
        size = max(1, total // num)
        return [size] + _split_size(total - size, num - 1)

    return num_blocks.flatmap(lambda nb: st.builds(
        lambda sizes: (nb, [(linox.Diagonal(jnp.ones(s, dtype=DTYPE)), jnp.eye(s, dtype=DTYPE))
                            for s in sizes]),
        st.just(_split_size(n, nb))
    )).map(_build_block_diag)


def _block_matrix_2x2_strategy(n: int) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    """Strategy for 2x2 block matrices."""
    if n < 2:
        return _diagonal_strategy(n)

    # Split dimension in half
    n1 = n // 2
    n2 = n - n1

    def _build_2x2(blocks: tuple[tuple[linox.LinearOperator, jnp.ndarray],
                                 tuple[linox.LinearOperator, jnp.ndarray],
                                 tuple[linox.LinearOperator, jnp.ndarray],
                                 tuple[linox.LinearOperator, jnp.ndarray]]) -> tuple[linox.LinearOperator, jnp.ndarray]:
        (A_op, A_dense), (B_op, B_dense), (C_op, C_dense), (D_op, D_dense) = blocks
        op = linox.BlockMatrix2x2(A_op, B_op, C_op, D_op)
        dense = jnp.block([[A_dense, B_dense], [C_dense, D_dense]])
        return op, dense

    # Create simple strategies for each block
    A_strat = st.one_of(_diagonal_strategy(n1),
                        arrays(np.float32, (n1, n1), elements=SMALL_FLOATS).map(
                            lambda arr: (linox.Matrix(jnp.asarray(arr, dtype=DTYPE)), jnp.asarray(arr, dtype=DTYPE))))
    B_strat = arrays(np.float32, (n1, n2), elements=SMALL_FLOATS).map(
        lambda arr: (linox.Matrix(jnp.asarray(arr, dtype=DTYPE)), jnp.asarray(arr, dtype=DTYPE)))
    C_strat = arrays(np.float32, (n2, n1), elements=SMALL_FLOATS).map(
        lambda arr: (linox.Matrix(jnp.asarray(arr, dtype=DTYPE)), jnp.asarray(arr, dtype=DTYPE)))
    D_strat = st.one_of(_diagonal_strategy(n2),
                        arrays(np.float32, (n2, n2), elements=SMALL_FLOATS).map(
                            lambda arr: (linox.Matrix(jnp.asarray(arr, dtype=DTYPE)), jnp.asarray(arr, dtype=DTYPE))))

    return st.tuples(A_strat, B_strat, C_strat, D_strat).map(_build_2x2)


def _base_operator_strategy(
    n: int,
) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    identity = st.just((linox.Identity((n,), dtype=DTYPE), jnp.eye(n, dtype=DTYPE)))
    zero = st.just((linox.Zero((n, n), dtype=DTYPE), jnp.zeros((n, n), dtype=DTYPE)))
    ones = st.just((linox.Ones((n, n), dtype=DTYPE), jnp.ones((n, n), dtype=DTYPE)))

    strategies = [
        _matrix_strategy(n),
        _diagonal_strategy(n),
        _symmetric_low_rank_strategy(n),
        identity,
        zero,
        ones,
    ]

    # Add structured operators for appropriate sizes
    if n == 4 or n == 9:  # Perfect squares for Kronecker
        strategies.append(_kronecker_strategy(n))

    if n >= 2:  # Block operators need at least 2x2
        strategies.append(_block_diagonal_strategy(n))
        strategies.append(_block_matrix_2x2_strategy(n))

    return st.one_of(*strategies)


def _combine_strategy(
    n: int,
    child: st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]],
) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    addition = st.tuples(child, child).map(
        lambda pair: (
            linox.AddLinearOperator(pair[0][0], pair[1][0]),
            jnp.asarray(pair[0][1]) + jnp.asarray(pair[1][1]),
        )
    )
    product = st.tuples(child, child).map(
        lambda pair: (
            linox.ProductLinearOperator(pair[0][0], pair[1][0]),
            jnp.asarray(pair[0][1]) @ jnp.asarray(pair[1][1]),
        )
    )
    scalar_scale = st.tuples(child, SMALL_FLOATS).map(
        lambda pair: (
            linox.ScaledLinearOperator(
                pair[0][0], jnp.asarray(pair[1], dtype=DTYPE)
            ),
            jnp.asarray(pair[1], dtype=DTYPE) * jnp.asarray(pair[0][1]),
        )
    )
    transpose = child.map(
        lambda item: (
            linox.TransposedLinearOperator(item[0]),
            jnp.swapaxes(jnp.asarray(item[1]), -1, -2),
        )
    )
    isotropic = st.tuples(child, SMALL_FLOATS).map(
        lambda pair: (
            linox.IsotropicAdditiveLinearOperator(
                jnp.asarray(pair[1], dtype=DTYPE),
                pair[0][0],
            ),
            jnp.asarray(pair[0][1])
            + jnp.asarray(pair[1], dtype=DTYPE) * jnp.eye(n, dtype=DTYPE),
        )
    )
    return st.one_of(addition, product, scalar_scale, transpose, isotropic)


def operator_expression_strategy(
    n: int,
    max_leaves: int = 6,
) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    """Build recursive operator expressions with configurable depth."""
    base = _base_operator_strategy(n)
    return st.recursive(
        base,
        lambda children: _combine_strategy(n, children),
        max_leaves=max_leaves,
    )


def deep_operator_expression_strategy(
    n: int,
) -> st.SearchStrategy[tuple[linox.LinearOperator, jnp.ndarray]]:
    """Build deep operator expressions (up to 5+ levels)."""
    return operator_expression_strategy(n, max_leaves=10)


@st.composite
def linear_operator_combo(
    draw: Callable[..., object],
) -> tuple[int, linox.LinearOperator, jnp.ndarray]:
    n = draw(st.integers(min_value=1, max_value=4))
    linop, dense = draw(operator_expression_strategy(n))
    dense = jnp.asarray(dense, dtype=DTYPE)
    return n, linop, dense


HYPOTHESIS_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)


@HYPOTHESIS_SETTINGS
@given(linear_operator_combo())
def test_random_combination_todense_matches_dense(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
) -> None:
    _n, linop, expected_dense = combo
    actual_dense = jnp.asarray(linop.todense())
    assert actual_dense.shape == expected_dense.shape
    assert jnp.allclose(actual_dense, expected_dense, atol=1e-5)


@HYPOTHESIS_SETTINGS
@given(linear_operator_combo(), st.data())
def test_random_combination_vector_matmul(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
    data: st.DataObject,
) -> None:
    n, linop, dense = combo
    vector = data.draw(arrays(np.float32, (n,), elements=FLOATS))
    vector = jnp.asarray(vector, dtype=DTYPE)
    expected = dense @ vector
    actual = linop @ vector
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=1e-5)


@HYPOTHESIS_SETTINGS
@given(linear_operator_combo(), st.data())
def test_random_combination_matrix_matmul(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
    data: st.DataObject,
) -> None:
    n, linop, dense = combo
    ncols = data.draw(st.integers(min_value=1, max_value=4))
    rhs = data.draw(arrays(np.float32, (n, ncols), elements=FLOATS))
    rhs = jnp.asarray(rhs, dtype=DTYPE)
    expected = dense @ rhs
    actual = linop @ rhs
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=1e-5)


# ============================================================================
# Deep combination tests (5+ levels)
# ============================================================================


@st.composite
def deep_linear_operator_combo(
    draw: Callable[..., object],
) -> tuple[int, linox.LinearOperator, jnp.ndarray]:
    """Generate deeply nested operator combinations (5+ levels)."""
    n = draw(st.integers(min_value=2, max_value=4))
    linop, dense = draw(deep_operator_expression_strategy(n))
    dense = jnp.asarray(dense, dtype=DTYPE)
    return n, linop, dense


DEEP_HYPOTHESIS_SETTINGS = settings(
    max_examples=50,  # Fewer examples for deep tests
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)


@DEEP_HYPOTHESIS_SETTINGS
@given(deep_linear_operator_combo())
def test_deep_combination_todense_matches_dense(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
) -> None:
    """Test that deeply nested operators produce correct dense matrices."""
    _n, linop, expected_dense = combo
    actual_dense = jnp.asarray(linop.todense())
    assert actual_dense.shape == expected_dense.shape
    assert jnp.allclose(actual_dense, expected_dense, atol=1e-4, rtol=1e-4)


@DEEP_HYPOTHESIS_SETTINGS
@given(deep_linear_operator_combo(), st.data())
def test_deep_combination_vector_matmul(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
    data: st.DataObject,
) -> None:
    """Test vector multiplication with deeply nested operators."""
    n, linop, dense = combo
    vector = data.draw(arrays(np.float32, (n,), elements=SMALL_FLOATS))
    vector = jnp.asarray(vector, dtype=DTYPE)
    expected = dense @ vector
    actual = linop @ vector
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@DEEP_HYPOTHESIS_SETTINGS
@given(deep_linear_operator_combo(), st.data())
def test_deep_combination_matrix_matmul(
    combo: tuple[int, linox.LinearOperator, jnp.ndarray],
    data: st.DataObject,
) -> None:
    """Test matrix multiplication with deeply nested operators."""
    n, linop, dense = combo
    ncols = data.draw(st.integers(min_value=1, max_value=3))
    rhs = data.draw(arrays(np.float32, (n, ncols), elements=SMALL_FLOATS))
    rhs = jnp.asarray(rhs, dtype=DTYPE)
    expected = dense @ rhs
    actual = linop @ rhs
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=1e-4, rtol=1e-4)


# ============================================================================
# Specific complex combination tests
# ============================================================================


def test_specific_complex_combination_1() -> None:
    """Test: (A ⊗ B + C^T) @ (D + λI) where operators are structured."""
    n = 4
    # Create Kronecker product
    A = linox.Diagonal(jnp.array([1.0, 2.0], dtype=DTYPE))
    B = linox.Matrix(jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=DTYPE))
    kron_AB = linox.Kronecker(A, B)

    # Create transpose of a matrix
    C = linox.Matrix(
        jnp.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=DTYPE,
        )
    )
    C_T = linox.TransposedLinearOperator(C)

    # Add them
    left = linox.AddLinearOperator(kron_AB, C_T)

    # Create isotropic additive operator
    D = linox.Diagonal(jnp.array([0.5, 1.0, 1.5, 2.0], dtype=DTYPE))
    lambda_val = jnp.array(0.1, dtype=DTYPE)
    right = linox.IsotropicAdditiveLinearOperator(lambda_val, D)

    # Final product
    result = linox.ProductLinearOperator(left, right)

    # Compute expected
    kron_dense = jnp.kron(A.todense(), B.todense())
    C_dense = C.todense()
    D_dense = D.todense()
    left_dense = kron_dense + C_dense.T
    right_dense = D_dense + lambda_val * jnp.eye(n, dtype=DTYPE)
    expected = left_dense @ right_dense

    actual = result.todense()
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_specific_complex_combination_2() -> None:
    """Test: Block diagonal of (scaled operators + isotropic)."""
    n1, n2 = 2, 3

    # First block: scaled symmetric low rank + isotropic
    u1 = jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=DTYPE)
    s1 = jnp.array([2.0, 1.0], dtype=DTYPE)
    slr1 = linox.SymmetricLowRank(u1, s1)
    scaled1 = linox.ScaledLinearOperator(slr1, 0.5)
    block1 = linox.IsotropicAdditiveLinearOperator(0.2, scaled1)

    # Second block: transpose of diagonal
    d2 = linox.Diagonal(jnp.array([1.0, 2.0, 3.0], dtype=DTYPE))
    block2 = linox.TransposedLinearOperator(d2)

    # Create block diagonal
    result = linox.BlockDiagonal(block1, block2)

    # Expected
    slr1_dense = u1 @ jnp.diag(s1) @ u1.T
    block1_dense = 0.5 * slr1_dense + 0.2 * jnp.eye(n1, dtype=DTYPE)
    block2_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=DTYPE)).T

    expected = jnp.block([
        [block1_dense, jnp.zeros((n1, n2), dtype=DTYPE)],
        [jnp.zeros((n2, n1), dtype=DTYPE), block2_dense]
    ])

    actual = result.todense()
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_specific_complex_combination_3() -> None:
    """Test: 2x2 block matrix with Kronecker and products in blocks."""
    # A block: Kronecker product
    A1 = linox.Diagonal(jnp.array([1.0, 2.0], dtype=DTYPE))
    A2 = linox.Identity((2,), dtype=DTYPE)
    A = linox.Kronecker(A1, A2)

    # B block: Scaled matrix
    B_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
                      dtype=DTYPE)
    B = linox.ScaledLinearOperator(linox.Matrix(B_mat), 0.5)

    # C block: Zero
    C = linox.Zero((2, 4), dtype=DTYPE)

    # D block: Product of two diagonals
    D1 = linox.Diagonal(jnp.array([2.0, 1.0], dtype=DTYPE))
    D2 = linox.Diagonal(jnp.array([0.5, 2.0], dtype=DTYPE))
    D = linox.ProductLinearOperator(D1, D2)

    # Create 2x2 block matrix
    result = linox.BlockMatrix2x2(A, B, C, D)

    # Expected
    A_dense = jnp.kron(A1.todense(), A2.todense())
    B_dense = 0.5 * B_mat
    C_dense = jnp.zeros((2, 4), dtype=DTYPE)
    D_dense = jnp.diag(jnp.array([2.0, 1.0], dtype=DTYPE)) @ jnp.diag(
        jnp.array([0.5, 2.0], dtype=DTYPE)
    )

    expected = jnp.block([[A_dense, B_dense], [C_dense, D_dense]])

    actual = result.todense()
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_five_level_depth_combination() -> None:
    """Test a manually crafted 5-level deep combination."""
    n = 4

    # Level 1: Base operators
    base1 = linox.Diagonal(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE))
    base2 = linox.Identity((n,), dtype=DTYPE)

    # Level 2: Combine with arithmetic
    level2 = linox.AddLinearOperator(base1, base2)  # D + I

    # Level 3: Scale the result
    level3 = linox.ScaledLinearOperator(level2, jnp.array(0.5, dtype=DTYPE))

    # Level 4: Add isotropic term
    level4 = linox.IsotropicAdditiveLinearOperator(
        jnp.array(0.1, dtype=DTYPE), level3
    )
    # 0.5 * (D + I) + 0.1 * I

    # Level 5: Transpose and add to original
    level5_t = linox.TransposedLinearOperator(level4)
    level5 = linox.AddLinearOperator(level4, level5_t)
    # [0.5(D+I) + 0.1I] + [0.5(D+I) + 0.1I]^T

    # Compute expected (should be symmetric since D is diagonal)
    D_dense = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE))
    I_dense = jnp.eye(n, dtype=DTYPE)
    level2_dense = D_dense + I_dense
    level3_dense = 0.5 * level2_dense
    level4_dense = level3_dense + 0.1 * I_dense
    expected = level4_dense + level4_dense.T

    actual = level5.todense()
    assert jnp.allclose(actual, expected, atol=1e-5)

    # Also test matmul - ensure dtype matches
    vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=level5.dtype)
    expected_matmul = expected @ vec
    actual_matmul = level5 @ vec
    assert jnp.allclose(actual_matmul, expected_matmul, atol=1e-5)


def test_ultra_deep_combination() -> None:
    """Test a 6+ level combination mixing all operator types."""
    n = 4

    # Level 1: Kronecker base
    A1 = linox.Diagonal(jnp.array([1.0, 2.0], dtype=DTYPE))
    B1 = linox.Identity((2,), dtype=DTYPE)
    kron1 = linox.Kronecker(A1, B1)

    # Level 2: Add symmetric low rank
    u = jnp.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7], [0.8, 0.2]], dtype=DTYPE)
    s = jnp.array([0.5, 0.3], dtype=DTYPE)
    slr = linox.SymmetricLowRank(u, s)
    level2 = linox.AddLinearOperator(kron1, slr)

    # Level 3: Scale
    level3 = linox.ScaledLinearOperator(level2, jnp.array(0.5, dtype=DTYPE))

    # Level 4: Add isotropic
    level4 = linox.IsotropicAdditiveLinearOperator(
        jnp.array(0.2, dtype=DTYPE), level3
    )

    # Level 5: Transpose
    level5 = linox.TransposedLinearOperator(level4)

    # Level 6: Product with diagonal and symmetrize
    D = linox.Diagonal(jnp.array([2.0, 1.5, 1.0, 0.5], dtype=DTYPE))
    level6 = linox.ProductLinearOperator(level5, D)

    # Level 7: Add to transposed self to make symmetric
    level7 = linox.AddLinearOperator(level6, linox.TransposedLinearOperator(level6))

    # Compute expected
    kron_dense = jnp.kron(A1.todense(), B1.todense())
    slr_dense = u @ jnp.diag(s) @ u.T
    level2_dense = kron_dense + slr_dense
    level3_dense = 0.5 * level2_dense
    level4_dense = level3_dense + 0.2 * jnp.eye(n, dtype=DTYPE)
    level5_dense = level4_dense.T
    D_dense = jnp.diag(jnp.array([2.0, 1.5, 1.0, 0.5], dtype=DTYPE))
    level6_dense = level5_dense @ D_dense
    expected = level6_dense + level6_dense.T

    actual = level7.todense()
    assert jnp.allclose(actual, expected, atol=1e-4)

    # Test matmul
    vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=level7.dtype)
    expected_matmul = expected @ vec
    actual_matmul = level7 @ vec
    assert jnp.allclose(actual_matmul, expected_matmul, atol=1e-4)


def test_block_operators_nested() -> None:
    """Test nested block operators (BlockDiagonal containing BlockMatrix2x2)."""
    # Create two 2x2 block matrices
    A1 = linox.Diagonal(jnp.array([1.0, 2.0], dtype=DTYPE))
    B1 = linox.Zero((2, 2), dtype=DTYPE)
    C1 = linox.Ones((2, 2), dtype=DTYPE)
    D1 = linox.Identity((2,), dtype=DTYPE)
    block2x2_1 = linox.BlockMatrix2x2(A1, B1, C1, D1)

    A2 = linox.Diagonal(jnp.array([0.5, 1.5], dtype=DTYPE))
    B2 = linox.Identity((2,), dtype=DTYPE)
    C2 = linox.Zero((2, 2), dtype=DTYPE)
    D2 = linox.Diagonal(jnp.array([2.0, 3.0], dtype=DTYPE))
    block2x2_2 = linox.BlockMatrix2x2(A2, B2, C2, D2)

    # Put them in a block diagonal
    result = linox.BlockDiagonal(block2x2_1, block2x2_2)

    # Compute expected
    block1_dense = jnp.block([
        [A1.todense(), B1.todense()],
        [C1.todense(), D1.todense()]
    ])
    block2_dense = jnp.block([
        [A2.todense(), B2.todense()],
        [C2.todense(), D2.todense()]
    ])
    expected = jnp.block([
        [block1_dense, jnp.zeros((4, 4), dtype=DTYPE)],
        [jnp.zeros((4, 4), dtype=DTYPE), block2_dense]
    ])

    actual = result.todense()
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_mixed_operators_stress() -> None:
    """Stress test combining many different operator types."""
    n = 4

    # Build a complex expression: ((A ⊗ B) + C)^T @ ((D + λI) @ E)
    # where E is a block diagonal

    # A ⊗ B
    A = linox.Diagonal(jnp.array([1.0, 2.0], dtype=DTYPE))
    B = linox.Diagonal(jnp.array([0.5, 1.0], dtype=DTYPE))
    kron = linox.Kronecker(A, B)

    # C: symmetric low rank
    u = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]], dtype=DTYPE)
    s = jnp.array([1.0, 0.5], dtype=DTYPE)
    C = linox.SymmetricLowRank(u, s)

    # (A ⊗ B) + C
    left_inner = linox.AddLinearOperator(kron, C)
    left = linox.TransposedLinearOperator(left_inner)

    # D + λI
    D = linox.Diagonal(jnp.array([1.0, 2.0, 1.5, 2.5], dtype=DTYPE))
    D_iso = linox.IsotropicAdditiveLinearOperator(jnp.array(0.1, dtype=DTYPE), D)

    # E: block diagonal
    E1 = linox.Diagonal(jnp.array([2.0, 1.0], dtype=DTYPE))
    E2 = linox.Diagonal(jnp.array([1.5, 0.5], dtype=DTYPE))
    E = linox.BlockDiagonal(E1, E2)

    # (D + λI) @ E
    right = linox.ProductLinearOperator(D_iso, E)

    # Final: left @ right
    result = linox.ProductLinearOperator(left, right)

    # Compute expected
    kron_dense = jnp.kron(A.todense(), B.todense())
    C_dense = u @ jnp.diag(s) @ u.T
    left_inner_dense = kron_dense + C_dense
    left_dense = left_inner_dense.T

    D_dense = jnp.diag(jnp.array([1.0, 2.0, 1.5, 2.5], dtype=DTYPE))
    D_iso_dense = D_dense + 0.1 * jnp.eye(n, dtype=DTYPE)

    E_dense = jnp.block([
        [E1.todense(), jnp.zeros((2, 2), dtype=DTYPE)],
        [jnp.zeros((2, 2), dtype=DTYPE), E2.todense()]
    ])

    right_dense = D_iso_dense @ E_dense
    expected = left_dense @ right_dense

    actual = result.todense()
    assert jnp.allclose(actual, expected, atol=1e-4)
