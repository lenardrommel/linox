# test_utils.py

"""Tests for :mod:`linox.utils`."""

from collections.abc import Iterable

import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox import utils
from linox.typing import ScalarLike, ShapeLike
from tests.test_kernel import case_kernel
from tests.test_linox_cases._matrix_cases import (
    case_add_operator,
    case_diagonal,
    case_identity,
    case_matrix,
    case_ones,
    case_product_operator,
    case_scaled_operator,
    case_transposed_operator,
    case_zero,
)

DTYPE = jnp.float32


def _toeplitz_dense(v: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.arange(v.shape[0])
    return v[jnp.abs(jnp.subtract.outer(indices, indices))]


@pytest_cases.case(id="block-matrix")
def case_block_matrix() -> tuple[linox.LinearOperator, jnp.ndarray]:
    block11 = linox.Matrix(jnp.array([[1.0]], dtype=DTYPE))
    block12 = linox.Matrix(jnp.array([[2.0]], dtype=DTYPE))
    block21 = linox.Matrix(jnp.array([[3.0]], dtype=DTYPE))
    block22 = linox.Matrix(jnp.array([[4.0]], dtype=DTYPE))
    linop = linox.BlockMatrix([[block11, block12], [block21, block22]])
    dense = jnp.block([
        [jnp.array([[1.0]], dtype=DTYPE), jnp.array([[2.0]], dtype=DTYPE)],
        [jnp.array([[3.0]], dtype=DTYPE), jnp.array([[4.0]], dtype=DTYPE)],
    ])
    return linop, dense


@pytest_cases.case(id="block-matrix-2x2")
def case_block_matrix_2x2() -> tuple[linox.LinearOperator, jnp.ndarray]:
    A = linox.Matrix(jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE))
    B = linox.Matrix(jnp.array([[0.5], [1.5]], dtype=DTYPE))
    C = linox.Matrix(jnp.array([[2.0, -1.0]], dtype=DTYPE))
    D = linox.Matrix(jnp.array([[3.0]], dtype=DTYPE))
    linop = linox.BlockMatrix2x2(A, B, C, D)
    dense = jnp.block([[A.todense(), B.todense()], [C.todense(), D.todense()]])
    return linop, dense


@pytest_cases.case(id="block-diagonal")
def case_block_diagonal() -> tuple[linox.LinearOperator, jnp.ndarray]:
    block1 = linox.Matrix(jnp.array([[2.0]], dtype=DTYPE))
    block2 = linox.Matrix(jnp.array([[1.0, 0.1], [0.1, 3.0]], dtype=DTYPE))
    linop = linox.BlockDiagonal(block1, block2)
    dense = jnp.block([
        [block1.todense(), jnp.zeros((1, 2), dtype=DTYPE)],
        [jnp.zeros((2, 1), dtype=DTYPE), block2.todense()],
    ])
    return linop, dense


@pytest_cases.case(id="low-rank")
def case_low_rank() -> tuple[linox.LinearOperator, jnp.ndarray]:
    U = jnp.array([[1.0, 0.0], [0.5, 1.0]], dtype=DTYPE)
    S = jnp.array([2.0, 0.3], dtype=DTYPE)
    V = jnp.array([[0.5, 1.0], [1.0, -0.5]], dtype=DTYPE)
    linop = linox.LowRank(U, S, V)
    dense = U @ jnp.diag(S) @ V.T
    return linop, dense


@pytest_cases.case(id="symmetric-low-rank")
def case_symmetric_low_rank() -> tuple[linox.LinearOperator, jnp.ndarray]:
    U = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    S = jnp.array([1.5, 0.4], dtype=DTYPE)
    linop = linox.SymmetricLowRank(U, S)
    dense = U @ jnp.diag(S) @ U.T
    return linop, dense


@pytest_cases.case(id="isotropic-symmetric-low-rank")
def case_isotropic_scaling_plus_symmetric_low_rank() -> tuple[
    linox.LinearOperator, jnp.ndarray
]:
    scalar = jnp.array(0.5, dtype=DTYPE)
    U = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    S = jnp.array([0.7, 0.2], dtype=DTYPE)
    linop = linox.IsotropicScalingPlusSymmetricLowRank(scalar, U, S)
    dense = scalar * jnp.eye(U.shape[0], dtype=DTYPE) + U @ jnp.diag(S) @ U.T
    return linop, dense


@pytest_cases.case(id="positive-diagonal-symmetric-low-rank")
def case_positive_diagonal_plus_symmetric_low_rank() -> tuple[
    linox.LinearOperator, jnp.ndarray
]:
    diag_entries = jnp.array([2.0, 3.0], dtype=DTYPE)
    diagonal = linox.Diagonal(diag_entries)
    U = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    S = jnp.array([0.6, 0.1], dtype=DTYPE)
    low_rank = linox.SymmetricLowRank(U, S)
    scale = 0.25
    linop = linox.PositiveDiagonalPlusSymmetricLowRank(diagonal, low_rank, scale)
    dense = jnp.diag(diag_entries) + scale * (U @ jnp.diag(S) @ U.T)
    return linop, dense


@pytest_cases.case(id="permutation")
def case_permutation() -> tuple[linox.LinearOperator, jnp.ndarray]:
    perm = jnp.array([2, 0, 1], dtype=jnp.int32)
    linop = linox.Permutation(perm)
    dense = jnp.eye(perm.shape[0], dtype=DTYPE)[perm]
    return linop, dense


@pytest_cases.case(id="toeplitz")
def case_toeplitz() -> tuple[linox.LinearOperator, jnp.ndarray]:
    v = jnp.array([1.0, -0.5, 0.25], dtype=DTYPE)
    linop = linox.Toeplitz(v)
    dense = _toeplitz_dense(v)
    return linop, dense


@pytest_cases.case(id="kronecker")
def case_kronecker_basic() -> tuple[linox.LinearOperator, jnp.ndarray]:
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    B = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=DTYPE)
    linop = linox.Kronecker(A, B)
    dense = jnp.kron(A, B)
    return linop, dense


@pytest_cases.case(id="isotropic-additive")
def case_isotropic_additive() -> tuple[linox.LinearOperator, jnp.ndarray]:
    base_matrix = jnp.array([[2.0, 0.5], [0.5, 1.5]], dtype=DTYPE)
    base_linop = linox.Matrix(base_matrix)
    scalar = jnp.array(0.3, dtype=DTYPE)
    linop = linox.IsotropicAdditiveLinearOperator(scalar, base_linop)
    dense = scalar * jnp.eye(base_matrix.shape[0], dtype=DTYPE) + base_matrix
    return linop, dense


@pytest_cases.case(id="isotropic-additive-array-kernel")
def case_isotropic_additive_array_kernel() -> tuple[
    linox.LinearOperator, jnp.ndarray
]:
    x = jnp.array([[0.1, 0.2], [0.3, -0.1]], dtype=DTYPE)
    kernel_linop = linox.ArrayKernel(_inner_product_kernel, x)
    scalar = jnp.array(0.2, dtype=DTYPE)
    linop = linox.IsotropicAdditiveLinearOperator(scalar, kernel_linop)
    dense = scalar * jnp.eye(x.shape[0], dtype=DTYPE) + kernel_linop.todense()
    return linop, dense


@pytest_cases.case(id="inverse")
def case_inverse_linear_operator() -> tuple[linox.LinearOperator, jnp.ndarray]:
    matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=DTYPE)
    base = linox.Matrix(matrix)
    linop = linox.InverseLinearOperator(base)
    dense = jnp.linalg.inv(matrix)
    return linop, dense


@pytest_cases.case(id="pseudo-inverse")
def case_pseudo_inverse_linear_operator() -> tuple[linox.LinearOperator, jnp.ndarray]:
    matrix = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=DTYPE,
    )
    base = linox.Matrix(matrix)
    linop = linox.PseudoInverseLinearOperator(base)
    dense = jnp.linalg.pinv(matrix)
    return linop, dense


@pytest_cases.case(id="congruence-transform")
def case_congruence_transform() -> tuple[linox.LinearOperator, jnp.ndarray]:
    A = jnp.array([[1.0, 2.0], [0.0, 1.0]], dtype=DTYPE)
    B = jnp.array([[3.0, 0.0], [0.0, 4.0]], dtype=DTYPE)
    linop = linox.congruence_transform(A, B)
    dense = A @ B @ A.T
    return linop, dense


def _inner_product_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x, y) + jnp.array(1e-6, dtype=DTYPE)


@pytest_cases.case(id="add-operator-array-kernel")
def case_add_operator_with_array_kernel() -> tuple[linox.LinearOperator, jnp.ndarray]:
    x = jnp.array([[0.1, 0.2], [0.3, -0.1]], dtype=DTYPE)
    kernel_linop = linox.ArrayKernel(_inner_product_kernel, x)
    base_matrix = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=DTYPE)
    matrix_linop = 1 * linox.Matrix(base_matrix)
    linop = linox.AddLinearOperator(kernel_linop, matrix_linop)
    dense = jnp.asarray(kernel_linop.todense()) + jnp.asarray(matrix_linop.todense())
    return linop, dense


@pytest_cases.case(id="isotropicadd-operator-array-kernel")
def case_isotropicadd_operator_with_array_kernel() -> tuple[
    linox.LinearOperator, jnp.ndarray
]:
    x = jnp.array([[0.1, 0.2], [0.3, -0.1]], dtype=DTYPE)
    kernel_linop = linox.ArrayKernel(_inner_product_kernel, x)
    base_matrix = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=DTYPE)
    matrix_linop = 1 * linox.Matrix(base_matrix)
    scalar = jnp.array(10.0, dtype=DTYPE)
    iso_linop = linox.IsotropicAdditiveLinearOperator(scalar, kernel_linop)
    linop = linox.AddLinearOperator(matrix_linop, iso_linop)
    dense = jnp.asarray(matrix_linop.todense()) + jnp.asarray(iso_linop.todense())
    return linop, dense


LINOP_CASES = [
    case_matrix,
    case_identity,
    case_zero,
    case_ones,
    case_diagonal,
    case_add_operator,
    case_scaled_operator,
    case_product_operator,
    case_transposed_operator,
    case_kernel,
    case_block_matrix,
    case_block_matrix_2x2,
    case_block_diagonal,
    case_low_rank,
    case_symmetric_low_rank,
    case_isotropic_scaling_plus_symmetric_low_rank,
    case_positive_diagonal_plus_symmetric_low_rank,
    case_permutation,
    case_toeplitz,
    case_kronecker_basic,
    case_isotropic_additive,
    case_isotropic_additive_array_kernel,
    case_inverse_linear_operator,
    case_pseudo_inverse_linear_operator,
    case_congruence_transform,
    case_add_operator_with_array_kernel,
    case_isotropicadd_operator_with_array_kernel,
]


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_as_linop_preserves_linear_operator(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    converted = utils.as_linop(linop)
    assert converted is linop
    assert isinstance(converted, linox.LinearOperator)
    converted_dense = jnp.asarray(converted.todense())
    expected_dense_array = jnp.asarray(expected_dense)
    assert jnp.allclose(converted_dense, expected_dense_array)


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(jnp.arange(4.0, dtype=jnp.float32).reshape(2, 2), id="square"),
        pytest.param(
            jnp.linspace(0.0, 5.0, num=6, dtype=jnp.float64).reshape(2, 3),
            id="rectangular",
        ),
    ],
)
def test_as_linop_from_array(array: jnp.ndarray) -> None:
    converted = utils.as_linop(array)
    assert isinstance(converted, linox.Matrix)
    assert jnp.allclose(converted.todense(), array)


def test_as_linop_rejects_unsupported_type() -> None:
    class NotALinearOperator:
        pass

    with pytest.raises(TypeError):
        utils.as_linop(NotALinearOperator())


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_todense_matches_dense(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    dense = utils.todense(linop)
    assert jnp.allclose(jnp.asarray(dense), jnp.asarray(expected_dense))


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_as_dense_matches_dense(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    dense = utils.as_dense(linop)
    assert jnp.allclose(jnp.asarray(dense), jnp.asarray(expected_dense))


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_allclose_handles_linear_operators(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    assert utils.allclose(linop, expected_dense)
    assert utils.allclose(expected_dense, linop)


def test_allclose_detects_difference() -> None:
    mat_a = jnp.eye(3, dtype=jnp.float32)
    mat_b = mat_a.at[0, 0].set(2.0)
    linop = linox.Matrix(mat_a)
    assert not utils.allclose(linop, mat_b)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        pytest.param(3.0, jnp.float32, id="python-float"),
        pytest.param(jnp.array(2.0, dtype=jnp.float64), jnp.float64, id="jax-array"),
        pytest.param(
            linox.Scalar(jnp.array(5.0, dtype=jnp.float32)),
            jnp.float32,
            id="scalar-operator",
        ),
    ],
)
def test_as_scalar_accepts_scalar_like(value: ScalarLike, dtype: jnp.dtype) -> None:
    scalar = utils.as_scalar(value, dtype=dtype)
    assert scalar.shape == ()
    assert scalar.dtype == dtype


def test_as_scalar_rejects_array() -> None:
    with pytest.raises(ValueError):
        utils.as_scalar(jnp.ones((2, 2)))


@pytest.mark.parametrize(
    ("input_shape", "expected"),
    [
        pytest.param(3, (3,), id="int"),
        pytest.param((2, 5), (2, 5), id="tuple"),
        pytest.param([4, 1], (4, 1), id="list"),
    ],
)
def test_as_shape_normalizes(input_shape: ShapeLike, expected: tuple[int, ...]) -> None:
    assert utils.as_shape(input_shape) == expected


def test_as_shape_validates_ndim() -> None:
    with pytest.raises(TypeError):
        utils.as_shape((2, 3), ndim=3)


def test_broadcast_shapes_matches_jax() -> None:
    shapes: Iterable[ShapeLike] = [(2, 1, 3), (1, 3)]
    assert utils._broadcast_shapes(shapes) == jnp.broadcast_shapes(
        *(tuple(s) for s in shapes)
    )


def test_broadcast_shapes_raises_on_incompatible() -> None:
    with pytest.raises(ValueError):
        utils._broadcast_shapes([(3, 4), (2, 5)])
