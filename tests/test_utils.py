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
]


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_as_linop_preserves_linear_operator(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    converted = utils.as_linop(linop)
    assert converted is linop
    assert isinstance(converted, linox.LinearOperator)
    assert jnp.allclose(converted.todense(), expected_dense)


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
    assert jnp.allclose(dense, expected_dense)


@pytest_cases.parametrize_with_cases("linop, expected_dense", cases=LINOP_CASES)
def test_as_dense_matches_dense(
    linop: linox.LinearOperator,
    expected_dense: jnp.ndarray,
) -> None:
    dense = utils.as_dense(linop)
    assert jnp.allclose(dense, expected_dense)


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
