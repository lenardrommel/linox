# _matrix_cases.py

from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import pytest

import linox
from linox import LinearOperator
from linox._arithmetic import (
    AddLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
)
from linox._isotropicadd import IsotropicAdditiveLinearOperator
from linox.typing import ShapeType

DType = jnp.float32
CaseType = tuple[LinearOperator, jax.Array]
KeyType = jax.random.PRNGKey

# Set testing options
IsTestBatchShape = False

# No batch dimensions
basic_shapes = [
    (2, 3),
    (3, 3),
    (4, 3),
]

add_shapes = [
    (2, 3),
    (3, 4),
    (4, 3),
]

symmetric_shapes = [
    (2,),
    (3,),
]


basic_matmul_shapes = [
    ((2, 3), (3, 4)),
    ((3, 4), (4, 3)),
]

symmetric_matmul_shapes = [
    ((2, 2), (2, 2)),
    ((3, 3), (3, 3)),
]

# Arbitrary batch dimensions
if IsTestBatchShape:
    batch_shapes_basic = [(1, 2), (2, 3), (1, 1, 1), (1, 2, 1)]
    batch_shapes_general = [
        ((1, 2), (1,)),
        ((2, 1, 3), (2, 3)),
    ]
else:
    batch_shapes_basic = [()]
    batch_shapes_general = [((), ())]


# Sample matrices and operators
def sample_matrices(shape: ShapeType) -> CaseType:
    key = jax.random.PRNGKey(1)
    arr = jax.random.normal(key, shape)
    return linox.Matrix(arr), arr


def sample_identity(shape: ShapeType) -> CaseType:
    return linox.Identity(shape), linox.Identity(shape).todense()


# Arithmetic Fallbacks
def sample_add_operator(shape: ShapeType) -> CaseType:
    arr1 = jax.random.normal(jax.random.PRNGKey(18974), shape)
    arr2 = jax.random.normal(jax.random.PRNGKey(4221), shape)
    linop = AddLinearOperator(
        arr1,
        arr2,
    )
    return linop, arr1 + arr2


def sample_scaled_operator(shape: ShapeType) -> CaseType:
    arr = jax.random.normal(jax.random.PRNGKey(18974), shape)
    scalar = jax.random.normal(jax.random.PRNGKey(4221), ())
    linop = ScaledLinearOperator(
        operator=arr,
        scalar=scalar,
    )
    return linop, scalar * arr


def sample_product_operator(shape: ShapeType) -> CaseType:
    arr1 = jax.random.normal(jax.random.PRNGKey(18974), (*shape[:-2], shape[-2], 4))
    arr2 = jax.random.normal(jax.random.PRNGKey(4221), (*shape[:-2], 4, shape[-1]))
    linop = ProductLinearOperator(
        arr1,
        arr2,
    )
    return linop, arr1 @ arr2


def sample_transposed_operator(shape: ShapeType) -> CaseType:
    arr = jax.random.normal(jax.random.PRNGKey(18974), shape)
    linop = TransposedLinearOperator(
        arr,
    )
    return linop, arr.swapaxes(-1, -2)


def sample_diagonal(shape: ShapeType) -> CaseType:
    key = jax.random.PRNGKey(1)
    arr = jax.random.normal(key, shape)
    return linox.Diagonal(arr), linox._matrix._batch_jnp_diag(arr)


def sample_zero(shape: ShapeType) -> CaseType:
    return linox.Zero(shape), jnp.zeros(shape)


def sample_ones(shape: ShapeType) -> CaseType:
    return linox.Ones(shape), jnp.ones(shape)


def sample_isotropic_additive(shape: ShapeType) -> CaseType:
    arr = jax.random.normal(jax.random.PRNGKey(18974), shape)
    scalar = jax.random.normal(jax.random.PRNGKey(4221), ())
    linop = IsotropicAdditiveLinearOperator(
        scalar,
        arr,
    )
    return linop, scalar * jnp.eye(shape[-1]) + arr


linops_options = [
    sample_zero,
    sample_ones,
    sample_matrices,
    sample_add_operator,
    sample_scaled_operator,
    sample_product_operator,
    sample_transposed_operator,
]

symmetric_options = [
    sample_diagonal,
    sample_identity,
]


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
# @pytest.mark.parametrize("key", random_key)
def case_matrix(
    shape: ShapeType,
    batch_shape: ShapeType,
    # key: KeyType,
) -> CaseType:
    linop, matrix = sample_matrices(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_identity(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_identity((
        *batch_shape,
        shape[-1],
    ))  # (b1, ..., bk, n, n)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_zero(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_zero(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_ones(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_ones(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
# @pytest.mark.parametrize("key", random_key)
def case_diagonal(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_diagonal(batch_shape + shape[-1:])
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_add_operator(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_add_operator(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_scaled_operator(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_scaled_operator(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_product_operator(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_product_operator(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shape", batch_shapes_basic)
@pytest.mark.parametrize("shape", basic_shapes)
def case_transposed_operator(shape: ShapeType, batch_shape: ShapeType) -> CaseType:
    linop, matrix = sample_transposed_operator(batch_shape + shape)
    return linop, matrix


@pytest.mark.parametrize("batch_shapes", batch_shapes_general)
@pytest.mark.parametrize(
    ("shape", "get_linop"),
    list(product(symmetric_shapes, product(symmetric_options, symmetric_options)))
    + list(product(basic_shapes, product(linops_options, linops_options))),
)
def case_add(
    shape: ShapeType,
    batch_shapes: tuple[ShapeType, ShapeType],
    get_linop: tuple[Callable, Callable],
) -> tuple[CaseType, CaseType]:
    get_linop1, get_linop2 = get_linop
    shape1 = batch_shapes[0] + shape
    shape2 = batch_shapes[1] + shape
    return get_linop1(shape1), get_linop2(shape2)


@pytest.mark.parametrize("batch_shapes", batch_shapes_general)
@pytest.mark.parametrize(
    ("shapes", "get_linop"),
    list(
        product(symmetric_matmul_shapes, product(symmetric_options, symmetric_options))
    )
    + list(product(basic_matmul_shapes, product(linops_options, linops_options))),
)
def case_matmul(
    shapes: tuple[ShapeType, ShapeType],
    batch_shapes: tuple[ShapeType, ShapeType],
    get_linop: tuple[Callable, Callable],
    # get_linop2: Callable,
) -> tuple[CaseType, CaseType]:
    get_linop1, get_linop2 = get_linop
    shape1 = batch_shapes[0] + shapes[0]
    shape2 = batch_shapes[1] + shapes[1]
    return get_linop1(shape1), get_linop2(shape2)
