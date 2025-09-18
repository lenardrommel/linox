from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import pytest

from linox._kronecker import Kronecker
from linox.typing import ShapeLike, ShapeType

DType = jnp.float32
CaseType = tuple[Kronecker, jax.Array]
KeyType = jax.random.PRNGKey

basic_shapes = [((2, 2), (4, 4)), ((4, 4), (4, 4)), ((1, 1), (3, 3)), ((1, 2), (3, 4))]

add_shapes = [
    (((2, 2), (4, 4)), ((2, 2), (4, 4))),
    (((2, 2), (4, 4)), ((4, 4), (2, 2))),
    (((2, 2), (2, 2)), ((2, 2), (2, 2))),
]

basic_dims = [1, 2, 3]

add_dims = [(1, 1), (2, 2), (1, 2), (2, 1), (3, 2), (2, 3)]


def sample_kronecker(shape: ShapeLike) -> CaseType:
    (shapeA, shapeB) = shape

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, shapeA)
    B = jax.random.normal(key2, shapeB)

    op = Kronecker(A, B)
    matrix = jnp.kron(A, B)

    assert op.shape == matrix.shape, "Shape mismatch"

    return op, matrix


def sample_nested_kronecker(dim: int, shape: ShapeLike) -> CaseType:
    op = None
    M = None
    for _ in range(dim):
        op, M = sample_kronecker(shape)
        op = Kronecker(op, op)
        M = jnp.kron(M, M)

    assert op.shape == M.shape, "Shape mismatch"

    return op, M


linops_options = [sample_kronecker, sample_nested_kronecker]

# ============================================================================
# Case Generators
# ============================================================================


@pytest.mark.parametrize("dim", basic_dims)
@pytest.mark.parametrize("shape", basic_shapes)
def case_kronecker(shape: ShapeLike, dim: int) -> CaseType:
    """Test case for Kronecker product."""
    return sample_nested_kronecker(dim, shape)


@pytest.mark.parametrize(
    ("dims", "shape", "get_linop"),
    list(
        product(
            add_dims,
            add_shapes,
            [(sample_nested_kronecker, sample_nested_kronecker)],
        )
    ),
)
def case_add(
    dims: tuple[int, int],
    shape: tuple[tuple[ShapeLike, ShapeLike], tuple[ShapeLike, ShapeLike]],
    get_linop: tuple[Callable, Callable],
) -> tuple[CaseType, CaseType]:
    dim1, dim2 = dims
    get_linop1, get_linop2 = get_linop
    shape1, shape2 = shape
    return get_linop1(dim1, shape1), get_linop2(dim2, shape2)


@pytest.mark.parametrize(
    ("dims", "shape", "get_linop"),
    list(
        product(
            add_dims,
            add_shapes,
            [(sample_nested_kronecker, sample_nested_kronecker)],
        )
    ),
)
def case_matmul(
    dims: tuple[int, int],
    shape: tuple[ShapeType, ShapeType],
    get_linop: tuple[Callable, Callable],
) -> tuple[CaseType, CaseType]:
    dim1, dim2 = dims
    get_linop1, get_linop2 = get_linop
    shape1, shape2 = shape
    return get_linop1(dim1, shape1), get_linop2(dim2, shape2)
