import jax
import jax.numpy as jnp
import pytest
import pytest_cases

import linox
from linox import LinearOperator

DTYPE = jnp.float32
CASE_TYPE = tuple[LinearOperator, jax.Array]
SHAPE_TYPE = tuple[int, int]

matrices = [
    jnp.array([[-1.5, 3], [0, -230]], dtype=DTYPE),
    jnp.array([[2, 0], [1, 3]], dtype=DTYPE),
    jnp.array([[2, 0, -1.5], [1, 3, -230]], dtype=DTYPE),
]
spd_matrices = [
    jnp.array([[1.0]], dtype=DTYPE),
    jnp.array([[1.0, -2.0], [-2.0, 5.0]], dtype=DTYPE),
    # random_spd_matrix(np.random.default_rng(597), dim=10),
]
lower_triangular_matrices = [
    jnp.array([[5.0]], dtype=DTYPE),
    jnp.array([[1.0, 0.0], [2.0, 3.0]], dtype=DTYPE),
    jnp.array([[4.0, 0.0, 0.0], [48.0, 60.0, 0.0], [21.0, 39.0, 7.0]], dtype=DTYPE),
]
upper_triangular_matrices = [mat.T for mat in lower_triangular_matrices]

dim = [2, 3, 4, 5]

shapes = [
    (2, 2),
    (2, 3),
    (1, 3),
]

add_shapes = [
    (2, 3),
    (3, 4),
    (4, 3),
]

mul_shapes = [
    ((2, 2), (2, 3)),
    ((2, 3), (3, 3)),
    ((3, 3), (3, 4)),
    ((3, 4), (4, 3)),
]

symmetric_shapes = [((2, 2), (2, 2)), ((3, 3), (3, 3))]


def draw_random_onb(shape: SHAPE_TYPE) -> jnp.ndarray:
    random_key1 = jax.random.PRNGKey(42)
    random_key2 = jax.random.PRNGKey(52)
    A = jax.random.uniform(random_key1, 2 * shape[:1])
    A = 0.5 * (A + A.T)
    A += jnp.eye(shape[0])
    U, _ = jnp.linalg.qr(A)
    S = jax.random.uniform(random_key2, (shape[1],))
    return U[: shape[1], :].T, S


def draw_random_eigen_decomposition(shape: SHAPE_TYPE) -> jnp.ndarray:
    U, S = draw_random_eigen_decomposition(2 * shape[:1])
    return linox.EigenD(U=U, S=S), U @ jnp.diag(S) @ U.T


def draw_random_permutation(n: int) -> jnp.ndarray:
    key = jax.random.PRNGKey(0)
    return jax.random.permutation(key, jnp.arange(0, n))


def get_identity(shape: tuple[int, int]) -> CASE_TYPE:
    dim = shape[0]
    return linox.Identity(dim, dtype=DTYPE), jnp.eye(dim, dtype=DTYPE)


def get_zero(shape: tuple[int, int]) -> CASE_TYPE:
    return linox.Zero(shape=shape, dtype=DTYPE), jnp.zeros(shape, dtype=DTYPE)


def get_ones(shape: tuple[int, int]) -> CASE_TYPE:
    return linox.Ones(shape=shape, dtype=DTYPE), jnp.ones(shape, dtype=DTYPE)


def get_low_rank_plus_isotropic(shape: SHAPE_TYPE) -> CASE_TYPE:
    random_key = jax.random.PRNGKey(2)
    scalar = jax.random.uniform(random_key, ())
    U, S = draw_random_onb(shape)
    return linox.IsotropicScalingPlusLowRank(scalar=scalar, U=U, S=S), scalar * jnp.eye(
        shape[0]
    ) + U @ jnp.diag(S) @ U.T


def get_permutation(shape: SHAPE_TYPE) -> CASE_TYPE:
    perm = draw_random_permutation(shape[0])
    return linox.Permutation(perm=perm), jnp.eye(shape[0], dtype=DTYPE)[perm]


def get_product(linop1: LinearOperator, linop2: LinearOperator) -> CASE_TYPE:
    return linox._arithmetic.ProductLinearOperator(
        linop1, linop2
    ), linop1.todense() @ linop2.todense()


def get_add(linop1: LinearOperator, linop2: LinearOperator) -> CASE_TYPE:
    return linox._arithmetic.AddLinearOperator(
        linop1, linop2
    ), linop1.todense() + linop2.todense()


def get_transpose(linop1: LinearOperator) -> CASE_TYPE:
    return linox._arithmetic.TransposedLinearOperator(linop1), linop1.todense().T


linops_options = [
    # get_identity,
    get_zero,
    get_ones,
]

symmetric_options = [get_permutation]
# isotropic_options = [get_low_rank_plus_isotropic]

all_base_options = linops_options + symmetric_options

# arithmetic_options = []  # [get_transpose]


@pytest.mark.parametrize("matrix", matrices)
def case_matrix(matrix: jnp.ndarray) -> tuple[linox.LinearOperator, jnp.ndarray]:
    linop = linox.Matrix(matrix)
    return linop, matrix


@pytest_cases.case(id="identity")
@pytest.mark.parametrize("dim", dim)
def case_identity(dim: int) -> tuple[linox.LinearOperator, jnp.ndarray]:
    return get_identity((dim, dim))


@pytest.mark.parametrize("dim", dim)
def case_zero(dim: int) -> tuple[linox.LinearOperator, jnp.ndarray]:
    return get_zero((dim, dim))


@pytest.mark.parametrize("dim", dim)
def case_ones(dim: int) -> tuple[linox.LinearOperator, jnp.ndarray]:
    return get_ones((dim, dim))


@pytest.mark.parametrize(("shape1", "shape2"), mul_shapes)
@pytest.mark.parametrize("linop1_fn", linops_options)
@pytest.mark.parametrize("linop2_fn", linops_options)
def case_product(shape1, shape2, linop1_fn, linop2_fn):
    linop1, _ = linop1_fn(shape1)
    linop2, _ = linop2_fn(shape2)
    return get_product(linop1, linop2)


@pytest.mark.parametrize("shape", add_shapes)
@pytest.mark.parametrize("linop1_fn", linops_options)
@pytest.mark.parametrize("linop2_fn", linops_options)
def case_add(shape, linop1_fn, linop2_fn):
    linop1, _ = linop1_fn(shape)
    linop2, _ = linop2_fn(shape)
    return get_add(linop1, linop2)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("linop", linops_options)
def case_transpose(shape, linop_fn):
    linop, _ = linop_fn(shape)
    return get_transpose(linop)


@pytest.mark.parametrize("linop1", linops_options)
@pytest.mark.parametrize("linop2", linops_options)
@pytest.mark.parametrize("shape", add_shapes)
def case_add_combinations(
    linop1, linop2, shape
) -> tuple[linox.LinearOperator, linox.LinearOperator]:
    return linop1(shape), linop2(shape)


@pytest.mark.parametrize("linop1", all_base_options)
@pytest.mark.parametrize("linop2", all_base_options)
@pytest.mark.parametrize("shape", symmetric_shapes)
def case_symmetric_combinations(
    linop1, linop2, shape
) -> tuple[linox.LinearOperator, linox.LinearOperator]:
    shape1, shape2 = shape[0], shape[1]
    return linop1(shape1), linop2(shape2)


@pytest.mark.parametrize("linop1", linops_options)
@pytest.mark.parametrize("linop2", linops_options)
@pytest.mark.parametrize("shapes", mul_shapes)
def case_mul_combinations(
    linop1, linop2, shapes
) -> tuple[linox.LinearOperator, linox.LinearOperator]:
    shape1, shape2 = shapes
    return linop1(shape1), linop2(shape2)
