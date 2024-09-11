import jax
import jax.numpy as jnp
import probnum as pn
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
    return linox.Identity(shape=(dim, dim), dtype=DTYPE), jnp.eye(dim, dtype=DTYPE)


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
    return linox._arithmetic.TransposedLinearOperator(linop1), linop1.todense().T  # noqa: SLF001


linops_options = [
    #    get_identity,
    get_zero,
    get_ones,
]

symmetric_options = [get_permutation]
# isotropic_options = [get_low_rank_plus_isotropic]

all_base_options = linops_options + symmetric_options

# arithmetic_options = []  # [get_transpose]


@pytest.mark.parametrize("matrix", matrices)
def case_matrix(matrix: jnp.ndarray) -> tuple[pn.linops.LinearOperator, jnp.ndarray]:
    linop = linox.Matrix(matrix)
    return linop, matrix


@pytest_cases.case(id="identity")
@pytest.mark.parametrize("dim", dim)
def case_identity(dim: int) -> tuple[pn.linops.LinearOperator, jnp.ndarray]:
    return get_identity((dim, dim))


@pytest.mark.parametrize("dim", dim)
def case_zero(dim: int) -> tuple[pn.linops.LinearOperator, jnp.ndarray]:
    return get_zero((dim, dim))


@pytest.mark.parametrize("dim", dim)
def case_ones(dim: int) -> tuple[pn.linops.LinearOperator, jnp.ndarray]:
    return get_ones((dim, dim))


@pytest.mark.parametrize("shape", mul_shapes)
@pytest.mark.parametrize("linop1", linops_options)
@pytest.mark.parametrize("linop2", linops_options)
def case_product(shape, linop1_fn, linop2_fn):
    linop1, _ = linop1_fn(shape[0])
    linop2, _ = linop2_fn(shape[1])
    return get_product(linop1, linop2)


@pytest.mark.parametrize("shape", add_shapes)
@pytest.mark.parametrize("linop1", linops_options)
@pytest.mark.parametrize("linop2", linops_options)
def case_add(shape, linop1_fn, linop2_fn):
    linop1, _ = linop1_fn(shape[0])
    linop2, _ = linop2_fn(shape[1])
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


# @pytest.mark.parametrize()

# @pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
# @pytest_cases.parametrize("matrix", spd_matrices)
# def case_matvec_spd(matrix: np.ndarray):
#     @pn.linops.LinearOperator.broadcast_matvec
#     def _matmul(vec: np.ndarray):
#         return matrix @ vec

#     linop = pn.linops.LambdaLinearOperator(
#         shape=matrix.shape, dtype=matrix.dtype, matmul=_matmul
#     )

#     linop.is_symmetric = True

#     return linop, matrix


# @pytest.mark.parametrize("matrix", matrices)
# def case_matrix(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     return pn.linops.Matrix(matrix), matrix


# @pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
# @pytest_cases.parametrize("matrix", spd_matrices)
# def case_matrix_spd(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     linop = pn.linops.Matrix(matrix)
#     linop.is_symmetric = True

#     return linop, matrix


# @pytest_cases.case(tags=("square", "lower-triangular"))
# @pytest_cases.parametrize("matrix", lower_triangular_matrices)
# def case_matrix_lower_triangular(
#     matrix: np.ndarray,
# ) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     linop = pn.linops.Matrix(matrix)
#     linop.is_lower_triangular = True

#     return linop, matrix


# @pytest_cases.case(tags=("square", "upper-triangular"))
# @pytest_cases.parametrize("matrix", upper_triangular_matrices)
# def case_matrix_upper_triangular(
#     matrix: np.ndarray,
# ) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     linop = pn.linops.Matrix(matrix)
#     linop.is_upper_triangular = True

#     return linop, matrix


# @pytest_cases.case(tags=("square", "symmetric"))
# def case_matrix_symmetric_indefinite() -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     matrix = np.diag((2.1, 1.3, -0.5))

#     linop = pn.linops.aslinop(matrix)
#     linop.is_symmetric = True

#     return linop, matrix


# @pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
# @pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
# def case_identity(n: int) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     return pn.linops.Identity(shape=n), np.eye(n)


# @pytest.mark.parametrize("shape", [(3, 3), (3, 4), (6, 5)])
# def case_zero(shape: tuple) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     return pn.linops.Zero(shape=shape), np.zeros(shape)


# @pytest.mark.parametrize("rng", [np.random.default_rng(42)])
# def case_sparse_matrix(
#     rng: np.random.Generator,
# ) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     matrix = scipy.sparse.rand(
#         10, 10, density=0.1, format="coo", dtype=np.double, random_state=rng
#     )
#     matrix.setdiag(2)
#     matrix = matrix.tocsr()

#     return pn.linops.Matrix(matrix), matrix.toarray()


# @pytest.mark.parametrize("rng", [np.random.default_rng(42)])
# def case_sparse_matrix_singular(
#     rng: np.random.Generator,
# ) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     matrix = scipy.sparse.rand(
#         10, 10, density=0.01, format="csr", dtype=np.double, random_state=rng
#     )

#     return pn.linops.Matrix(matrix), matrix.toarray()


# @pytest.mark.parametrize("rng", [np.random.default_rng(422)])
# def case_inverse(
#     rng: np.random.Generator,
# ) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
#     N = 21

#     v = rng.uniform(0.2, 0.5, N)

#     linop = pn.linops.LambdaLinearOperator(
#         shape=(N, N),
#         dtype=np.double,
#         matmul=lambda x: 2.0 * x + v[:, None] @ (v[None, :] @ x),
#     )

#     return linop.inv(), linop.inv().todense()
