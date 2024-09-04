"""General tests for the linear operator base class."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

import linox

case_modules = [
    ".test_linox_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_linox_cases").glob("*_cases.py")
]


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def key(request) -> np.random.Generator:
    return jax.random.PRNGKey(request.param)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_case_valid(linop: linox.LinearOperator, matrix: jnp.ndarray):
    assert isinstance(linop, linox.LinearOperator)
    assert isinstance(matrix, jnp.ndarray)
    assert linop.ndim == matrix.ndim == 2
    assert linop.shape == matrix.shape
    assert linop.size == matrix.size
    assert linop.dtype == matrix.dtype


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matvec(
    linop: linox.LinearOperator,
    matrix: jnp.ndarray,
    key: jax.random.PRNGKey,
):
    vec = jax.random.normal(key=key, shape=(linop.shape[1],), dtype=jnp.float32)

    linop_matvec = linop @ vec
    matrix_matvec = matrix @ vec

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    jnp.allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("ncols", [1, 2, 15, 100])
@pytest.mark.parametrize("order", ["K"])  # NotImplementedError for order='F' or 'C'
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matmat(
    linop: linox.LinearOperator,
    matrix: jnp.ndarray,
    key: jax.random.PRNGKey,
    ncols: int,
    order: str,
):
    val = jax.random.normal(key, shape=(linop.shape[1], ncols))
    mat = jnp.asarray(val, order=order)

    linop_matmat = linop @ mat
    matrix_matmat = matrix @ mat

    print(linop_matmat.ndim)
    print(linop_matmat.shape)
    print(linop_matmat.dtype)

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape
    assert linop_matmat.dtype == matrix_matmat.dtype

    jnp.allclose(linop_matmat, matrix_matmat)


# FIXME(2bys): This test is failing because of the shape mismatch.
# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_matmat_shape_mismatch(
#     linop: linox.LinearOperator,
#     matrix: jnp.ndarray,
# ):
#     mat = jnp.zeros((linop.shape[1] + 1, 10))

#     # with pytest.raises(ValueError) as excinfo:
#     #     matrix @ mat  # pylint: disable=pointless-statement

#     with pytest.raises(ValueError):
#         linop @ mat  # pylint: disable=pointless-statement


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatvec(
    linop: linox.LinearOperator,
    matrix: jnp.ndarray,
    key: jax.random.PRNGKey,
):
    vec = jax.random.normal(key=key, shape=(linop.shape[0],))

    linop_matvec = vec @ linop
    matrix_matvec = vec @ matrix

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    jnp.allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("nrows", [1, 2, 15, 100])
@pytest.mark.parametrize("order", ["K"])  # NotImplementedError for order='F' or 'C'
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatmat(
    linop: linox.LinearOperator,
    matrix: np.ndarray,
    key: jax.random.PRNGKey,
    nrows: int,
    order: str,
):
    val = jax.random.normal(key, shape=(nrows, linop.shape[-2]))
    mat = jnp.asarray(val, order=order)
    linop_matmat = mat @ linop
    matrix_matmat = mat @ matrix

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape
    assert linop_matmat.dtype == matrix_matmat.dtype

    jnp.allclose(linop_matmat, matrix_matmat)


# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_rmatmat_shape_mismatch(
#     linop: linox.LinearOperator,
#     matrix: jnp.ndarray,
# ):
#     mat = jnp.zeros((10, linop.shape[0] + 1))

#     with pytest.raises(Exception) as excinfo:
#         mat @ matrix  # pylint: disable=pointless-statement

#     with pytest.raises(excinfo.type):
#         mat @ linop  # pylint: disable=pointless-statement


# -------------------------------------------------------------------------------------------

# NotImplementedArray - Matrix object is not callable. Maybe replace this by a permutation matrix.
# @pytest.mark.parametrize(
#     "shape",
#     [
#         # axis=-2 (__matmul__)
#         (1, 1, None, 1),
#         (2, 8, None, 2),
#         # axis=-2 (__matmul__ on arr[..., np.newaxis])
#         (1, 1, 1, None),
#         (5, 2, 3, None),
#         (3, 5, 3, None),
#         # axis < -2
#         (1, None, 1, 1),
#         (5, None, 3, 3),
#         (None, 3, 4, 2),
#     ],
# )
# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_call(
#     linop: linox.LinearOperator,
#     matrix: jnp.ndarray,
#     key: jax.random.PRNGKey,
#     shape: Tuple[Optional[int], ...],
# ):
#     axis = shape.index(None) - len(shape)
#     shape = tuple(entry if entry is not None else linop.shape[1] for entry in shape)

#     arr = jax.random.normal(key=key, shape=shape)

#     linop_call = linop(arr, axis=axis)
#     matrix_call = jnp.moveaxis(jnp.tensordot(matrix, arr, axes=(-1, axis)), 0, axis)

#     assert linop_call.ndim == 4
#     assert linop_call.shape == matrix_call.shape
#     assert linop_call.dtype == matrix_call.dtype

#     jnp.allclose(linop_call, matrix_call)

# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no solve function for the linear operator class.
# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_solve_vector(
#     linop: pn.linops.LinearOperator,
#     matrix: np.ndarray,
#     rng: np.random.Generator,
# ):
#     vec = rng.normal(size=linop.shape[0])

#     if linop.is_square:
#         expected_exception = None

#         try:
#             np_linalg_solve = np.linalg.solve(matrix, vec)
#         except Exception as e:  # pylint: disable=broad-except
#             expected_exception = e

#         if expected_exception is None:
#             linop_solve = linop.solve(vec)

#             assert linop_solve.ndim == 1
#             assert linop_solve.shape == np_linalg_solve.shape
#             assert linop_solve.dtype == np_linalg_solve.dtype

#             np.testing.assert_allclose(linop_solve, np_linalg_solve)
#         else:
#             with pytest.raises(type(expected_exception)):
#                 linop.solve(vec)
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.solve(vec)


# @pytest.mark.parametrize(
#     "batch_shape,ncols", [((), 1), ((), 5), ((1,), 2), ((3, 1), 3)]
# )
# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_solve(
#     linop: pn.linops.LinearOperator,
#     matrix: np.ndarray,
#     rng: np.random.Generator,
#     batch_shape: Tuple[int],
#     ncols: int,
# ):
#     mat = rng.normal(size=batch_shape + (linop.shape[0], ncols))

#     if linop.is_square:
#         expected_exception = None

#         try:
#             np_linalg_solve = np.linalg.solve(matrix, mat)
#         except Exception as e:  # pylint: disable=broad-except
#             expected_exception = e

#         if expected_exception is None:
#             linop_solve = linop.solve(mat)

#             assert linop_solve.ndim == mat.ndim
#             assert linop_solve.shape == np_linalg_solve.shape
#             assert linop_solve.dtype == np_linalg_solve.dtype

#             np.testing.assert_allclose(linop_solve, np_linalg_solve)
#         else:
#             with pytest.raises(type(expected_exception)):
#                 linop.solve(mat)
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.solve(mat)


# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_solve_shape_mismatch(
#     linop: pn.linops.LinearOperator,
#     matrix: np.ndarray,
# ):
#     # Solve with scalar right-hand side
#     b = 42.0

#     with pytest.raises(Exception) as excinfo:
#         np.linalg.solve(matrix, b)

#     with pytest.raises(excinfo.type):
#         linop.solve(b)

#     # Solve with dimension mismatch
#     b = np.ones((2, matrix.shape[1] + 1, 2))

#     with pytest.raises(Exception) as excinfo:
#         np.linalg.solve(matrix, b)

#     with pytest.raises(excinfo.type):
#         linop.solve(b)

# -------------------------------------------------------------------------------------------


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_todense(linop: linox.LinearOperator, matrix: jnp.ndarray):
    linop_dense = linop.todense()

    assert isinstance(linop_dense, jnp.ndarray)
    assert linop_dense.shape == matrix.shape
    assert linop_dense.dtype == matrix.dtype

    jnp.allclose(linop_dense, matrix)


# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no rank function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_rank(linop: linox.LinearOperator, matrix: jnp.ndarray):
#     linop_rank = linop.rank()
#     matrix_rank = np.linalg.matrix_rank(matrix)

#     assert isinstance(linop_rank, np.intp)
#     assert linop_rank.shape == ()
#     assert linop_rank.dtype == matrix_rank.dtype

#     assert linop_rank == matrix_rank

# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no eigvals function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_eigvals(linop: pn.linops.LinearOperator, matrix: np.ndarray):
#     if linop.is_square:
#         linop_eigvals = linop.eigvals()
#         matrix_eigvals = np.linalg.eigvals(matrix)

#         assert isinstance(linop_eigvals, np.ndarray)
#         assert linop_eigvals.shape == matrix_eigvals.shape
#         assert linop_eigvals.dtype == matrix_eigvals.dtype

#         np.testing.assert_allclose(np.sort(linop_eigvals), np.sort(matrix_eigvals))
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.eigvals()
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no cond function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest.mark.parametrize("p", [None, 1, 2, np.inf, "fro", -1, -2, -np.inf])
# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_cond(
#     linop: pn.linops.LinearOperator, matrix: np.ndarray, p: Union[None, int, float, str]
# ):
#     if linop.is_square:
#         linop_cond = linop.cond(p=p)
#         matrix_cond = np.linalg.cond(matrix, p=p)

#         assert linop_cond.shape == ()
#         assert linop_cond.dtype == matrix_cond.dtype

#         try:
#             np.testing.assert_allclose(linop_cond, matrix_cond)
#         except AssertionError as e:
#             if p == -2 and 0 < linop.rank() < linop.shape[0] and linop_cond == np.inf:
#                 # `np.linalg.cond` returns 0.0 for p = -2 if the matrix is singular but
#                 # not zero. This is a bug.
#                 pass
#             else:
#                 raise e
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.cond(p=p)

# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no det function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_det(linop: pn.linops.LinearOperator, matrix: np.ndarray):
#     if linop.is_square:
#         linop_det = linop.det()
#         matrix_det = np.linalg.det(matrix)

#         assert linop_det.shape == ()
#         assert linop_det.dtype == matrix_det.dtype

#         np.testing.assert_allclose(linop_det, matrix_det)
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.det()
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no logabsdet function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_logabsdet(linop: pn.linops.LinearOperator, matrix: np.ndarray):
#     if linop.is_square:
#         linop_logabsdet = linop.logabsdet()
#         _, matrix_logabsdet = np.linalg.slogdet(matrix)

#         assert linop_logabsdet.shape == ()
#         assert linop_logabsdet.dtype == matrix_logabsdet.dtype

#         np.testing.assert_allclose(linop_logabsdet, matrix_logabsdet)
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.logabsdet()
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# NotImplementedError - There is no trace function available for the linear operator class.
# -------------------------------------------------------------------------------------------

# @pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
# def test_trace(linop: pn.linops.LinearOperator, matrix: np.ndarray):
#     if linop.is_square:
#         linop_trace = linop.trace()
#         matrix_trace = np.trace(matrix)

#         assert linop_trace.shape == ()
#         assert linop_trace.dtype == matrix_trace.dtype

#         np.testing.assert_allclose(linop_trace, matrix_trace)
#     else:
#         with pytest.raises(np.linalg.LinAlgError):
#             linop.trace()
# -------------------------------------------------------------------------------------------


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_diagonal(linop: linox.LinearOperator, matrix: jnp.ndarray):
    linop_diagonal = linox.diagonal(linop)
    matrix_diagonal = jnp.diagonal(matrix)

    assert isinstance(linop_diagonal, jnp.ndarray)
    assert linop_diagonal.shape == matrix_diagonal.shape
    assert linop_diagonal.dtype == matrix_diagonal.dtype

    jnp.allclose(linop_diagonal, matrix_diagonal)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_transpose(linop: linox.LinearOperator, matrix: jnp.ndarray):
    # NotImplementedError - There is no axis argument for the transpose function.
    matrix_transpose = matrix.transpose()
    linop_transpose = linop.transpose()

    assert isinstance(linop_transpose, linox.LinearOperator)
    assert linop_transpose.shape == matrix_transpose.shape
    assert linop_transpose.dtype == matrix_transpose.dtype

    jnp.allclose(linop_transpose.todense(), matrix_transpose)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_inv(linop: linox.LinearOperator, matrix: np.ndarray):
    if linox.is_square(linop):
        expected_exception = None

        try:
            matrix_inv = jnp.linalg.inv(matrix)
        except Exception as e:  # pylint: disable=broad-except
            expected_exception = e

        if expected_exception is None:
            linop_inv = linox.inverse(linop)

            assert isinstance(linop_inv, linox.LinearOperator)
            assert linop_inv.shape == matrix_inv.shape
            assert linop_inv.dtype == matrix_inv.dtype

            jnp.allclose(linop_inv.todense(), matrix_inv, atol=1e-12)
        else:
            with pytest.raises(type(expected_exception)):
                linox.inverse(linop).todense()
    else:
        with pytest.raises(ValueError):
            linox.inverse(linop).todense()


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_symmetrize(linop: linox.LinearOperator, matrix: jnp.ndarray):
    if linox.is_square(linop):
        sym_linop = linox.symmetrize(linop)

        assert linox.is_symmetric(sym_linop)

        assert jnp.array_equal(sym_linop.todense(), sym_linop.todense().T)
    # else: #TODO(2bys): Fix this test case.
    #     with pytest.raises(ValueError):
    #         linox.symmetrize(linop)
