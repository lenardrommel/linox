# test_linear_operator.py

"""General tests for the linear operator base class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

import linox

from .test_linox_cases._linops_cases import (
    case_add,
    case_identity,
    case_matrix,
    case_ones,
    case_product,
    case_zero,
)

case_modules = [
    case_zero,
    case_ones,
    case_product,
    case_add,
    case_matrix,
    case_identity,
]

inverse_cases = [case_identity, case_ones, case_matrix]


@pytest.fixture(
    params=[
        pytest.param(seed, id=f"seed{seed}")
        for seed in [
            42,
        ]
    ],
)
def key(request) -> np.random.Generator:
    return jax.random.PRNGKey(request.param)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_case_valid(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    assert isinstance(linop, linox.LinearOperator)
    assert isinstance(matrix, jnp.ndarray)
    assert linop.shape == matrix.shape
    assert linop.size == matrix.size
    assert linop.dtype == matrix.dtype


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matvec(
    linop: linox.LinearOperator,
    matrix: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> None:
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
) -> None:
    val = jax.random.normal(key, shape=(linop.shape[1], ncols))
    mat = jnp.asarray(val, order=order, dtype=linop.dtype)

    linop_matmat = linop @ mat
    matrix_matmat = matrix @ mat

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape, "Shape does not match"
    assert linop_matmat.dtype == matrix_matmat.dtype, (
        f"Expected {linop_matmat.dtype}, got {matrix_matmat.dtype}"
    )

    assert jnp.allclose(linop_matmat, matrix_matmat), (
        "Matrix-matrix product does not match"
    )


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatvec(
    linop: linox.LinearOperator,
    matrix: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> None:
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
) -> None:
    val = jax.random.normal(key, shape=(nrows, linop.shape[-2]))
    mat = jnp.asarray(val, order=order)
    linop_matmat = linox.todense(mat @ linop)
    matrix_matmat = mat @ matrix

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape
    assert linop_matmat.dtype == matrix_matmat.dtype

    jnp.allclose(linop_matmat, matrix_matmat)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_todense(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    linop_dense = linop.todense()

    assert isinstance(linop_dense, jnp.ndarray)
    assert linop_dense.shape == matrix.shape
    assert linop_dense.dtype == matrix.dtype

    jnp.allclose(linop_dense, matrix)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_diagonal(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    linop_diagonal = linox.diagonal(linop)
    matrix_diagonal = jnp.diagonal(matrix)

    assert isinstance(linop_diagonal, jnp.ndarray)
    assert linop_diagonal.ndim == 1
    assert linop_diagonal.shape == matrix_diagonal.shape
    assert linop_diagonal.dtype == matrix_diagonal.dtype

    jnp.allclose(linop_diagonal, matrix_diagonal)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_transpose(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    # NotImplementedError - There is no axis argument for the transpose function.
    matrix_transpose = matrix.transpose()
    linop_transpose = linop.transpose()

    assert isinstance(linop_transpose, linox.LinearOperator)
    assert linop_transpose.shape == matrix_transpose.shape
    assert linop_transpose.dtype == matrix_transpose.dtype

    jnp.allclose(linox.todense(linop_transpose), matrix_transpose)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=inverse_cases)
def test_inv(linop: linox.LinearOperator, matrix: np.ndarray) -> None:
    if linox.is_square(linop):
        expected_exception = None

        try:
            matrix_inv = jnp.linalg.inv(matrix)
        except Exception as e:  # noqa: BLE001
            expected_exception = e

        if expected_exception is None:
            linop_inv = linox.linverse(linop)

            assert isinstance(linop_inv, linox.LinearOperator)
            assert linop_inv.shape == matrix_inv.shape
            assert linop_inv.dtype == matrix_inv.dtype

            jnp.allclose(linop_inv.todense(), matrix_inv, atol=1e-12)
        else:
            with pytest.raises(type(expected_exception)):
                linox.linverse(linop).todense()
    else:
        with pytest.raises(ValueError, match="Argument to inv must have shape"):
            linox.linverse(linop).todense()


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_symmetrize(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    del matrix
    if linox.is_square(linop):
        sym_linop = linox.symmetrize(linop)

        assert linox.is_symmetric(sym_linop)

        assert jnp.array_equal(sym_linop.todense(), sym_linop.todense().T)
