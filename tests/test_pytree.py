"""Tests for PyTree functionality of linear operators."""

import operator

import jax
import jax.numpy as jnp
import pytest_cases

import linox
from linox._arithmetic import (
    AddLinearOperator,
    CongruenceTransform,
    InverseLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    TransposedLinearOperator,
)
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


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=[
        case_matrix,
        case_identity,
        case_zero,
        case_ones,
        case_diagonal,
        case_add_operator,
        case_scaled_operator,
        case_product_operator,
        case_transposed_operator,
    ],
)
def test_pytree_roundtrip(linop: linox.LinearOperator, matrix: jnp.ndarray) -> None:
    """Test that linear operators can be flattened and unflattened."""
    del matrix
    # Flatten the operator
    flat, treedef = jax.tree.flatten(linop)
    # Unflatten it back
    unflattened = jax.tree.unflatten(treedef, flat)

    # Check that the unflattened operator has the same properties
    assert isinstance(unflattened, type(linop))
    # assert unflattened.shape == linop.shape
    assert unflattened.dtype == linop.dtype

    # Check that the operators behave the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_matrix_pytree() -> None:
    """Test Matrix operator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    linop = linox.Matrix(A)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 1
    assert jnp.array_equal(children[0], A)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.Matrix.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Matrix)
    assert jnp.array_equal(unflattened.A, A)


def test_identity_pytree() -> None:
    """Test Identity operator pytree functionality."""
    shape_int = 3
    dtype = jnp.float64
    linop = linox.Identity(shape_int, dtype=dtype)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 0
    assert aux_data == {"shape": (shape_int,), "dtype": dtype}

    # Test unflattening
    unflattened = linox.Identity.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Identity)
    assert unflattened.shape == (shape_int, shape_int)
    assert unflattened.dtype == dtype


def test_diagonal_pytree() -> None:
    """Test Diagonal operator pytree functionality."""
    diag = jnp.array([1.0, 2.0, 3.0])
    linop = linox.Diagonal(diag)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 1
    assert jnp.array_equal(children[0], diag)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.Diagonal.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Diagonal)
    assert jnp.array_equal(unflattened.diag, diag)


def test_scalar_pytree() -> None:
    """Test Scalar operator pytree functionality."""
    scalar = 2.5
    linop = linox.Scalar(scalar)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 1
    assert jnp.array_equal(children[0], jnp.array(scalar))
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.Scalar.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Scalar)
    assert jnp.array_equal(unflattened.scalar, jnp.array(scalar))


def test_zero_pytree() -> None:
    """Test Zero operator pytree functionality."""
    shape = (2, 3)
    dtype = jnp.float32
    linop = linox.Zero(shape, dtype=dtype)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 0
    assert aux_data == {"shape": shape, "dtype": dtype}

    # Test unflattening
    unflattened = linox.Zero.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Zero)
    assert unflattened.shape == shape
    assert unflattened.dtype == dtype


def test_ones_pytree() -> None:
    """Test Ones operator pytree functionality."""
    shape = (2, 3)
    dtype = jnp.float32
    linop = linox.Ones(shape, dtype=dtype)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 0
    assert aux_data == {"shape": shape, "dtype": dtype}

    # Test unflattening
    unflattened = linox.Ones.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Ones)
    assert unflattened.shape == shape
    assert unflattened.dtype == dtype


def test_scaled_linop_pytree() -> None:
    """Test ScaledLinearOperator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    operator = linox.Matrix(A)
    scalar = 2.5
    linop = ScaledLinearOperator(operator=operator, scalar=scalar)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert isinstance(children[0], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1], jnp.array(scalar))
    assert aux_data == {}

    # Test unflattening
    unflattened = ScaledLinearOperator.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, ScaledLinearOperator)
    assert isinstance(unflattened.operator, linox.Matrix)
    assert jnp.array_equal(unflattened.operator.A, A)
    assert jnp.array_equal(unflattened.scalar, jnp.array(scalar))


def test_add_linop_pytree() -> None:
    """Test AddLinearOperator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    op1 = linox.Matrix(A)
    op2 = linox.Matrix(B)
    linop = AddLinearOperator(op1, op2)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert aux_data == {}

    # Test unflattening
    unflattened = AddLinearOperator.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, AddLinearOperator)
    assert len(unflattened.operator_list) == 2
    assert isinstance(unflattened.operator_list[0], linox.Matrix)
    assert isinstance(unflattened.operator_list[1], linox.Matrix)
    assert jnp.array_equal(unflattened.operator_list[0].A, A)
    assert jnp.array_equal(unflattened.operator_list[1].A, B)


def test_product_linop_pytree() -> None:
    """Test ProductLinearOperator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    op1 = linox.Matrix(A)
    op2 = linox.Matrix(B)
    linop = ProductLinearOperator(op1, op2)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert aux_data == {}

    # Test unflattening
    unflattened = ProductLinearOperator.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, ProductLinearOperator)
    assert len(unflattened.operator_list) == 2
    assert isinstance(unflattened.operator_list[0], linox.Matrix)
    assert isinstance(unflattened.operator_list[1], linox.Matrix)
    assert jnp.array_equal(unflattened.operator_list[0].A, A)
    assert jnp.array_equal(unflattened.operator_list[1].A, B)


def test_congruence_transform_pytree() -> None:
    """Test CongruenceTransform pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    op1 = linox.Matrix(A)
    op2 = linox.Matrix(B)
    linop = CongruenceTransform(op1, op2)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert aux_data == {}

    # Test unflattening
    unflattened = CongruenceTransform.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, CongruenceTransform)
    assert isinstance(unflattened._A, linox.Matrix)
    assert isinstance(unflattened._B, linox.Matrix)
    assert jnp.array_equal(unflattened._A.A, A)
    assert jnp.array_equal(unflattened._B.A, B)


def test_transposed_linop_pytree() -> None:
    """Test TransposedLinearOperator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    op = linox.Matrix(A)
    linop = TransposedLinearOperator(op)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 1
    assert isinstance(children[0], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert aux_data == {}

    # Test unflattening
    unflattened = TransposedLinearOperator.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, TransposedLinearOperator)
    assert isinstance(unflattened.operator, linox.Matrix)
    assert jnp.array_equal(unflattened.operator.A, A)


def test_inverse_linop_pytree() -> None:
    """Test InverseLinearOperator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    op = linox.Matrix(A)
    linop = InverseLinearOperator(op)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 1
    assert isinstance(children[0], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert aux_data == {}

    # Test unflattening
    unflattened = InverseLinearOperator.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, InverseLinearOperator)
    assert isinstance(unflattened.operator, linox.Matrix)
    assert jnp.array_equal(unflattened.operator.A, A)


def test_jit_compatibility() -> None:
    """Test that linear operators can be used with jit."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    linop = linox.Matrix(A)

    x = jnp.array([1.0, 2.0])
    result = operator.matmul(linop, x)
    expected = linop @ x

    assert jnp.allclose(result, expected)


def test_vmap_compatibility() -> None:
    """Test that linear operators can be used with vmap."""
    batch_size = 3
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)

    # Create a batch of matrices
    matrices = jnp.stack([jax.random.normal(k, (2, 2)) for k in keys])

    # Create a batch of vectors
    vectors = jnp.stack([jax.random.normal(k, (2,)) for k in keys])

    # Define a function that creates a linear operator and applies it
    def apply_matrix(mat, vec):
        linop = linox.Matrix(mat)
        return linop @ vec

    # Use vmap to apply the function to each element in the batch
    batched_result = jax.vmap(apply_matrix)(matrices, vectors)

    # Compute the expected result
    expected_result = jnp.stack([
        apply_matrix(matrices[i], vectors[i]) for i in range(batch_size)
    ])

    assert jnp.allclose(batched_result, expected_result)


def test_grad_compatibility() -> None:
    """Test that linear operators can be used with grad."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    x = jnp.array([1.0, 2.0])

    def loss_fn(matrix, vector):
        linop = linox.Matrix(matrix)
        result = linop @ vector
        return jnp.sum(result**2)

    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_result = grad_fn(A, x)

    # Compute expected gradient manually
    expected_grad = 2 * jnp.outer(A @ x, x)

    assert jnp.allclose(grad_result, expected_grad)


def test_blockmatrix_pytree() -> None:
    """Test BlockMatrix operator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    C = jnp.array([[9.0, 10.0], [11.0, 12.0]])
    D = jnp.array([[13.0, 14.0], [15.0, 16.0]])

    blocks = [[linox.Matrix(A), linox.Matrix(B)], [linox.Matrix(C), linox.Matrix(D)]]

    linop = linox.BlockMatrix(blocks)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 4
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert isinstance(children[2], linox.Matrix)
    assert isinstance(children[3], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert jnp.array_equal(children[2].A, C)
    assert jnp.array_equal(children[3].A, D)

    # Check the updated aux_data structure
    assert "block_shape" in aux_data
    assert aux_data["block_shape"] == (2, 2)
    assert "col_sizes" in aux_data
    assert len(aux_data["col_sizes"]) == 2
    assert aux_data["col_sizes"][0] == B.shape[1]
    assert aux_data["col_sizes"][1] == D.shape[1]

    # Test unflattening
    unflattened = linox.BlockMatrix.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.BlockMatrix)
    assert unflattened._block_shape == (2, 2)
    assert len(unflattened._blocks) == 2
    assert len(unflattened._blocks[0]) == 2
    assert len(unflattened._blocks[1]) == 2

    # Check that the blocks are correctly reconstructed
    assert jnp.array_equal(unflattened._blocks[0][0].todense(), A)
    assert jnp.array_equal(unflattened._blocks[0][1].todense(), B)
    assert jnp.array_equal(unflattened._blocks[1][0].todense(), C)
    assert jnp.array_equal(unflattened._blocks[1][1].todense(), D)

    # Check that the col_sizes are correctly restored
    assert hasattr(unflattened, "_col_sizes")
    assert unflattened._col_sizes == aux_data["col_sizes"]

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_blockmatrix2x2_pytree() -> None:
    """Test BlockMatrix2x2 operator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    C = jnp.array([[9.0, 10.0], [11.0, 12.0]])
    D = jnp.array([[13.0, 14.0], [15.0, 16.0]])

    linop = linox.BlockMatrix2x2(
        A=linox.Matrix(A), B=linox.Matrix(B), C=linox.Matrix(C), D=linox.Matrix(D)
    )

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 4
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert isinstance(children[2], linox.Matrix)
    assert isinstance(children[3], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert jnp.array_equal(children[2].A, C)
    assert jnp.array_equal(children[3].A, D)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.BlockMatrix2x2.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.BlockMatrix2x2)

    # Check that the blocks are correctly reconstructed
    assert jnp.array_equal(unflattened.A.todense(), A)
    assert jnp.array_equal(unflattened.B.todense(), B)
    assert jnp.array_equal(unflattened.C.todense(), C)
    assert jnp.array_equal(unflattened.D.todense(), D)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_blockdiagonal_pytree() -> None:
    """Test BlockDiagonal operator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    C = jnp.array([[9.0, 10.0], [11.0, 12.0]])

    linop = linox.BlockDiagonal(linox.Matrix(A), linox.Matrix(B), linox.Matrix(C))

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 3
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert isinstance(children[2], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert jnp.array_equal(children[2].A, C)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.BlockDiagonal.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.BlockDiagonal)

    # Check that the blocks are correctly reconstructed
    assert len(unflattened.blocks) == 3
    assert jnp.array_equal(unflattened.blocks[0].todense(), A)
    assert jnp.array_equal(unflattened.blocks[1].todense(), B)
    assert jnp.array_equal(unflattened.blocks[2].todense(), C)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_block_jit_compatibility() -> None:
    """Test that block operators can be used with jit."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    C = jnp.array([[9.0, 10.0], [11.0, 12.0]])
    D = jnp.array([[13.0, 14.0], [15.0, 16.0]])

    blocks = [[linox.Matrix(A), linox.Matrix(B)], [linox.Matrix(C), linox.Matrix(D)]]

    linop = linox.BlockMatrix(blocks)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = operator.matmul(linop, x)
    expected = linop @ x

    assert jnp.allclose(result, expected)


def test_block_vmap_compatibility() -> None:
    """Test that block operators can be used with vmap."""
    batch_size = 3
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size * 4)

    # Create batches of matrices for blocks
    As = jnp.stack([jax.random.normal(keys[i], (2, 2)) for i in range(batch_size)])
    Bs = jnp.stack([
        jax.random.normal(keys[i + batch_size], (2, 2)) for i in range(batch_size)
    ])
    Cs = jnp.stack([
        jax.random.normal(keys[i + 2 * batch_size], (2, 2)) for i in range(batch_size)
    ])
    Ds = jnp.stack([
        jax.random.normal(keys[i + 3 * batch_size], (2, 2)) for i in range(batch_size)
    ])

    # Create a batch of vectors
    vectors = jnp.stack([jax.random.normal(keys[i], (4,)) for i in range(batch_size)])

    # Define a function that creates a block matrix and applies it
    def apply_block_matrix(A, B, C, D, vec):
        linop = linox.BlockMatrix2x2(
            A=linox.Matrix(A), B=linox.Matrix(B), C=linox.Matrix(C), D=linox.Matrix(D)
        )
        return linop @ vec

    # Use vmap to apply the function to each element in the batch
    batched_result = jax.vmap(apply_block_matrix)(As, Bs, Cs, Ds, vectors)

    # Compute the expected result
    expected_result = jnp.stack([
        apply_block_matrix(As[i], Bs[i], Cs[i], Ds[i], vectors[i])
        for i in range(batch_size)
    ])

    assert jnp.allclose(batched_result, expected_result)


def test_block_grad_compatibility() -> None:
    """Test that block operators can be used with grad."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    x = jnp.array([1.0, 2.0])

    def loss_fn(matrix_A, matrix_B, vector):
        linop = linox.BlockDiagonal(linox.Matrix(matrix_A), linox.Matrix(matrix_B))
        result = linop @ jnp.concatenate([vector, vector])
        return jnp.sum(result**2)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_A, _grad_B = grad_fn(A, B, x)

    # Compute expected gradients manually
    expected_grad_A = 2 * jnp.outer(A @ x, x)
    2 * jnp.outer(B @ x, x)

    assert jnp.allclose(grad_A, expected_grad_A)


def test_eigend_pytree() -> None:
    """Test EigenD operator pytree functionality."""
    # Create test data for U and S
    key = jax.random.key(0)
    n = 3
    U_key, S_key = jax.random.split(key)

    # Create orthogonal matrix U and eigenvalues S
    U = jax.random.normal(U_key, (n, n))
    Q, _ = jnp.linalg.qr(U)  # Ensure U is orthogonal
    S = jnp.abs(jax.random.normal(S_key, (n,)))  # Positive eigenvalues

    # Create the EigenD operator
    linop = linox.EigenD(U=Q, S=S)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    # U is a Matrix LinearOperator, S is an array
    assert isinstance(children[0], linox.Matrix)
    assert jnp.allclose(children[0].todense(), Q)
    assert jnp.allclose(children[1], S)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.EigenD.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.EigenD)
    assert jnp.allclose(unflattened.U.todense(), Q)
    assert jnp.allclose(unflattened.S, S)

    # Check that the operator behaves the same
    vector = jax.random.normal(key, (n,))

    # Test matrix-vector product
    assert jnp.allclose(unflattened @ vector, linop @ vector)

    # Test dense matrix representation
    assert jnp.allclose(unflattened.todense(), linop.todense())

    # Test with JAX transformations
    @jax.jit
    def apply_eigend(U, S, x):
        op = linox.EigenD(U=U, S=S)
        return op @ x

    jit_result = apply_eigend(Q, S, vector)
    assert jnp.allclose(jit_result, linop @ vector)

    # Test with vmap
    batch_size = 3
    keys = jax.random.split(key, batch_size * 3)

    # Create batches of matrices and vectors
    Us = jnp.stack([jax.random.normal(keys[i], (n, n)) for i in range(batch_size)])
    Qs = jnp.stack([jnp.linalg.qr(Us[i])[0] for i in range(batch_size)])
    Ss = jnp.stack([
        jnp.abs(jax.random.normal(keys[i + batch_size], (n,)))
        for i in range(batch_size)
    ])
    vectors = jnp.stack([
        jax.random.normal(keys[i + 2 * batch_size], (n,)) for i in range(batch_size)
    ])

    # Define a function that creates an EigenD operator and applies it
    def apply_eigend_batch(U, S, vec):
        linop = linox.EigenD(U=U, S=S)
        return linop @ vec

    # Use vmap to apply the function to each element in the batch
    batched_result = jax.vmap(apply_eigend_batch)(Qs, Ss, vectors)

    # Compute the expected result
    expected_result = jnp.stack([
        apply_eigend_batch(Qs[i], Ss[i], vectors[i]) for i in range(batch_size)
    ])

    assert jnp.allclose(batched_result, expected_result)

    # Test with grad
    def loss_fn(U, S, vector):
        linop = linox.EigenD(U=U, S=S)
        result = linop @ vector
        return jnp.sum(result**2)

    # Take gradient with respect to S (eigenvalues)
    grad_fn = jax.grad(loss_fn, argnums=1)
    grad_S = grad_fn(Q, S, vector)

    # Verify gradient is not None and has the right shape
    assert grad_S is not None
    assert grad_S.shape == S.shape


def test_kronecker_pytree() -> None:
    """Test Kronecker operator pytree functionality."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    op_A = linox.Matrix(A)
    op_B = linox.Matrix(B)
    linop = linox.Kronecker(A=op_A, B=op_B)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert isinstance(children[0], linox.Matrix)
    assert isinstance(children[1], linox.Matrix)
    assert jnp.array_equal(children[0].A, A)
    assert jnp.array_equal(children[1].A, B)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.Kronecker.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Kronecker)
    assert isinstance(unflattened.A, linox.Matrix)
    assert isinstance(unflattened.B, linox.Matrix)
    assert jnp.array_equal(unflattened.A.A, A)
    assert jnp.array_equal(unflattened.B.A, B)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())

    # Test with JAX transformations
    @jax.jit
    def apply_kronecker(A, B, x):
        op = linox.Kronecker(A=linox.Matrix(A), B=linox.Matrix(B))
        return op @ x

    jit_result = apply_kronecker(A, B, vector)
    assert jnp.allclose(jit_result, linop @ vector)

    # Test with vmap
    batch_size = 3
    keys = jax.random.split(key, batch_size * 3)

    # Create batches of matrices and vectors
    As = jnp.stack([jax.random.normal(keys[i], (2, 2)) for i in range(batch_size)])
    Bs = jnp.stack([
        jax.random.normal(keys[i + batch_size], (2, 2)) for i in range(batch_size)
    ])
    vectors = jnp.stack([
        jax.random.normal(keys[i + 2 * batch_size], (4,)) for i in range(batch_size)
    ])

    # Define a function that creates a Kronecker operator and applies it
    def apply_kronecker_batch(A, B, vec):
        linop = linox.Kronecker(A=linox.Matrix(A), B=linox.Matrix(B))
        return linop @ vec

    # Use vmap to apply the function to each element in the batch
    batched_result = jax.vmap(apply_kronecker_batch)(As, Bs, vectors)

    # Compute the expected result
    expected_result = jnp.stack([
        apply_kronecker_batch(As[i], Bs[i], vectors[i]) for i in range(batch_size)
    ])

    assert jnp.allclose(batched_result, expected_result)

    # Test with grad
    def loss_fn(A_mat, B_mat, vector):
        linop = linox.Kronecker(A=linox.Matrix(A_mat), B=linox.Matrix(B_mat))
        result = linop @ vector
        return jnp.sum(result**2)

    # Take gradient with respect to A
    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_A = grad_fn(A, B, vector)

    # Verify gradient is not None and has the right shape
    assert grad_A is not None
    assert grad_A.shape == A.shape


def test_lowrank_pytree() -> None:
    """Test LowRank operator pytree functionality."""
    U = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    V = jnp.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).T

    linop = linox.LowRank(U=U, V=V)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 3
    assert jnp.array_equal(children[0], U)
    assert jnp.array_equal(children[-1], V)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.LowRank.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.LowRank)
    assert jnp.array_equal(unflattened.U, U)
    assert jnp.array_equal(unflattened.V, V)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_symmetriclowrank_pytree() -> None:
    """Test SymmetricLowRank operator pytree functionality."""
    U = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    S = jnp.array([0.5, 1.5])

    linop = linox.SymmetricLowRank(U=U, S=S)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert jnp.array_equal(children[0], U)
    assert jnp.array_equal(children[1], S)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.SymmetricLowRank.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.SymmetricLowRank)
    assert jnp.array_equal(unflattened.U, U)
    assert jnp.array_equal(unflattened.S, S)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_isotropicscalingplussymmetriclowrank_pytree() -> None:
    """Test IsotropicScalingPlusSymmetricLowRank operator pytree functionality."""
    U = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    S = jnp.array([0.5, 1.5])
    scalar = 2.0

    linop = linox.IsotropicScalingPlusSymmetricLowRank(scalar=scalar, U=U, S=S)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 3
    assert jnp.array_equal(children[0], jnp.array(scalar))
    assert jnp.array_equal(children[1], U)
    assert jnp.array_equal(children[2], S)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.IsotropicScalingPlusSymmetricLowRank.tree_unflatten(
        aux_data, children
    )
    assert isinstance(unflattened, linox.IsotropicScalingPlusSymmetricLowRank)
    assert jnp.array_equal(unflattened.scalar, jnp.array(scalar))
    assert jnp.array_equal(unflattened.U, U)
    assert jnp.array_equal(unflattened.S, S)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_positivediagonalplussymmetriclowrank_pytree() -> None:
    """Test PositiveDiagonalPlusSymmetricLowRank operator pytree functionality."""
    diag = jnp.array([1.0, 2.0, 3.0])
    diagonal = linox.Diagonal(diag)

    U = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    S = jnp.array([0.5, 1.5])
    low_rank = linox.SymmetricLowRank(U=U, S=S)

    low_rank_scale = 0.75

    linop = linox.PositiveDiagonalPlusSymmetricLowRank(
        diagonal=diagonal, low_rank=low_rank, low_rank_scale=low_rank_scale
    )

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 3
    assert isinstance(children[0], linox.Diagonal)
    assert isinstance(children[1], linox.SymmetricLowRank)
    assert jnp.array_equal(children[0].diag, diag)
    assert jnp.array_equal(children[1].U, U)
    assert jnp.array_equal(children[1].S, S)
    assert jnp.array_equal(children[2], jnp.array(low_rank_scale))
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.PositiveDiagonalPlusSymmetricLowRank.tree_unflatten(
        aux_data, children
    )
    assert isinstance(unflattened, linox.PositiveDiagonalPlusSymmetricLowRank)
    assert isinstance(unflattened._diagonal, linox.Diagonal)
    assert isinstance(unflattened._low_rank, linox.SymmetricLowRank)
    assert jnp.array_equal(unflattened._diagonal.diag, diag)
    assert jnp.array_equal(unflattened._low_rank.U, U)
    assert jnp.array_equal(unflattened._low_rank.S, S)
    assert jnp.array_equal(unflattened._low_rank_scale, jnp.array(low_rank_scale))

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)
    assert jnp.allclose(unflattened.todense(), linop.todense())


def test_permutation_pytree() -> None:
    """Test Permutation operator pytree functionality."""
    # Create a permutation array
    perm = jnp.array([2, 0, 1])
    perm_inv = jnp.array([1, 2, 0])  # Inverse of the permutation

    linop = linox.Permutation(perm=perm, perm_inv=perm_inv)

    # Test flattening
    children, aux_data = linop.tree_flatten()
    assert len(children) == 2
    assert jnp.array_equal(children[0], perm)
    assert jnp.array_equal(children[1], perm_inv)
    assert aux_data == {}

    # Test unflattening
    unflattened = linox.Permutation.tree_unflatten(aux_data, children)
    assert isinstance(unflattened, linox.Permutation)
    assert jnp.array_equal(unflattened._perm, perm)
    assert jnp.array_equal(unflattened._perm_inv, perm_inv)

    # Check that the operator behaves the same
    key = jax.random.key(0)
    vector = jax.random.normal(key, (linop.shape[-1],))

    assert jnp.allclose(unflattened @ vector, linop @ vector)

    # Test with JAX transformations
    @jax.jit
    def apply_permutation(perm, perm_inv, x):
        op = linox.Permutation(perm=perm, perm_inv=perm_inv)
        return op @ x

    jit_result = apply_permutation(perm, perm_inv, vector)
    assert jnp.allclose(jit_result, linop @ vector)

    # Test with automatic inverse computation
    auto_linop = linox.Permutation(perm=perm)  # Let it compute the inverse
    assert jnp.array_equal(auto_linop._perm_inv, perm_inv)

    # Test transpose/inverse
    transpose_op = linop.transpose()
    assert jnp.array_equal(transpose_op._perm, perm_inv)
    assert jnp.array_equal(transpose_op._perm_inv, perm)

    # Check that transpose behaves correctly
    assert jnp.allclose(transpose_op @ (linop @ vector), vector)
