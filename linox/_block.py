"""Block class for Linox."""

from functools import reduce
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from linox import _utils as utils
from linox._linear_operator import LinearOperator

LinearOperatorLike = LinearOperator | jax.Array


class BlockMatrix(LinearOperator):
    """A block matrix where each block is represented by a linear operator.

    Parameters
    ----------
    blocks : List[List[LinearOperatorLike]]
        A nested list of `LinearOperatorLike` instances representing the blocks
        of the matrix.
        Shapes must be valid such that creating a block matrix is possible.
    """

    def __init__(self, blocks: list[list[LinearOperatorLike]]) -> None:
        self._blocks = [[utils.as_linop(x) for x in sub_list] for sub_list in blocks]
        self._block_shape = (len(self._blocks), len(self._blocks[0]))

        # Determine the dtype
        dtype = self._blocks[0][0].dtype
        for i, j in product(range(len(self._blocks)), range(len(self._blocks[0]))):
            assert (  # noqa: S101
                self._blocks[i][j].dtype == dtype
            ), "All blocks must have the same dtype."
            expected_shape = (self._blocks[i][0].shape[0], self._blocks[0][j].shape[1])
            if self._blocks[i][j].shape != expected_shape:
                msg = (
                    f"Shape error in block [{i}, {j}]: "
                    f"Expected shape {expected_shape}, "
                    f"got shape {self._blocks[i, j].shape}."
                )
                raise ValueError(msg)

            # Compute the total shape of the block matrix
            num_rows = sum(row[0].shape[0] for row in self._blocks)
            num_cols = sum(block.shape[1] for block in self._blocks[0][:])

            # Initialize the super class
            super().__init__(shape=(num_rows, num_cols), dtype=dtype)

            # Calculate the split indices for splitting input
            self._split_indices = jnp.array(
                tuple(block.shape[1] for block in self._blocks[0])
            ).cumsum()[:-1]

    @property
    def blocks(self) -> jnp.ndarray:
        """The blocks of the block matrix."""
        return self._blocks

    def _split_input(self, x: jnp.ndarray) -> list[jnp.ndarray]:
        """Split the input into blocks."""
        return jnp.split(x, self._split_indices, axis=-2)

    def _matmul(self, arr: jnp.ndarray) -> jnp.ndarray:
        # Split the input according to the block structure
        arr_split = self._split_input(arr)
        row_wise_results = []

        # Perform matrix multiplication for each row of blocks
        for i in range(self._block_shape[0]):
            row_wise_results.append(  # noqa: PERF401
                jnp.sum(
                    jnp.array([
                        block @ cur_x
                        for block, cur_x in zip(self.blocks[i], arr_split, strict=False)
                    ]),
                    axis=0,
                )
            )

        # Concatenate the results to form the final matrix
        return jnp.concatenate(row_wise_results, axis=-2)

    def todense(self) -> jax.Array:
        """Convert the block matrix to a dense matrix."""
        _blocks = [
            [None for _ in range(self._block_shape[1])]
            for _ in range(self._block_shape[0])
        ]
        for i, j in np.ndindex(self._block_shape):
            _blocks[i][j] = self._blocks[i][j].todense()
        return jnp.block(_blocks)

    def transpose(self) -> "BlockMatrix":
        """Transpose the block matrix."""
        blocks_t = [
            [self._blocks[i][j].transpose() for i in range(self._block_shape[0])]
            for j in range(self._block_shape[1])
        ]

        # for i, j in np.ndindex(self._blocks.shape):
        #     blocks_t[i][j] = self._blocks[i][j].transpose()

        return BlockMatrix(blocks_t)


class BlockMatrix2x2(LinearOperator):
    """2x2 Block Matrix.

    A linear operator that represents a linear system of the form:

        | A B | | x | = | u |
        | C D | | y | = | v |

    Parameters
    ----------
    A : LinearOperatorLike
        The top-left block of the matrix.
    B : LinearOperatorLike
        The top-right block of the matrix.
    C : LinearOperatorLike
        The bottom-left block of the matrix.
    D : LinearOperatorLike
        The bottom-right block of the matrix.
    """

    def __init__(
        self,
        A: LinearOperatorLike,
        B: LinearOperatorLike,
        C: LinearOperatorLike,
        D: LinearOperatorLike,
    ) -> None:
        self.A = utils.as_linop(A)
        self.B = utils.as_linop(B)
        self.C = utils.as_linop(C)
        self.D = utils.as_linop(D)

        dtype = reduce(
            jnp.promote_types, (self.A.dtype, self.B.dtype, self.C.dtype, self.D.dtype)
        )

        super().__init__(
            shape=(A.shape[0] + D.shape[0], A.shape[1] + D.shape[1]), dtype=dtype
        )

    def _split_input(self, arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.split(arr, [self.A.shape[1]], axis=-2)

    def _matmul(self, arr: jnp.ndarray) -> jnp.ndarray:
        arr0, arr1 = self._split_input(arr)
        return jnp.concatenate(
            [
                self.A @ arr0 + self.B @ arr1,
                self.C @ arr0 + self.D @ arr1,
            ],
            axis=-2,
        )

    def todense(self) -> jax.Array:
        """Convert the block matrix to a dense matrix."""
        A = self.A.todense()
        B = self.B.todense()
        C = self.C.todense()
        D = self.D.todense()

        return jnp.block([[A, B], [C, D]])

    def transpose(self) -> "BlockMatrix2x2":
        return BlockMatrix2x2(
            A=self.A.transpose(),
            B=self.C.transpose(),
            C=self.B.transpose(),
            D=self.D.transpose(),
        )


class BlockDiagonal(LinearOperator):
    def __init__(self, *blocks: LinearOperator) -> None:
        if len(blocks) < 1:
            msg = "At least one block must be given."
            raise ValueError(msg)

        self.blocks = [utils.as_linop(block) for block in blocks]
        self._all_blocks_square = all(
            block.shape[0] == block.shape[1] for block in blocks
        )

        dtype = reduce(jnp.promote_types, (block.dtype for block in blocks))
        shape_0 = sum(block.shape[0] for block in blocks)
        shape_1 = sum(block.shape[1] for block in blocks)
        self.split_indices = tuple(
            jnp.array([block.shape[1] for block in blocks]).cumsum()[:-1]
        )

        super().__init__((shape_0, shape_1), dtype)

    def _split_input(self, x: jax.Array) -> list[jax.Array]:
        return jnp.split(x, self.split_indices, axis=-2)

    def _matmul(self, x: jax.Array) -> jax.Array:
        res = jnp.concatenate(
            [
                block @ cur_x
                for block, cur_x in zip(self.blocks, self._split_input(x), strict=False)
            ],
            axis=-2,
        )
        return res

    def todense(self) -> jax.Array:
        return jax.scipy.linalg.block_diag(*[block.todense() for block in self.blocks])

    def transpose(self) -> "BlockDiagonal":
        return BlockDiagonal(*[block.transpose() for block in self.blocks])
