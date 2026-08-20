import jax.numpy as jnp
import linox
from linox.operators import (
    Diagonal,
    Identity,
    IsotropicAdditiveLinearOperator,
    Matrix,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
    Zero,
)


def test_rewrite_isotropic_add() -> None:
    A = Matrix(jnp.eye(3))
    op = 2.0 * Identity(3) + A

    assert isinstance(op, IsotropicAdditiveLinearOperator)
    assert op.shape == (3, 3)
    # Check if scalar is correctly extracted
    op_dense = op.todense()
    expected = 2.0 * jnp.eye(3) + jnp.eye(3)
    assert jnp.allclose(op_dense, expected)


def test_rewrite_zero_elimination() -> None:
    A = Matrix(jnp.ones((2, 2)))
    Z = Zero((2, 2))

    op = A + Z
    assert isinstance(op, Matrix)  # Should be just A (Matrix)
    assert not isinstance(op, linox.operators.AddLinearOperator)


def test_rewrite_diag_plus_lowrank() -> None:
    D = Diagonal(jnp.array([1.0, 2.0, 3.0]))
    U = jnp.ones((3, 1))
    LR = SymmetricLowRank(U)

    op = D + LR
    assert isinstance(op, PositiveDiagonalPlusSymmetricLowRank)

    op2 = D + 2.0 * LR
    assert isinstance(op2, PositiveDiagonalPlusSymmetricLowRank)
    assert op2.low_rank_scale == 2.0


def test_rewrite_multi_add() -> None:
    # 2*I + A + Z -> IsotropicAdd(2, A)
    A = Matrix(jnp.eye(3))
    Z = Zero((3, 3))
    I = Identity(3)

    op = 2.0 * I + A + Z
    # Depending on order and recursion:
    # (2I + A) + Z -> Isotropic + Z -> Isotropic
    assert isinstance(op, IsotropicAdditiveLinearOperator)
