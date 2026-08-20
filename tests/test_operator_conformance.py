"""Universal semantic properties, asserted against every operator type.

The rest of the suite tests operators one at a time, which is how three
silently-wrong results reached a release: `svd_partial` pairing the wrong
singular vectors, `lsqrt` returning inverses, and `s*I + A` inverting a
non-symmetric operand via `eigh`. Each was invisible because the test asserted
the same thing the implementation did.

These tests instead assert *mathematical properties* that hold for any linear
operator, against a registry of every operator linox defines. A property test
cannot agree with a buggy implementation, because it never mentions one.

Adding an operator to `OPERATORS` opts it into all of them.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import linox
import pytest
from linox import (
    PSD,
    SPD,
    BlockDiagonal,
    BlockMatrix2x2,
    Diagonal,
    EigenD,
    Identity,
    Kronecker,
    LowRank,
    Matrix,
    Ones,
    Permutation,
    Scalar,
    Sym,
    SymmetricLowRank,
    Toeplitz,
    Zero,
)
from linox.operators.arithmetic import (
    AddLinearOperator,
    CongruenceTransform,
    ProductLinearOperator,
    ScaledLinearOperator,
    lsqrt,
)

jax.config.update("jax_enable_x64", True)

F64 = jnp.float64
KEY = jax.random.PRNGKey(20240819)


def _spd(n, seed=0):
    key = jax.random.fold_in(KEY, seed)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n), dtype=F64))
    return Q @ jnp.diag(jnp.linspace(1.0, 5.0, n)) @ Q.T


def _dense(n, m=None, seed=1):
    return jax.random.normal(jax.random.fold_in(KEY, seed), (n, m or n), dtype=F64)


@dataclasses.dataclass(frozen=True)
class Case:
    """An operator plus what is true about it.

    The flags are deliberately narrow: they say what the *operator* supports,
    not what a particular implementation happens to return.
    """

    name: str
    build: callable
    invertible: bool = False
    has_sqrt: bool = False
    #: Documented matrix-free: a matvec must not materialise the dense matrix.
    matrix_free: bool = False
    #: `jax.grad` through a matvec, w.r.t. the vector.
    differentiable: bool = True

    def op(self):
        return self.build()

    def dense(self):
        return jnp.asarray(linox.todense(self.build()))


OPERATORS: list[Case] = [
    Case("Matrix", lambda: Matrix(_spd(4)), invertible=True, has_sqrt=True),
    Case("Matrix(nonsym)", lambda: Matrix(_dense(4)), invertible=True),
    Case("Matrix(rect 5x3)", lambda: Matrix(_dense(5, 3))),
    Case(
        "Diagonal",
        lambda: Diagonal(jnp.arange(1.0, 5.0, dtype=F64)),
        invertible=True,
        has_sqrt=True,
        matrix_free=True,
    ),
    Case("Identity", lambda: Identity(4), invertible=True, has_sqrt=True, matrix_free=True),
    Case("Zero", lambda: Zero((4, 4)), matrix_free=True),
    Case("Ones", lambda: Ones((4, 4)), matrix_free=True),
    Case(
        "Kronecker",
        lambda: Kronecker(Matrix(_spd(3)), Matrix(_spd(3, seed=2))),
        invertible=True,
        has_sqrt=True,
        matrix_free=True,
    ),
    Case(
        "Toeplitz",
        lambda: Toeplitz(jnp.array([4.0, 1.0, 0.5, 0.1], dtype=F64)),
        invertible=True,
        matrix_free=True,
    ),
    Case("Permutation", lambda: Permutation(jnp.array([2, 0, 3, 1])), invertible=True, matrix_free=True),
    Case("LowRank(square)", lambda: LowRank(_dense(4, 2), jnp.ones(2, dtype=F64), _dense(4, 2, seed=3))),
    Case("SymmetricLowRank", lambda: SymmetricLowRank(_dense(4, 2), jnp.ones(2, dtype=F64))),
    Case("BlockDiagonal", lambda: BlockDiagonal(Matrix(_spd(2)), Matrix(_spd(2, seed=4))), invertible=True),
    Case(
        "BlockMatrix2x2",
        lambda: BlockMatrix2x2(Matrix(_spd(2)), Zero((2, 2)), Zero((2, 2)), Matrix(_spd(2, seed=5))),
        invertible=True,
    ),
    Case("EigenD", lambda: _eigend(), invertible=True, has_sqrt=True),
    Case("IsotropicAdditive", lambda: Matrix(_spd(4)) + 0.5 * Identity(4), invertible=True, has_sqrt=True),
    Case("Scaled", lambda: 2.5 * Matrix(_spd(4)), invertible=True, has_sqrt=True),
    Case("Add", lambda: AddLinearOperator(Matrix(_spd(4)), Matrix(_spd(4, seed=6))), invertible=True),
    Case("Product", lambda: ProductLinearOperator(Matrix(_spd(4)), Matrix(_spd(4, seed=7))), invertible=True),
    Case("Transposed", lambda: Matrix(_dense(4)).T, invertible=True),
    Case("CongruenceTransform", lambda: CongruenceTransform(Matrix(_spd(4)), Matrix(_spd(4, seed=8)))),
    Case("Sym", lambda: Sym(Matrix(_spd(4))), invertible=True, matrix_free=True),
    Case("PSD", lambda: PSD(Matrix(_spd(4))), invertible=True, matrix_free=True),
    Case("SPD", lambda: SPD(Matrix(_spd(4))), invertible=True, matrix_free=True),
]


def _eigend():
    n = 4
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.fold_in(KEY, 9), (n, n), dtype=F64))
    return EigenD(Matrix(Q), Diagonal(jnp.linspace(1.0, 4.0, n)))


IDS = [c.name for c in OPERATORS]


def _vec(op, seed=0):
    return jax.random.normal(jax.random.fold_in(KEY, 100 + seed), (op.shape[-1],), dtype=F64)


def _count_densify(fn):
    import linox.config as config

    events = []
    config.set_debug_hook(lambda e: events.append(e.kind))
    try:
        fn()
    finally:
        config.set_debug_hook(None)
    return events.count("densify")


@pytest.mark.parametrize("case", OPERATORS, ids=IDS)
class TestUniversalProperties:
    """Properties every linear operator must satisfy, whatever it is."""

    def test_linearity(self, case: Case) -> None:
        """A(ax + by) == a(Ax) + b(Ay)."""
        op = case.op()
        x, y = _vec(op, 1), _vec(op, 2)
        a, b = 2.5, -1.75

        combined = op @ (a * x + b * y)
        separate = a * (op @ x) + b * (op @ y)

        assert jnp.allclose(combined, separate, atol=1e-9)

    def test_adjoint_identity(self, case: Case) -> None:
        """<Ax, y> == <x, A^T y> -- the defining property of the transpose."""
        op = case.op()
        x = jax.random.normal(jax.random.fold_in(KEY, 3), (op.shape[-1],), dtype=F64)
        y = jax.random.normal(jax.random.fold_in(KEY, 4), (op.shape[-2],), dtype=F64)

        lhs = jnp.vdot(op @ x, y)
        rhs = jnp.vdot(x, op.T @ y)

        assert jnp.allclose(lhs, rhs, atol=1e-9), f"{case.name}: transpose is not the adjoint"

    def test_zero_vector_maps_to_zero(self, case: Case) -> None:
        op = case.op()
        result = op @ jnp.zeros((op.shape[-1],), dtype=F64)
        assert jnp.all(result == 0)
        assert jnp.all(jnp.isfinite(result))

    def test_matvec_matches_dense(self, case: Case) -> None:
        """The lazy matvec and the materialized matrix must agree."""
        op, dense = case.op(), case.dense()
        x = _vec(case.op(), 5)
        assert jnp.allclose(op @ x, dense @ x, atol=1e-9)

    def test_determinism(self, case: Case) -> None:
        """Exact methods are deterministic: same input, same bits."""
        x = _vec(case.op(), 6)
        first, second = case.op() @ x, case.op() @ x
        assert jnp.array_equal(first, second)

    def test_jit_matches_eager(self, case: Case) -> None:
        op = case.op()
        x = _vec(op, 7)
        assert jnp.allclose(jax.jit(lambda v: case.op() @ v)(x), op @ x, atol=1e-9)

    def test_dtype_is_preserved(self, case: Case) -> None:
        op = case.op()
        assert (op @ _vec(op, 8)).dtype == F64

    def test_pytree_round_trip(self, case: Case) -> None:
        op = case.op()
        leaves, treedef = jax.tree_util.tree_flatten(op)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert type(rebuilt) is type(op)
        x = _vec(op, 9)
        assert jnp.allclose(rebuilt @ x, op @ x, atol=1e-9)


@pytest.mark.parametrize("case", OPERATORS, ids=IDS)
class TestRightHandSideShapes:
    """A matvec, a single column and a block of columns must agree."""

    def test_vector_and_single_column_agree(self, case: Case) -> None:
        op = case.op()
        x = _vec(op, 10)

        as_vector = op @ x
        as_column = op @ x[:, None]

        assert as_column.shape == (op.shape[-2], 1)
        assert jnp.allclose(as_column[:, 0], as_vector, atol=1e-9)

    def test_matrix_rhs_equals_column_wise(self, case: Case) -> None:
        op = case.op()
        rhs = jax.random.normal(jax.random.fold_in(KEY, 11), (op.shape[-1], 3), dtype=F64)

        block = op @ rhs
        columns = jnp.stack([op @ rhs[:, j] for j in range(3)], axis=-1)

        assert block.shape == (op.shape[-2], 3)
        assert jnp.allclose(block, columns, atol=1e-9)


@pytest.mark.parametrize("case", [c for c in OPERATORS if c.differentiable], ids=[c.name for c in OPERATORS if c.differentiable])
class TestGradients:
    def test_grad_matches_finite_differences(self, case: Case) -> None:
        """d/dx sum(Ax) == sum of A's columns, checked against a finite difference."""
        op = case.op()
        x = _vec(op, 12)

        grad = jax.grad(lambda v: (case.op() @ v).sum())(x)

        eps = 1e-6
        expected = jnp.array(
            [
                ((op @ x.at[i].add(eps)).sum() - (op @ x.at[i].add(-eps)).sum())
                / (2 * eps)
                for i in range(x.shape[0])
            ]
        )
        assert jnp.allclose(grad, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("case", [c for c in OPERATORS if c.invertible], ids=[c.name for c in OPERATORS if c.invertible])
class TestSolve:
    def test_solve_residual(self, case: Case) -> None:
        op = case.op()
        b = _vec(op, 13)
        x = linox.solve(op, b)
        assert jnp.linalg.norm(case.dense() @ x - b) / jnp.linalg.norm(b) < 1e-8


@pytest.mark.parametrize("case", [c for c in OPERATORS if c.has_sqrt], ids=[c.name for c in OPERATORS if c.has_sqrt])
class TestSquareRoot:
    def test_reconstruction(self, case: Case) -> None:
        """The documented contract: a left square root, S @ S.T == A."""
        factor = jnp.asarray(linox.todense(lsqrt(case.op())))
        assert jnp.allclose(factor @ factor.T, case.dense(), atol=1e-7)


@pytest.mark.parametrize("case", [c for c in OPERATORS if c.matrix_free], ids=[c.name for c in OPERATORS if c.matrix_free])
class TestMatrixFree:
    def test_matvec_does_not_densify(self, case: Case) -> None:
        op = case.op()
        x = _vec(op, 14)
        assert _count_densify(lambda: op @ x) == 0

    def test_transpose_matvec_does_not_densify(self, case: Case) -> None:
        op = case.op()
        y = jax.random.normal(jax.random.fold_in(KEY, 15), (op.shape[-2],), dtype=F64)
        assert _count_densify(lambda: op.T @ y) == 0


class TestStructurePreservingRewrites:
    """`smart_add` and friends rewrite expressions; the maths must not change."""

    def _same(self, lhs, rhs) -> None:
        assert jnp.allclose(
            jnp.asarray(linox.todense(lhs)), jnp.asarray(linox.todense(rhs)), atol=1e-9
        )

    def test_adding_zero_is_identity(self) -> None:
        A = Matrix(_spd(4))
        self._same(A + Zero((4, 4)), A)

    def test_isotropic_rewrite_matches_the_plain_sum(self) -> None:
        A = Matrix(_spd(4))
        rewritten = A + 0.5 * Identity(4)
        assert type(rewritten).__name__ == "IsotropicAdditiveLinearOperator"
        self._same(rewritten, Matrix(_spd(4) + 0.5 * jnp.eye(4)))

    def test_diagonal_plus_low_rank_rewrite(self) -> None:
        d = Diagonal(jnp.arange(1.0, 5.0, dtype=F64))
        lr = SymmetricLowRank(_dense(4, 2), jnp.ones(2, dtype=F64))
        self._same(d + lr, Matrix(linox.todense(d) + linox.todense(lr)))

    def test_scaling_composes(self) -> None:
        A = Matrix(_spd(4))
        self._same(2.0 * (3.0 * A), 6.0 * A)

    def test_transpose_of_transpose(self) -> None:
        A = Matrix(_dense(4))
        self._same(A.T.T, A)

    def test_kron_matches_dense_kron(self) -> None:
        a, b = _spd(3), _spd(3, seed=2)
        self._same(Kronecker(Matrix(a), Matrix(b)), Matrix(jnp.kron(a, b)))
