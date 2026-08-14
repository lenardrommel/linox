"""linox operators must work as drop-in linear operators for `skerch`.

skerch is a PyTorch sketching library with no JAX support of its own, so
`linox.interop.skerch` bridges the two. These tests pin the contract skerch
actually relies on, which is wider than the `.shape` + `@` interface it
advertises: results are slice-assigned into preallocated torch buffers, handed
to `torch.linalg.qr`, and mutated in place, so they must come back as torch
tensors of the caller's declared dtype.

Every case is set up in a regime where the sketch is *exact* -- a low-rank
operator sketched with more measurements than its rank, or Girard-Hutchinson on
a diagonal operator, where the off-diagonal variance that makes it an estimator
is identically zero. That is deliberate: these test the bridge, not skerch's
statistical quality, and an exact regime turns any wiring bug into an
unambiguous failure instead of a tolerance argument.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("skerch", reason="requires the `interop` dependency group")
torch = pytest.importorskip("torch", reason="requires the `interop` dependency group")

import linox  # noqa: E402
from linox.interop import SkerchLinOp, to_skerch  # noqa: E402
from linox.operators.base import LinearOperator  # noqa: E402
from skerch.algorithms import hutch, seigh, snorm, ssvd  # noqa: E402

jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float64
TORCH_DTYPE = torch.float64
DEVICE = "cpu"


def _dense(op):
    """The operator as a numpy array, for comparison against skerch's output."""
    return np.asarray(linox.todense(op))


@pytest.fixture
def low_rank():
    """A rank-3 40x40 operator. Sketching it with >3 measurements is exact."""
    rank, n = 3, 40
    u = jax.random.normal(jax.random.PRNGKey(0), (n, rank), dtype=DTYPE)
    v = jax.random.normal(jax.random.PRNGKey(1), (n, rank), dtype=DTYPE)
    return linox.Matrix(u @ v.T), rank


@pytest.fixture
def psd():
    """A rank-4 PSD operator, for the Hermitian sketch."""
    rank, n = 4, 40
    b = jax.random.normal(jax.random.PRNGKey(2), (n, rank), dtype=DTYPE)
    return linox.Matrix(b @ b.T), rank


class Opaque(LinearOperator):
    """An operator with only the minimum a subclass must provide.

    Records every dense materialization, so a test can assert there are none.
    Mirrors the operator in `test_transpose_matrix_free.py`, which is the change
    that makes the adjoint half of skerch's contract matrix-free.
    """

    def __init__(self, array, log):
        self._array = array
        self._log = log
        super().__init__(array.shape, array.dtype)

    def _matmul(self, other):
        return self._array @ other

    def _todense(self):
        self._log.append(1)
        return self._array


class TestAdapter:
    """The bridge itself, independent of any skerch algorithm."""

    def test_shape_is_plain_ints(self) -> None:
        # skerch unpacks this and feeds it straight to `torch.empty`.
        op = linox.Matrix(jnp.zeros((5, 3), dtype=DTYPE))
        assert to_skerch(op).shape == (5, 3)
        assert all(type(d) is int for d in to_skerch(op).shape)

    @pytest.mark.parametrize("ncols", [1, 4])
    def test_forward_matmul(self, ncols) -> None:
        op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(3), (5, 3), dtype=DTYPE))
        dense = torch.as_tensor(_dense(op).copy())
        x = torch.randn(3, ncols, dtype=TORCH_DTYPE)

        result = to_skerch(op) @ x

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, ncols)
        assert result.dtype == TORCH_DTYPE
        assert torch.allclose(result, dense @ x)

    def test_forward_matvec(self) -> None:
        op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(4), (5, 3), dtype=DTYPE))
        dense = torch.as_tensor(_dense(op).copy())
        x = torch.randn(3, dtype=TORCH_DTYPE)

        result = to_skerch(op) @ x

        assert result.shape == (5,)
        assert torch.allclose(result, dense @ x)

    @pytest.mark.parametrize("nrows", [1, 7])
    def test_adjoint_matmul(self, nrows) -> None:
        # The path linox's own `__rmatmul__` cannot serve: it returns a lazy
        # LinearOperator for 2-D left operands, and skerch needs an array.
        op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(5), (5, 3), dtype=DTYPE))
        dense = torch.as_tensor(_dense(op).copy())
        x = torch.randn(nrows, 5, dtype=TORCH_DTYPE)

        result = x @ to_skerch(op)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (nrows, 3)
        assert torch.allclose(result, x @ dense)

    def test_adjoint_matvec(self) -> None:
        op = linox.Matrix(jax.random.normal(jax.random.PRNGKey(6), (5, 3), dtype=DTYPE))
        dense = torch.as_tensor(_dense(op).copy())
        x = torch.randn(5, dtype=TORCH_DTYPE)

        result = x @ to_skerch(op)

        assert result.shape == (3,)
        assert torch.allclose(result, x @ dense)

    def test_output_dtype_follows_the_input(self) -> None:
        # Importing linox turns on JAX x64 globally, so a float32 sketch could
        # otherwise get float64 back and break skerch's buffer assignment.
        op = linox.Matrix(jnp.eye(3, dtype=jnp.float32))
        assert (to_skerch(op) @ torch.randn(3, dtype=torch.float32)).dtype == torch.float32

    def test_result_is_writable(self) -> None:
        # skerch does `meas1 -= meas2` on results (a_posteriori.py:159), which
        # is undefined behaviour on a tensor wrapping a read-only numpy array.
        op = linox.Matrix(jnp.eye(3, dtype=DTYPE))
        result = to_skerch(op) @ torch.ones(3, dtype=TORCH_DTYPE)
        result -= 1.0
        assert torch.allclose(result, torch.zeros(3, dtype=TORCH_DTYPE))

    def test_rejects_non_operator(self) -> None:
        with pytest.raises(TypeError, match="linox LinearOperator"):
            to_skerch(jnp.eye(3))

    def test_rejects_complex(self) -> None:
        with pytest.raises(ValueError, match="[Cc]omplex"):
            to_skerch(linox.Matrix(jnp.eye(3, dtype=jnp.complex128)))

    def test_repr_names_the_wrapped_operator(self) -> None:
        assert "Identity" in repr(to_skerch(linox.Identity(3)))


class TestSketchedAlgorithms:
    """End-to-end runs of skerch's algorithms against linox operators."""

    def test_ssvd_recovers_low_rank(self, low_rank) -> None:
        op, rank = low_rank

        u, s, vh = ssvd(
            to_skerch(op), DEVICE, TORCH_DTYPE, outer_dims=rank + 2, seed=42
        )
        reconstructed = ((u * s) @ vh).numpy()

        dense = _dense(op)
        rel_err = np.linalg.norm(reconstructed - dense) / np.linalg.norm(dense)
        assert rel_err < 1e-12

    def test_seigh_recovers_psd_spectrum(self, psd) -> None:
        op, rank = psd

        ews, _ = seigh(to_skerch(op), DEVICE, TORCH_DTYPE, outer_dims=rank + 2, seed=7)

        top = np.sort(ews.numpy())[::-1][:rank]
        reference = np.sort(np.linalg.eigvalsh(_dense(op)))[::-1][:rank]
        assert np.allclose(top, reference, rtol=1e-10)

    def test_snorm_matches_dense_norms(self, low_rank) -> None:
        op, rank = low_rank
        dense = _dense(op)

        norms, _ = snorm(
            to_skerch(op),
            DEVICE,
            TORCH_DTYPE,
            num_meas=rank + 10,
            seed=11,
            norm_types=("fro", "op"),
        )

        assert np.isclose(float(norms["fro"]), np.linalg.norm(dense, "fro"), rtol=1e-10)
        assert np.isclose(
            float(norms["op"]), np.linalg.svd(dense, compute_uv=False)[0], rtol=1e-10
        )

    def test_hutch_recovers_diagonal(self) -> None:
        # Girard-Hutchinson is exact on a diagonal operator: the variance comes
        # from off-diagonal mass, and there is none.
        d = jax.random.normal(jax.random.PRNGKey(3), (24,), dtype=DTYPE)
        op = linox.Diagonal(d)

        result = hutch(
            to_skerch(op), DEVICE, TORCH_DTYPE, num_meas=8, seed=5, noise_type="rademacher"
        )

        assert np.allclose(result["diag"].numpy(), np.asarray(d), atol=1e-12)
        assert np.isclose(float(result["tr"]), float(d.sum()), rtol=1e-12)


class TestSketchingIsMatrixFree:
    """Sketching must never materialize the operator it is sketching."""

    def test_ssvd_never_densifies(self) -> None:
        # The point of sketching: an operator that only knows how to apply
        # itself should be decomposable without ever forming its matrix. This
        # holds only because `A.T @ x` is matrix-free -- ssvd measures the
        # adjoint as `block.conj().T @ lop`.
        rank, n = 3, 30
        u = jax.random.normal(jax.random.PRNGKey(0), (n, rank), dtype=DTYPE)
        v = jax.random.normal(jax.random.PRNGKey(1), (n, rank), dtype=DTYPE)
        array = u @ v.T
        log = []
        op = Opaque(array, log)

        result = ssvd(to_skerch(op), DEVICE, TORCH_DTYPE, outer_dims=rank + 2, seed=42)

        assert log == [], "sketching materialized the dense matrix"

        # And it is still the right answer.
        u_s, s, vh = result
        dense = np.asarray(array)
        reconstructed = ((u_s * s) @ vh).numpy()
        rel_err = np.linalg.norm(reconstructed - dense) / np.linalg.norm(dense)
        assert rel_err < 1e-12

    def test_adjoint_alone_never_densifies(self) -> None:
        array = jax.random.normal(jax.random.PRNGKey(8), (5, 3), dtype=DTYPE)
        log = []
        lop = to_skerch(Opaque(array, log))

        _ = torch.randn(7, 5, dtype=TORCH_DTYPE) @ lop
        _ = torch.randn(5, dtype=TORCH_DTYPE) @ lop

        assert log == [], "the adjoint materialized the dense matrix"


def test_structured_operators_survive_the_round_trip() -> None:
    """The adapter is structure-agnostic: it only ever applies the operator."""
    key = jax.random.PRNGKey(9)
    operators = [
        linox.Identity(6, dtype=DTYPE),
        linox.Diagonal(jax.random.normal(key, (6,), dtype=DTYPE)),
        linox.Matrix(jax.random.normal(key, (6, 6), dtype=DTYPE)),
        linox.Kronecker(
            linox.Matrix(jax.random.normal(key, (2, 2), dtype=DTYPE)),
            linox.Matrix(jax.random.normal(key, (3, 3), dtype=DTYPE)),
        ),
    ]
    x = torch.randn(6, 2, dtype=TORCH_DTYPE)

    for op in operators:
        wrapped = to_skerch(op)
        assert isinstance(wrapped, SkerchLinOp)
        dense = torch.as_tensor(_dense(op).copy())
        assert torch.allclose(wrapped @ x, dense @ x), f"forward failed for {op!r}"
        assert torch.allclose(x.T @ wrapped, x.T @ dense), f"adjoint failed for {op!r}"
