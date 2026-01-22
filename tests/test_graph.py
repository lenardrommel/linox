# test_graph.py

import jax
import jax.numpy as jnp
import pytest

import linox
from linox import utils
from linox._graph import inspect_run

jax.config.update("jax_enable_x64", True)
linox.config.set_debug(True)


def _get_trace(obj):
    """
    Accepts either:
      - a tuple (out, trace)
      - an object with .out and .steps
      - a dict {"out": ..., "steps": ...}
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        out, trace = obj
        return out, trace

    if isinstance(obj, dict):
        return obj["out"], obj.get("steps", obj.get("trace", None))

    # object style
    out = getattr(obj, "out", None)
    steps = getattr(obj, "steps", None)
    if out is not None and steps is not None:
        return out, steps

    raise AssertionError(
        "inspect_run return type not understood. "
        "Expected (out, trace) or object with .out/.steps or dict."
    )


def _trace_steps(trace):
    # accept either list-like or object with .steps
    if trace is None:
        return None
    if isinstance(trace, (list, tuple)):
        return trace
    steps = getattr(trace, "steps", None)
    if steps is not None:
        return steps
    return trace  # last resort


def _step_fields(step):
    """Extract common fields from step that tests can assert on."""
    # dict step
    if isinstance(step, dict):
        return {
            "op_type": step.get("op_type") or step.get("type") or step.get("name"),
            "in_shape": step.get("in_shape") or step.get("input_shape"),
            "out_shape": step.get("out_shape") or step.get("output_shape"),
        }

    # object step
    return {
        "op_type": getattr(step, "op_type", None)
        or getattr(step, "type", None)
        or getattr(step, "name", None),
        "in_shape": getattr(step, "in_shape", None)
        or getattr(step, "input_shape", None),
        "out_shape": getattr(step, "out_shape", None)
        or getattr(step, "output_shape", None),
    }


@pytest.mark.parametrize("rhs_kind", ["vec", "mat"])
def test_inspect_run_matches_matmul(rhs_kind):
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[2.0, 0.0], [0.0, 0.5]])
    linox.config.set_debug(True)
    op = linox.kron(utils.as_linop(A), utils.as_linop(B))

    # Kronecker of 2x2 @ 2x2 gives shape (4, 4), so rhs needs 4 elements
    rhs = (
        jnp.array([1.0, -1.0, 0.5, 2.0])
        if rhs_kind == "vec"
        else jnp.array([[1.0, 0.0], [-1.0, 2.0], [0.5, -0.5], [1.0, 1.0]])
    )

    # --- call inspect_run (ADAPT THIS LINE) ---
    res = linox._graph.inspect_run(op, rhs)  # or op.inspect_run(rhs)

    out, trace = _get_trace(res)
    expected = op @ rhs
    assert jnp.allclose(out, expected, atol=1e-8)

    steps = _trace_steps(trace)
    assert steps is not None
    assert len(steps) >= 1


def test_inspect_run_has_step_metadata():
    op = utils.as_linop(jnp.eye(3))
    rhs = jnp.ones((3,))

    res = linox.inspect_run(op, rhs)
    out, trace = _get_trace(res)

    assert out.shape == (3,)
    steps = _trace_steps(trace)
    assert len(steps) >= 1

    s0 = _step_fields(steps[0])
    assert s0["op_type"] is not None, "Each step should identify operator type/name"
    # shapes are optional but strongly recommended
    if s0["in_shape"] is not None:
        assert tuple(s0["in_shape"])[-1] == 1 or tuple(s0["in_shape"])[-1] == 3
    if s0["out_shape"] is not None:
        assert tuple(s0["out_shape"])[0] in (3,)


def test_inspect_run_composition_shows_multiple_steps():
    A = utils.as_linop(jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    B = utils.as_linop(jnp.array([[3.0, 1.0], [0.0, 1.0]]))
    C = utils.as_linop(jnp.array([[1.0, 1.0], [1.0, 0.0]]))

    op = (A @ B) + (2.0 * C)  # Product + Scaled + Add
    rhs = jnp.array([1.0, 2.0])

    out, trace = _get_trace(linox.inspect_run(op, rhs))
    assert jnp.allclose(out, op @ rhs, atol=1e-8)

    steps = _trace_steps(trace)
    # Expect >1 for composed operator
    assert len(steps) >= 2, "Composed operator should yield multiple inspect steps"


def test_inspect_run_handles_kronecker_nested():
    from linox._kronecker import Kronecker

    A = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    B = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    op = Kronecker(Kronecker(A, A), Kronecker(B, B))  # nested kron

    rhs = jnp.ones((op.shape[-1],))
    out, trace = _get_trace(linox.inspect_run(op, rhs))

    assert jnp.allclose(out, op @ rhs, atol=1e-8)
    steps = _trace_steps(trace)
    assert len(steps) >= 1


def test_inspect_run_pinv_rhs_matrix_does_not_break():
    # this catches exactly the class of bugs you hit earlier:
    # pinv(op).todense() triggers pinv @ I -> rhs is matrix.
    op = utils.as_linop(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    pinv_op = linox.lpinverse(op)

    I = jnp.eye(pinv_op.shape[-1], dtype=pinv_op.dtype)
    out, trace = _get_trace(linox.inspect_run(pinv_op, I))

    assert out.shape == (pinv_op.shape[-2], pinv_op.shape[-1])
    assert jnp.allclose(out, pinv_op @ I, atol=1e-8)


@pytest.mark.parametrize(
    "op_builder",
    [
        lambda: utils.as_linop(jnp.eye(3)),
        lambda: (utils.as_linop(jnp.eye(3)) + utils.as_linop(jnp.eye(3))),
        lambda: (2.0 * utils.as_linop(jnp.eye(3))),
        lambda: (utils.as_linop(jnp.eye(3)).T),
        lambda: linox.lpinverse(utils.as_linop(jnp.array([[1.0, 2.0], [3.0, 4.0]]))),
    ],
)
def test_inspect_run_supports_common_ops(op_builder):
    op = op_builder()
    rhs = jnp.ones((op.shape[-1],))
    out, trace = _get_trace(linox.inspect_run(op, rhs))
    assert jnp.allclose(out, op @ rhs, atol=1e-8)
    steps = _trace_steps(trace)
    assert steps is not None and len(steps) >= 1
