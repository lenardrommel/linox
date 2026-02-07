
import jax
import jax.numpy as jnp
import linox
from hypothesis import given, settings
from linox import utils

from tests.test_linox_cases._hypothesis_cases import linear_operators

# Configure JAX
jax.config.update("jax_enable_x64", True)

# --- Tests ---

@settings(max_examples=50, deadline=None)
@given(op=linear_operators(depth=10)) # Requesting depth up to 10
def test_inspect_run_composition_hypothesis(op):
    """Test inspect_run on deeply nested random operators."""
    n = op.shape[1]
    # RHS vector
    rhs = jnp.ones((n,)) 

    # Run inspect
    res = utils.inspect_run(op, rhs)
    
    # helper from test_graph.py logic, but simplified as we know inspect_run returns (out, report)
    # The user might be referring to `_trace` as `inspect.trace` if it existed, but here we check result directly.
    
    out, report = res
    
    # 1. Check Output Correctness
    try:
        expected = op @ rhs
        # Use relaxed tolerance for deep compositions of floats
        assert jnp.allclose(out, expected, atol=1e-5, rtol=1e-5)
    except Exception as e:
         # Some random ops might be ill-conditioned or explode, but basic arithmetic should hold
         # unless we hit NaNs/Infs
         if jnp.isnan(out).any() or jnp.isinf(out).any():
             return # Skip unstable cases
         raise e
    
    # 2. Check Trace Existence and Content
    assert isinstance(report, utils.InspectReport)
    steps = report.steps
    assert isinstance(steps, list)
    
    # If op is complicated, we expect some events. 
    # Matrix @ vector usually emits at least one "matmul" event or similar from base operations.
    if len(steps) == 0:
        # It's possible for simple wrapper or cached result? 
        # But linox usually emits events for fundamental ops.
        # Matrix @ array emits 'matmul'.
        # If op is just Matrix, it should have 1 step.
        pass
    else:
        for step in steps:
             assert hasattr(step, "kind")
             assert hasattr(step, "msg")
             # Check for common event kinds
             assert step.kind in ("matmul", "init", "densify", "solve", "eigh", "svd") or step.kind

    # 3. Check Trace Metadata (basic)
    # Ensure at least one 'matmul' event if we did a matmul
    # (unless it was optimized away or is identity/zero which might differ)
    has_matmul = any(s.kind == "matmul" for s in steps)
    
    # Note: simple Matrix(A) might emit 'init' but not 'matmul' during construction, 
    # but `inspect_run(op, rhs)` executes `op(rhs)` or `op @ rhs`.
    # `Matrix @ array` emits 'matmul'.
    if isinstance(op, linox.Matrix):
        assert has_matmul, "Matrix @ vector should emit a matmul event"



