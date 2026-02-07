import jax
import jax.numpy as jnp

import linox
from linox import Diagonal, Identity, Matrix


def test_trace_dispatch() -> None:
    key = jax.random.PRNGKey(0)
    # 5x5 Identity
    op = Identity(5)

    # Exact trace
    tr_exact = linox.trace(op, method="exact")
    assert tr_exact == 5.0

    # Hutchinson trace
    tr_hutch = linox.trace(op, method="hutchinson", key=key, num_samples=100)
    # Should be close to 5.0 (var=0 for identity with Rademacher actually, but general case approx)
    # Identity with Rademacher gives exact trace in one sample actually?
    # v^T I v = v^T v = n. Mean is n. Var is 0.
    assert jnp.allclose(tr_hutch, 5.0)


def test_eigh_dispatch_exact() -> None:
    # Diagonal matrix Diag([1, 2, 3])
    diag = jnp.array([1.0, 2.0, 3.0])
    op = Diagonal(diag)

    # Exact eigh
    evals, evecs = linox.eigh(op, method="exact")
    # Sorted? leigh sorts?
    # Usually leigh returns eigenvalues.
    assert jnp.allclose(jnp.sort(evals), diag)

    # Check correctness
    # Av = v*lambda
    # evecs might be Identity for diagonal
    evecs_dense = evecs.todense() if isinstance(evecs, linox.LinearOperator) else evecs

    # Check A @ V = V @ Lambda
    assert jnp.allclose(op @ evecs_dense, evecs_dense * evals)


def test_eigh_dispatch_lanczos() -> None:
    # Symmetric matrix
    key = jax.random.PRNGKey(1)
    A = jax.random.normal(key, (10, 10))
    A += A.T  # Symmetric
    op = Matrix(A)

    # Exact ground truth
    true_evals, _ = jnp.linalg.eigh(A)

    # Lanczos top-2 (Largest Magnitude)
    # k=2
    # Increase num_iters for better accuracy
    evals, _evecs = linox.eigh(op, k=2, method="lanczos", which="LM", num_iters=30)

    # Sort true evals by magnitude to compare
    true_sorted_idx = jnp.argsort(jnp.abs(true_evals))[::-1]
    true_top2 = true_evals[true_sorted_idx][:2]

    # Lanczos might return them sorted differently or slightly approx
    # But for 10x10 and k=2 it should work decent.
    # Check if close to true_top2 (order might vary)
    # Sort both to compare
    assert jnp.allclose(
        jnp.sort(jnp.abs(evals)), jnp.sort(jnp.abs(true_top2)), rtol=0.05
    )


def test_solve_dispatch() -> None:
    # Simple solve
    jax.random.PRNGKey(2)
    A = jnp.eye(3) * 2.0
    op = Matrix(A)
    b = jnp.ones(3)

    x = linox.solve(op, b, method="exact")
    assert jnp.allclose(x, b * 0.5)

    # method="auto"
    x_auto = linox.solve(op, b, method="auto")
    assert jnp.allclose(x_auto, b * 0.5)
