# test_diagonal_lazy.py

import jax.numpy as jnp

import linox


def test_diagonal_kronecker_no_densify_simple(monkeypatch) -> None:
    # Build a simple Kronecker product of two small dense matrices
    A = linox.Matrix(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64))
    B = linox.Matrix(jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float64))

    K = linox.Kronecker(A, B)

    # Ensure we never densify the Kronecker to compute the diagonal
    def fail_if_called(_self):  # pragma: no cover
        msg = "Kronecker.todense must not be called for diagonal"
        raise AssertionError(msg)

    monkeypatch.setattr(linox._kronecker.Kronecker, "todense", fail_if_called)

    diag_K = linox.diagonal(K)
    expected = jnp.kron(jnp.diag(A.A), jnp.diag(B.A))

    assert jnp.allclose(diag_K, expected)


def test_diagonal_isotropic_add_scaled_product_in_kronecker_no_densify(
    monkeypatch,
) -> None:
    # Match the scenario from tests/test_arithmetics.py and assert we don't densify
    mat_a = jnp.arange(4.0, dtype=jnp.float64).reshape(2, 2)
    mat_b = jnp.linspace(1.0, 4.0, num=4).reshape(2, 2)
    mat_c = jnp.ones((2, 2), dtype=jnp.float64)

    factor_left = linox.Matrix(mat_a)
    factor_right = linox.Matrix(mat_b)
    product = factor_left @ factor_right  # ProductLinearOperator
    scaled_product = linox.ScaledLinearOperator(product, jnp.array(1.3))
    kron_wrapper = linox.Kronecker(scaled_product, linox.Matrix(jnp.ones((1, 1))))
    kron_wrapper = linox.kron(
        kron_wrapper, linox.kron(linox.Matrix(mat_c), linox.Matrix(mat_c))
    )
    additive = linox.AddLinearOperator(
        kron_wrapper, linox.Matrix(jnp.eye(kron_wrapper.shape[0]))
    )
    iso = linox.IsotropicAdditiveLinearOperator(jnp.array(0.5), additive)

    # Forbid densification on Kronecker and Add during diagonal evaluation
    def fail_if_called(_self):  # pragma: no cover
        msg = "Unexpected densify called during diagonal computation"
        raise AssertionError(msg)

    monkeypatch.setattr(linox._kronecker.Kronecker, "todense", fail_if_called)
    monkeypatch.setattr(linox._arithmetic.AddLinearOperator, "todense", fail_if_called)

    result = linox.diagonal(iso)

    # Compute expected diagonal via dense reference on small factors only
    mat_prod = 1.3 * (mat_a @ mat_b)
    kron1 = jnp.kron(mat_prod, jnp.ones((1, 1)))
    kron2 = jnp.kron(kron1, jnp.kron(mat_c, mat_c))
    expected = jnp.diag(kron2 + (1.5) * jnp.eye(kron2.shape[0]))

    assert jnp.allclose(result, expected, atol=1e-6)
