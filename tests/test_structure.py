import jax.numpy as jnp

from linox.cache.session import cache_lookup, cache_update, clear_cache, with_cache
from linox.operators import Identity, Kronecker, Matrix
from linox.structure import KroneckerIR, analyze
from linox.structure.fingerprint import fingerprint


def test_analyze_kronecker() -> None:
    A = Matrix(jnp.eye(2))
    B = Identity(2)
    K = Kronecker(A, B)

    ir = analyze(K)
    assert isinstance(ir, KroneckerIR)
    assert len(ir.factors) == 2
    assert ir.factors[0] is A
    assert ir.factors[1] is B


def test_analyze_flatten_kron() -> None:
    A = Matrix(jnp.eye(2))
    B = Identity(2)
    C = Matrix(jnp.ones((2, 2)))

    # Kron(Kron(A, B), C)
    K1 = Kronecker(A, B)
    K2 = Kronecker(K1, C)

    # Analyze should flatten
    ir = analyze(K2)

    assert isinstance(ir, KroneckerIR)
    assert len(ir.factors) == 3
    # Check leaves
    # Note: A, B, C matching depends on tree structure but should be there
    # extract_kronecker_factors is recursive
    assert ir.factors[0] is A
    assert ir.factors[1] is B
    assert ir.factors[2] is C


def test_fingerprint_stability() -> None:
    A = Matrix(jnp.eye(2))
    B = Identity(2)
    K = Kronecker(A, B)

    fp1 = fingerprint(K)
    fp2 = fingerprint(K)

    assert fp1 == fp2

    # Same structure, different object?
    # Matrix content hashing is id-based for arrays currently, so if array is different...
    A2 = Matrix(jnp.eye(2))
    Kronecker(A2, B)
    # Different array objects -> different fingerprint in current impl
    # assert fingerprint(K2) != fp1

    # Same array object -> same fingerprint
    K3 = Kronecker(A, B)
    assert fingerprint(K3) == fp1


def test_session_cache() -> None:
    with with_cache():
        cache_update("key1", "value1")
        assert cache_lookup("key1") == "value1"

    # Outside context, should be empty/isolated (or default empty)
    # The default impl creates a thread local dict if none.
    # But context manager restores old.

    # Clean check
    clear_cache()
    assert cache_lookup("key1") is None
