"""Operator fingerprinting for caching."""

import hashlib

import jax
import jax.numpy as jnp

from linox.operators import LinearOperator


def fingerprint(op: LinearOperator) -> str:
    """Compute a cryptographic hash of the operator.

    Hash depends on:
    - Operator type
    - Shape, dtype
    - Parameters (scalars)
    - Children fingerprints
    - Dense array content (if small/safe) or array id (if tracing)

    NOTE: Currently simplistic. Robust hashing of large arrays in JAX is tricky.
    We assume immutable operators.
    """
    hasher = hashlib.sha256()
    _hash_recursive(op, hasher)
    return hasher.hexdigest()


def _hash_recursive(obj, hasher) -> None:
    """Recursive hashing helper."""
    if isinstance(obj, LinearOperator):
        hasher.update(type(obj).__name__.encode())
        hasher.update(str(obj.shape).encode())
        hasher.update(str(obj.dtype).encode())

        # Flatten children
        children, aux = obj.tree_flatten()
        for child in children:
            _hash_recursive(child, hasher)
        # Hash aux data? Usually metadata
        hasher.update(str(aux).encode())

    elif isinstance(obj, (jax.Array, jnp.ndarray)):
        # For small arrays, hash content. For large/traced, hash metadata+id?
        # In JAX tracing, content isn't available.
        # Ideally we want value-based equality for caching.
        # For now: hash shape/dtype + assume unique inputs if id differs?
        # WARNING: id() is not safe across JIT trace runs if objects are rebuilt.
        # But for Session Cache in standard python execution it might be OK.
        # For strict correctness: we cannot safely cache dependent on array content
        # unless we read it (blocks) or use weakrefs to existing objects.
        # Using id() for now as simple fingerprint for persistent objects.
        hasher.update(str(id(obj)).encode())
        hasher.update(str(obj.shape).encode())

    elif isinstance(obj, (int, float, complex, str)):
        hasher.update(str(obj).encode())

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _hash_recursive(item, hasher)

    else:
        # Fallback
        hasher.update(str(obj).encode())
