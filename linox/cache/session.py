"""Session-based caching mechanism."""

import contextlib
import threading
from typing import Any

# Global thread-local storage for session cache
_thread_local = threading.local()


def _get_store() -> dict[str, Any]:
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}
    return _thread_local.cache


def get_analysis_cache() -> dict[str, Any]:
    """Get the current thread's analysis cache."""
    return _get_store()


def clear_cache() -> None:
    """Clear the current thread's cache."""
    _thread_local.cache = {}


@contextlib.contextmanager
def with_cache(cache_dict: dict[str, Any] | None = None):
    """Context manager to scope a cache session.

    If None, creates a new empty cache for the block.
    """
    old_cache = getattr(_thread_local, "cache", None)

    if cache_dict is None:
        cache_dict = {}

    _thread_local.cache = cache_dict
    try:
        yield cache_dict
    finally:
        if old_cache is None:
            delattr(_thread_local, "cache")
        else:
            _thread_local.cache = old_cache


def cache_lookup(key: str) -> Any | None:
    return _get_store().get(key)


def cache_update(key: str, value: Any) -> None:
    _get_store()[key] = value


def cache_key(op: Any) -> str:
    """Generate a cache key for a linear operator based on object identity.

    Args:
        op: The operator (or any object) to key.

    Returns:
        A unique string key (e.g. valid for the lifetime of the object).
    """
    return str(id(op))
