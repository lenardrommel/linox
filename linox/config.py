# config.py

"""Global configuration for linox warnings and debug behavior.

Usage:
- Toggle debug prints (e.g., densification warnings):
    from linox.config import set_debug
    set_debug(True)

- Or via environment variable:
    export LINOX_DEBUG=1
"""

from __future__ import annotations

import os

_DEBUG: bool = os.getenv("LINOX_DEBUG", "0") not in {"0", "false", "False", ""}


def set_debug(value: bool) -> None:
    """Enable or disable debug mode (controls warning prints)."""
    global _DEBUG
    _DEBUG = bool(value)


def is_debug() -> bool:
    """Return whether debug mode is enabled."""
    return _DEBUG


def warn(msg: str, *, prefix: str = "Warning") -> None:
    """Conditionally print a warning message if debug is enabled.

    Args:
        msg: Message to print.
        prefix: Optional prefix for the message, defaults to 'Warning'.
    """
    if _DEBUG:
        pass
