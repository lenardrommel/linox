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
import time
from dataclasses import dataclass

from linox.typing import AnyType, CallableType

_DEBUG: bool = os.getenv("LINOX_DEBUG", "0") not in ("0", "false", "False", "")
_DEBUG_HOOK: CallableType[["DebugEvent"], None] | None = None


@dataclass(frozen=True)
class DebugEvent:
    kind: str  # e.g. "densify", "solve_fallback", "eigh_dense"
    msg: str
    op_type: str | None = None
    op_id: int | None = None
    shape: AnyType = None
    dtype: AnyType = None
    meta: dict[str, AnyType] | None = None
    t: float = 0.0


def set_debug(value: bool) -> None:
    """Enable or disable debug mode (controls warning prints)."""
    global _DEBUG
    _DEBUG = bool(value)


def is_debug() -> bool:
    """Return whether debug mode is enabled."""
    return _DEBUG


def set_debug_hook(hook: CallableType[[DebugEvent], None] | None) -> None:
    """Register/unregister a debug hook that receives DebugEvent objects."""
    global _DEBUG_HOOK
    _DEBUG_HOOK = hook


def emit(event: DebugEvent) -> None:
    """Emit a structured debug event to the hook (if any)."""
    if _DEBUG_HOOK is not None:
        # set timestamp lazily
        if event.t == 0.0:
            object.__setattr__(event, "t", time.time())  # dataclass frozen workaround
        _DEBUG_HOOK(event)


def warn(msg: str, *, prefix: str = "Warning") -> None:
    """Conditionally print a warning message if debug is enabled.

    Args:
        msg: Message to print.
        prefix: Optional prefix for the message, defaults to 'Warning'.
    """
    if _DEBUG:
        print(f"{prefix}: {msg}")
        emit(DebugEvent(kind="warn", msg=f"{prefix}: {msg}"))
