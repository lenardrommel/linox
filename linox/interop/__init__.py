"""Adapters for using linox operators with other linear-algebra libraries.

This subpackage is deliberately *not* imported by ``linox/__init__.py``: its
adapters depend on frameworks linox does not require, and importing linox must
not drag PyTorch into the process. Reach for it explicitly::

    from linox.interop import to_skerch

which is the point where the optional dependency is imported, and where a
missing one is reported.
"""

from .skerch import SkerchLinOp, to_skerch

__all__ = ["SkerchLinOp", "to_skerch"]
