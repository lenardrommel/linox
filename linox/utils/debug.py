# _graph.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from linox import config

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class LinOpNode:
    kind: str
    shape: tuple[int, int] | tuple[Any, ...]
    dtype: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    children: list[LinOpNode] = field(default_factory=list)

    def pretty(self, indent: str = "", last: bool = True) -> str:
        branch = "└─ " if last else "├─ "
        hdr = f"{indent}{branch}{self.kind}(shape={self.shape}, dtype={self.dtype})"
        if self.extra:
            hdr += f" {self.extra}"
        lines = [hdr]
        nxt = indent + ("   " if last else "│  ")
        for i, ch in enumerate(self.children):
            lines.append(ch.pretty(nxt, i == len(self.children) - 1))
        return "\n".join(lines)


def linop_graph(
    op, *, show_extra: bool = True, max_depth: int | None = None
) -> LinOpNode:
    seen = {}

    def _node(x, depth: int) -> LinOpNode:
        obj_id = id(x)
        if obj_id in seen:
            # cycle / shared subgraph
            return LinOpNode(
                kind=f"{type(x).__name__} [shared]",
                shape=getattr(x, "shape", "?"),
                dtype=getattr(x, "dtype", None),
            )

        if max_depth is not None and depth > max_depth:
            return LinOpNode(
                kind=f"{type(x).__name__} [cut]",
                shape=getattr(x, "shape", "?"),
                dtype=getattr(x, "dtype", None),
            )

        kind = type(x).__name__
        shape = getattr(x, "shape", "?")
        dtype = getattr(x, "dtype", None)

        extra = {}
        if show_extra:
            # optional: tags, requires, ...
            pass

        node = LinOpNode(kind=kind, shape=shape, dtype=dtype, extra=extra, children=[])
        seen[obj_id] = node

        # Try to descend using tree_flatten if available
        tf = getattr(x, "tree_flatten", None)
        if callable(tf):
            children, aux = tf()

            if show_extra and aux:
                node.extra = {**node.extra, **aux}
            node.children = [_node(c, depth + 1) for c in children]
        return node

    return _node(op, 0)


@dataclass
class InspectReport:
    events: list[config.DebugEvent]

    @property
    def steps(self) -> list[config.DebugEvent]:
        """Alias for events, for compatibility with test expectations."""
        return self.events

    def summary(self) -> str:
        # simple summary: counts per kind
        c = Counter(e.kind for e in self.events)
        lines = ["Inspect summary:"]
        for k, v in c.most_common():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class _Collector:
    def __init__(self) -> None:
        self.events: list[config.DebugEvent] = []

    def __call__(self, e: config.DebugEvent):
        self.events.append(e)


def inspect_run(fn: Callable[..., Any], *args, **kwargs):
    collector = _Collector()
    old = config._DEBUG_HOOK  # or better: add getter
    config.set_debug_hook(collector)
    try:
        out = fn(*args, **kwargs)
    finally:
        config.set_debug_hook(old)
    return out, InspectReport(collector.events)
