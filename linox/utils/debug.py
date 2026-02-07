# _graph.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from linox import config

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class LinOpNode:
    """A node in the linear operator graph visualization.

    Attributes
    ----------
        kind: The class name or type of the operator.
        shape: The shape of the operator (rows, cols).
        dtype: The data type of the operator elements.
        extra: Dictionary of additional metadata (e.g., parameters).
        children: List of child nodes (operands).
    """

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
    """A report containing the trace of operations executed during `inspect_run`.

    Use `.steps` or iterate over the report to access `DebugEvent` objects.
    """

    events: list[config.DebugEvent]

    @property
    def steps(self) -> list[config.DebugEvent]:
        """Alias for events, for compatibility with test expectations."""
        return self.events

    def summary(self) -> str:
        # Aggregate stats
        stats = {}  # kind -> {starts, ends, point, time}

        for e in self.events:
            if e.kind not in stats:
                stats[e.kind] = {"starts": 0, "ends": 0, "point": 0, "time": 0.0}

            s = stats[e.kind]
            if getattr(e, "phase", None) == "start":
                s["starts"] += 1
            elif getattr(e, "phase", None) == "end":
                s["ends"] += 1
                if getattr(e, "duration", None) is not None:
                    s["time"] += e.duration
            else:
                s["point"] += 1

        lines = ["Inspect summary:"]
        # sort by time descending, then count
        items = sorted(
            stats.items(),
            key=lambda x: (x[1]["time"], x[1]["starts"] + x[1]["point"]),
            reverse=True,
        )

        for k, s in items:
            total_invocations = s["starts"] + s["point"]
            if total_invocations == 0:
                 continue

            msg = f"  {k}: {total_invocations} calls"
            if s["time"] > 0:
                avg = s["time"] / max(s["ends"], 1)
                msg += f", {s['time']:.4f}s total ({avg:.4f}s avg)"

            lines.append(msg)

        return "\n".join(lines)


class _Collector:
    def __init__(self) -> None:
        self.events: list[config.DebugEvent] = []

    def __call__(self, e: config.DebugEvent):
        self.events.append(e)


def inspect_run(fn: Callable[..., Any], *args, **kwargs):
    """Run a function (or operator call) with debug tracing enabled.

    Captures all `config.DebugEvent`s emitted during execution, such as
    internal matrix multiplications, densification warnings, and solver steps.

    Args:
        fn: The function or callable (e.g., operator) to execute.
        *args: Positional arguments for `fn`.
        **kwargs: Keyword arguments for `fn`.

    Returns
    -------
        A tuple `(result, report)` where:
        - `result`: The return value of `fn(*args, **kwargs)`.
        - `report`: An `InspectReport` containing the list of debug events.
    """
    collector = _Collector()
    old = config._DEBUG_HOOK  # or better: add getter
    config.set_debug_hook(collector)
    try:
        out = fn(*args, **kwargs)
    finally:
        config.set_debug_hook(old)
    return out, InspectReport(collector.events)
