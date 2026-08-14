"""Tracing infrastructure for debugging and profiling."""

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field

from linox import config
from linox.config import DebugEvent


@dataclass
class TraceReport:
    """Report generated from a trace session."""

    events: list[DebugEvent]
    summary: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dense_ops: list[DebugEvent] = field(default_factory=list)

    def analyze(self):
        """Analyze events and populate summary."""
        for event in self.events:
            self.summary[event.kind] += 1
            if event.kind == "densify":
                self.dense_ops.append(event)

    def __str__(self) -> str:
        s = ["Trace Report:"]
        for k, v in self.summary.items():
            s.append(f"  {k}: {v}")
        if self.dense_ops:
            s.append(f"\nDense Operations ({len(self.dense_ops)}):")
            for ev in self.dense_ops[:10]:
                s.append(f"  {ev.msg} (Shape: {ev.shape})")
            if len(self.dense_ops) > 10:
                s.append(f"  ... (+{len(self.dense_ops) - 10} more)")
        return "\n".join(s)


class TraceContext:
    """Context manager for tracing linox operations."""

    def __init__(self):
        self.events: list[DebugEvent] = []
        self._prev_hook = None

    def __enter__(self) -> "TraceContext":
        # Hook into config
        self._prev_hook = config._DEBUG_HOOK
        config.set_debug_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        config.set_debug_hook(self._prev_hook)

    def _hook(self, event: DebugEvent):
        self.events.append(event)
        # Chain previous hook if it existed?
        # Usually hooks are exclusive or chained manually.
        # For now, simplistic exclusive (with manual chain if needed, but not implemented).
        if self._prev_hook:
            self._prev_hook(event)

    @property
    def report(self) -> TraceReport:
        """Return the events collected so far as an analyzed :class:`TraceReport`."""
        r = TraceReport(events=list(self.events))
        r.analyze()
        return r


@contextlib.contextmanager
def trace():
    """Convenience context manager yielding the report/context."""
    ctx = TraceContext()
    with ctx:
        yield ctx
