# _registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from linox import LinearOperator

Tag = Literal[
    "square",
    "rectangular",
    "symmetric",
    "spd",
    "orthogonal",
    "triangular",
    "diagonal",
    "psd",
    "normal",
    "banded",
]


@dataclass(frozen=True)
class OpSpec:
    name: str
    maker: Callable[..., LinearOperator]  # e.g. maker(key, shape, dtype, **kw)
    tags: set[Tag]
    # Optional: default kwargs or constraints
    min_ndim: int = 2


_REGISTRY: dict[str, OpSpec] = {}


def register(
    name: str, *, tags: Iterable[Tag] = ()
) -> Callable[[Callable[..., LinearOperator]], Callable[..., LinearOperator]]:
    def deco(maker: Callable[..., LinearOperator]) -> Callable[..., LinearOperator]:
        if name in _REGISTRY:
            msg = f"Operator '{name}' already registered"
            raise ValueError(msg)
        _REGISTRY[name] = OpSpec(name, maker, set(tags))
        return maker

    return deco


def list_ops(*required: Tag, exclude: Iterable[Tag] = ()) -> list[OpSpec]:
    ex = set(exclude)
    req = set(required)
    return [
        spec
        for spec in _REGISTRY.values()
        if req.issubset(spec.tags) and spec.tags.isdisjoint(ex)
    ]


def get(name: str) -> OpSpec:
    return _REGISTRY[name]
