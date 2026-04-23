from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable


@dataclass(slots=True)
class Packet:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class Blackboard:
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._store: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def require(self, key: str) -> Any:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def increment(self, key: str, amount: int | float = 1) -> None:
        self._store[key] = self._store.get(key, 0) + amount


@dataclass(slots=True)
class Layer:
    name: str
    packet_kinds: tuple[str, ...]
    handler: Callable[[Packet, Blackboard], Packet]
    subsystem: str | None = None


class Arbiter:
    def __init__(self, switches: dict[str, bool] | None = None) -> None:
        self._switches = dict(switches or {})

    def allows(self, subsystem: str | None, board: Blackboard) -> bool:
        if subsystem is None:
            return True
        default = self._switches.get(subsystem, False)
        return bool(board.get(f"switches.{subsystem}", default))


class LayerStack:
    def __init__(self, layers: list[Layer], arbiter: Arbiter | None = None) -> None:
        self._layers = layers
        self._arbiter = arbiter or Arbiter()

    def dispatch(self, packet: Packet, board: Blackboard) -> Packet:
        current = packet
        for layer in self._layers:
            if current.kind not in layer.packet_kinds:
                continue
            if not self._arbiter.allows(layer.subsystem, board):
                continue
            current = layer.handler(current, board)
        return current


__all__ = ["Arbiter", "Blackboard", "Layer", "LayerStack", "Packet"]
