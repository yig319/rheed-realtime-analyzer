from __future__ import annotations

"""Lightweight buffering primitives for streaming analytics."""

from collections import deque
from typing import Deque, Generic, Iterable, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Typed deque wrapper with convenience helpers for recent-history access."""

    def __init__(self, maxlen: int):
        """Create a fixed-capacity ring buffer."""
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._data: Deque[T] = deque(maxlen=maxlen)

    def append(self, item: T) -> None:
        """Append one item, evicting the oldest if full."""
        self._data.append(item)

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items in order."""
        self._data.extend(items)

    def clear(self) -> None:
        """Remove all items."""
        self._data.clear()

    def latest(self) -> T | None:
        """Return most recent item or None when empty."""
        return self._data[-1] if self._data else None

    def as_list(self) -> list[T]:
        """Materialize the current buffer contents."""
        return list(self._data)

    def __len__(self) -> int:
        """Return current number of buffered items."""
        return len(self._data)
