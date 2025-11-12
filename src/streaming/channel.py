"""Lightweight in-memory channel for streaming rows."""

from __future__ import annotations

from queue import Empty, Queue
from typing import Generic, List, Optional, Sequence, TypeVar

T = TypeVar("T")


class EventChannel(Generic[T]):
    """Minimal interface for publishing and consuming streaming events."""

    def publish(self, event: T) -> None:
        raise NotImplementedError

    def consume_batch(
        self,
        max_items: int,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> List[T]:
        raise NotImplementedError

    def drain(self) -> List[T]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class InMemoryChannel(EventChannel[T]):
    """Queue-backed channel with best-effort batching."""

    def __init__(self) -> None:
        self._queue: "Queue[T]" = Queue()
        self._closed = False

    def publish(self, event: T) -> None:
        if self._closed:
            raise RuntimeError("Channel is closed")
        self._queue.put(event)

    def consume_batch(
        self,
        max_items: int,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> List[T]:
        if max_items <= 0:
            return []

        items: List[T] = []
        try:
            if block:
                item = self._queue.get(timeout=timeout)
            else:
                item = self._queue.get_nowait()
            items.append(item)
        except Empty:
            return items

        while len(items) < max_items:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                break
        return items

    def drain(self) -> List[T]:
        items: List[T] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                break
        return items

    def close(self) -> None:
        self._closed = True
        self.drain()
