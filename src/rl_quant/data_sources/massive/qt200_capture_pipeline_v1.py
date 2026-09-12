"""Bounded CPU/I/O concurrency; no change to source or economic semantics.

One process owns a shared request pacer and at most four disjoint histories.
Publication remains serialized by the caller's single publication lock. No
credentials are serialized, and a failure stops scheduling or retrying work.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event, Lock
from typing import Callable, TypeVar

T = TypeVar("T")


class PipelineStopped(RuntimeError):
    """A failed or cancelled owner may not issue further provider requests."""


class RequestPacer:
    """Globally space request grants by at least the existing 0.3 seconds."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stop = Event()
        self._next = 0.0
        self._grants = 0

    def cancel(self) -> None:
        self._stop.set()

    def check(self) -> None:
        if self._stop.is_set():
            raise PipelineStopped("Capture pipeline stopped; no automatic retry")

    def acquire(self, *, deadline: float) -> float:
        with self._lock:
            self.check()
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            if now + delay >= deadline:
                raise PipelineStopped("Request deadline exhausted before grant")
            if self._stop.wait(delay):
                self.check()
            now = time.monotonic()
            if now >= deadline:
                raise PipelineStopped("Request deadline exhausted before grant")
            self._next = now + 0.3
            self._grants += 1
            return now

    @property
    def granted_requests(self) -> int:
        with self._lock:
            return self._grants


def run_bounded(*, keys: tuple[str, ...], operation: Callable[[str], T],
                pacer: RequestPacer, workers: int = 4) -> tuple[T, ...]:
    """Keep only ``workers`` tasks in flight, return in declared key order.

    Running operations may finish an already-started request/publication on
    failure, but no replacement task is submitted and the shared pacer closes.
    Partial source transactions are retained by their normal source owners.
    """
    if (type(workers) is not int or not 1 <= workers <= 4
            or type(keys) is not tuple or any(type(k) is not str or not k for k in keys)
            or len(set(keys)) != len(keys) or type(pacer) is not RequestPacer):
        raise ValueError("Invalid bounded capture population or worker count")
    pacer.check()
    pending_keys = iter(keys)
    results: dict[str, T] = {}

    def invoke(key: str) -> T:
        try:
            pacer.check()
            return operation(key)
        except BaseException:
            pacer.cancel()
            raise

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qt200-capture") as pool:
        pending = {}

        def submit_one() -> None:
            key = next(pending_keys, None)
            if key is not None:
                pacer.check()
                pending[pool.submit(invoke, key)] = key

        try:
            for _ in range(min(workers, len(keys))):
                submit_one()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Inspect all finished futures before scheduling any successor.
                for future in finished:
                    results[pending.pop(future)] = future.result()
                pacer.check()
                for _ in finished:
                    submit_one()
        except BaseException:
            pacer.cancel()
            for future in pending:
                future.cancel()
            raise
    return tuple(results[key] for key in keys)
