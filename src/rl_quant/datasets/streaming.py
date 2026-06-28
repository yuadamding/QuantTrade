"""Lazy, disk-streaming day handles for large universes (e.g. TOP2000) whose full second-bar tensors do not fit
in RAM. The in-RAM path (build_windows -> list of window dicts -> flatten_days) is UNCHANGED; streaming swaps the
materialized window dicts for ``LazyWindow`` handles that load their cached ``.pt`` on demand (LRU-bounded) and
slice the requested day.

Equivalence is exact by construction: a LazyWindow loads the SAME per-window ``.pt`` the in-RAM path built and
returns the SAME ``field[di]`` slice -- only the lifetime differs (loaded on access, evicted under the LRU). A
returned slice is a view that keeps its parent tensor's storage alive until the caller is done (the trainers
``torch.stack`` it, which copies), so LRU eviction is safe (no use-after-free).
"""
from __future__ import annotations

from collections import OrderedDict

import torch

# the per-day tensor fields a window cache carries (leading n_days axis); everything else is light meta
TENSOR_KEYS = ("bars", "bar_mask", "cov_blocks", "news_raw", "news_mask",
               "avail", "ret", "ret_valid", "day_open", "day_close")
META_KEYS = ("n_days", "n_blocks", "dates", "window")


class _WindowLRU:
    """Process-global LRU of loaded per-window cache dicts -- bounds streaming RAM to ~maxn windows at once."""

    def __init__(self, maxn: int = 4) -> None:
        self.maxn = max(1, int(maxn))
        self.od: "OrderedDict[str, dict]" = OrderedDict()

    def load(self, path: str) -> dict:
        p = str(path)
        hit = self.od.get(p)
        if hit is not None:
            self.od.move_to_end(p)
            return hit
        d = torch.load(p, weights_only=False)
        self.od[p] = d
        while len(self.od) > self.maxn:
            self.od.popitem(last=False)
        return d


_LRU = _WindowLRU()


def set_cache_windows(n: int) -> None:
    """Set how many per-window cache files the streaming LRU may hold resident at once (RAM ~= n * window size)."""
    _LRU.maxn = max(1, int(n))


def lru_size() -> int:
    return len(_LRU.od)


class LazyWindow:
    """Dict-like handle to a per-window cache ``.pt``. Light meta (n_days/n_blocks/dates/window) is served from the
    sidecar passed at construction; tensor fields are loaded lazily via the LRU and returned whole ([n_days, ...])."""

    def __init__(self, path, meta: dict) -> None:
        self._path = str(path)
        self._meta = {k: meta[k] for k in META_KEYS if k in meta}

    def __contains__(self, k) -> bool:
        return k in self._meta or k in TENSOR_KEYS

    def __getitem__(self, k):
        if k in self._meta:
            return self._meta[k]
        return _LRU.load(self._path)[k]

    def get(self, k, default=None):
        return self[k] if k in self else default

    @property
    def n_days(self) -> int:
        return int(self._meta["n_days"])


class LazyDay:
    """Dict-like per-DAY view onto a LazyWindow at index ``di``, plus optional materialized OVERRIDES (e.g. the
    encoded market/per_stock embeddings, which are small and stay in RAM). Tensor fields not overridden load
    lazily from the window (LRU) and slice ``[di]``; meta (date/n_blocks) is served from the window."""

    def __init__(self, win: LazyWindow, di: int, overrides: dict | None = None) -> None:
        self._win = win
        self._di = int(di)
        self._ov = dict(overrides or {})

    def __contains__(self, k) -> bool:
        return k in self._ov or k in ("date", "n_blocks") or k in TENSOR_KEYS

    def __getitem__(self, k):
        if k in self._ov:
            return self._ov[k]
        if k == "date":
            return self._win["dates"][self._di]
        if k == "n_blocks":
            return self._win["n_blocks"]
        return self._win[k][self._di]              # lazy load of the window, then slice this day

    def get(self, k, default=None):
        return self[k] if k in self else default

    def with_overrides(self, **kw) -> "LazyDay":
        """Return a new view with extra materialized fields (used by streaming encode_days to attach embeddings)."""
        o = dict(self._ov)
        o.update(kw)
        return LazyDay(self._win, self._di, o)
