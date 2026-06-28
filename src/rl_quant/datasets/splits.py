"""Leak-free train/val/test splitting of built windows into per-day units.

The split is CHRONOLOGICAL (no shuffling) and, in daily/cross-day mode, episodes are assembled PER-SPLIT (see
rl_quant.datasets.daily.build_daily_episodes), so a day's T+1 label can never reference an open outside its own
split. These helpers live in the package (not the driver) so the leak-critical logic is unit-tested.
"""
from __future__ import annotations

from rl_quant.datasets.streaming import LazyDay, LazyWindow

# the per-day fields carried out of a built window (everything build_window stores with a leading n_days axis)
_DAY_KEYS = ("bars", "bar_mask", "cov_blocks", "news_raw", "news_mask", "avail", "ret", "ret_valid",
             "day_open", "day_close")


def flatten_days(windows: list) -> list:
    """Expand built windows (leading n_days axis) into per-day session dicts. For an in-RAM window dict, indexing
    returns a view (cheap; the window keeps the storage alive). For a LazyWindow (streaming), produce LazyDay
    handles that load the window .pt on demand (LRU-bounded) -- so the whole dataset is never resident at once."""
    out = []
    for w in windows:
        if isinstance(w, LazyWindow):
            out.extend(LazyDay(w, di) for di in range(w.n_days))   # streaming: lazy per-day handles
        else:
            for di in range(w["n_days"]):
                d = {k: w[k][di] for k in _DAY_KEYS if k in w}     # carry the fields present (e.g. day_close)
                d["date"] = w["dates"][di]
                out.append(d)
    return out


def time_split(built: list, train_frac: float, val_frac: float):
    """Split a chronologically-ordered list (windows or days) into train/val/test by fraction (test = remainder)."""
    n = len(built)
    n_tr, n_va = int(n * train_frac), int(n * val_frac)
    return built[:n_tr], built[n_tr:n_tr + n_va], built[n_tr + n_va:]


def day_sequence(built: list[dict]) -> list[dict]:
    """One continuous, date-sorted, deduped sequence of per-day sessions across all windows (the cross-day unit).
    Overlapping windows are deduped by date (keep first)."""
    seen: dict = {}
    for d in flatten_days(built):
        seen.setdefault(d["date"], d)
    return [seen[k] for k in sorted(seen)]


def split_days(built: list[dict], mode: str, train_frac: float, val_frac: float):
    """Chronological train/val/test as lists of per-day sessions. intraday: split WINDOWS (time-ordered) then
    flatten. daily: build the continuous deduped day sequence then split it. No date is shared across splits."""
    if mode in ("daily", "daily_raw"):
        return time_split(day_sequence(built), train_frac, val_frac)
    tr, va, te = time_split(built, train_frac, val_frac)
    return flatten_days(tr), flatten_days(va), flatten_days(te)
