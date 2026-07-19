"""Leak-free train/val/test splitting of built windows into per-day units.

The split is CHRONOLOGICAL (no shuffling) and, in daily/cross-day mode, episodes are assembled PER-SPLIT (see
rl_quant.datasets.daily.build_daily_episodes), so a day's T+1 label can never reference an open outside its own
split. These helpers live in the package (not the driver) so the leak-critical logic is unit-tested.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

from rl_quant.datasets.streaming import LazyDay, LazyWindow

# the per-day fields carried out of a built window (everything build_window stores with a leading n_days axis)
_DAY_KEYS = ("bars", "bar_mask", "cov_blocks", "cov_valid_blocks", "news_raw", "news_mask", "avail",
             "universe_member", "ret", "ret_valid",
             "day_open", "day_close", "day_close_valid", "session_close_block")


def flatten_days(windows: list) -> list[Any]:
    """Expand built windows (leading n_days axis) into per-day session dicts. For an in-RAM window dict, indexing
    returns a view (cheap; the window keeps the storage alive). For a LazyWindow (streaming), produce LazyDay
    handles that load the window .pt on demand (LRU-bounded) -- so the whole dataset is never resident at once."""
    out: list[Any] = []
    for w in windows:
        if isinstance(w, LazyWindow):
            out.extend(LazyDay(w, di) for di in range(w.n_days))   # streaming: lazy per-day handles
        else:
            for di in range(w["n_days"]):
                d = {k: w[k][di] for k in _DAY_KEYS if k in w}     # carry the fields present (e.g. day_close)
                d["date"] = w["dates"][di]
                out.append(d)
    return out


def _validate_split_fractions(train_frac: float, val_frac: float) -> None:
    for name, value in (("train_frac", train_frac), ("val_frac", val_frac)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number in [0, 1]; got {value!r}.")
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError(f"train_frac must be strictly between 0 and 1; got {train_frac!r}.")
    if not 0.0 <= float(val_frac) < 1.0:
        raise ValueError(f"val_frac must be in [0, 1); got {val_frac!r}.")
    if float(train_frac) + float(val_frac) >= 1.0:
        raise ValueError(
            "train_frac + val_frac must be < 1 so the chronological test fraction is positive; "
            f"got {train_frac!r} + {val_frac!r}."
        )


def time_split(built: list, train_frac: float, val_frac: float):
    """Split a chronologically-ordered list (windows or days) into train/val/test by fraction (test = remainder)."""
    _validate_split_fractions(train_frac, val_frac)
    n = len(built)
    n_tr, n_va = int(n * train_frac), int(n * val_frac)
    n_te = n - n_tr - n_va
    if n_tr == 0 or n_va == 0 or n_te == 0:
        raise ValueError(
            "fraction split must produce non-empty train, validation, and test sets; "
            f"got sizes train={n_tr}, validation={n_va}, test={n_te} from n={n}."
        )
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
    if mode not in ("daily", "daily_raw", "intraday"):
        raise ValueError(f"mode must be 'intraday', 'daily', or 'daily_raw'; got {mode!r}.")
    if mode in ("daily", "daily_raw"):
        return time_split(day_sequence(built), train_frac, val_frac)
    tr, va, te = time_split(built, train_frac, val_frac)
    day_splits = flatten_days(tr), flatten_days(va), flatten_days(te)
    date_sets = tuple({day["date"] for day in split} for split in day_splits)
    names = ("train", "validation", "test")
    collisions: list[str] = []
    for left_index, left_dates in enumerate(date_sets):
        for right_index in range(left_index + 1, len(date_sets)):
            overlap = sorted(left_dates.intersection(date_sets[right_index]), key=str)
            if overlap:
                collisions.append(
                    f"{names[left_index]}/{names[right_index]} share {overlap[:5]!r}"
                )
    if collisions:
        raise ValueError(
            "intraday window split leaks duplicate decision dates across split boundaries; "
            + "; ".join(collisions)
        )
    return day_splits
