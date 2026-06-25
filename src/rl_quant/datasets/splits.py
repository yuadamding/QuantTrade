"""Leak-free train/val/test splitting of built windows into per-day units.

The split is CHRONOLOGICAL (no shuffling) and, in daily/cross-day mode, episodes are assembled PER-SPLIT (see
rl_quant.datasets.daily.build_daily_episodes), so a day's T+1 label can never reference an open outside its own
split. These helpers live in the package (not the driver) so the leak-critical logic is unit-tested.
"""
from __future__ import annotations

# the per-day fields carried out of a built window (everything build_window stores with a leading n_days axis)
_DAY_KEYS = ("bars", "bar_mask", "cov_blocks", "news_raw", "news_mask", "avail", "ret", "ret_valid", "day_open")


def flatten_days(windows: list[dict]) -> list[dict]:
    """Expand built windows (leading n_days axis) into per-day session dicts. Indexing a window tensor returns a
    view, so this is cheap -- the window tensors keep the storage alive. The per-day `date` is attached."""
    out = []
    for w in windows:
        for di in range(w["n_days"]):
            d = {k: w[k][di] for k in _DAY_KEYS}
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
    if mode == "daily":
        return time_split(day_sequence(built), train_frac, val_frac)
    tr, va, te = time_split(built, train_frac, val_frac)
    return flatten_days(tr), flatten_days(va), flatten_days(te)
