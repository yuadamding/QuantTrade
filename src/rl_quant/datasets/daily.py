"""Cross-day (DAILY) assembly for the event-timed framework.

Intraday cross-section is ~efficient in TOP50 (the price IC is ~0 at every horizon up to daily), so to capture
CROSS-DAY profit the policy must hold positions across days and be scored on cross-day returns. This module turns
the per-day end-of-day context (the encoder's last-block output) into a continuous day SEQUENCE with cross-day
labels, chunked into fixed-length episodes that are shaped EXACTLY like the intraday per-day dicts -- so the same
horizon-agnostic Stage-2 rollout (positions carried across the sequence axis) trains/evaluates them unchanged.

Label convention (point-in-time clean, T+1): decide at the end of day d using day-d's end-of-day context; execute
at the next session OPEN (day d+1) and exit at the following OPEN (day d+2) -> ret_d = open_{d+2}/open_{d+1} - 1
(one full day, including one overnight; all execution strictly after the decision). CASH (action 0) return = 0.
"""
from __future__ import annotations

import torch

CASH_INDEX = 0


def cross_day_returns(day_open: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """T+1 open-to-open cross-day return + validity from a date-sorted per-stock open series.
    day_open [N,A] (NaN where a stock has no bars that day) -> (ret [N,A], valid [N,A]).
    ret_d = open_{d+2}/open_{d+1} - 1; valid only where both opens are finite & positive. CASH = 0/valid."""
    N, A = day_open.shape
    ret = torch.zeros(N, A)
    valid = torch.zeros(N, A, dtype=torch.bool)
    valid[:, CASH_INDEX] = True                                  # CASH tradeable every day, return 0
    if N >= 3:
        o1, o2 = day_open[1:N - 1], day_open[2:N]                # exec open (d+1), exit open (d+2), for d=0..N-3
        good = torch.isfinite(o1) & torch.isfinite(o2) & (o1 > 0)
        r = torch.where(good, o2 / torch.where(o1 > 0, o1, torch.ones_like(o1)) - 1.0, torch.zeros_like(o1))
        ret[: N - 2] = torch.where(good, r.clamp(-1.0, 1.0), torch.zeros_like(r))
        valid[: N - 2] = good
        ret[:, CASH_INDEX] = 0.0
        valid[:, CASH_INDEX] = True
    return ret, valid


def horizon_close_returns(day_close: torch.Tensor, horizon: int, exec_delay: int = 1
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Close-to-close H-day forward return, POINT-IN-TIME clean. Decide at END of day d (using context through
    close d), EXECUTE at close[d+exec_delay], EXIT at close[d+exec_delay+horizon]:
        ret_d = close[d+exec_delay+horizon] / close[d+exec_delay] - 1.
    day_close [N,A] (NaN where a stock has no bars that day) -> (ret [N,A], valid [N,A]). CASH (action 0) = 0/valid.
    `exec_delay>=1` removes the T+0 look-ahead (the decision never trades at the close it observed). The policy may
    still CARRY a position far beyond `horizon` (gate=hold) -- `horizon` only sets the per-decision credit signal."""
    N, A = day_close.shape
    ret = torch.zeros(N, A)
    valid = torch.zeros(N, A, dtype=torch.bool)
    valid[:, CASH_INDEX] = True                                  # CASH tradeable every day, return 0
    e, x = exec_delay, exec_delay + horizon
    last = N - x                                                 # decisions d=0..last-1 have an in-range exit
    if last >= 1:
        entry, exit_ = day_close[e:e + last], day_close[x:x + last]
        good = torch.isfinite(entry) & torch.isfinite(exit_) & (entry > 0)
        safe = torch.where(entry > 0, entry, torch.ones_like(entry))
        r = torch.where(good, exit_ / safe - 1.0, torch.zeros_like(entry))
        ret[:last] = torch.where(good, r.clamp(-1.0, 1.0), torch.zeros_like(r))
        valid[:last] = good
        ret[:, CASH_INDEX] = 0.0
        valid[:, CASH_INDEX] = True
    return ret, valid


def build_daily_raw_episodes(records: list[dict], episode_len: int, stride: int | None = None,
                             horizon: int = 21, exec_delay: int = 1) -> list[dict]:
    """daily_raw episodes: close-to-close H-day labels over the FULL split day-sequence, sliced into episodes that
    carry the FROZEN end-of-day context + the FULL-day raw bars + news + avail for the cross-day policy.

    records: a DATE-SORTED list of per-day dicts, each with
        {market [dc], per_stock [A,dc], bars [A,S,F], bar_mask [A,S], news_raw [A,M,1], news_mask [A,M],
         day_close [A], avail [A]}.
    Labels are computed once over the whole split (so a day's H-day label is drawn from the global close series, not
    limited to its episode). Episodes are [s:s+L]; for a CONTINUOUS evaluation rollout pass episode_len>=N (one
    episode spanning the split, so every day has its full causal cross-day history)."""
    N = len(records)
    if N < exec_delay + horizon + 1:
        return []
    day_close = torch.stack([r["day_close"] for r in records])   # [N,A]
    ret, valid = horizon_close_returns(day_close, horizon, exec_delay)
    market = torch.stack([r["market"] for r in records])
    per_stock = torch.stack([r["per_stock"] for r in records])
    bars = torch.stack([r["bars"] for r in records])
    bar_mask = torch.stack([r["bar_mask"] for r in records])
    news_raw = torch.stack([r["news_raw"] for r in records])
    news_mask = torch.stack([r["news_mask"] for r in records])
    avail = torch.stack([r["avail"] for r in records])
    usable = N - (exec_delay + horizon)                          # days d=0..usable-1 carry an in-range H-day label
    L = min(episode_len, usable)
    st = stride if (stride and stride > 0) else L
    starts = list(range(0, usable - L + 1, st)) or [0]
    episodes = []
    for s in starts:
        e = s + L
        episodes.append({
            "market": market[s:e], "per_stock": per_stock[s:e], "bars": bars[s:e], "bar_mask": bar_mask[s:e],
            "news_raw": news_raw[s:e], "news_mask": news_mask[s:e], "avail": avail[s:e],
            "ret": ret[s:e], "ret_valid": valid[s:e], "n_blocks": L,
        })
    return episodes


def build_daily_episodes(records: list[dict], episode_len: int, stride: int | None = None) -> list[dict]:
    """records: a DATE-SORTED list of per-day dicts, each with the end-of-day context + day-open + availability:
        {market [d], per_stock [A,d], bars [A,S,F], bar_mask [A,S], news_raw [A,M,1], news_mask [A,M],
         day_open [A], avail [A]}.
    Returns equal-length episodes shaped like the intraday per-day dicts (sequence axis = DAYS), so Stage-2's
    rollout carries positions ACROSS days -- a policy that holds (gate=0) keeps a position for the WHOLE episode,
    which is how LONG holds (e.g. two trades >=180 days apart) are expressed. `episode_len` sets the max hold; a
    short `stride` yields OVERLAPPING sliding windows so long episodes still give many training samples (use
    stride=episode_len for non-overlapping evaluation). If the sequence is shorter than `episode_len`, ONE episode
    of the full usable length is emitted (so a short val/test split is not starved). Only the first N-2 days carry
    a T+1 label."""
    N = len(records)
    if N < 3:
        return []
    day_open = torch.stack([r["day_open"] for r in records])     # [N,A]
    ret, valid = cross_day_returns(day_open)
    market = torch.stack([r["market"] for r in records])         # [N,d]
    per_stock = torch.stack([r["per_stock"] for r in records])   # [N,A,d]
    news_raw = torch.stack([r["news_raw"] for r in records])     # [N,A,M,1]
    news_mask = torch.stack([r["news_mask"] for r in records])   # [N,A,M]
    bars = torch.stack([r["bars"] for r in records]) if "bars" in records[0] else None        # [N,A,S,F]
    bar_mask = torch.stack([r["bar_mask"] for r in records]) if "bar_mask" in records[0] else None  # [N,A,S]
    avail = torch.stack([r["avail"] for r in records])           # [N,A] as-of tradeability (traded that day)
    usable = N - 2                                               # labelled days
    L = min(episode_len, usable)                                 # don't starve a short split: one full episode
    st = stride if (stride and stride > 0) else L
    starts = list(range(0, usable - L + 1, st)) or [0]
    episodes = []
    for s in starts:
        episode = {"market": market[s:s + L], "per_stock": per_stock[s:s + L],
                   "news_raw": news_raw[s:s + L], "news_mask": news_mask[s:s + L], "avail": avail[s:s + L],
                   "ret": ret[s:s + L], "ret_valid": valid[s:s + L], "n_blocks": L}
        if bars is not None and bar_mask is not None:
            episode["bars"] = bars[s:s + L]
            episode["bar_mask"] = bar_mask[s:s + L]
        episodes.append(episode)
    return episodes
