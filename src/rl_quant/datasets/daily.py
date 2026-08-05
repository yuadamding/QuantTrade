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

from rl_quant.datasets.streaming import LazyDay

CASH_INDEX = 0


def _compact_detached(value: torch.Tensor) -> torch.Tensor:
    """Detach ``value`` without copying unless its view pins a larger backing allocation.

    ``encode_days(last_only=True)`` already returns compact, detached EOD tensors.  Cloning those again in the
    daily adapter briefly doubled TOP2000's largest host allocation before the chronological backing was built.
    Conversely, an EOD slice of a full block tensor *must* be copied or it retains all intraday blocks.  Storage
    size, rather than contiguity alone, distinguishes those cases and also lets broadcast zero/news tensors keep
    their intentionally tiny backing until the builder stacks the timeline once.
    """

    value = value.detach()
    logical_nbytes = value.numel() * value.element_size()
    if value.untyped_storage().nbytes() <= logical_nbytes:
        return value
    return value.clone(memory_format=torch.contiguous_format)


def _owned_eod(value: torch.Tensor, full_ndim: int, key: str, block_index: int = -1) -> torch.Tensor:
    """Accept a full ``[n_blocks, ...]`` field or an already-EOD field.

    A full block tensor must be compacted immediately.  An already-EOD tensor
    may intentionally be a day view into the distributed chronological cache;
    retain that view until :func:`build_daily_raw_episodes` creates the one
    owned timeline, instead of allocating one copy per day and then stacking a
    second copy of the same chronology.
    """
    if value.ndim == full_ndim:
        if not -value.shape[0] <= block_index < value.shape[0]:
            raise ValueError(
                f"daily field {key!r} close block {block_index} is outside {value.shape[0]} blocks"
            )
        return value[block_index].detach().contiguous().clone()
    elif value.ndim != full_ndim - 1:
        raise ValueError(
            f"daily field {key!r} must have {full_ndim} dims (per-block) or {full_ndim - 1} dims (EOD); "
            f"got shape {tuple(value.shape)}"
        )
    return value.detach()


def _timeline_view(value: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Return an episode view into a single chronological backing (never a per-episode copy)."""

    return value.narrow(0, start, stop - start)


def _stack_news_timeline(records: list[dict], key: str) -> torch.Tensor:
    """Stack enabled news, but keep the disabled-news chronology backed by one zero scalar."""

    values = [record[key] for record in records]
    first = values[0]

    def scalar_backed_zero(value: torch.Tensor) -> bool:
        if (
            value.shape != first.shape
            or value.dtype != first.dtype
            or value.device != first.device
            or value.numel() <= 1
            or value.untyped_storage().nbytes() != value.element_size()
        ):
            return False
        return not bool(value[(0,) * value.ndim])

    if all(scalar_backed_zero(value) for value in values):
        return torch.zeros((), dtype=first.dtype, device=first.device).expand(len(values), *first.shape)
    return torch.stack(values)


def to_daily_raw_records(encoded: list) -> list[dict]:
    """EOD adapter (the single, explicit place daily_raw assembles records). Each `encoded` day carries per-BLOCK
    context ([nB, ...]) OR already-selected EOD context + raw bars + day_close (from encode_days). This selects or
    accepts the END-OF-DAY fields for the daily decision. Full-block slices are compacted immediately; existing
    EOD chronology views are stacked once by the episode builder. Raw bars stay LAZY when the day is a LazyDay
    (streaming) -- so the shape contract is in-repo + unit-tested rather than living in an external driver.

    encoded[i]: {market [nB,d], per_stock [nB,A,d], avail [nB,A], news_raw [nB,A,M,1], news_mask [nB,A,M],
                 day_close [A], date, + bars/bar_mask (materialized) OR a lazy bars handle if a LazyDay}.
    -> records consumable by build_daily_raw_episodes (each end-of-day; bars materialized or via "_bars_day")."""
    if not encoded:
        return []
    recs = []
    for e in encoded:
        close_block = int(e["session_close_block"]) if "session_close_block" in e else -1
        r = {
            "date": e["date"],
            "day_close": _compact_detached(e["day_close"]),
            "avail": _owned_eod(e["avail"], 2, "avail", close_block),
            "market": _owned_eod(e["market"], 2, "market", close_block),
            "per_stock": _owned_eod(e["per_stock"], 3, "per_stock", close_block),
            "news_raw": _owned_eod(e["news_raw"], 4, "news_raw", close_block),
            "news_mask": _owned_eod(e["news_mask"], 3, "news_mask", close_block),
        }
        if "day_close_valid" in e:
            r["day_close_valid"] = _compact_detached(e["day_close_valid"])
        if isinstance(e, LazyDay):                              # streaming: keep full-day bars lazy via the handle
            r["_bars_day"] = e.raw_handle()                     # context overrides must not pin full storage
        else:
            r["bars"], r["bar_mask"] = e["bars"], e["bar_mask"]
        recs.append(r)
    return recs


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


def build_daily_raw_episodes(
    records: list[dict],
    episode_len: int,
    stride: int | None = None,
    horizon: int = 21,
    exec_delay: int = 1,
    *,
    auxiliary_horizons: tuple[int, ...] | None = None,
    entry_credit_horizon_days: int | None = None,
    require_aux_labels: bool = False,
    score_start: int = 0,
    score_tail: int | None = None,
) -> list[dict]:
    """Build daily_raw episodes with a canonical one-step transition reward and optional burn-in prefix.

    ``ret``/``ret_valid`` and ``real_ret``/``real_ret_valid`` are both the next one-day realized wealth-change
    basis used by policy training and evaluation. The H-day forecasting target is retained separately as
    ``aux_ret``/``aux_ret_valid``; it must not silently redefine the environment reward. By default episode
    eligibility therefore retains every one-step-valid decision. ``require_aux_labels=True`` limits the usable
    tail to decisions having an H-day auxiliary label, which is useful for an explicitly auxiliary-only training
    experiment but should not be used for continuous validation/test coverage.

    ``score_start`` is a record index: earlier records remain in the episode as causal observation burn-in but
    have ``score_mask=False``. Consumers should AND this mask with their loss/reporting label, keep the burn-in
    book in CASH, and use the prefix only to build temporal state.

    ``score_tail`` is the training-window mode.  When set, only a non-overlapping tail of at most this many
    decisions is scored in each overlapping episode; the earlier rows are causal observation-only burn-in.  A
    final offset window covers any stride remainder without scoring a date twice.  This makes a length-252,
    stride-15 episode train on decisions with 238--252 days of history instead of counting every short prefix,
    and keeps the scored current day inside a two-speed policy's recent-raw tail.  Leave it ``None`` for the
    continuous validation/test behavior.

    ``auxiliary_horizons`` optionally materializes an ordered multi-horizon label tensor under
    ``aux_ret_multi``/``aux_ret_valid_multi`` while retaining ``aux_ret`` as the compatibility alias for
    ``horizon``. ``entry_credit_horizon_days`` emits scored ``entry_credit_mask`` and ``entry_censored_mask``
    fields.  A decision is creditable only when its delayed fill and the complete requested holding lifecycle fit
    inside the episode; this prevents a right boundary from being misreported as an intentionally short hold.

    Episodes carry the FROZEN end-of-day context + FULL-day raw bars + news + availability for the cross-day
    policy. ``records`` is a DATE-SORTED list of per-day dicts, each with
        {market [dc], per_stock [A,dc], bars [A,S,F], bar_mask [A,S], news_raw [A,M,1], news_mask [A,M],
         day_close [A], avail [A]}.
    Labels are computed once over the whole record sequence (so a target is never limited to its episode). Episodes
    are [s:s+L]; for a CONTINUOUS evaluation rollout pass episode_len>=N (one episode spanning the usable sequence,
    so every scored day has its full causal cross-day history)."""
    N = len(records)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if auxiliary_horizons is None:
        materialized_horizons = (horizon,)
    else:
        materialized_horizons = tuple(auxiliary_horizons)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in materialized_horizons
        ):
            raise ValueError("auxiliary_horizons must contain positive integers")
        if tuple(sorted(set(materialized_horizons))) != materialized_horizons:
            raise ValueError("auxiliary_horizons must be strictly increasing and unique")
        if horizon not in materialized_horizons:
            raise ValueError("auxiliary_horizons must include horizon")
    if entry_credit_horizon_days is not None and (
        isinstance(entry_credit_horizon_days, bool)
        or not isinstance(entry_credit_horizon_days, int)
        or entry_credit_horizon_days <= 0
    ):
        raise ValueError("entry_credit_horizon_days must be a positive integer or None")
    if exec_delay < 1:
        raise ValueError(f"exec_delay must be at least one for PIT-clean execution, got {exec_delay}")
    if episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {episode_len}")
    if not 0 <= score_start <= N:
        raise ValueError(f"score_start must be in [0, {N}], got {score_start}")
    if score_tail is not None and (
        isinstance(score_tail, bool) or not isinstance(score_tail, int) or score_tail <= 0
    ):
        raise ValueError(f"score_tail must be a positive integer or None, got {score_tail!r}")
    if N < exec_delay + 2:
        return []
    day_close = torch.stack([r["day_close"] for r in records])   # [N,A]
    have_close_valid = ["day_close_valid" in record for record in records]
    if any(have_close_valid) and not all(have_close_valid):
        raise ValueError("daily records cannot mix explicit and implicit day_close_valid semantics")
    if all(have_close_valid):
        day_close_valid = torch.stack([r["day_close_valid"] for r in records]).bool()
        if day_close_valid.shape != day_close.shape:
            raise ValueError(
                f"day_close_valid shape {tuple(day_close_valid.shape)} must match day_close {tuple(day_close.shape)}"
            )
        day_close = day_close.masked_fill(~day_close_valid, float("nan"))
    auxiliary = {
        label_horizon: horizon_close_returns(day_close, label_horizon, exec_delay)
        for label_horizon in materialized_horizons
    }
    aux_ret, aux_valid = auxiliary[horizon]                    # compatibility alias for the primary H-day target
    aux_ret_multi = aux_valid_multi = None
    if auxiliary_horizons is not None:
        aux_ret_multi = torch.stack([auxiliary[value][0] for value in materialized_horizons], dim=1)
        aux_valid_multi = torch.stack([auxiliary[value][1] for value in materialized_horizons], dim=1)
    ret, valid = horizon_close_returns(day_close, 1, exec_delay)                   # canonical 1-day MDP transition
    # PAST-return INPUT channel (PIT-clean): day d carries its OWN 1-day close-to-close return close_d/close_{d-1}-1,
    # fully known at the EOD-d decision. This is the raw close series under a scale-invariant normalization (the
    # 'level'-norm spirit) -- it lets the cross-day temporal encoder compute momentum/reversal over ITS OWN window
    # (e.g. ~12-month momentum at 252d reach) instead of asking a within-day encoder to reconstruct price history.
    past_ret = torch.zeros(day_close.shape)
    past_valid = torch.zeros(day_close.shape, dtype=torch.bool)
    past_valid[:, CASH_INDEX] = True
    if N >= 2:
        c0, c1 = day_close[:-1], day_close[1:]
        good = torch.isfinite(c0) & torch.isfinite(c1) & (c0 > 0)
        pr = torch.where(good, c1 / torch.where(c0 > 0, c0, torch.ones_like(c0)) - 1.0, torch.zeros_like(c0))
        past_ret[1:] = torch.where(good, pr.clamp(-1.0, 1.0), torch.zeros_like(pr))
        past_valid[1:] = good
        past_ret[:, CASH_INDEX] = 0.0
        past_valid[:, CASH_INDEX] = True
    market = torch.stack([r["market"] for r in records])
    per_stock = torch.stack([r["per_stock"] for r in records])
    # Disabled news arrives as scalar-backed broadcast zeros. Preserve that representation across the date axis;
    # a normal torch.stack would silently turn it into ~0.25 GiB of TOP2000 host storage per rank. Enabled news
    # remains dense and follows the ordinary stack path.
    news_raw = _stack_news_timeline(records, "news_raw")
    news_mask = _stack_news_timeline(records, "news_mask")
    avail = torch.stack([r["avail"] for r in records])
    # STREAMING: a record carries "_bars_day" (a lazy per-day handle exposing ["bars"]/["bar_mask"]) instead of a
    # materialized "bars". DON'T stack full-day bars (a 171-day episode would be hundreds of GB) -- keep the per-day
    # handles; the rollout loads day-t bars on demand. The small fields above are already in RAM (materialized at
    # encode), so stacking them is cheap.
    stream = "_bars_day" in records[0]
    bars = bar_mask = None
    if not stream:
        bars = torch.stack([r["bars"] for r in records])
        bar_mask = torch.stack([r["bar_mask"] for r in records])
    required_horizon = max(materialized_horizons) if require_aux_labels else 1
    usable = N - (exec_delay + required_horizon)                 # eligible decisions on the selected label basis
    if usable <= 0:
        return []
    L = min(episode_len, usable)
    if score_tail is not None and score_tail > L:
        raise ValueError(f"score_tail {score_tail} exceeds the usable episode length {L}")
    st = stride if (stride and stride > 0) else L
    starts = list(range(0, usable - L + 1, st)) or [0]
    if score_tail is not None:
        # A regular stride can leave a short tail.  End one final window at the
        # data boundary; ``next_score_index`` below admits only the remainder,
        # so already-scored dates are not duplicated.
        final_start = usable - L
        if starts[-1] != final_start:
            starts.append(final_start)
    episodes = []
    next_score_index = max(score_start, L - score_tail) if score_tail is not None else score_start
    for s in starts:
        e = s + L
        indices = torch.arange(s, e)
        if score_tail is None:
            score_mask = indices >= score_start
        else:
            score_from = max(s, score_start, e - score_tail, next_score_index)
            score_mask = indices >= score_from
            next_score_index = max(next_score_index, e)
        ep = {
            # Every large field is a view of one date-sorted backing. With TOP2000/252d/stride15 this is the
            # difference between one chronology per rank and roughly forty overlapping episode copies.
            "market": _timeline_view(market, s, e),
            "per_stock": _timeline_view(per_stock, s, e),
            "news_raw": _timeline_view(news_raw, s, e),
            "news_mask": _timeline_view(news_mask, s, e),
            "avail": _timeline_view(avail, s, e),
            "ret": _timeline_view(ret, s, e),
            "ret_valid": _timeline_view(valid, s, e),                       # canonical 1-day train reward
            "real_ret": _timeline_view(ret, s, e),
            "real_ret_valid": _timeline_view(valid, s, e),                  # same basis, compatibility alias
            "aux_ret": _timeline_view(aux_ret, s, e),
            "aux_ret_valid": _timeline_view(aux_valid, s, e),               # H-day auxiliary forecast label
            "past_ret": _timeline_view(past_ret, s, e),
            "past_ret_valid": _timeline_view(past_valid, s, e),             # PIT input: own 1-day past return
            "score_mask": score_mask,                                        # input-only prefix is not scored
            "decision_ids": tuple(str(records[index].get("date", index)) for index in range(s, e)),
            "n_blocks": L,
        }
        if auxiliary_horizons is not None:
            assert aux_ret_multi is not None and aux_valid_multi is not None
            ep["auxiliary_horizons"] = materialized_horizons
            ep["aux_ret_multi"] = _timeline_view(aux_ret_multi, s, e)
            ep["aux_ret_valid_multi"] = _timeline_view(aux_valid_multi, s, e)
        if entry_credit_horizon_days is not None:
            # Decision i fills after ``exec_delay`` states and needs a terminal
            # state after the requested number of holding transitions.  The
            # strict comparison is equivalent to i + delay + horizon <= e - 1.
            lifecycle_fits = indices + exec_delay + entry_credit_horizon_days < e
            ep["entry_credit_horizon_days"] = entry_credit_horizon_days
            ep["entry_credit_mask"] = score_mask & lifecycle_fits
            ep["entry_censored_mask"] = score_mask & ~lifecycle_fits
        if stream:
            ep["bars_days"] = [r["_bars_day"] for r in records[s:e]]   # lazy per-day handles (load bars on demand)
        else:
            assert bars is not None and bar_mask is not None
            ep["bars"] = _timeline_view(bars, s, e)
            ep["bar_mask"] = _timeline_view(bar_mask, s, e)
        episodes.append(ep)
    return episodes


def build_daily_episodes(
    records: list[dict],
    episode_len: int,
    stride: int | None = None,
    *,
    raw_block_steps: int | None = None,
) -> list[dict]:
    """records: a DATE-SORTED list of per-day dicts, each with the end-of-day context + day-open + availability:
        {market [d], per_stock [A,d], bars [A,S,F], bar_mask [A,S], news_raw [A,M,1], news_mask [A,M],
         day_open [A], avail [A]}.
    ``raw_block_steps`` compacts each day's raw input to its final block *before* stacking. This is semantically
    exact for the generic daily DecisionPolicy, whose daily raw step consumes only that block, and avoids copying
    an otherwise unused full session into every episode. Pass the configured block length in bar slots; already
    compact inputs are accepted unchanged.

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
    if raw_block_steps is not None and raw_block_steps <= 0:
        raise ValueError(f"raw_block_steps must be positive, got {raw_block_steps}")
    bars = bar_mask = None
    if "bars" in records[0] and "bar_mask" in records[0]:
        if raw_block_steps is None:
            bars = torch.stack([r["bars"] for r in records])                  # [N,A,S,F]
            bar_mask = torch.stack([r["bar_mask"] for r in records])          # [N,A,S]
        else:
            if any(r["bars"].shape[-2] < raw_block_steps for r in records):
                shortest = min(r["bars"].shape[-2] for r in records)
                raise ValueError(f"raw daily input has only {shortest} bar slots, need {raw_block_steps}")
            # Slice before stack: torch.stack creates the compact owned episode storage, so no tail view pins a
            # full-session backing allocation after the input records are released.
            bars = torch.stack([r["bars"][..., -raw_block_steps:, :] for r in records])
            bar_mask = torch.stack([r["bar_mask"][..., -raw_block_steps:] for r in records])
    avail = torch.stack([r["avail"] for r in records])           # [N,A] as-of tradeability (traded that day)
    usable = N - 2                                               # labelled days
    L = min(episode_len, usable)                                 # don't starve a short split: one full episode
    st = stride if (stride and stride > 0) else L
    starts = list(range(0, usable - L + 1, st)) or [0]
    episodes = []
    for s in starts:
        episode = {"market": market[s:s + L], "per_stock": per_stock[s:s + L],
                   "news_raw": news_raw[s:s + L], "news_mask": news_mask[s:s + L], "avail": avail[s:s + L],
                   "ret": ret[s:s + L], "ret_valid": valid[s:s + L],
                   "decision_ids": tuple(
                       str(records[index].get("date", index)) for index in range(s, s + L)
                   ),
                   "n_blocks": L}
        if bars is not None and bar_mask is not None:
            episode["bars"] = bars[s:s + L]
            episode["bar_mask"] = bar_mask[s:s + L]
        episodes.append(episode)
    return episodes
