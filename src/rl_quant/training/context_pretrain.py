"""Stage-1 training: self-supervised CONTEXT LEARNING over full sessions, then freeze + encode.

The unit is a trading DAY (a full RTH session = nB blocks). The two-tier encoder turns each day into a context
at EVERY block; the SSL pretext has TWO heads (both targets derived from the T+1 labels -- no extra inputs):
  * MARKET head: from each block's market context predict that block's next-interval equal-weight market return
    + realized vol.
  * PER-STOCK head: from each stock's per-block context predict that stock's next-block CROSS-SECTIONALLY
    DEMEANED return (r_i - equal-weight). The market head alone would train only the pooled mean, leaving the
    per-stock embeddings with no gradient rewarding which-stock-wins discrimination; this head makes the FROZEN
    context carry the relative-value signal the policy needs (without it the policy has nothing to select on).
The encoder is then FROZEN and used to ENCODE every day into detached per-block context embeddings while carrying
the raw bars forward for Stage 2. The policy trains on those detached contexts plus its own trainable raw-second
encoder (it never holds a Stage-1 encoder reference -> the split is literal).

Days stream from CPU-resident storage to ``device`` per micro-batch (full sessions are too big to hold all on
GPU) and gradients are ACCUMULATED over ``accum_steps`` micro-batches. Resumability is delegated to the caller
(start_step + optimizer + an on_checkpoint that persists model+opt+step+RNG); RNG is the global torch RNG.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from rl_quant.datasets.streaming import LazyDay
from rl_quant.training._optim import apply_lr, lr_scale, make_adamw


def ssl_targets(ret: torch.Tensor, ret_valid: torch.Tensor) -> torch.Tensor:
    """Per-block [equal-weight return, realized vol] over the valid non-CASH actions. ret/ret_valid [nB,A] -> [nB,2]."""
    r, v = ret[:, 1:], ret_valid[:, 1:]
    n = v.float().sum(1).clamp_min(1.0)
    ew = torch.where(v, r, torch.zeros_like(r)).sum(1) / n
    vol = torch.sqrt(torch.where(v, r * r, torch.zeros_like(r)).sum(1) / n)
    return torch.stack([ew, vol], dim=-1)


def ssl_targets_perstock(ret: torch.Tensor, ret_valid: torch.Tensor):
    """Per-stock CROSS-SECTIONALLY-DEMEANED next-block return (r_i - equal-weight) + its validity mask, over the
    full action axis (CASH column is invalid). ret/ret_valid [nB,A] -> (tgt [nB,A], mask [nB,A])."""
    v = ret_valid.clone()
    v[:, 0] = False                                                            # CASH carries no relative signal
    n = v[:, 1:].float().sum(1, keepdim=True).clamp_min(1.0)                    # valid non-CASH per block
    ew = torch.where(v[:, 1:], ret[:, 1:], torch.zeros_like(ret[:, 1:])).sum(1, keepdim=True) / n
    tgt = torch.zeros_like(ret)
    tgt[:, 1:] = ret[:, 1:] - ew                                               # demeaned -> the relative-value target
    return torch.where(v, tgt, torch.zeros_like(tgt)), v


def ssl_targets_daily(day_close: torch.Tensor, horizon: int, exec_delay: int = 1):
    """DAILY per-stock SSL target: each day's next-H-day CROSS-SECTIONALLY-DEMEANED close-to-close return -- the
    relative-value signal a DAILY cross-sectional policy actually needs (vs the intraday next-block target). Built
    over a chronological day_close sequence [N,A]; PIT-clean (uses only the close series within the split, so the
    last exec_delay+horizon days are invalid rather than peeking ahead). -> (tgt [N,A], mask [N,A]) (CASH invalid)."""
    from rl_quant.datasets.daily import horizon_close_returns
    ret, valid = horizon_close_returns(day_close, horizon, exec_delay)
    valid = valid.clone()
    valid[:, 0] = False                                                        # CASH carries no relative signal
    n = valid[:, 1:].float().sum(1, keepdim=True).clamp_min(1.0)
    ew = torch.where(valid[:, 1:], ret[:, 1:], torch.zeros_like(ret[:, 1:])).sum(1, keepdim=True) / n
    tgt = torch.zeros_like(ret)
    tgt[:, 1:] = ret[:, 1:] - ew
    return torch.where(valid, tgt, torch.zeros_like(tgt)), valid


def train_context_encoder(
    encoder, head, train_days, *, device, perstock_head=None, perstock_coef: float = 1.0,
    daily_head=None, daily_targets=None, daily_coef: float = 1.0,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 1, accum_steps: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
    grad_reduce: Callable[[list], None] | None = None,
    prepare_checkpoint: Callable[[], None] | None = None,
    sync_after_checkpoint: Callable[[], None] | None = None,
):
    """Fit the encoder + the market SSL head (+ optional per-stock and DAILY SSL heads) over full sessions.
    ``daily_head`` + ``daily_targets`` (a list aligned with train_days of (tgt [A], mask [A]) = each day's next-H-day
    cross-sectional return) add a DAILY relative-value pretext on the END-OF-DAY context -- the target a daily
    cross-sectional policy needs (see ssl_targets_daily). STREAMS ``batch_size`` days/micro-batch and
    GRADIENT-ACCUMULATES ``accum_steps`` micro-batches per step. Distributed
    callers may provide ``prepare_checkpoint`` for an all-rank snapshot (for
    example per-rank RNG gathering), then ``sync_after_checkpoint`` so all
    ranks wait for rank-0 checkpoint I/O before starting the next collective.
    Returns the optimizer."""
    # A zero-weight auxiliary objective is disabled, not merely multiplied by
    # zero after its dense head has run.  This matters at TOP2000 scale: the
    # per-stock head's hidden activation has shape [B,nB,A,d].  Keep disabled
    # heads in ``heads``/``params`` so existing optimizer and checkpoint layouts
    # remain compatible; their gradients stay None on every rank and the
    # distributed reducer already skips parameters unused globally.
    use_perstock = perstock_head is not None and perstock_coef != 0.0
    use_daily = daily_head is not None and daily_targets is not None and daily_coef != 0.0
    heads = [head] + ([perstock_head] if perstock_head is not None else []) + \
            ([daily_head] if daily_head is not None else [])
    params = list(encoder.parameters()) + [p for h in heads for p in h.parameters()]
    if optimizer is None:
        optimizer = make_adamw(params, lr=lr, weight_decay=weight_decay)
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    targets = [ssl_targets(d["ret"], d["ret_valid"]) for d in train_days]      # per day [nB,2]
    valid = [d["ret_valid"][:, 1:].any(1) for d in train_days]                 # per day [nB] block has a target
    ps = (
        [ssl_targets_perstock(d["ret"], d["ret_valid"]) for d in train_days]
        if use_perstock else None
    )                                                                          # per day ([nB,A], [nB,A])
    n = len(train_days)
    if not 0 < batch_size <= n:
        raise ValueError(f"batch_size must be in [1, {n}], got {batch_size}")
    encoder.train()
    for h in heads:
        h.train()

    def micro():
        # A local micro-batch should spend every slot on a distinct trading
        # day.  Rank date shards are disjoint, so this also avoids cross-rank
        # duplicates without adding a distributed sampling collective.
        idx = torch.randperm(n)[:batch_size].tolist()
        st = lambda src: torch.stack([src[i] for i in idx]).to(device, non_blocking=True)  # noqa: E731
        bars = torch.stack([train_days[i]["bars"] for i in idx]).to(device, non_blocking=True)
        mask = torch.stack([train_days[i]["bar_mask"] for i in idx]).to(device, non_blocking=True)
        cov = torch.stack([train_days[i]["cov_blocks"] for i in idx]).to(device, non_blocking=True)
        cov_valid = None
        if all("cov_valid_blocks" in train_days[i] for i in idx):
            cov_valid = torch.stack([train_days[i]["cov_valid_blocks"] for i in idx]).to(
                device, non_blocking=True
            )
        close_blocks = torch.tensor(
            [
                int(train_days[i]["session_close_block"])
                if "session_close_block" in train_days[i]
                else int(train_days[i]["ret"].shape[0]) - 1
                for i in idx
            ],
            dtype=torch.long,
            device=device,
        )
        tgt, vm = st(targets), st(valid)                                          # [b,nB,2], [b,nB]
        ps_tgt = ps_vm = None
        if use_perstock:
            assert ps is not None
            ps_tgt = torch.stack([ps[i][0] for i in idx]).to(device, non_blocking=True)   # [b,nB,A]
            ps_vm = torch.stack([ps[i][1] for i in idx]).to(device, non_blocking=True)    # [b,nB,A]
        d_tgt = d_vm = None
        if use_daily:
            d_tgt = torch.stack([daily_targets[i][0] for i in idx]).to(device, non_blocking=True)  # [b,A]
            d_vm = torch.stack([daily_targets[i][1] for i in idx]).to(device, non_blocking=True)   # [b,A]
        return bars, mask, cov, cov_valid, close_blocks, tgt, vm, ps_tgt, ps_vm, d_tgt, d_vm

    for step in range(start_step, steps):
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            bars, mask, cov, cov_valid, close_blocks, tgt, vm, ps_tgt, ps_vm, d_tgt, d_vm = micro()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                encoded = encoder(bars, mask, cov, cov_valid) if cov_valid is not None else encoder(bars, mask, cov)
                per_stock, market = encoded                          # [b,nB,A,d], [b,nB,d]
                pred = head(market)                                   # [b,nB,2]
                loss = F.smooth_l1_loss(pred[vm], tgt[vm]) if vm.any() else (pred.sum() * 0.0)
                if use_perstock and ps_vm is not None and ps_tgt is not None and ps_vm.any():
                    ps_pred = perstock_head(per_stock)                # [b,nB,A]
                    loss = loss + perstock_coef * F.smooth_l1_loss(ps_pred[ps_vm], ps_tgt[ps_vm])
                if use_daily and d_vm.any():                          # DAILY next-H-day relative-value pretext
                    rows = torch.arange(per_stock.shape[0], device=per_stock.device)
                    d_pred = daily_head(per_stock[rows, close_blocks])  # session-boundary context -> [b,A]
                    loss = loss + daily_coef * F.smooth_l1_loss(d_pred[d_vm], d_tgt[d_vm])
                loss = loss / accum_steps
            loss.backward()
        if grad_reduce is not None:                  # data-parallel: average grads across ranks before the step
            grad_reduce(params)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        checkpoint_due = bool(checkpoint_every and (step + 1) % checkpoint_every == 0)
        if checkpoint_due:
            if prepare_checkpoint is not None:
                prepare_checkpoint()
            if on_checkpoint is not None:
                on_checkpoint(step + 1, optimizer)
            if sync_after_checkpoint is not None:
                sync_after_checkpoint()
    return optimizer


def freeze_encoder(encoder) -> None:
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()


@torch.no_grad()
def encode_days(
    encoder,
    days,
    device,
    batch: int = 2,
    amp: bool = False,
    *,
    last_only: bool = False,
    output_dtype: torch.dtype = torch.float32,
) -> list[dict | LazyDay]:
    """Encode frozen context in bounded day chunks.

    By default, return every block for intraday policies. ``last_only=True`` is the daily-policy storage path: the
    full session is still encoded causally, but encoders advertising ``supports_last_only`` avoid retaining and
    fusing unused earlier-block outputs. Only the end-of-day market/per-stock context and matching availability/news
    fields leave the encode chunk. Dense selected tensors are cloned after slicing so they own a compact storage
    allocation instead of pinning a full ``[n_blocks, ...]`` backing tensor (especially important for
    :class:`LazyDay` overrides). Disabled-news tensors are scalar-backed broadcast-zero views and retain that compact
    representation. ``output_dtype`` controls only the cached context embedding dtype; raw inputs, masks, labels,
    and prices retain their source dtypes.

    Returned records carry no encoder reference. Raw bars remain materialized for in-memory days and lazy for
    ``LazyDay`` inputs; the daily adapters decide whether a policy needs a full session or only its final raw block.
    """
    if batch <= 0:
        raise ValueError(f"encode batch must be positive, got {batch}")
    if not isinstance(output_dtype, torch.dtype) or not output_dtype.is_floating_point:
        raise TypeError(f"output_dtype must be a floating torch.dtype, got {output_dtype!r}")

    def owned_cpu(t: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Detach and give a (possibly sliced) tensor its own compact CPU storage."""
        return t.detach().to(device="cpu", dtype=dtype or t.dtype).contiguous().clone()

    def eod_field(day, key: str) -> torch.Tensor:
        block = int(day["session_close_block"]) if "session_close_block" in day else -1
        selected = day[key][block]
        # The reportable no-news cache represents the full logical tensor as an expanded zero scalar. Cloning its
        # EOD view would materialize [A,max_news,...] once per day before the episode builder stacks it again. Keep
        # the broadcast representation; dense fields still take the compact owned-copy path below.
        scalar_backed = selected.numel() > 1 and selected.untyped_storage().nbytes() == selected.element_size()
        if key in ("news_raw", "news_mask") and scalar_backed:
            origin = selected[(0,) * selected.ndim]
            if not bool(origin):
                return torch.zeros((), dtype=selected.dtype, device="cpu").expand(selected.shape)
        return owned_cpu(selected)

    encoder.eval()
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    out: list[dict | LazyDay] = []
    for i in range(0, len(days), batch):
        chunk = days[i:i + batch]
        bars = torch.stack([d["bars"] for d in chunk]).to(device, non_blocking=True)
        mask = torch.stack([d["bar_mask"] for d in chunk]).to(device, non_blocking=True)
        cov = torch.stack([d["cov_blocks"] for d in chunk]).to(device, non_blocking=True)
        cov_valid = None
        if all("cov_valid_blocks" in d for d in chunk):
            cov_valid = torch.stack([d["cov_valid_blocks"] for d in chunk]).to(device, non_blocking=True)
        close_blocks = None
        if last_only:
            have_close_blocks = ["session_close_block" in d for d in chunk]
            if any(have_close_blocks) and not all(have_close_blocks):
                raise ValueError("a daily encode chunk cannot mix days with and without session_close_block")
            if all(have_close_blocks):
                close_blocks = torch.stack([d["session_close_block"] for d in chunk]).to(
                    device=device, dtype=torch.long, non_blocking=True
                )
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            supports_fast_last = bool(getattr(encoder, "supports_last_only", False))
            if cov_valid is not None and supports_fast_last:
                encoded = encoder(
                    bars,
                    mask,
                    cov,
                    cov_valid,
                    last_only=last_only,
                    last_block_index=close_blocks,
                )
            elif cov_valid is not None:
                encoded = encoder(bars, mask, cov, cov_valid)
            elif supports_fast_last:
                encoded = encoder(
                    bars,
                    mask,
                    cov,
                    last_only=last_only,
                    last_block_index=close_blocks,
                )
            else:
                encoded = encoder(bars, mask, cov)
            per_stock, market = encoded                              # [b,nB,A,d], [b,nB,d]
        if last_only:
            # Slice on the accelerator before the device transfer. ``owned_cpu`` also clones CPU encodes, where a
            # same-device ``to`` would otherwise preserve the full output storage behind this one-block view.
            if supports_fast_last or close_blocks is None:
                per_stock = owned_cpu(per_stock[:, -1], dtype=output_dtype)   # [b,A,d]
                market = owned_cpu(market[:, -1], dtype=output_dtype)         # [b,d]
            else:
                rows = torch.arange(len(chunk), device=per_stock.device)
                per_stock = owned_cpu(per_stock[rows, close_blocks], dtype=output_dtype)
                market = owned_cpu(market[rows, close_blocks], dtype=output_dtype)
        else:
            per_stock = owned_cpu(per_stock, dtype=output_dtype)          # [b,nB,A,d]
            market = owned_cpu(market, dtype=output_dtype)                # [b,nB,d]
        for j, d in enumerate(chunk):
            if last_only:
                small = {
                    "market": owned_cpu(market[j]),
                    "per_stock": owned_cpu(per_stock[j]),
                    "avail": eod_field(d, "avail"),
                    "news_raw": eod_field(d, "news_raw"),
                    "news_mask": eod_field(d, "news_mask"),
                }
                # DAILY rewards are rebuilt causally from the cross-day price sequence. Keep only the EOD copies
                # of the intraday labels for dict compatibility; do not retain a full-block label backing tensor.
                if "ret" in d:
                    small["ret"] = eod_field(d, "ret")
                if "ret_valid" in d:
                    small["ret_valid"] = eod_field(d, "ret_valid")
                for key in ("day_open", "day_close", "day_close_valid", "session_close_block"):
                    if key in d:
                        small[key] = owned_cpu(d[key])
                if isinstance(d, LazyDay):
                    # The lazy window remains solely as the raw-bar handle. All frequently accessed small fields
                    # are owned overrides and therefore cannot keep an evicted full-window tensor alive as views.
                    out.append(d.with_overrides(**small))
                else:
                    out.append({
                        **small,
                        "bars": d["bars"],
                        "bar_mask": d["bar_mask"],
                        "n_blocks": 1,
                        **({"date": d["date"]} if "date" in d else {}),
                    })
                continue
            if isinstance(d, LazyDay):
                # STREAMING: attach the embeddings + materialize the SMALL per-day fields in RAM (the window is
                # already resident here for the encode, so this is an LRU hit, not a reload). ONLY bars/bar_mask
                # stay lazy -> downstream (episode build, rollout) never reloads a 1GB window for a label or mask.
                out.append(d.with_overrides(
                    market=market[j].clone(), per_stock=per_stock[j].clone(),
                    avail=d["avail"].clone(), news_raw=d["news_raw"].clone(), news_mask=d["news_mask"].clone(),
                    ret=d["ret"].clone(), ret_valid=d["ret_valid"].clone(),
                    day_open=d["day_open"].clone(), day_close=d["day_close"].clone(),
                    **({"day_close_valid": d["day_close_valid"].clone()} if "day_close_valid" in d else {}),
                    **({"session_close_block": d["session_close_block"].clone()}
                       if "session_close_block" in d else {})))
            else:
                out.append({
                    "market": market[j], "per_stock": per_stock[j],
                    "bars": d["bars"], "bar_mask": d["bar_mask"],
                    "news_raw": d["news_raw"], "news_mask": d["news_mask"], "avail": d["avail"],
                    "ret": d["ret"], "ret_valid": d["ret_valid"], "n_blocks": d["ret"].shape[0],
                    # carry the cross-day execution prices + date so the DAILY episode builders are self-contained
                    # (no reliance on an external adapter to re-attach day_close); harmless for intraday.
                    **({"day_open": d["day_open"]} if "day_open" in d else {}),
                    **({"day_close": d["day_close"]} if "day_close" in d else {}),
                    **({"day_close_valid": d["day_close_valid"]} if "day_close_valid" in d else {}),
                    **({"session_close_block": d["session_close_block"]} if "session_close_block" in d else {}),
                    **({"date": d["date"]} if "date" in d else {}),
                })
    return out
