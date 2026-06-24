"""Train-time organizer for the RAW time-partitioned dataset (e.g. TOP50 `top50_raw_time_partitioned_v1`).

Reads ONLY raw inputs from a dataset root: ``partitions/<S_to_E>/{bars.parquet, covariates.parquet, news.jsonl}``
+ ``universe.json``. NOTHING is precomputed/stored as features -- this module ORGANIZES the raw inputs into the
tensors the learning framework consumes, all at train time. The design is BLOCK-ALIGNED and EVENT-TIMED: one full
RTH session per trading day is stored ONCE, SESSION-ALIGNED (index s = second s after the 09:30 open); the encoder
turns it into a context at every ``block_seconds`` block, and the policy chooses WHEN to act over those blocks.

  * context bars: the RAW 1-second OHLCV bars directly (one token per second), session-aligned over the whole
    ``session_seconds`` RTH session (no pooling, no hand-computed features; the encoder normalizes + compresses
    them itself, causally per block).
  * as-of covariates: the latest point-in-time covariate record available at each block's end.
  * news: the RAW per-article sentiment scores available by each block's end (the model aggregates them at train
    time -- no precomputed count/mean).
  * T+1 forward-return labels per (day, block): decide at block b (context <= block-b end), EXECUTE at block
    b+1's end, hold to block b+2's end -- close@(b+1 end)+latency -> close@(b+2 end)+latency (the reward signal).

The block grid (78 blocks/day at 300s) is DST-aware (zoneinfo). Output tensors carry a leading n_days axis; the
driver flattens windows to per-day units (the training unit).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

BAR_FIELDS = ("open", "high", "low", "close", "volume")
BAR_FEATS = len(BAR_FIELDS)  # the encoder consumes the RAW bar fields directly (one token per second)
COV_FIELDS = (
    "market_cap", "share_class_shares_outstanding", "financial_revenue", "financial_net_income",
    "financial_assets", "financial_liabilities", "financial_cash", "financial_operating_cashflow",
    "dividend_cash_amount", "split_ratio", "is_common_stock", "is_adr_or_foreign",
)
NEWS_RAW_DIM = 1  # raw fields kept per news article (the qwen3 sentiment_score) -- NO precomputed aggregate
MAX_NEWS = 32     # most-recent articles kept per (stock, decision); the model aggregates them at train time


@dataclass
class RawWindowConfig:
    session_seconds: int = 23400  # full RTH session (09:30->16:00) stored once per day (session-aligned)
    block_seconds: int = 300      # candidate/decision cadence = the encoder's tier-1 block (must match it)
    max_news: int = MAX_NEWS
    open_et_hhmm: tuple[int, int] = (9, 30)
    exec_latency_ms: int = 1000
    cache_version: int = 1
    bar_fields: tuple[str, ...] = field(default=BAR_FIELDS)
    cov_fields: tuple[str, ...] = field(default=COV_FIELDS)


def load_universe(root: Path) -> tuple[list[str], int]:
    u = json.loads((Path(root) / "universe.json").read_text())
    return list(u["actions"]), int(u["cash_index"])  # actions[0] == CASH


def list_windows(root: Path) -> list[str]:
    parts = Path(root) / "partitions"
    return sorted(p.name for p in parts.iterdir() if (p / "bars.parquet").exists())


try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # zoneinfo/tzdata unavailable -> fall back to the month heuristic below
    _ET = None


def _et_offset_hours(date_iso: str) -> int:
    """UTC offset (hours) of US/Eastern on `date_iso`. Uses the real DST calendar (2nd Sun Mar / 1st Sun Nov)
    via zoneinfo; the month heuristic is only a fallback if tzdata is missing. Sampled at noon ET so the RTH
    decision/open times (all after the 02:00 transition) get the correct post-transition offset."""
    y, m, d = (int(x) for x in date_iso[:10].split("-"))
    if _ET is not None:
        off = dt.datetime(y, m, d, 12, tzinfo=_ET).utcoffset()
        if off is not None:
            return int(off.total_seconds() // 3600)
    return -4 if 3 <= m <= 11 else -5


def _load_window_raw(root: Path, window: str, cfg: RawWindowConfig):
    base = Path(root) / "partitions" / window
    bt = pq.read_table(base / "bars.parquet",
                       columns=["symbol", "timestamp_ms", "date_exchange", *cfg.bar_fields])
    bars = {c: (bt.column(c).to_numpy() if c not in ("symbol", "date_exchange") else bt.column(c).to_pylist())
            for c in bt.column_names}
    cov = None
    if (base / "covariates.parquet").exists():
        ct = pq.read_table(base / "covariates.parquet")
        cov = {c: ct.column(c).to_pylist() for c in ct.column_names}
    news = []
    if (base / "news.jsonl").exists():
        for line in (base / "news.jsonl").read_text().splitlines():
            if line.strip():
                news.append(json.loads(line))
    return bars, cov, news


def _open_ms(date_iso: str, cfg: RawWindowConfig) -> int:
    """UTC ms of the 09:30 ET session open on `date_iso` (DST-aware)."""
    off = _et_offset_hours(date_iso)
    y, m, day = map(int, date_iso.split("-"))
    return int(dt.datetime(y, m, day, cfg.open_et_hhmm[0] - off, cfg.open_et_hhmm[1],
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def build_window(root: Path, window: str, stock_to_idx: dict[str, int], n_actions: int,
                 cfg: RawWindowConfig) -> dict | None:
    """BLOCK-ALIGNED organizer. One full RTH session per trading day is stored session-aligned (index s = second
    s after the 09:30 open); the encoder turns it into a context at every `block_seconds` block, and the policy
    chooses WHEN to act over those blocks. Per (day, block) we also store as-of covariates, raw news, and the
    T+1 forward-return label: decide at block b (context <= block-b end), EXECUTE at block b+1's end, hold to
    block b+2's end. Returns per-window tensors with a leading n_days axis."""
    bars, cov, news = _load_window_raw(root, window, cfg)
    days = sorted(set(bars["date_exchange"]))
    if not days:
        return None
    S, bl = cfg.session_seconds, cfg.block_seconds
    nB = S // bl
    Dd, A, F, M, NC = len(days), n_actions, len(cfg.bar_fields), cfg.max_news, len(cfg.cov_fields)
    day_idx = {d: i for i, d in enumerate(days)}
    open_ms = np.array([_open_ms(d, cfg) for d in days], dtype=np.int64)              # [Dd]
    block_end = open_ms[:, None] + (np.arange(1, nB + 1) * bl * 1000)                 # [Dd,nB] block-b end (ms)
    lat = cfg.exec_latency_ms

    bars_t = np.zeros((Dd, A, S, F), dtype=np.float32)
    bar_mask = np.zeros((Dd, A, S), dtype=bool)
    covt = np.zeros((Dd, nB, A, NC), dtype=np.float32)
    news_raw = np.zeros((Dd, nB, A, M, NEWS_RAW_DIM), dtype=np.float32)
    news_mask = np.zeros((Dd, nB, A, M), dtype=bool)
    ret = np.full((Dd, nB, A), np.nan, dtype=np.float32)
    ret_valid = np.zeros((Dd, nB, A), dtype=bool)
    ret[:, :, 0] = 0.0          # CASH return is identically 0 at every block
    ret_valid[:, :, 0] = True

    # --- bars: vectorized scatter of every bar into [day, stock, second-offset from the open] ---
    ts = bars["timestamp_ms"].astype(np.int64)
    b_sym = np.array([stock_to_idx.get(s, -1) for s in bars["symbol"]], dtype=np.int64)
    b_day = np.array([day_idx[d] for d in bars["date_exchange"]], dtype=np.int64)
    b_soff = (ts - open_ms[b_day]) // 1000
    ok = (b_sym >= 1) & (b_soff >= 0) & (b_soff < S)
    ohlcv = np.stack([bars[f].astype(np.float32) for f in cfg.bar_fields], axis=1)    # [N,F]
    bars_t[b_day[ok], b_sym[ok], b_soff[ok]] = ohlcv[ok]
    bar_mask[b_day[ok], b_sym[ok], b_soff[ok]] = True

    # --- per-stock as-of covariates / raw news / T+1 labels at each (day, block) ---
    order = np.lexsort((ts, b_sym))
    sym_s, ts_s = b_sym[order], ts[order]
    close_s = ohlcv[order][:, 3].astype(np.float64)
    cs = cav = None
    if cov is not None and cov.get("symbol"):
        cs = np.array([stock_to_idx.get(s, -1) for s in cov["symbol"]])
        cav = np.array(cov["available_timestamp_ms"], dtype=np.int64)
    if news:
        nt = np.array([stock_to_idx.get(r.get("ticker"), -1) for r in news])
        nav = np.array([int(r.get("llm_feature_available_timestamp_ms", r.get("published_timestamp_ms", 0)))
                        for r in news], dtype=np.int64)
        nsent = np.array([float(r.get("sentiment_score", 0.0)) for r in news], dtype=np.float32)
    for ai in range(1, A):
        m = sym_s == ai
        a_ts, a_close = ts_s[m], close_s[m]
        cav_a = cidx_a = None
        if cs is not None:
            cm = np.nonzero(cs == ai)[0]
            if len(cm):
                co = np.argsort(cav[cm])
                cav_a, cidx_a = cav[cm][co], cm[co]
        nav_a = nse_a = None
        if news:
            nm = np.nonzero(nt == ai)[0]
            if len(nm):
                no = np.argsort(nav[nm])
                nav_a, nse_a = nav[nm][no], nsent[nm][no]
        for d in range(Dd):
            for b in range(nB):
                te = int(block_end[d, b])
                if cav_a is not None:                                # as-of covariates at block-b end
                    k = np.searchsorted(cav_a, te, "right") - 1
                    if k >= 0:
                        j = cidx_a[k]
                        covt[d, b, ai] = [float(cov[f][j]) if isinstance(cov[f][j], (int, float)) else 0.0
                                          for f in cfg.cov_fields]
                if nav_a is not None:                                # RAW news available by block-b end
                    k = np.searchsorted(nav_a, te, "right")
                    if k > 0:
                        take = nse_a[max(0, k - M):k]
                        kk = len(take)
                        news_raw[d, b, ai, :kk, 0] = take
                        news_mask[d, b, ai, :kk] = True
                if b <= nB - 3 and len(a_close):                     # T+1 label: execute b+1 end -> b+2 end
                    ei = np.searchsorted(a_ts, int(block_end[d, b + 1]) + lat, "left")
                    xi = np.searchsorted(a_ts, int(block_end[d, b + 2]) + lat, "left")
                    if ei < len(a_close) and xi < len(a_close) and a_close[ei] > 0:
                        r = a_close[xi] / a_close[ei] - 1.0
                        if np.isfinite(r):
                            ret[d, b, ai] = float(np.clip(r, -1.0, 1.0))
                            ret_valid[d, b, ai] = True

    return {
        "bars": torch.from_numpy(bars_t), "bar_mask": torch.from_numpy(bar_mask),
        "cov_blocks": torch.from_numpy(covt), "news_raw": torch.from_numpy(news_raw),
        "news_mask": torch.from_numpy(news_mask),
        "ret": torch.from_numpy(ret), "ret_valid": torch.from_numpy(ret_valid),
        "window": window, "n_days": Dd, "n_blocks": nB,
    }
