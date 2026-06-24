"""Train-time organizer for the RAW time-partitioned dataset (e.g. TOP50 `top50_raw_time_partitioned_v1`).

Reads ONLY raw inputs from a dataset root: ``partitions/<S_to_E>/{bars.parquet, covariates.parquet, news.jsonl}``
+ ``universe.json``. NOTHING is precomputed/stored as features -- this module ORGANIZES the raw inputs into the
tensors the learning framework consumes, all at train time:

  * context bars: the RAW 1-second OHLCV bars directly (one token per second), LEFT-aligned -- the most recent
    ``max_seconds`` in ``[09:30 session open, decision)`` (no pooling, no hand-computed features; the encoder
    normalizes + compresses them itself). Rolls from the session open.
  * as-of covariates: the latest point-in-time covariate record available at the decision.
  * news: the RAW per-article sentiment scores available at the decision (the model aggregates them at train
    time -- no precomputed count/mean).
  * forward-return labels: close@decision+latency -> close@next-decision+latency (the reward signal).

The decision grid is the 5 hourly RTH decisions/day, DST-aware (zoneinfo), built from the trading days.
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
    max_seconds: int = 3600  # raw 1s bars per decision (most recent max_seconds in [session open, decision))
    max_news: int = MAX_NEWS
    decision_et_hhmm: tuple[tuple[int, int], ...] = ((10, 30), (11, 30), (12, 30), (13, 30), (14, 30))
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


def _decision_grid(date_exchange, cfg: RawWindowConfig):
    """Sorted [(decision_ms, session_open_ms)] for the window's trading days (DST-aware ET->UTC)."""
    pairs = []
    for d in sorted(set(date_exchange)):
        off = _et_offset_hours(d)
        y, m, day = map(int, d.split("-"))
        open_ms = int(dt.datetime(y, m, day, cfg.open_et_hhmm[0] - off, cfg.open_et_hhmm[1],
                                  tzinfo=dt.timezone.utc).timestamp() * 1000)
        for h, mm in cfg.decision_et_hhmm:
            t = int(dt.datetime(y, m, day, h - off, mm, tzinfo=dt.timezone.utc).timestamp() * 1000)
            pairs.append((t, open_ms))
    return sorted(pairs)


def build_window(root: Path, window: str, stock_to_idx: dict[str, int], n_actions: int,
                 cfg: RawWindowConfig) -> dict | None:
    """Organize one raw time window: RAW per-second bars + as-of covariates + news + forward-return labels."""
    bars, cov, news = _load_window_raw(root, window, cfg)
    grid = _decision_grid(bars["date_exchange"], cfg)
    if len(grid) < 2:
        return None
    ts = bars["timestamp_ms"].astype(np.int64)
    sym = np.array([stock_to_idx.get(s, -1) for s in bars["symbol"]], dtype=np.int64)
    order = np.lexsort((ts, sym))
    sym_s, ts_s = sym[order], ts[order]
    ohlcv_s = np.stack([bars[f].astype(np.float32)[order] for f in cfg.bar_fields], axis=1)  # RAW [N,F]
    close_s = ohlcv_s[:, 3].astype(np.float64)
    dms_arr = np.array([g[0] for g in grid], dtype=np.int64)
    open_arr = np.array([g[1] for g in grid], dtype=np.int64)

    D, A, S, F = len(grid) - 1, n_actions, cfg.max_seconds, len(cfg.bar_fields)
    bars_t = np.zeros((D, A, S, F), dtype=np.float32)        # RAW 1s bars, one token per second (left-aligned)
    bar_mask = np.zeros((D, A, S), dtype=bool)
    ret = np.full((D, A), np.nan, dtype=np.float32)
    ret_valid = np.zeros((D, A), dtype=bool)
    ret[:, 0] = 0.0          # CASH return is identically 0
    ret_valid[:, 0] = True
    for ai in range(1, A):
        m = sym_s == ai
        if not m.any():
            continue
        a_ts, a_ohlcv, a_close = ts_s[m], ohlcv_s[m], close_s[m]
        for di in range(D):
            dms, day_open = dms_arr[di], open_arr[di]
            lo = np.searchsorted(a_ts, day_open, "left")     # bars in [session open, decision)
            hi = np.searchsorted(a_ts, dms, "left")
            win = a_ohlcv[lo:hi][-S:]                         # most recent S raw bars, oldest-first (left-aligned)
            k = len(win)
            if k:
                bars_t[di, ai, :k] = win
                bar_mask[di, ai, :k] = True
            ei = np.searchsorted(a_ts, dms + cfg.exec_latency_ms, "left")
            xi = np.searchsorted(a_ts, dms_arr[di + 1] + cfg.exec_latency_ms, "left")
            if ei < len(a_close) and xi < len(a_close) and a_close[ei] > 0:
                r = a_close[xi] / a_close[ei] - 1.0
                if np.isfinite(r):
                    ret[di, ai] = float(np.clip(r, -1.0, 1.0))
                    ret_valid[di, ai] = True

    covt = np.zeros((D, A, len(cfg.cov_fields)), dtype=np.float32)
    if cov is not None and cov.get("symbol"):
        cs = np.array([stock_to_idx.get(s, -1) for s in cov["symbol"]])
        cav = np.array(cov["available_timestamp_ms"], dtype=np.int64)
        for ai in range(1, A):
            m = cs == ai
            if not m.any():
                continue
            idxs = np.nonzero(m)[0]
            o = np.argsort(cav[m])
            av = cav[m][o]
            idxs = idxs[o]
            for di in range(D):
                k = np.searchsorted(av, dms_arr[di], "right") - 1
                if k >= 0:
                    j = idxs[k]
                    covt[di, ai] = [float(cov[f][j]) if isinstance(cov[f][j], (int, float)) else 0.0
                                    for f in cfg.cov_fields]

    # RAW news: the per-article qwen3 sentiment scores available as-of each decision (NO precomputed count/mean;
    # the model aggregates them at train time). Most-recent cfg.max_news articles, chronological (oldest-first).
    M = cfg.max_news
    news_raw = np.zeros((D, A, M, NEWS_RAW_DIM), dtype=np.float32)
    news_mask = np.zeros((D, A, M), dtype=bool)
    if news:
        nt = np.array([stock_to_idx.get(r.get("ticker"), -1) for r in news])
        nav = np.array([int(r.get("llm_feature_available_timestamp_ms", r.get("published_timestamp_ms", 0)))
                        for r in news], dtype=np.int64)
        nsent = np.array([float(r.get("sentiment_score", 0.0)) for r in news], dtype=np.float32)
        for ai in range(1, A):
            m = nt == ai
            if not m.any():
                continue
            o = np.argsort(nav[m])                       # chronological
            av, se = nav[m][o], nsent[m][o]
            for di in range(D):
                k = np.searchsorted(av, dms_arr[di], "right")     # articles available before the decision
                if k > 0:
                    take = se[max(0, k - M):k]                    # most recent M raw scores (oldest-first)
                    kk = len(take)
                    news_raw[di, ai, :kk, 0] = take
                    news_mask[di, ai, :kk] = True

    return {
        "bars": torch.from_numpy(bars_t), "bar_mask": torch.from_numpy(bar_mask),
        "cov": torch.from_numpy(covt), "news_raw": torch.from_numpy(news_raw),
        "news_mask": torch.from_numpy(news_mask),
        "ret": torch.from_numpy(ret), "ret_valid": torch.from_numpy(ret_valid),
        "window": window, "decisions": D,
    }
