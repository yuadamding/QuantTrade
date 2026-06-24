"""Train-time organizer for the RAW time-partitioned dataset (e.g. TOP50 `top50_raw_time_partitioned_v1`).

Reads ONLY raw inputs from a dataset root: ``partitions/<S_to_E>/{bars.parquet, covariates.parquet, news.jsonl}``
+ ``universe.json``. NOTHING is precomputed/stored as features -- this module ORGANIZES the raw seconds into
the tensors the learning framework consumes, all at train time:

  * context chunk tokens: per stock per trading day, the raw 1s bars are turned into scale-free per-second
    features (close-to-close log-return, open gap, intrabar high/low, log-volume; reset across overnight gaps)
    and pooled into fixed ``chunk_sec`` bins. Each decision sees the LEFT-aligned tokens of that day's chunks
    fully closed before the decision -> context that rolls from the 09:30 session open.
  * as-of covariates: the latest point-in-time covariate record available at the decision.
  * news: count + mean sentiment of items available at the decision.
  * forward-return labels: close@decision+latency -> close@next-decision+latency (the reward signal).

Pooling raw seconds is organizing, not feature engineering -- nothing is persisted; the model still consumes
raw-derived inputs. The decision grid is the 5 hourly RTH decisions/day, DST-aware, built from the trading days.
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
CHUNK_FEATS = 6  # mean(close_ret, gap, hi, lo, logv) + log1p(seconds in the chunk)
COV_FIELDS = (
    "market_cap", "share_class_shares_outstanding", "financial_revenue", "financial_net_income",
    "financial_assets", "financial_liabilities", "financial_cash", "financial_operating_cashflow",
    "dividend_cash_amount", "split_ratio", "is_common_stock", "is_adr_or_foreign",
)
NEWS_FEATS = 2  # log1p(count), mean sentiment


@dataclass
class RawWindowConfig:
    chunk_sec: int = 300
    max_chunks: int = 80
    decision_et_hhmm: tuple[tuple[int, int], ...] = ((10, 30), (11, 30), (12, 30), (13, 30), (14, 30))
    open_et_hhmm: tuple[int, int] = (9, 30)
    exec_latency_ms: int = 1000
    gap_reset_ms: int = 5000
    cache_version: int = 1
    bar_fields: tuple[str, ...] = field(default=BAR_FIELDS)
    cov_fields: tuple[str, ...] = field(default=COV_FIELDS)


def load_universe(root: Path) -> tuple[list[str], int]:
    u = json.loads((Path(root) / "universe.json").read_text())
    return list(u["actions"]), int(u["cash_index"])  # actions[0] == CASH


def list_windows(root: Path) -> list[str]:
    parts = Path(root) / "partitions"
    return sorted(p.name for p in parts.iterdir() if (p / "bars.parquet").exists())


def _et_offset_hours(date_iso: str) -> int:
    # US Eastern DST: EDT (-4) ~ Mar..Nov, EST (-5) otherwise. Sufficient for RTH decision/open placement.
    return -4 if 3 <= int(date_iso[5:7]) <= 11 else -5


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


def _norm_seconds(a_ts: np.ndarray, ohlcv: np.ndarray, gap_reset_ms: int) -> np.ndarray:
    """Scale-free per-second features from RAW OHLCV; returns reset across gaps. -> [N,5]."""
    o, h, low, c, v = (ohlcv[:, i] for i in range(5))
    eps = 1e-6
    lc = np.log(np.maximum(c, eps))
    prev_lc = np.empty_like(lc)
    prev_lc[0] = lc[0]
    prev_lc[1:] = lc[:-1]
    close_ret = lc - prev_lc
    gap = np.log(np.maximum(o, eps)) - prev_lc
    hi = np.log(np.maximum(h, eps)) - lc
    lo = np.log(np.maximum(low, eps)) - lc
    logv = np.log1p(np.maximum(v, 0.0))
    dts = np.empty_like(a_ts)
    dts[0] = gap_reset_ms + 1
    dts[1:] = a_ts[1:] - a_ts[:-1]
    bad = dts > gap_reset_ms
    close_ret[bad] = 0.0
    gap[bad] = 0.0
    return np.nan_to_num(np.stack([close_ret, gap, hi, lo, logv], axis=1)).astype(np.float32)


def _chunk_tokens(a_ts: np.ndarray, feats: np.ndarray, chunk_ms: int):
    """Pool per-second features into absolute chunk bins. -> (tokens [nb,6], bin_start, bin_end)."""
    cid = a_ts // chunk_ms
    uniq, inv = np.unique(cid, return_inverse=True)
    nb = len(uniq)
    sums = np.zeros((nb, 5), np.float64)
    counts = np.zeros(nb, np.float64)
    np.add.at(sums, inv, feats.astype(np.float64))
    np.add.at(counts, inv, 1.0)
    means = sums / counts[:, None]
    tok = np.concatenate([means, np.log1p(counts)[:, None]], axis=1).astype(np.float32)
    bin_start = (uniq * chunk_ms).astype(np.int64)
    return tok, bin_start, bin_start + chunk_ms


def build_window(root: Path, window: str, stock_to_idx: dict[str, int], n_actions: int,
                 cfg: RawWindowConfig) -> dict | None:
    """Organize one raw time window into the framework's tensors (context tokens, covariates, news, labels)."""
    bars, cov, news = _load_window_raw(root, window, cfg)
    grid = _decision_grid(bars["date_exchange"], cfg)
    if len(grid) < 2:
        return None
    chunk_ms = cfg.chunk_sec * 1000
    ts = bars["timestamp_ms"].astype(np.int64)
    sym = np.array([stock_to_idx.get(s, -1) for s in bars["symbol"]], dtype=np.int64)
    order = np.lexsort((ts, sym))
    sym_s, ts_s = sym[order], ts[order]
    ohlcv_s = np.stack([bars[f].astype(np.float64)[order] for f in cfg.bar_fields], axis=1)
    close_s = ohlcv_s[:, 3]
    dms_arr = np.array([g[0] for g in grid], dtype=np.int64)
    open_arr = np.array([g[1] for g in grid], dtype=np.int64)

    D, A, C = len(grid) - 1, n_actions, cfg.max_chunks
    chunk = np.zeros((D, A, C, CHUNK_FEATS), dtype=np.float32)
    chunk_mask = np.zeros((D, A, C), dtype=bool)
    ret = np.full((D, A), np.nan, dtype=np.float32)
    ret_valid = np.zeros((D, A), dtype=bool)
    ret[:, 0] = 0.0          # CASH return is identically 0
    ret_valid[:, 0] = True
    for ai in range(1, A):
        m = sym_s == ai
        if not m.any():
            continue
        a_ts, a_ohlcv, a_close = ts_s[m], ohlcv_s[m], close_s[m]
        feats = _norm_seconds(a_ts, a_ohlcv, cfg.gap_reset_ms)
        tok, bstart, bend = _chunk_tokens(a_ts, feats, chunk_ms)
        for di in range(D):
            dms, day_open = dms_arr[di], open_arr[di]
            sel = np.nonzero((bstart >= day_open) & (bend <= dms))[0]  # this day's chunks closed before decision
            if len(sel):
                sel = sel[:C]                                          # left-aligned: keep earliest C tokens
                k = len(sel)
                chunk[di, ai, :k] = tok[sel]
                chunk_mask[di, ai, :k] = True
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

    nfeat = np.zeros((D, A, NEWS_FEATS), dtype=np.float32)
    if news:
        nt = np.array([stock_to_idx.get(r.get("ticker"), -1) for r in news])
        nav = np.array([int(r.get("llm_feature_available_timestamp_ms", r.get("published_timestamp_ms", 0)))
                        for r in news], dtype=np.int64)
        nsent = np.array([float(r.get("sentiment_score", 0.0)) for r in news], dtype=np.float64)
        for ai in range(1, A):
            m = nt == ai
            if not m.any():
                continue
            av, se = nav[m], nsent[m]
            for di in range(D):
                sel = av <= dms_arr[di]
                if sel.any():
                    nfeat[di, ai, 0] = np.log1p(int(sel.sum()))
                    nfeat[di, ai, 1] = float(se[sel].mean())

    return {
        "chunk": torch.from_numpy(chunk), "chunk_mask": torch.from_numpy(chunk_mask),
        "cov": torch.from_numpy(covt), "news": torch.from_numpy(nfeat),
        "ret": torch.from_numpy(ret), "ret_valid": torch.from_numpy(ret_valid),
        "window": window, "decisions": D,
    }
