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
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch

from rl_quant.datasets.provenance import (
    declared_universe_actions,
    point_in_time_membership,
    source_symbol_to_action_index,
)

BAR_FIELDS = ("open", "high", "low", "close", "volume")
BAR_FEATS = len(BAR_FIELDS)  # the encoder consumes the RAW bar fields directly (one token per second)
COV_FIELDS = (
    "market_cap", "share_class_shares_outstanding", "financial_revenue", "financial_net_income",
    "financial_assets", "financial_liabilities", "financial_cash", "financial_operating_cashflow",
    "dividend_cash_amount", "split_ratio", "is_common_stock", "is_adr_or_foreign",
)
NEWS_RAW_DIM = 1  # raw fields kept per news article (the qwen3 sentiment_score) -- NO precomputed aggregate
MAX_NEWS = 32     # most-recent articles kept per (stock, decision); the model aggregates them at train time
NEWS_SENTIMENT_RANGE = (-1.0, 1.0)

# Audited XNYS 13:00 ET closes inside the checked-in TOP50/TOP2000 sample
# (2022-01-03 through 2026-06-23), plus the already-published remainder of
# 2026.  Keep this table explicit: an approximate weekday/holiday algorithm is
# not an acceptable source of execution timestamps.  A future data extension
# must append dates from the venue calendar before it can claim the same
# session-close contract.
XNYS_EARLY_CLOSE_DATES_2022_2026 = (
    "2022-11-25",
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-11-27",
    "2026-12-24",
)
XNYS_AUDITED_CALENDAR_START = "2022-01-01"
XNYS_AUDITED_CALENDAR_END = "2026-12-31"

# One canonical version for the raw-window payload and cache identity.  Bump
# whenever tensor semantics or the cache dependency contract changes.  Version
# 11 makes the session end calendar-aware, removes post-close bars/features,
# and requires a fresh completed regular-session aggregate. Older caches can
# therefore never retain a stale daily mark or half-day extended-hours data.
RAW_WINDOW_CACHE_VERSION = 11


@dataclass
class RawWindowConfig:
    session_seconds: int = 23400  # full RTH session (09:30->16:00) stored once per day (session-aligned)
    block_seconds: int = 300      # candidate/decision cadence = the encoder's tier-1 block (must match it)
    bar_seconds: int = 1          # bar GRID resolution: 1 = raw 1-second rows scattered as-is; >1 = the raw rows
    #                               are RESAMPLED to bar_seconds-OHLCV at load time (open=first, high=max, low=min,
    #                               close=last, volume=sum per slot). Same raw fields on a coarser grid -- input
    #                               organization, NOT a stored/engineered feature. T+1 LABELS, day open/close, and
    #                               all PIT joins keep using the raw 1-second rows' real timestamps (unchanged
    #                               accuracy); only the model's bar tokens coarsen. 60 makes TOP2000 ~40x cheaper
    #                               (the lever that fits training in a day). Must divide block_seconds.
    max_news: int = MAX_NEWS
    open_et_hhmm: tuple[int, int] = (9, 30)
    exec_latency_ms: int = 1000
    close_recency_seconds: int = 300  # EOD proxy must print in the final five minutes before the session boundary
    cache_version: int = RAW_WINDOW_CACHE_VERSION
    use_news: bool = False        # opt-in research input; current scored-news artifacts are not PIT reportable
    cov_carry_days: int = 400     # as-of covariates CARRY from prior windows up to this many calendar days back
    #                               (fundamentals publish ~quarterly into event-time partitions; without the carry
    #                               a window only sees records published INSIDE its ~3 days -> the model saw
    #                               market_cap on ~10% of stock-days and financials on ~2.5%. 0 disables carry.)
    bar_fields: tuple[str, ...] = field(default=BAR_FIELDS)
    cov_fields: tuple[str, ...] = field(default=COV_FIELDS)

    def __post_init__(self) -> None:
        if isinstance(self.cache_version, bool) or not isinstance(self.cache_version, int) or self.cache_version <= 0:
            raise ValueError("cache_version must be a positive integer")
        if (
            isinstance(self.close_recency_seconds, bool)
            or not isinstance(self.close_recency_seconds, int)
            or self.close_recency_seconds <= 0
        ):
            raise ValueError("close_recency_seconds must be a positive integer")


def raw_window_dependency_paths(root: str | Path, window: str, cfg: RawWindowConfig) -> tuple[Path, ...]:
    """Return every upstream file whose state can affect ``build_window``.

    Missing paths are deliberately retained: creating a previously absent
    carried covariate, membership, or news file must change the signature just
    as modifying an existing file does.  Prior-window *bars contents* are not a
    dependency; their filenames/existence only select covariate partitions, and
    that selection is represented by the returned covariate paths themselves.
    """

    root = Path(root)
    base = root / "partitions" / window
    paths: list[Path] = [
        root / "universe.json",
        root / "universe_membership.parquet",
        base / "bars.parquet",
    ]
    paths.extend(
        root / "partitions" / source_window / "covariates.parquet"
        for source_window in _cov_source_windows(root, window, cfg)
    )
    if cfg.use_news:
        paths.append(base / "news.jsonl")
    # Preserve dependency order while removing the current covariate path if a
    # defensive caller supplied a duplicate source window.
    return tuple(dict.fromkeys(paths))


def _metadata_signature_record(root: Path, path: Path) -> tuple[object, ...]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = str(path.resolve())
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (relative, "missing")
    # ctime + inode catch atomic replacements and same-size rewrites whose mtime
    # is restored. No file content is read on cache lookup.
    return (
        relative,
        "present",
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_mode),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def raw_window_source_signature(root: str | Path, window: str, cfg: RawWindowConfig) -> str:
    """Metadata digest of all active file dependencies for one raw window.

    The digest covers current bars, every covariate partition eligible for
    as-of carry (including missing sentinels), dynamic universe membership, the
    universe declaration, and news only when enabled.  It uses filesystem
    metadata rather than reading/hashing multi-GB parquet contents on each load.
    """

    root = Path(root)
    records: list[tuple[object, ...]] = [("signature_schema", 2)]
    records.extend(
        _metadata_signature_record(root, path)
        for path in raw_window_dependency_paths(root, window, cfg)
    )
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_config_signature(cfg: RawWindowConfig) -> str:
    payload = {
        "session_seconds": cfg.session_seconds,
        "block_seconds": cfg.block_seconds,
        "bar_seconds": cfg.bar_seconds,
        "max_news": cfg.max_news,
        "open_et_hhmm": cfg.open_et_hhmm,
        "exec_latency_ms": cfg.exec_latency_ms,
        "close_recency_seconds": cfg.close_recency_seconds,
        # Treat the audited venue schedule as part of tensor semantics, so
        # extending/correcting the table invalidates affected cache identities
        # even if a maintainer forgets to bump the coarse cache version.
        "xnys_early_close_dates": XNYS_EARLY_CLOSE_DATES_2022_2026,
        "xnys_audited_calendar_bounds": (XNYS_AUDITED_CALENDAR_START, XNYS_AUDITED_CALENDAR_END),
        "use_news": cfg.use_news,
        "cov_carry_days": cfg.cov_carry_days,
        "bar_fields": cfg.bar_fields,
        "cov_fields": cfg.cov_fields,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def raw_window_cache_key(
    root: str | Path,
    window: str,
    cfg: RawWindowConfig,
    *,
    universe_signature: str = "",
) -> str:
    """Canonical cache filename for one raw-window build.

    ``universe_signature`` identifies the caller's ordered action mapping; the
    source signature independently covers the on-disk universe declaration and
    point-in-time membership events.
    """

    universe_digest = hashlib.sha256(universe_signature.encode()).hexdigest()[:12]
    source_digest = raw_window_source_signature(root, window, cfg)
    return (
        f"{window}__S{cfg.session_seconds}b{cfg.block_seconds}g{cfg.bar_seconds}"
        f"_v{cfg.cache_version}_{_cache_config_signature(cfg)}_{universe_digest}_{source_digest}.pt"
    )


def load_universe(root: Path) -> tuple[list[str], int]:
    return declared_universe_actions(root), 0


def _minimum_parquet_int(path: Path, column: str) -> int:
    parquet = pq.ParquetFile(path)
    try:
        column_index = parquet.schema_arrow.names.index(column)
    except ValueError as exc:
        raise ValueError(f"{path} is missing required column {column!r}") from exc
    minima: list[int] = []
    for row_group in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(row_group).column(column_index).statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(int(statistics.min))
    if minima:
        return min(minima)
    values = pq.read_table(path, columns=[column]).column(column).to_numpy()
    if values.size == 0:
        raise ValueError(f"{path} contains no rows")
    return int(np.min(values))


def _utc_timestamp_ms(value: str) -> int:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def _validated_news_sentiment(article: object, *, location: str) -> float:
    """Return the sole model-facing news value, rejecting missing/corrupt rows.

    A missing score is not neutral sentiment: silently substituting zero would
    make malformed inputs observationally indistinguishable from an extractor
    that actually emitted a neutral score.
    """

    if not isinstance(article, dict):
        raise ValueError(f"{location} must contain a JSON object")
    ticker = article.get("ticker")
    if not isinstance(ticker, str) or not ticker or ticker != ticker.strip():
        raise ValueError(f"{location} lacks a canonical non-empty ticker")
    value = article.get("sentiment_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} lacks a numeric sentiment_score")
    score = float(value)
    lower, upper = NEWS_SENTIMENT_RANGE
    if not np.isfinite(score) or not lower <= score <= upper:
        raise ValueError(
            f"{location} sentiment_score must be finite and in [{lower:g}, {upper:g}]"
        )
    return score


def _validated_news_timestamp(article: object, name: str, *, location: str) -> int:
    if not isinstance(article, dict):
        raise ValueError(f"{location} must contain a JSON object")
    value = article.get(name)
    if value is None:
        raise ValueError(f"{location} lacks required integer {name}")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"{location} has invalid integer {name}")
    return value


def news_is_reportable(
    root: Path,
    windows: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Fail-closed chronology/provenance audit for every active news article.

    A reportable extractor must already exist by the first market decision and
    by article publication, while its scored feature can become available only
    after both. Required deterministic-extraction metadata must be explicit;
    missing news files are not silently treated as known-zero coverage.
    """

    root = Path(root)
    active_windows = list(list_windows(root) if windows is None else windows)
    if not active_windows:
        return False, "no active bars windows are available for the news chronology audit"
    try:
        known_tickers = set(source_symbol_to_action_index(root))
    except (OSError, ValueError) as exc:
        return False, f"cannot establish declared ticker universe for news audit: {exc}"
    try:
        first_decision_ms = min(
            _minimum_parquet_int(root / "partitions" / window / "bars.parquet", "timestamp_ms")
            for window in active_windows
        )
    except (OSError, ValueError) as exc:
        return False, f"cannot establish first market decision for news audit: {exc}"

    modern_model_floor_ms = 946_684_800_000  # 2000-01-01; rejects epoch/sentinel availability values.
    total = 0
    for window in active_windows:
        news_path = root / "partitions" / window / "news.jsonl"
        if not news_path.exists():
            return False, f"active window {window} is missing news.jsonl (coverage is unknown, not zero)"
        with news_path.open() as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, f"{window}/news.jsonl:{line_number} is invalid JSON: {exc}"
                location = f"{window}/news.jsonl:{line_number}"
                try:
                    _validated_news_sentiment(article, location=location)
                except ValueError as exc:
                    return False, str(exc)
                if article["ticker"] not in known_tickers:
                    return False, f"{location} ticker {article['ticker']!r} is not a declared non-CASH action"
                required_timestamps = (
                    "published_timestamp_ms",
                    "llm_feature_available_timestamp_ms",
                    "model_available_timestamp_ms",
                )
                try:
                    published, feature_available, model_available = (
                        _validated_news_timestamp(article, name, location=location)
                        for name in required_timestamps
                    )
                except ValueError as exc:
                    return False, str(exc)
                if model_available < modern_model_floor_ms:
                    return False, (
                        f"{window}/news.jsonl:{line_number} has an implausible/sentinel model availability "
                        f"timestamp {model_available}"
                    )
                if model_available > first_decision_ms or model_available > published:
                    return False, (
                        f"{window}/news.jsonl:{line_number} uses a model unavailable by the first decision/article"
                    )
                if feature_available < max(model_available, published):
                    return False, (
                        f"{window}/news.jsonl:{line_number} exposes the feature before its model/article exists"
                    )
                temperature = article.get("extractor_temperature")
                if (
                    isinstance(temperature, bool)
                    or not isinstance(temperature, (int, float))
                    or not np.isfinite(float(temperature))
                    or float(temperature) != 0.0
                ):
                    return False, f"{window}/news.jsonl:{line_number} lacks deterministic temperature=0"
                if article.get("extractor_no_retrieval") is not True:
                    return False, f"{window}/news.jsonl:{line_number} does not certify no external retrieval"
                for name in ("llm_model_id", "llm_prompt_hash", "llm_schema_hash"):
                    value = article.get(name)
                    if not isinstance(value, str) or not value or value != value.strip():
                        return False, f"{window}/news.jsonl:{line_number} lacks {name}"
                provider_value = article.get("extractor_provider")
                if (
                    not isinstance(provider_value, str)
                    or not provider_value
                    or provider_value != provider_value.strip()
                ):
                    return False, f"{window}/news.jsonl:{line_number} lacks extractor_provider"
                provider = provider_value
                cutoff_value = article.get("model_training_cutoff_utc")
                if cutoff_value is not None and not isinstance(cutoff_value, str):
                    return False, f"{window}/news.jsonl:{line_number} has a non-string model training cutoff"
                cutoff = "" if cutoff_value is None else cutoff_value.strip()
                deterministic_baseline = provider.startswith("deterministic")
                if not deterministic_baseline and (not cutoff or cutoff.lower().startswith("unknown")):
                    return False, f"{window}/news.jsonl:{line_number} lacks a known model training cutoff"
                if not deterministic_baseline:
                    try:
                        cutoff_ms = _utc_timestamp_ms(cutoff)
                    except (OSError, OverflowError, ValueError):
                        return False, f"{window}/news.jsonl:{line_number} has an invalid model training cutoff"
                    if cutoff_ms > model_available:
                        return False, (
                            f"{window}/news.jsonl:{line_number} has a training cutoff after model availability"
                        )
                total += 1
    return True, f"ok ({total} articles across {len(active_windows)} active windows passed chronology/provenance)"


def list_windows(root: Path) -> list[str]:
    parts = Path(root) / "partitions"
    return sorted(p.name for p in parts.iterdir() if (p / "bars.parquet").exists())


_ET: dt.tzinfo | None
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


def _window_start_date(window: str) -> str:
    return window.split("_to_")[0]


def _cov_source_windows(root: Path, window: str, cfg: RawWindowConfig) -> list[str]:
    """Windows whose covariate records can be as-of visible in `window`: the window itself plus PRIOR windows back
    to `cov_carry_days` calendar days before its start. Fundamentals are event-sparse (quarterly filings, monthly
    snapshots) and partitions are event-time, so the LAST-KNOWN record usually lives in an EARLIER partition;
    point-in-time safety is unchanged because visibility is still gated per record on available_timestamp_ms."""
    wins = list_windows(root)
    if window not in wins or cfg.cov_carry_days <= 0:
        return [window]
    y, m, d = (int(x) for x in _window_start_date(window).split("-"))
    horizon = (dt.date(y, m, d) - dt.timedelta(days=cfg.cov_carry_days)).isoformat()
    i = wins.index(window)
    return [w for w in wins[:i] if _window_start_date(w) >= horizon] + [window]


def _load_window_raw(root: Path, window: str, cfg: RawWindowConfig):
    base = Path(root) / "partitions" / window
    bt = pq.read_table(base / "bars.parquet",
                       columns=["symbol", "timestamp_ms", "date_exchange", *cfg.bar_fields])
    bars = {c: (bt.column(c).to_numpy() if c not in ("symbol", "date_exchange") else bt.column(c).to_pylist())
            for c in bt.column_names}
    # covariates: concat this window's records with prior windows' (the as-of carry); availability stays
    # per-record point-in-time (available_timestamp_ms), so older records add reach, never look-ahead.
    cov_cols = ("symbol", "available_timestamp_ms", *cfg.cov_fields)
    cov_parts = []
    for w in _cov_source_windows(root, window, cfg):
        f = Path(root) / "partitions" / w / "covariates.parquet"
        if f.exists():
            ct = pq.read_table(f, columns=[c for c in cov_cols if c in pq.read_schema(f).names])
            cov_parts.append({c: ct.column(c).to_pylist() for c in ct.column_names})
    cov = None
    if cov_parts:
        cov = {c: [v for p in cov_parts for v in p.get(c, [None] * len(p["symbol"]))] for c in cov_cols
               if any(c in p for p in cov_parts)}
    # News is disabled by default for reportable runs.  Do not parse a potentially
    # multi-GB side input only to discard it in ``build_window``; this also keeps a
    # no-news cache build independent of the availability of the research-only
    # news file.
    news = []
    if cfg.use_news and (base / "news.jsonl").exists():
        with (base / "news.jsonl").open() as source:
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    article = json.loads(line)
                    _validated_news_sentiment(
                        article,
                        location=f"{window}/news.jsonl:{line_number}",
                    )
                    _validated_news_timestamp(
                        article,
                        "llm_feature_available_timestamp_ms",
                        location=f"{window}/news.jsonl:{line_number}",
                    )
                    news.append(article)
    return bars, cov, news


def _open_ms(date_iso: str, cfg: RawWindowConfig) -> int:
    """UTC ms of the 09:30 ET session open on `date_iso` (DST-aware)."""
    off = _et_offset_hours(date_iso)
    y, m, day = map(int, date_iso.split("-"))
    return int(dt.datetime(y, m, day, cfg.open_et_hhmm[0] - off, cfg.open_et_hhmm[1],
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def _session_end_ms(date_iso: str, cfg: RawWindowConfig) -> int:
    """Official session end for the audited XNYS sample, in UTC milliseconds.

    ``session_seconds`` remains the tensor/grid capacity.  On an audited
    13:00-ET half day the effective session is shorter; ``min`` preserves
    intentionally short synthetic/custom grids without extending them.
    """

    open_ms = _open_ms(date_iso, cfg)
    if not XNYS_AUDITED_CALENDAR_START <= date_iso <= XNYS_AUDITED_CALENDAR_END:
        raise ValueError(
            "audited XNYS calendar covers only "
            f"{XNYS_AUDITED_CALENDAR_START} through {XNYS_AUDITED_CALENDAR_END}; got {date_iso}. "
            "Extend the versioned venue schedule before building this date."
        )
    nominal_end = open_ms + cfg.session_seconds * 1000
    if date_iso not in XNYS_EARLY_CLOSE_DATES_2022_2026:
        return nominal_end
    off = _et_offset_hours(date_iso)
    year, month, day = map(int, date_iso.split("-"))
    early_end = int(
        dt.datetime(year, month, day, 13 - off, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    return min(nominal_end, early_end)


def build_window(root: Path, window: str, stock_to_idx: dict[str, int], n_actions: int,
                 cfg: RawWindowConfig) -> dict | None:
    """BLOCK-ALIGNED organizer. One full RTH session per trading day is stored session-aligned (index s = second
    s after the 09:30 open); the encoder turns it into a context at every `block_seconds` block, and the policy
    chooses WHEN to act over those blocks. Per (day, block) we also store as-of covariates, raw news, and the
    T+1 forward-return label: decide at block b (context <= block-b end), EXECUTE at block b+1's end, hold to
    block b+2's end. Returns per-window tensors with a leading n_days axis."""
    bars, cov, news = _load_window_raw(root, window, cfg)
    if not cfg.use_news:
        news = []                  # reportable ablation: leave news_raw zero / news_mask False (the model sees none)
    days = sorted(set(bars["date_exchange"]))
    if not days:
        return None
    gs = max(1, int(cfg.bar_seconds))                        # bar-grid resolution (1 = raw seconds)
    if cfg.block_seconds % gs:
        raise ValueError(f"bar_seconds {gs} must divide block_seconds {cfg.block_seconds}")
    S = cfg.session_seconds // gs                            # bar SLOTS per session (tokens the encoder sees)
    bl = cfg.block_seconds // gs                             # bar slots per tier-1 block
    nB = cfg.session_seconds // cfg.block_seconds
    Dd, A, F, M, NC = len(days), n_actions, len(cfg.bar_fields), cfg.max_news, len(cfg.cov_fields)
    day_idx = {d: i for i, d in enumerate(days)}
    open_ms = np.array([_open_ms(d, cfg) for d in days], dtype=np.int64)              # [Dd]
    session_end = np.array([_session_end_ms(d, cfg) for d in days], dtype=np.int64)   # [Dd], early-close aware
    block_end = open_ms[:, None] + (np.arange(1, nB + 1) * cfg.block_seconds * 1000)  # [Dd,nB] block-b end (ms)
    # Cap every PIT join at the official session boundary. The fixed-size tensor remains
    # 78 blocks on a half day, but its post-close suffix can neither reveal
    # news/covariates/membership events nor introduce extended-hours bars.
    effective_block_end = np.minimum(block_end, session_end[:, None])
    session_close_block = ((session_end - open_ms - 1) // (cfg.block_seconds * 1000)).astype(np.int64)
    lat = cfg.exec_latency_ms

    bars_t = np.zeros((Dd, A, S, F), dtype=np.float32)
    bar_mask = np.zeros((Dd, A, S), dtype=bool)
    covt = np.zeros((Dd, nB, A, NC), dtype=np.float32)
    cov_valid = np.zeros((Dd, nB, A, NC), dtype=bool)
    # The default reportable path disables news.  Keep those tensors logical
    # but do not allocate/write ~71 MiB of zeros per TOP2000 window: an
    # expanded scalar has identical values/shapes and torch.save persists only
    # its one-element storage.  The opt-in news path remains dense and mutable.
    news_raw = np.zeros((Dd, nB, A, M, NEWS_RAW_DIM), dtype=np.float32) if cfg.use_news else None
    news_mask = np.zeros((Dd, nB, A, M), dtype=bool) if cfg.use_news else None
    ret = np.full((Dd, nB, A), np.nan, dtype=np.float32)
    ret_valid = np.zeros((Dd, nB, A), dtype=bool)
    ret[:, :, 0] = 0.0          # CASH return is identically 0 at every block
    ret_valid[:, :, 0] = True

    # --- bars: vectorized scatter of every raw row into [day, stock, grid-slot offset from the open]. At gs=1
    # the slot IS the second (at most one raw row per slot -> plain scatter). At gs>1 the rows in a slot are
    # RESAMPLED to one OHLCV bar: open = first row's open, high = max, low = min, close = last row's close,
    # volume = sum -- the standard aggregation of the same raw fields, computed here at load time. ---
    ts = bars["timestamp_ms"].astype(np.int64)
    b_sym = np.array([stock_to_idx.get(s, -1) for s in bars["symbol"]], dtype=np.int64)
    b_day = np.array([day_idx[d] for d in bars["date_exchange"]], dtype=np.int64)
    b_soff = (ts - open_ms[b_day]) // (1000 * gs)
    ok = (b_sym >= 1) & (b_soff >= 0) & (b_soff < S) & (ts < session_end[b_day])
    ohlcv = np.stack([bars[f].astype(np.float32) for f in cfg.bar_fields], axis=1)    # [N,F]
    if gs == 1:
        bars_t[b_day[ok], b_sym[ok], b_soff[ok]] = ohlcv[ok]
        bar_mask[b_day[ok], b_sym[ok], b_soff[ok]] = True
    else:
        oi = np.nonzero(ok)[0]
        slot_key = (b_day[oi] * A + b_sym[oi]) * S + b_soff[oi]                      # unique (day,stock,slot) id
        so = oi[np.lexsort((ts[oi], slot_key))]                                      # slot-major, time-ascending
        k_sorted = (b_day[so] * A + b_sym[so]) * S + b_soff[so]
        uniq, first_pos, counts = np.unique(k_sorted, return_index=True, return_counts=True)
        first_row = so[first_pos]                                                    # earliest raw row per slot
        last_row = so[first_pos + counts - 1]                                        # latest raw row per slot
        d_u, s_u, o_u = b_day[first_row], b_sym[first_row], b_soff[first_row]
        fields = {f: i for i, f in enumerate(cfg.bar_fields)}
        bars_t[d_u, s_u, o_u, fields["open"]] = ohlcv[first_row, fields["open"]]
        bars_t[d_u, s_u, o_u, fields["close"]] = ohlcv[last_row, fields["close"]]
        # high/low/volume reduce over ALL rows in the slot (order-independent)
        hi = np.full(len(uniq), -np.inf, dtype=np.float64)
        lo_ = np.full(len(uniq), np.inf, dtype=np.float64)
        vol = np.zeros(len(uniq), dtype=np.float64)
        pos = np.searchsorted(uniq, k_sorted)                                        # row -> slot group
        np.maximum.at(hi, pos, ohlcv[so, fields["high"]].astype(np.float64))
        np.minimum.at(lo_, pos, ohlcv[so, fields["low"]].astype(np.float64))
        np.add.at(vol, pos, ohlcv[so, fields["volume"]].astype(np.float64))
        bars_t[d_u, s_u, o_u, fields["high"]] = hi.astype(np.float32)
        bars_t[d_u, s_u, o_u, fields["low"]] = lo_.astype(np.float32)
        bars_t[d_u, s_u, o_u, fields["volume"]] = vol.astype(np.float32)
        bar_mask[d_u, s_u, o_u] = True

    # --- per-stock as-of covariates / raw news / T+1 labels at each (day, block) ---
    order = np.lexsort((ts, b_sym))
    sym_s, ts_s = b_sym[order], ts[order]
    close_s = ohlcv[order][:, 3].astype(np.float64)
    cs = cav = cvals = None
    if cov is not None and cov.get("symbol"):
        cs = np.array([stock_to_idx.get(s, -1) for s in cov["symbol"]])
        cav = np.array([int(v) if v is not None else -1 for v in cov["available_timestamp_ms"]], dtype=np.int64)
        cs = np.where(cav >= 0, cs, -1)              # a record without an availability timestamp is never PIT-usable
        # raw per-record field values, NaN where null: records are event-sparse and PARTIAL (a monthly market-cap
        # snapshot has null financials), so the as-of state must FORWARD-FILL per FIELD -- taking the latest row
        # wholesale would erase previously published fields with 0s.
        cvals = np.stack([np.array([float(v) if isinstance(v, (int, float)) else np.nan for v in cov[f]],
                                   dtype=np.float64) for f in cfg.cov_fields], axis=1)   # [n_rows, NC]
        co = np.lexsort((cav, cs))
        cs, cav, cvals = cs[co], cav[co], cvals[co]
    if news:
        unknown_news_tickers = sorted({
            r["ticker"] for r in news
            if r["ticker"] not in stock_to_idx or stock_to_idx[r["ticker"]] <= 0
        })
        if unknown_news_tickers:
            raise ValueError(
                f"{window} news contains tickers outside the non-CASH action universe: "
                + ", ".join(unknown_news_tickers)
            )
        nt = np.array([stock_to_idx.get(r.get("ticker"), -1) for r in news])
        nav = np.array([
            _validated_news_timestamp(
                r,
                "llm_feature_available_timestamp_ms",
                location=f"{window}/news article {index}",
            )
            for index, r in enumerate(news, start=1)
        ], dtype=np.int64)
        nsent = np.array([
            _validated_news_sentiment(r, location=f"{window}/news article {index}")
            for index, r in enumerate(news, start=1)
        ], dtype=np.float32)
        no = np.lexsort((nav, nt))
        nt, nav, nsent = nt[no], nav[no], nsent[no]
    flat_block_end = effective_block_end.reshape(-1)
    for ai in range(1, A):
        # ``sym_s`` is symbol-major.  Binary-searching its contiguous slice
        # avoids scanning every raw row once per action (the old O(A * rows)
        # path is prohibitive for ~2,000 names).
        blo = np.searchsorted(sym_s, ai, side="left")
        bhi = np.searchsorted(sym_s, ai, side="right")
        a_ts, a_close = ts_s[blo:bhi], close_s[blo:bhi]
        cav_a = cfill_a = cvalid_a = None
        if cs is not None and cav is not None and cvals is not None:
            clo = np.searchsorted(cs, ai, side="left")
            chi = np.searchsorted(cs, ai, side="right")
            if chi > clo:
                cav_a = cav[clo:chi]
                vals_a = cvals[clo:chi]                              # [n_a, NC] in availability order, NaN=null
                fi = np.where(~np.isnan(vals_a), np.arange(len(cav_a))[:, None], 0)
                np.maximum.accumulate(fi, axis=0, out=fi)            # per-FIELD index of the last non-null so far
                cfill_a = np.nan_to_num(np.take_along_axis(vals_a, fi, axis=0)).astype(np.float32)
                cvalid_a = np.maximum.accumulate(~np.isnan(vals_a), axis=0)
        nav_a = nse_a = None
        if news:
            nlo = np.searchsorted(nt, ai, side="left")
            nhi = np.searchsorted(nt, ai, side="right")
            if nhi > nlo:
                nav_a, nse_a = nav[nlo:nhi], nsent[nlo:nhi]

        if cav_a is not None and cfill_a is not None and cvalid_a is not None:  # vectorized block-grid as-of join
            ck = np.searchsorted(cav_a, flat_block_end, side="right") - 1
            good = ck >= 0
            cov_flat = covt[:, :, ai].reshape(-1, NC)
            valid_flat = cov_valid[:, :, ai].reshape(-1, NC)
            cov_flat[good] = cfill_a[ck[good]]
            valid_flat[good] = cvalid_a[ck[good]]

        # T+1 labels are independent across days and blocks; vectorize the
        # timestamp searches and validity checks rather than repeating Python
        # search calls for every block.
        if len(a_close) and nB >= 3:
            entry_target = block_end[:, 1:nB - 1] + lat
            exit_target = block_end[:, 2:nB] + lat
            ei = np.searchsorted(a_ts, entry_target)
            xi = np.searchsorted(a_ts, exit_target)
            in_bounds = (ei < len(a_ts)) & (xi < len(a_ts))
            ei_safe = np.minimum(ei, len(a_ts) - 1)
            xi_safe = np.minimum(xi, len(a_ts) - 1)
            entry_px, exit_px = a_close[ei_safe], a_close[xi_safe]
            good = (in_bounds & (entry_px > 0)
                    & (a_ts[ei_safe] < session_end[:, None]) & (a_ts[xi_safe] < session_end[:, None])
                    & (a_ts[ei_safe] - entry_target <= cfg.block_seconds * 1000)
                    & (a_ts[xi_safe] - exit_target <= cfg.block_seconds * 1000))
            ratio = np.divide(exit_px, entry_px, out=np.zeros_like(exit_px), where=entry_px > 0) - 1.0
            good &= np.isfinite(ratio)
            ret[:, :nB - 2, ai] = np.where(good, np.clip(ratio, -1.0, 1.0), np.nan).astype(np.float32)
            ret_valid[:, :nB - 2, ai] = good

        for d in range(Dd):
            for b in range(nB):
                te = int(effective_block_end[d, b])
                if nav_a is not None and nse_a is not None:          # RAW news available by block-b end
                    k = int(np.searchsorted(nav_a, te, "right"))
                    if k > 0:
                        take = nse_a[max(0, k - M):k]
                        kk = len(take)
                        if news_raw is None or news_mask is None:  # guarded by ``if news``; defensive invariant
                            raise RuntimeError("news records require dense news tensors")
                        news_raw[d, b, ai, :kk, 0] = take
                        news_mask[d, b, ai, :kk] = True

    block_number = np.arange(nB, dtype=np.int64)[None, :]
    close_block = block_number == session_close_block[:, None]
    post_close = block_number > session_close_block[:, None]
    # The fixed 78-block shape is retained for batching, but a half-day's
    # padded suffix is not a sequence of repeated 13:00 observations.  Keeping
    # those repeats would overweight half-day covariates in normalization and
    # expose synthetic policy steps. The declared session-close block itself remains.
    covt[post_close] = 0.0
    cov_valid[post_close] = False
    ret_valid[post_close] = False
    if news_raw is not None and news_mask is not None:
        news_raw[post_close] = 0.0
        news_mask[post_close] = False

    # per-stock day-OPEN price (first valid bar's open) -- the cross-day (daily, open-to-open) execution price
    fv = bar_mask.argmax(axis=2)                                  # [Dd,A] first valid second (0 if none)
    has = bar_mask.any(axis=2)                                    # [Dd,A]
    opens = np.take_along_axis(bars_t[:, :, :, 0], fv[:, :, None], axis=2)[:, :, 0]
    day_open = np.where(has, opens, np.nan).astype(np.float32)    # [Dd,A] (NaN where the stock has no bars that day)
    # per-stock day-CLOSE price (LAST valid bar's close) -- the close-to-close (daily_raw) execution price.
    lv = (bar_mask.shape[2] - 1) - np.argmax(bar_mask[:, :, ::-1], axis=2)   # [Dd,A] last valid second (S-1 if none)
    closes = np.take_along_axis(bars_t[:, :, :, 3], lv[:, :, None], axis=2)[:, :, 0]   # field 3 = close
    # A daily close proxy is usable only when its final COMPLETED regular-session
    # aggregate is fresh relative to the official session boundary. Polygon
    # aggregate timestamps denote window starts, so the source's RTH contract is
    # half-open and a row exactly at 16:00/13:00 remains excluded. This proxy is
    # not a condition-qualified official auction close. Merely trading once in
    # the morning is not enough, and a half-day post-boundary row cannot rescue
    # a stale mark because those rows were excluded by ``ok``.
    last_trade_ms = np.full((Dd, A), -1, dtype=np.int64)
    np.maximum.at(last_trade_ms, (b_day[ok], b_sym[ok]), ts[ok])
    close_valid = (
        has
        & (last_trade_ms >= session_end[:, None] - cfg.close_recency_seconds * 1000)
        & np.isfinite(closes)
        & (closes > 0)
    )
    day_close = np.where(close_valid, closes, np.nan).astype(np.float32)

    # As-of AVAILABILITY: a stock must both have traded by the block end and belong to the universe according to
    # an event that was available by that time.  Datasets without a membership event table retain the historical
    # static-list behavior, but the provenance validator marks them development-only.
    block_present = bar_mask[:, :, :nB * bl].reshape(Dd, A, nB, bl).any(axis=3)   # [Dd,A,nB]
    avail = np.ascontiguousarray(np.maximum.accumulate(block_present, axis=2).transpose(0, 2, 1))  # [Dd,nB,A]
    if (root / "universe.json").exists():
        actions = declared_universe_actions(root)
        expected_mapping = source_symbol_to_action_index(root)
        if len(actions) != A or stock_to_idx != expected_mapping:
            raise ValueError("raw source-symbol mapping does not match the ordered universe declaration")
    else:
        actions = [""] * A
        for symbol, idx in stock_to_idx.items():
            if 0 <= idx < A:
                actions[idx] = symbol
        if actions:
            actions[0] = "CASH"
    member = point_in_time_membership(
        root,
        [date for date in days for _ in range(nB)],
        actions,
        effective_block_end.reshape(-1),
    ).reshape(Dd, nB, A)
    avail &= member
    # At the close (and in the fixed-grid suffix after an early close), policy
    # tradeability additionally requires a fresh pre-boundary close proxy. This
    # keeps ``avail`` aligned with daily label validity instead of advertising
    # an asset whose last price may be hours old.
    close_eligible = close_valid[:, None, :] & member
    avail = np.where(
        close_block[:, :, None],
        close_eligible,
        avail,
    )
    avail = np.where(post_close[:, :, None], False, avail)
    avail[:, :, 0] = True                                                         # CASH always available

    if news_raw is None or news_mask is None:
        news_raw_t = torch.zeros((), dtype=torch.float32).expand(Dd, nB, A, M, NEWS_RAW_DIM)
        news_mask_t = torch.zeros((), dtype=torch.bool).expand(Dd, nB, A, M)
    else:
        news_raw_t = torch.from_numpy(news_raw)
        news_mask_t = torch.from_numpy(news_mask)

    return {
        "bars": torch.from_numpy(bars_t), "bar_mask": torch.from_numpy(bar_mask),
        "cov_blocks": torch.from_numpy(covt), "cov_valid_blocks": torch.from_numpy(cov_valid),
        "news_raw": news_raw_t, "news_mask": news_mask_t,
        "avail": torch.from_numpy(avail),
        "universe_member": torch.from_numpy(member),
        "ret": torch.from_numpy(ret), "ret_valid": torch.from_numpy(ret_valid),
        "day_open": torch.from_numpy(day_open), "day_close": torch.from_numpy(day_close),
        "day_close_valid": torch.from_numpy(close_valid),
        "session_close_block": torch.from_numpy(session_close_block), "dates": days,
        "window": window, "n_days": Dd, "n_blocks": nB,
    }
