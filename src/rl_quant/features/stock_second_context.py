from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import torch

from rl_quant.data_sources.polygon_second_aggs import (
    DEFAULT_BAR_LATENCY_MS,
    available_timestamp_ms,
    iso_to_timestamp_ms,
    timestamp_ms_to_iso,
)
from rl_quant.research_protocol import stable_json_hash, utc_now_iso


EASTERN = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)
SUPPORTED_DECISION_INTERVALS = {"5m", "15m", "30m", "60m"}

MARKET_CONTEXT_FEATURE_NAMES = [
    "active_symbol_count",
    "active_fraction",
    "equal_weight_return",
    "dollar_volume_weighted_return",
    "median_return",
    "return_std",
    "up_fraction",
    "down_fraction",
    "top_decile_return",
    "bottom_decile_return",
    "top_minus_bottom_return",
    "abs_return_dollar_volume_weighted",
    "dollar_volume_concentration",
    "transaction_concentration",
    "log_total_dollar_volume",
    "log_total_volume",
    "log_total_transactions",
    "mean_range_bps",
    "range_bps_std",
    "mean_active_seconds",
    "missing_symbol_fraction",
    "large_move_fraction",
    "quality_score",
    "is_premarket",
    "is_regular_session",
    "is_postmarket",
    "seconds_since_open",
    "seconds_to_close",
]

ACTION_FEATURE_NAMES = [
    "action_index_scaled",
    "is_cash",
    "is_etf",
    "is_stock",
    "is_inverse",
    "is_leveraged",
    "leverage_factor",
    "target_weight",
    "valid_price_flag",
    "feature_staleness_seconds",
    "log_last_dollar_volume",
    "estimated_cost_bps",
]

PORTFOLIO_STATE_FEATURE_NAMES = ["cash_weight", "gross_exposure", "previous_action_index_scaled"]
CONSTRAINT_STATE_FEATURE_NAMES = ["data_quality_score", "valid_action_fraction", "minutes_to_close_scaled"]


@dataclass(frozen=True)
class StockSecondContextConfig:
    decision_interval: str = "15m"
    context_seconds: int = 3_600
    block_seconds: int = 300
    bar_latency_ms: int = DEFAULT_BAR_LATENCY_MS
    ingestion_latency_ms: int = 0
    min_active_symbols: int = 250
    max_action_staleness_seconds: int = 300
    execution_latency_ms: int = DEFAULT_BAR_LATENCY_MS
    execution_model: str = "optimistic_close_plus_estimated_cost_bps"
    default_action_cost_bps: float = 1.0
    max_action_cost_bps: float = 25.0
    rth_only: bool = True
    include_extended_hours: bool = False
    allow_post_close_exit: bool = False
    source_bar_interval: str = "1s"
    feature_set_id: str = "stock_second_context_v001"
    known_limitations: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.decision_interval not in SUPPORTED_DECISION_INTERVALS and not self.decision_interval.endswith("s"):
            raise ValueError(f"decision_interval must be one of {sorted(SUPPORTED_DECISION_INTERVALS)} or a second interval.")
        if self.context_seconds <= 0 or self.block_seconds <= 0:
            raise ValueError("context_seconds and block_seconds must be positive.")
        if self.context_seconds % self.block_seconds != 0:
            raise ValueError("context_seconds must be an integer multiple of block_seconds.")
        if self.bar_latency_ms < DEFAULT_BAR_LATENCY_MS:
            raise ValueError("bar_latency_ms must be at least 1000 for second aggregates.")
        if self.ingestion_latency_ms < 0:
            raise ValueError("ingestion_latency_ms must be non-negative.")
        if self.min_active_symbols <= 0:
            raise ValueError("min_active_symbols must be positive.")
        if self.max_action_staleness_seconds < 0:
            raise ValueError("max_action_staleness_seconds must be non-negative.")
        if self.execution_latency_ms < 0:
            raise ValueError("execution_latency_ms must be non-negative.")
        if self.source_bar_interval == "1s" and self.execution_latency_ms < DEFAULT_BAR_LATENCY_MS:
            # Causal invariant for 1s aggregates (mirrors bar_latency_ms above): the entry fill must occur at
            # least one bar AFTER the decision, never at the decision bar's already-observed close. A second
            # source with execution_latency_ms < 1000 would fill at/inside the decision bar -> look-ahead.
            raise ValueError("execution_latency_ms must be at least 1000 for second (1s) aggregates.")
        if self.default_action_cost_bps < 0 or self.max_action_cost_bps < self.default_action_cost_bps:
            raise ValueError("action cost bps settings are invalid.")
        if self.rth_only and self.include_extended_hours:
            raise ValueError("rth_only and include_extended_hours cannot both be true.")

    @property
    def decision_seconds(self) -> int:
        return parse_duration_seconds(self.decision_interval)

    @property
    def lookback_blocks(self) -> int:
        return self.context_seconds // self.block_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_duration_seconds(value: str) -> int:
    text = value.strip().lower()
    if text.endswith("s"):
        seconds = int(text[:-1])
    elif text.endswith("m"):
        seconds = int(text[:-1]) * 60
    elif text.endswith("h"):
        seconds = int(text[:-1]) * 3_600
    else:
        raise ValueError(f"Unsupported duration {value!r}; expected values like 5m, 15m, or 60m.")
    if seconds <= 0:
        raise ValueError("duration must be positive.")
    return seconds


def regular_session_decision_grid_ms(
    *,
    start: str,
    end_exclusive: str,
    decision_interval: str,
    execution_latency_ms: int = 0,
    allow_post_close_exit: bool = False,
    exchange_tz: ZoneInfo = EASTERN,
) -> list[int]:
    interval_seconds = parse_duration_seconds(decision_interval)
    if execution_latency_ms < 0:
        raise ValueError("execution_latency_ms must be non-negative.")
    def _aware_utc(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # A naive ISO string (e.g. a date-only "2024-03-15") must be treated as UTC, not the
        # host's system timezone, or .astimezone() would shift the grid by the local offset and
        # make decision-grid generation environment-dependent.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    start_dt = _aware_utc(start).astimezone(exchange_tz)
    end_dt = _aware_utc(end_exclusive).astimezone(exchange_tz)
    execution_latency = timedelta(milliseconds=execution_latency_ms)
    decisions: list[int] = []
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        # KNOWN LIMITATION: only weekends are excluded -- there is no market-holiday/early-close
        # calendar, so holiday weekdays still emit a full decision grid. Those rows come back
        # all-invalid (no second data) and are masked, so there is no leakage, but they inflate
        # row/segment counts and quality denominators. A trading calendar should gate sessions.
        if current_date.weekday() < 5:
            session_start = datetime.combine(current_date, RTH_START, tzinfo=exchange_tz)
            session_end = datetime.combine(current_date, RTH_END, tzinfo=exchange_tz)
            decision = session_start + timedelta(seconds=interval_seconds)
            while decision + timedelta(seconds=interval_seconds) <= session_end:
                reward_exit = decision + timedelta(seconds=interval_seconds) + execution_latency
                if not allow_post_close_exit and reward_exit > session_end:
                    break
                decision_utc = decision.astimezone(timezone.utc)
                if start_dt.astimezone(timezone.utc) <= decision_utc < end_dt.astimezone(timezone.utc):
                    decisions.append(int(decision_utc.timestamp() * 1000))
                decision += timedelta(seconds=interval_seconds)
        current_date += timedelta(days=1)
    return decisions


def _session_flags(timestamp_ms: int) -> tuple[float, float, float, float, float]:
    local = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).astimezone(EASTERN)
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    open_seconds = 9 * 3600 + 30 * 60
    close_seconds = 16 * 3600
    is_premarket = float(seconds < open_seconds)
    is_regular = float(open_seconds <= seconds < close_seconds)
    is_postmarket = float(seconds >= close_seconds)
    seconds_since_open = max(0.0, min(float(seconds - open_seconds), float(close_seconds - open_seconds)))
    seconds_to_close = max(0.0, float(close_seconds - seconds))
    return is_premarket, is_regular, is_postmarket, seconds_since_open, seconds_to_close


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        # KNOWN LIMITATION: when every weight (e.g. dollar volume) is zero, the dollar-volume-
        # weighted feature silently degrades to a plain equal-weight mean with no distinguishing
        # flag. Blocks with non-trivial volume are unaffected; price-only blocks fall back here.
        return sum(values) / max(len(values), 1)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _concentration(weights: list[float]) -> float:
    total = sum(weights)
    return max(weights) / total if total > 0 and weights else 0.0


def _symbol_block_summary(frame: Any) -> dict[str, float] | None:
    if len(frame) == 0:
        return None
    first_close = _finite(frame["close"].iloc[0], default=0.0)
    last_close = _finite(frame["close"].iloc[-1], default=0.0)
    if first_close <= 0 or last_close <= 0:
        return None
    volume = float(frame["volume"].fillna(0.0).clip(lower=0.0).sum()) if "volume" in frame.columns else 0.0
    if "vwap" in frame.columns:
        dollar_volume = float((frame["vwap"].fillna(frame["close"]) * frame.get("volume", 0.0)).fillna(0.0).sum())
    else:
        dollar_volume = float((frame["close"] * frame.get("volume", 0.0)).fillna(0.0).sum())
    transactions = float(frame["transactions"].fillna(0.0).clip(lower=0.0).sum()) if "transactions" in frame.columns else 0.0
    close = frame["close"].replace(0, float("nan"))
    ranges = ((frame["high"] - frame["low"]) / close * 10_000.0).replace([float("inf"), -float("inf")], float("nan"))
    clipped_return = max(min(last_close / first_close - 1.0, 1.0), -1.0)
    return {
        "return": clipped_return,
        # Derive abs_return from the CLIPPED return so a single bad-tick outlier cannot dominate
        # the dollar-volume-weighted abs-return / large-move features while the signed return is
        # bounded to [-1, 1].
        "abs_return": abs(clipped_return),
        "volume": max(volume, 0.0),
        "dollar_volume": max(dollar_volume, 0.0),
        "transactions": max(transactions, 0.0),
        "range_bps": _finite(ranges.mean(), default=0.0),
        "active_seconds": float(len(frame)),
    }


def _block_market_features(
    frames_by_symbol: Mapping[str, Any],
    *,
    block_start_ms: int,
    block_end_ms: int,
    total_symbols: int,
    min_active_symbols: int,
) -> tuple[list[float], bool]:
    summaries: list[dict[str, float]] = []
    for frame in frames_by_symbol.values():
        if len(frame) == 0:
            continue
        selected = frame.loc[
            (frame["timestamp_ms"] >= block_start_ms) & (frame["timestamp_ms"] <= block_end_ms)
        ]
        summary = _symbol_block_summary(selected)
        if summary is not None:
            summaries.append(summary)
    active = len(summaries)
    if not summaries:
        is_pre, is_regular, is_post, since_open, to_close = _session_flags(block_end_ms)
        empty = [0.0] * len(MARKET_CONTEXT_FEATURE_NAMES)
        # A block with no active symbols means every symbol is missing: missing_symbol_fraction
        # (index 20) must be 1.0, mirroring the populated path's (1.0 - active_fraction). Leaving
        # it 0.0 would report "all symbols present" -- the exact inverse of the truth.
        empty[20] = 1.0
        empty[22] = 0.0
        empty[23] = is_pre
        empty[24] = is_regular
        empty[25] = is_post
        empty[26] = since_open
        empty[27] = to_close
        return empty, False

    returns = [item["return"] for item in summaries]
    abs_returns = [item["abs_return"] for item in summaries]
    dollar_volumes = [item["dollar_volume"] for item in summaries]
    transactions = [item["transactions"] for item in summaries]
    volumes = [item["volume"] for item in summaries]
    ranges = [item["range_bps"] for item in summaries]
    active_seconds = [item["active_seconds"] for item in summaries]
    sorted_returns = sorted(returns)
    bucket = max(1, len(sorted_returns) // 10)
    top_decile = sum(sorted_returns[-bucket:]) / bucket
    bottom_decile = sum(sorted_returns[:bucket]) / bucket
    active_fraction = active / max(float(total_symbols), 1.0)
    quality_score = min(active / max(float(min_active_symbols), 1.0), 1.0)
    is_pre, is_regular, is_post, since_open, to_close = _session_flags(block_end_ms)
    row = [
        float(active),
        active_fraction,
        sum(returns) / len(returns),
        _weighted_mean(returns, dollar_volumes),
        sorted_returns[len(sorted_returns) // 2],
        _std(returns),
        sum(1.0 for value in returns if value > 0.0) / len(returns),
        sum(1.0 for value in returns if value < 0.0) / len(returns),
        top_decile,
        bottom_decile,
        top_decile - bottom_decile,
        _weighted_mean(abs_returns, dollar_volumes),
        _concentration(dollar_volumes),
        _concentration(transactions),
        math.log1p(sum(dollar_volumes)),
        math.log1p(sum(volumes)),
        math.log1p(sum(transactions)),
        sum(ranges) / len(ranges),
        _std(ranges),
        sum(active_seconds) / len(active_seconds),
        1.0 - active_fraction,
        sum(1.0 for value in abs_returns if value > 0.01) / len(abs_returns),
        quality_score,
        is_pre,
        is_regular,
        is_post,
        since_open,
        to_close,
    ]
    return row, active >= min_active_symbols


def build_market_context_from_frames(
    frames_by_symbol: Mapping[str, Any],
    *,
    decision_timestamps_ms: list[int],
    config: StockSecondContextConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    rows: list[list[list[float]]] = []
    masks: list[list[bool]] = []
    available: list[list[int]] = []
    total_symbols = max(len(frames_by_symbol), 1)
    block_ms = config.block_seconds * 1000
    latency_ms = config.bar_latency_ms + config.ingestion_latency_ms
    for decision_ms in decision_timestamps_ms:
        context_end_ms = int(decision_ms) - latency_ms
        decision_rows: list[list[float]] = []
        decision_mask: list[bool] = []
        decision_available: list[int] = []
        for block_index in range(config.lookback_blocks):
            blocks_from_end = config.lookback_blocks - 1 - block_index
            block_end_ms = context_end_ms - blocks_from_end * block_ms
            block_start_ms = block_end_ms - block_ms + 1_000
            features, valid = _block_market_features(
                frames_by_symbol,
                block_start_ms=block_start_ms,
                block_end_ms=block_end_ms,
                total_symbols=total_symbols,
                min_active_symbols=config.min_active_symbols,
            )
            available_ms = available_timestamp_ms(
                block_end_ms,
                bar_latency_ms=config.bar_latency_ms,
                ingestion_latency_ms=config.ingestion_latency_ms,
            )
            if available_ms > decision_ms:
                raise ValueError("Second-context block is not available by the decision timestamp.")
            decision_rows.append(features)
            decision_mask.append(valid)
            decision_available.append(available_ms)
        rows.append(decision_rows)
        masks.append(decision_mask)
        available.append(decision_available)
    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(available, dtype=torch.long),
    )


def _close_lookup(frame: Any) -> tuple[list[int], list[float], list[float]]:
    if len(frame) == 0:
        return [], [], []
    timestamps = [int(value) for value in frame["timestamp_ms"].tolist()]
    closes = [_finite(value, default=math.nan) for value in frame["close"].tolist()]
    if "volume" in frame.columns:
        volumes = [_finite(value, default=0.0) for value in frame["volume"].tolist()]
    else:
        volumes = [0.0] * len(timestamps)
    dollar_volume = [max(close, 0.0) * max(volume, 0.0) for close, volume in zip(closes, volumes)]
    return timestamps, closes, dollar_volume


def _close_at_or_after(
    lookup: tuple[list[int], list[float], list[float]],
    timestamp_ms: int,
    *,
    max_staleness_seconds: int,
) -> tuple[float, int, float] | None:
    timestamps, closes, dollar_volumes = lookup
    pos = bisect.bisect_left(timestamps, int(timestamp_ms))
    if pos >= len(timestamps):
        return None
    delay_ms = int(timestamps[pos]) - int(timestamp_ms)
    if delay_ms < 0 or delay_ms > max_staleness_seconds * 1000:
        return None
    close = closes[pos]
    if not math.isfinite(close) or close <= 0:
        return None
    return close, timestamps[pos], dollar_volumes[pos]


def _close_at_or_before(
    lookup: tuple[list[int], list[float], list[float]],
    timestamp_ms: int,
    *,
    max_staleness_seconds: int,
) -> tuple[float, int, float] | None:
    timestamps, closes, dollar_volumes = lookup
    pos = bisect.bisect_right(timestamps, int(timestamp_ms)) - 1
    if pos < 0:
        return None
    age_ms = int(timestamp_ms) - int(timestamps[pos])
    if age_ms < 0 or age_ms > max_staleness_seconds * 1000:
        return None
    close = closes[pos]
    if not math.isfinite(close) or close <= 0:
        return None
    return close, timestamps[pos], dollar_volumes[pos]


ETF_SYMBOLS = {
    "ARKK",
    "DIA",
    "EEM",
    "EFA",
    "GLD",
    "HYG",
    "IWM",
    "IVV",
    "LQD",
    "QQQ",
    "SQQQ",
    "SLV",
    "SMH",
    "SOXL",
    "SOXS",
    "SPY",
    "TBT",
    "TLT",
    "TQQQ",
    "UNG",
    "USO",
    "VTI",
    "XBI",
    "XLE",
    "XLF",
    "XLK",
    "XLU",
}

LEVERAGE_BY_SYMBOL = {
    "SOXL": 3.0,
    "SOXS": -3.0,
    "TQQQ": 3.0,
    "SQQQ": -3.0,
    "UPRO": 3.0,
    "SPXU": -3.0,
    "QLD": 2.0,
    "QID": -2.0,
    "SSO": 2.0,
    "SDS": -2.0,
    "TBT": -2.0,
}


def _action_type_flags(symbol: str) -> tuple[float, float]:
    if symbol.upper() in ETF_SYMBOLS:
        return 1.0, 0.0
    return 0.0, 1.0


def _action_metadata(symbol: str) -> dict[str, float]:
    upper = symbol.upper()
    is_cash = float(upper == "CASH")
    is_etf, is_stock = (0.0, 0.0) if is_cash else _action_type_flags(upper)
    leverage = 0.0 if is_cash else LEVERAGE_BY_SYMBOL.get(upper, 1.0)
    abs_leverage = abs(leverage)
    target_weight = 0.0 if is_cash else min(1.0, 1.0 / max(abs_leverage, 1.0))
    return {
        "is_cash": is_cash,
        "is_etf": is_etf,
        "is_stock": is_stock,
        "is_inverse": float(leverage < 0.0),
        "is_leveraged": float(abs_leverage > 1.0),
        "leverage_factor": leverage,
        "target_weight": target_weight,
    }


def _action_metadata_manifest(action_names: list[str]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for action_index, action_name in enumerate(action_names):
        metadata = _action_metadata(action_name)
        if metadata["is_cash"]:
            asset_type = "cash"
            group = "cash"
            underlying = None
            risk_bucket = "cash"
        elif metadata["is_etf"]:
            asset_type = "etf"
            group = "leveraged" if metadata["is_leveraged"] else "core_etf"
            underlying = action_name.upper()
            risk_bucket = "leveraged" if metadata["is_leveraged"] else "core_equity"
        else:
            asset_type = "stock"
            group = "single_stock"
            underlying = action_name.upper()
            risk_bucket = "single_stock"
        actions.append(
            {
                "action_index": action_index,
                "action_name": action_name,
                "symbol_id": "CASH" if metadata["is_cash"] else action_name.upper(),
                "asset_type": asset_type,
                "group": group,
                "underlying": underlying,
                "leverage_factor": metadata["leverage_factor"],
                "inverse": bool(metadata["is_inverse"]),
                "max_weight": metadata["target_weight"],
                "risk_bucket": risk_bucket,
            }
        )
    return {
        "actions": actions,
        "action_metadata_hash": stable_json_hash(actions),
    }


def _estimate_cost_bps(dollar_volume: float, config: StockSecondContextConfig) -> float:
    if dollar_volume <= 0:
        return min(config.max_action_cost_bps, config.default_action_cost_bps * 5.0)
    liquidity_discount = min(math.log1p(dollar_volume) / math.log1p(100_000_000.0), 1.0)
    extra = (1.0 - liquidity_discount) * config.default_action_cost_bps * 4.0
    return min(config.max_action_cost_bps, config.default_action_cost_bps + extra)


def _minutes_to_close_scaled(decision_ms: int) -> float:
    _is_pre, _is_regular, _is_post, _since_open, seconds_to_close = _session_flags(decision_ms)
    return max(0.0, min(seconds_to_close / (6.5 * 3600.0), 1.0))


def _session_id(timestamp_ms: int) -> str:
    local = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).astimezone(EASTERN)
    return local.date().isoformat()


def build_second_context_payload(
    *,
    stock_frames_by_symbol: Mapping[str, Any],
    action_frames_by_symbol: Mapping[str, Any],
    action_names: list[str],
    decision_timestamps_ms: list[int],
    config: StockSecondContextConfig,
    dataset_manifest: Mapping[str, Any] | None = None,
    data_quality_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config.validate()
    if not action_names or action_names[0].upper() != "CASH":
        raise ValueError("action_names must start with CASH.")
    decision_timestamps_ms = sorted({int(value) for value in decision_timestamps_ms})
    if not decision_timestamps_ms:
        raise ValueError("decision_timestamps_ms must not be empty.")
    decision_seconds = config.decision_seconds
    next_timestamps_ms = [value + decision_seconds * 1000 for value in decision_timestamps_ms]
    market_context, market_mask, available_ms = build_market_context_from_frames(
        stock_frames_by_symbol,
        decision_timestamps_ms=decision_timestamps_ms,
        config=config,
    )
    action_lookups = {
        symbol: _close_lookup(frame)
        for symbol, frame in action_frames_by_symbol.items()
    }
    action_features: list[list[list[float]]] = []
    action_returns: list[list[float]] = []
    decision_action_valid_mask: list[list[bool]] = []
    label_valid_mask: list[list[bool]] = []
    entry_fill_observed_mask: list[list[bool]] = []
    reward_exit_observed_mask: list[list[bool]] = []
    action_cost_bps: list[list[float]] = []
    action_target_weights: list[list[float]] = []
    action_features_available_timestamps_ms: list[list[int]] = []
    action_cost_available_timestamps_ms: list[list[int]] = []
    action_mask_reason_code: list[list[int]] = []
    action_quality_score: list[list[float]] = []
    execution_latency_ms = config.execution_latency_ms
    input_latency_ms = config.bar_latency_ms + config.ingestion_latency_ms
    entry_execution_timestamps_ms: list[list[int]] = []
    exit_execution_timestamps_ms: list[list[int]] = []
    action_count_denom = max(len(action_names) - 1, 1)
    for decision_ms, next_ms, context_mask in zip(decision_timestamps_ms, next_timestamps_ms, market_mask):
        decision_action_features: list[list[float]] = []
        decision_returns: list[float] = []
        decision_valid: list[bool] = []
        decision_label_valid: list[bool] = []
        decision_entry_observed: list[bool] = []
        decision_exit_observed: list[bool] = []
        decision_costs: list[float] = []
        decision_weights: list[float] = []
        decision_action_available: list[int] = []
        decision_cost_available: list[int] = []
        decision_reason_codes: list[int] = []
        decision_action_quality: list[float] = []
        decision_entry_ts: list[int] = []
        decision_exit_ts: list[int] = []
        # Per-action validity no longer depends on aggregate market-context coverage (see F3 fix);
        # market-context quality is reported separately via quality_by_row / decision_quality_score.
        for action_index, action in enumerate(action_names):
            symbol = action.upper()
            action_index_scaled = action_index / action_count_denom
            metadata = _action_metadata(symbol)
            if symbol == "CASH":
                decision_action_features.append(
                    [
                        action_index_scaled,
                        metadata["is_cash"],
                        metadata["is_etf"],
                        metadata["is_stock"],
                        metadata["is_inverse"],
                        metadata["is_leveraged"],
                        metadata["leverage_factor"],
                        metadata["target_weight"],
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                decision_returns.append(0.0)
                decision_valid.append(True)
                decision_label_valid.append(True)
                decision_entry_observed.append(True)
                decision_exit_observed.append(True)
                decision_costs.append(0.0)
                decision_weights.append(0.0)
                decision_action_available.append(int(decision_ms))
                decision_cost_available.append(int(decision_ms))
                decision_reason_codes.append(0)
                decision_action_quality.append(1.0)
                decision_entry_ts.append(int(decision_ms))
                decision_exit_ts.append(int(next_ms))
                continue
            lookup = action_lookups.get(symbol)
            feature_point = None if lookup is None else _close_at_or_before(
                lookup,
                decision_ms - input_latency_ms,
                max_staleness_seconds=config.max_action_staleness_seconds,
            )
            current = None if lookup is None else _close_at_or_after(
                lookup,
                decision_ms + execution_latency_ms,
                max_staleness_seconds=config.max_action_staleness_seconds,
            )
            future = None if lookup is None else _close_at_or_after(
                lookup,
                next_ms + execution_latency_ms,
                max_staleness_seconds=config.max_action_staleness_seconds,
            )
            # Per-action ex-ante validity depends ONLY on this action's own point-in-time feature
            # availability, NOT on global market-context coverage. A sparse market-context row
            # still forces cash at the ROW level via force_cash_mask (decision_quality_score
            # multiplies by quality_by_row), so a genuinely tradable action is no longer marked
            # invalid merely because the aggregate market context happened to be sparse.
            decision_known = feature_point is not None
            entry_observed = current is not None
            exit_observed = future is not None
            label_valid = decision_known and entry_observed and exit_observed
            if feature_point is None:
                feature_staleness = float(config.max_action_staleness_seconds + 1)
                last_dv = 0.0
                cost = config.max_action_cost_bps
                feature_available_ts = -1
            else:
                _feature_close, feature_ts, last_dv = feature_point
                feature_staleness = max(0.0, (decision_ms - input_latency_ms - feature_ts) / 1000.0)
                cost = _estimate_cost_bps(last_dv, config)
                feature_available_ts = available_timestamp_ms(
                    feature_ts,
                    bar_latency_ms=config.bar_latency_ms,
                    ingestion_latency_ms=config.ingestion_latency_ms,
                )
                if feature_available_ts > decision_ms:
                    raise ValueError("Action features are not available by the decision timestamp.")
            entry_ts = -1 if current is None else int(current[1])
            exit_ts = -1 if future is None else int(future[1])
            if decision_known:
                reason_code = 0
            else:
                reason_code = 2
            decision_action_features.append(
                [
                    action_index_scaled,
                    0.0,
                    metadata["is_etf"],
                    metadata["is_stock"],
                    metadata["is_inverse"],
                    metadata["is_leveraged"],
                    metadata["leverage_factor"],
                    metadata["target_weight"],
                    float(feature_point is not None),
                    feature_staleness,
                    math.log1p(max(last_dv, 0.0)),
                    cost,
                ]
            )
            decision_costs.append(cost)
            decision_valid.append(bool(decision_known))
            decision_label_valid.append(bool(label_valid))
            decision_entry_observed.append(bool(entry_observed))
            decision_exit_observed.append(bool(exit_observed))
            decision_weights.append(float(metadata["target_weight"]))
            decision_action_available.append(int(feature_available_ts))
            decision_cost_available.append(int(feature_available_ts))
            decision_reason_codes.append(reason_code)
            # Per-action quality reflects this action's own data readiness; market-context
            # coverage is reported separately in constraint_state and decision_quality_score.
            decision_action_quality.append(1.0 if feature_point is not None else 0.0)
            decision_entry_ts.append(int(entry_ts))
            decision_exit_ts.append(int(exit_ts))
            if label_valid and current is not None and future is not None:
                decision_returns.append(max(min(future[0] / current[0] - 1.0, 1.0), -1.0))
            else:
                decision_returns.append(math.nan)
        action_features.append(decision_action_features)
        action_returns.append(decision_returns)
        decision_action_valid_mask.append(decision_valid)
        label_valid_mask.append(decision_label_valid)
        entry_fill_observed_mask.append(decision_entry_observed)
        reward_exit_observed_mask.append(decision_exit_observed)
        action_cost_bps.append(decision_costs)
        action_target_weights.append(decision_weights)
        action_features_available_timestamps_ms.append(decision_action_available)
        action_cost_available_timestamps_ms.append(decision_cost_available)
        action_mask_reason_code.append(decision_reason_codes)
        action_quality_score.append(decision_action_quality)
        entry_execution_timestamps_ms.append(decision_entry_ts)
        exit_execution_timestamps_ms.append(decision_exit_ts)

    action_valid = torch.tensor(decision_action_valid_mask, dtype=torch.bool)
    label_valid = torch.tensor(label_valid_mask, dtype=torch.bool)
    entry_fill_observed = torch.tensor(entry_fill_observed_mask, dtype=torch.bool)
    reward_exit_observed = torch.tensor(reward_exit_observed_mask, dtype=torch.bool)
    quality_by_row = market_mask.float().mean(dim=1).clamp(0.0, 1.0)
    valid_action_fraction = action_valid.float().mean(dim=1).clamp(0.0, 1.0)
    decision_quality_score = (quality_by_row * valid_action_fraction).clamp(0.0, 1.0)
    force_cash_mask = decision_quality_score <= 0.0
    portfolio_state = torch.zeros((len(decision_timestamps_ms), len(PORTFOLIO_STATE_FEATURE_NAMES)), dtype=torch.float32)
    portfolio_state[:, 0] = 1.0
    constraint_state = torch.stack(
        [
            quality_by_row,
            valid_action_fraction,
            torch.tensor([_minutes_to_close_scaled(value) for value in decision_timestamps_ms], dtype=torch.float32),
        ],
        dim=1,
    )
    session_ids = [_session_id(value) for value in decision_timestamps_ms]
    segment_ids: list[int] = []
    segment = -1
    previous_session = None
    for session in session_ids:
        if session != previous_session:
            segment += 1
            previous_session = session
        segment_ids.append(segment)
    valid_start_indices = [index for index, valid in enumerate(action_valid.any(dim=1).tolist()) if valid]
    action_metadata = _action_metadata_manifest(list(action_names))
    feature_names = {
        "market_context": list(MARKET_CONTEXT_FEATURE_NAMES),
        "action_features": list(ACTION_FEATURE_NAMES),
        "portfolio_state": list(PORTFOLIO_STATE_FEATURE_NAMES),
        "constraint_state": list(CONSTRAINT_STATE_FEATURE_NAMES),
    }
    schema_registry = {
        "decision_tensor_protocol_version": "1.0.0",
        "dataset_schema_version": "second_context_gold_v1",
        "feature_schema_hash": stable_json_hash(feature_names),
        "action_schema_hash": stable_json_hash(list(action_names)),
        "action_metadata_hash": action_metadata["action_metadata_hash"],
        "constraint_schema_hash": stable_json_hash(CONSTRAINT_STATE_FEATURE_NAMES),
        "portfolio_state_schema_hash": stable_json_hash(PORTFOLIO_STATE_FEATURE_NAMES),
        "execution_schema_hash": stable_json_hash(
            {
                "execution_model": config.execution_model,
                "entry_price_source": "first_action_close_at_or_after_decision_plus_execution_latency",
                "exit_price_source": "first_action_close_at_or_after_reward_end_plus_execution_latency",
                "execution_latency_ms": config.execution_latency_ms,
                "allow_post_close_exit": config.allow_post_close_exit,
            }
        ),
    }
    tensor_availability = {
        "market_context": "market_context_available_timestamps_ms <= decision_timestamps_ms",
        "action_features": "action_features_available_timestamps_ms <= decision_timestamps_ms",
        "action_cost_bps": "action_cost_available_timestamps_ms <= decision_timestamps_ms",
        "portfolio_state": "known before the decision from prior fills",
        "constraint_state": "known before the decision from current masks and session state",
        "decision_action_valid_mask": "known before the decision; actions the policy may select",
        "action_valid_mask": "legacy alias for decision_action_valid_mask",
        "label_valid_mask": "known only after reward realization; used for training/evaluation labels, forbidden as model input",
        "entry_fill_observed_mask": "historical audit flag for whether the entry fill label was observed, forbidden as model input",
        "reward_exit_observed_mask": "historical audit flag for whether the reward exit label was observed, forbidden as model input",
        "action_returns": "label realized after decision; forbidden as model input",
    }
    model_input_keys = [
        "market_context",
        "market_context_mask",
        "action_features",
        "decision_action_valid_mask",
        "action_valid_mask",
        "action_cost_bps",
        "action_target_weights",
        "portfolio_state",
        "constraint_state",
        "decision_quality_score",
        "force_cash_mask",
    ]
    label_keys = [
        "action_returns",
        "label_valid_mask",
        "entry_fill_observed_mask",
        "reward_exit_observed_mask",
        "next_timestamps",
        "entry_execution_timestamps_ms",
        "exit_execution_timestamps_ms",
    ]
    forbidden_model_input_keys = [
        "action_returns",
        "label_valid_mask",
        "entry_fill_observed_mask",
        "reward_exit_observed_mask",
        "next_timestamps",
        "exit_execution_timestamps_ms",
    ]
    # Validate the REAL builder split against the protocol contract at BUILD time (not only a test fixture):
    # a model input must never be a label / forbidden (realized-outcome) key. No-op for the canonical split;
    # fails closed if an edit ever introduces leakage. (features -> protocol is a valid downward import.)
    from rl_quant.protocol import assert_no_model_input_leakage

    assert_no_model_input_leakage(
        model_input_keys=model_input_keys,
        label_keys=label_keys,
        forbidden_model_input_keys=forbidden_model_input_keys,
    )
    execution_model = {
        "name": config.execution_model,
        "entry_rule": "first action close at or after decision_ts + execution_latency_ms",
        "exit_rule": "first action close at or after next_ts + execution_latency_ms",
        "cost_rule": "action_cost_bps estimated from trailing liquidity and charged on switches",
        "liquidate_at_end": False,
        "allow_post_close_exit": bool(config.allow_post_close_exit),
        "allow_extended_hours_exit": bool(config.include_extended_hours),
    }
    base_manifest = dict(dataset_manifest or {})
    payload_hash = stable_json_hash(
        {
            "decision_timestamps": [timestamp_ms_to_iso(value) for value in decision_timestamps_ms],
            "action_names": list(action_names),
            "config": config.to_dict(),
            "feature_schema_hash": schema_registry["feature_schema_hash"],
            "action_schema_hash": schema_registry["action_schema_hash"],
            "universe_selection_timestamp": base_manifest.get(
                "universe_selection_timestamp",
                base_manifest.get("universe_selection_date"),
            ),
            "universe_source_hash": base_manifest.get("universe_source_hash"),
            "source_manifest_hash": base_manifest.get("source_manifest_hash", base_manifest.get("manifest_hash")),
            "intended_action_symbols": list(base_manifest.get("intended_action_symbols", action_names)),
            "realized_action_symbols": list(base_manifest.get("realized_action_symbols", action_names)),
        }
    )
    decision_timestamp_strings = [timestamp_ms_to_iso(value) for value in decision_timestamps_ms]
    next_timestamp_strings = [timestamp_ms_to_iso(value) for value in next_timestamps_ms]
    split_manifest = {
        "schema_version": "split_manifest_v1",
        "rule": "unsplit gold payload; train/validation/test runs must declare decision_ts in split and next_ts <= split_end",
        "embargo": None,
        "unsplit": {
            "start": decision_timestamp_strings[0] if decision_timestamp_strings else None,
            "end": decision_timestamp_strings[-1] if decision_timestamp_strings else None,
            "rows": len(decision_timestamp_strings),
            "valid_rows": len(valid_start_indices),
            "reward_end_max": max(next_timestamp_strings) if next_timestamp_strings else None,
        },
    }
    report = dict(data_quality_report or {})
    source_download_complete = bool(
        base_manifest.get("source_download_complete", report.get("source_download_complete", False))
    )
    reportability_errors = [
        *list(base_manifest.get("reportability_errors", [])),
        *list(report.get("reportability_errors", [])),
    ]
    if not source_download_complete:
        reportability_errors.append("source_download_incomplete")
    if not report:
        reportability_errors.append("data_quality_report_missing")
    if config.min_active_symbols < max(1, len(stock_frames_by_symbol) // 2):
        reportability_errors.append("min_active_symbols_too_low_for_reportable_dataset")
    if config.allow_post_close_exit:
        reportability_errors.append("post_close_reward_exit_allowed")
    universe_selection_timestamp = base_manifest.get(
        "universe_selection_timestamp",
        base_manifest.get("universe_selection_date"),
    )
    if not universe_selection_timestamp:
        reportability_errors.append("universe_selection_timestamp_missing")
    else:
        selection_ms = iso_to_timestamp_ms(str(universe_selection_timestamp))
        if decision_timestamps_ms and min(decision_timestamps_ms) < selection_ms:
            reportability_errors.append("future_universe_selection_timestamp")
    reportability_scope = (
        "extended_reward_exit_allowed" if config.allow_post_close_exit else "regular_session_reward_exits_only"
    )
    manifest = {
        **base_manifest,
        "schema_version": "stock_second_context_decision_v3",
        "protocol_version": "decision_tensor_v1",
        **schema_registry,
        "feature_set_id": config.feature_set_id,
        "created_at_utc": utc_now_iso(),
        "source_bar_interval": config.source_bar_interval,
        "first_decision_timestamp": decision_timestamp_strings[0] if decision_timestamp_strings else None,
        "last_decision_timestamp": decision_timestamp_strings[-1] if decision_timestamp_strings else None,
        "universe_selection_timestamp": universe_selection_timestamp,
        "universe_method": base_manifest.get("universe_method"),
        "universe_source_hash": base_manifest.get("universe_source_hash"),
        "retrospective_fixed_survivor_universe_diagnostic": bool(
            "future_universe_selection_timestamp" in reportability_errors
            or "universe_selection_timestamp_missing" in reportability_errors
        ),
        "intended_action_symbols": list(base_manifest.get("intended_action_symbols", action_names)),
        "realized_action_symbols": list(base_manifest.get("realized_action_symbols", action_names)),
        "missing_intended_action_source_symbols": list(base_manifest.get("missing_intended_action_source_symbols", [])),
        "action_schema_changed_due_to_missing_sources": bool(
            base_manifest.get("action_schema_changed_due_to_missing_sources", False)
        ),
        "action_mask_semantics": {
            "decision_action_valid_mask": "ex-ante tradability/data-readiness mask used for model action selection",
            "action_valid_mask": "legacy alias for decision_action_valid_mask",
            "label_valid_mask": "ex-post label availability mask used only for supervised loss and realized-return filtering",
            "entry_fill_observed_mask": "ex-post fill availability flag for realized label construction",
            "reward_exit_observed_mask": "ex-post reward-exit availability flag for realized label construction",
        },
        "return_semantics": "action_returns are raw asset returns from entry_execution_timestamps_ms to exit_execution_timestamps_ms; strategy PnL applies action_target_weights and costs separately.",
        "target_weight_semantics": "action_target_weights are signed target exposures relative to portfolio equity; generated v3 stock-second actions are long-only except CASH=0.",
        "cash_action_invariant": "CASH is action index 0, always decision-valid and label-valid, with zero return, zero cost, and zero target weight.",
        "decision_interval": config.decision_interval,
        "context_seconds": config.context_seconds,
        "block_seconds": config.block_seconds,
        "lookback_blocks": config.lookback_blocks,
        "bar_latency_ms": config.bar_latency_ms,
        "ingestion_latency_ms": config.ingestion_latency_ms,
        "execution_latency_ms": config.execution_latency_ms,
        "allow_post_close_exit": config.allow_post_close_exit,
        "execution_model": config.execution_model,
        "execution_model_detail": execution_model,
        "tensor_availability": tensor_availability,
        "model_input_keys": model_input_keys,
        "label_keys": label_keys,
        "forbidden_model_input_keys": forbidden_model_input_keys,
        "payload_hash": payload_hash,
        "split_manifest_hash": stable_json_hash(split_manifest),
        "source_download_complete": source_download_complete,
        "reportable": source_download_complete and not reportability_errors,
        "reportability_scope": reportability_scope,
        "reportability_errors": list(dict.fromkeys(reportability_errors)),
        "known_limitations": [
            *config.known_limitations,
            "Top-stock second bars provide market context; action returns require separate tradable action bars.",
            "Sparse seconds are preserved through masks and active-second features.",
            "Action-return labels use first available action bar at or after the decision and reward timestamps.",
        ],
    }
    payload = {
        "schema_version": "stock_second_context_decision_v3",
        "protocol_version": "decision_tensor_v1",
        **schema_registry,
        "decision_timestamps": decision_timestamp_strings,
        "next_timestamps": next_timestamp_strings,
        "decision_timestamps_ms": torch.tensor(decision_timestamps_ms, dtype=torch.long),
        "next_timestamps_ms": torch.tensor(next_timestamps_ms, dtype=torch.long),
        "market_context": market_context,
        "market_context_mask": market_mask,
        "market_context_available_timestamps_ms": available_ms,
        "action_features": torch.tensor(action_features, dtype=torch.float32),
        "action_returns": torch.tensor(action_returns, dtype=torch.float32),
        "decision_action_valid_mask": action_valid,
        "action_valid_mask": action_valid,
        "label_valid_mask": label_valid,
        "entry_fill_observed_mask": entry_fill_observed,
        "reward_exit_observed_mask": reward_exit_observed,
        "action_mask_reason_code": torch.tensor(action_mask_reason_code, dtype=torch.int32),
        "action_cost_bps": torch.tensor(action_cost_bps, dtype=torch.float32),
        "action_target_weights": torch.tensor(action_target_weights, dtype=torch.float32),
        "action_features_available_timestamps_ms": torch.tensor(action_features_available_timestamps_ms, dtype=torch.long),
        "action_cost_available_timestamps_ms": torch.tensor(action_cost_available_timestamps_ms, dtype=torch.long),
        "action_quality_score": torch.tensor(action_quality_score, dtype=torch.float32),
        "entry_execution_timestamps_ms": torch.tensor(entry_execution_timestamps_ms, dtype=torch.long),
        "exit_execution_timestamps_ms": torch.tensor(exit_execution_timestamps_ms, dtype=torch.long),
        "entry_price_source": "first_action_close_at_or_after_decision_plus_execution_latency",
        "exit_price_source": "first_action_close_at_or_after_reward_end_plus_execution_latency",
        "execution_model": config.execution_model,
        "portfolio_state": portfolio_state,
        "portfolio_state_available_timestamps_ms": torch.tensor(decision_timestamps_ms, dtype=torch.long),
        "constraint_state": constraint_state,
        "constraint_state_available_timestamps_ms": torch.tensor(decision_timestamps_ms, dtype=torch.long),
        "decision_quality_score": decision_quality_score,
        "force_cash_mask": force_cash_mask,
        "valid_start_indices": torch.tensor(valid_start_indices, dtype=torch.long),
        "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
        "session_ids": session_ids,
        "feature_names": feature_names,
        "feature_names_by_tensor": feature_names,
        "action_names": list(action_names),
        "action_metadata": action_metadata,
        "split_manifest": split_manifest,
        "dataset_manifest": manifest,
        "data_quality_report": report,
        "tensor_availability": tensor_availability,
        "model_input_keys": model_input_keys,
        "label_keys": label_keys,
        "forbidden_model_input_keys": forbidden_model_input_keys,
        "execution_model_detail": execution_model,
        "config": config.to_dict(),
        "payload_hash": payload_hash,
    }
    validate_second_context_payload(payload)
    return payload


def validate_second_context_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "decision_timestamps",
        "next_timestamps",
        "entry_execution_timestamps_ms",
        "exit_execution_timestamps_ms",
        "market_context",
        "market_context_mask",
        "market_context_available_timestamps_ms",
        "action_features",
        "action_returns",
        "action_valid_mask",
        "action_target_weights",
        "action_cost_bps",
        "portfolio_state",
        "constraint_state",
        "feature_names",
        "action_names",
        "dataset_manifest",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Second-context dataset is missing required keys: {sorted(missing)}")
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    if not decisions:
        raise ValueError("decision_timestamps must not be empty.")
    if len(decisions) != len(next_timestamps):
        raise ValueError("next_timestamps length must match decision_timestamps.")
    decision_ms = [iso_to_timestamp_ms(value) for value in decisions]
    next_ms = [iso_to_timestamp_ms(value) for value in next_timestamps]
    for index, (current, following) in enumerate(zip(decision_ms, next_ms)):
        if following <= current:
            raise ValueError(f"next_timestamps must be after decision_timestamps at row {index}.")
    market = payload["market_context"]
    market_mask = payload["market_context_mask"].bool()
    available = payload["market_context_available_timestamps_ms"].long()
    action_features = payload["action_features"]
    action_returns = payload["action_returns"].float()
    action_valid = payload["action_valid_mask"].bool()
    decision_action_valid = payload.get("decision_action_valid_mask")
    if decision_action_valid is not None:
        decision_action_valid = decision_action_valid.bool()
    label_valid = payload.get("label_valid_mask")
    if label_valid is not None:
        label_valid = label_valid.bool()
    entry_fill_observed = payload.get("entry_fill_observed_mask")
    if entry_fill_observed is not None:
        entry_fill_observed = entry_fill_observed.bool()
    reward_exit_observed = payload.get("reward_exit_observed_mask")
    if reward_exit_observed is not None:
        reward_exit_observed = reward_exit_observed.bool()
    action_costs = payload["action_cost_bps"].float()
    action_target_weights = payload["action_target_weights"].float()
    entry_execution = payload["entry_execution_timestamps_ms"].long()
    exit_execution = payload["exit_execution_timestamps_ms"].long()
    rows = len(decisions)
    if market.ndim != 3 or market.shape[0] != rows:
        raise ValueError("market_context must have shape [rows, lookback_blocks, features].")
    if tuple(market_mask.shape) != tuple(market.shape[:2]):
        raise ValueError("market_context_mask shape must match market_context first two dimensions.")
    if tuple(available.shape) != tuple(market_mask.shape):
        raise ValueError("market_context_available_timestamps_ms shape must match market_context_mask.")
    decision_ms_tensor = torch.tensor(decision_ms, dtype=torch.long).unsqueeze(1)
    if bool((available > decision_ms_tensor).any().item()):
        raise ValueError("market context contains blocks that are unavailable at the decision timestamp.")
    if action_features.ndim != 3 or action_features.shape[0] != rows:
        raise ValueError("action_features must have shape [rows, actions, features].")
    action_names = list(payload["action_names"])
    if len(action_names) != action_features.shape[1]:
        raise ValueError("action_names length must match the action dimension.")
    feature_names = payload["feature_names"]
    if not isinstance(feature_names, Mapping):
        raise ValueError("feature_names must be a dictionary.")
    expected_feature_widths = {
        "market_context": market.shape[-1],
        "action_features": action_features.shape[-1],
        "portfolio_state": payload["portfolio_state"].shape[-1],
        "constraint_state": payload["constraint_state"].shape[-1],
    }
    for group, width in expected_feature_widths.items():
        names = list(feature_names.get(group, []))
        if len(names) != int(width):
            raise ValueError(f"feature_names[{group!r}] length must match tensor width.")
    if tuple(action_returns.shape) != tuple(action_valid.shape):
        raise ValueError("action_valid_mask shape must match action_returns.")
    if decision_action_valid is not None:
        if tuple(decision_action_valid.shape) != tuple(action_returns.shape):
            raise ValueError("decision_action_valid_mask shape must match action_returns.")
        if not bool(torch.equal(decision_action_valid, action_valid)):
            raise ValueError("action_valid_mask must be the legacy alias of decision_action_valid_mask.")
    else:
        decision_action_valid = action_valid
    if label_valid is not None and tuple(label_valid.shape) != tuple(action_returns.shape):
        raise ValueError("label_valid_mask shape must match action_returns.")
    if label_valid is None:
        label_valid = action_valid
    if bool((label_valid & ~action_valid).any().item()):
        raise ValueError("label_valid_mask must be a subset of decision action validity.")
    if entry_fill_observed is not None and tuple(entry_fill_observed.shape) != tuple(action_returns.shape):
        raise ValueError("entry_fill_observed_mask shape must match action_returns.")
    if reward_exit_observed is not None and tuple(reward_exit_observed.shape) != tuple(action_returns.shape):
        raise ValueError("reward_exit_observed_mask shape must match action_returns.")
    if tuple(action_costs.shape) != tuple(action_returns.shape):
        raise ValueError("action_cost_bps shape must match action_returns.")
    if tuple(action_target_weights.shape) != tuple(action_returns.shape):
        raise ValueError("action_target_weights shape must match action_returns.")
    action_feature_available = payload.get("action_features_available_timestamps_ms")
    if action_feature_available is not None:
        action_feature_available = action_feature_available.long()
        if tuple(action_feature_available.shape) != tuple(action_returns.shape):
            raise ValueError("action_features_available_timestamps_ms shape must match action_returns.")
        known = action_feature_available >= 0
        if bool((action_feature_available[known] > decision_ms_tensor.expand_as(action_valid)[known]).any().item()):
            raise ValueError("action features contain values unavailable at the decision timestamp.")
    action_feature_available_by_feature = payload.get("action_feature_available_timestamps_ms")
    if action_feature_available_by_feature is not None:
        action_feature_available_by_feature = action_feature_available_by_feature.long()
        if tuple(action_feature_available_by_feature.shape) != tuple(action_features.shape):
            raise ValueError("action_feature_available_timestamps_ms shape must match action_features.")
        decision_feature_ms = torch.tensor(decision_ms, dtype=torch.long).view(rows, 1, 1).expand_as(action_feature_available_by_feature)
        known = action_feature_available_by_feature >= 0
        if bool((action_feature_available_by_feature[known] > decision_feature_ms[known]).any().item()):
            raise ValueError("action feature tensor contains per-feature values unavailable at the decision timestamp.")
    action_covariates = payload.get("action_covariates")
    action_covariate_feature_names: list[str] = []
    if action_covariates is not None:
        required_covariate_keys = {
            "action_covariate_mask",
            "action_covariate_available_timestamps_ms",
            "action_covariate_feature_names",
            "action_covariate_schema_hash",
        }
        missing_covariate_keys = required_covariate_keys - set(payload)
        if missing_covariate_keys:
            raise ValueError(f"action_covariates missing required keys: {sorted(missing_covariate_keys)}")
        action_covariates = action_covariates.float()
        action_covariate_mask = payload["action_covariate_mask"].bool()
        action_covariate_available = payload["action_covariate_available_timestamps_ms"].long()
        action_covariate_feature_names = list(payload["action_covariate_feature_names"])
        if action_covariates.ndim != 3 or tuple(action_covariates.shape[:2]) != tuple(action_returns.shape):
            raise ValueError("action_covariates must have shape [rows, actions, features].")
        if tuple(action_covariate_mask.shape) != tuple(action_covariates.shape):
            raise ValueError("action_covariate_mask shape must match action_covariates.")
        if tuple(action_covariate_available.shape) != tuple(action_covariates.shape):
            raise ValueError("action_covariate_available_timestamps_ms shape must match action_covariates.")
        if len(action_covariate_feature_names) != int(action_covariates.shape[-1]):
            raise ValueError("action_covariate_feature_names length must match action_covariates width.")
        if payload.get("action_covariate_schema_hash") != stable_json_hash(action_covariate_feature_names):
            raise ValueError("action_covariate_schema_hash does not match action_covariate_feature_names.")
        decision_covariate_ms = torch.tensor(decision_ms, dtype=torch.long).view(rows, 1, 1).expand_as(action_covariate_available)
        known_covariates = action_covariate_mask & (action_covariate_available >= 0)
        if bool((action_covariate_available[known_covariates] > decision_covariate_ms[known_covariates]).any().item()):
            raise ValueError("action covariates contain values unavailable at the decision timestamp.")
        if action_covariates[known_covariates].numel() and not bool(torch.isfinite(action_covariates[known_covariates]).all().item()):
            raise ValueError("Known action_covariates must be finite.")
        action_covariate_age = payload.get("action_covariate_age_seconds")
        if action_covariate_age is not None and tuple(action_covariate_age.shape) != tuple(action_covariates.shape):
            raise ValueError("action_covariate_age_seconds shape must match action_covariates.")
    action_cost_available = payload.get("action_cost_available_timestamps_ms")
    if action_cost_available is not None:
        action_cost_available = action_cost_available.long()
        if tuple(action_cost_available.shape) != tuple(action_returns.shape):
            raise ValueError("action_cost_available_timestamps_ms shape must match action_returns.")
        known = action_cost_available >= 0
        if bool((action_cost_available[known] > decision_ms_tensor.expand_as(action_valid)[known]).any().item()):
            raise ValueError("action costs contain values unavailable at the decision timestamp.")
    action_mask_reason_code = payload.get("action_mask_reason_code")
    if action_mask_reason_code is not None and tuple(action_mask_reason_code.shape) != tuple(action_returns.shape):
        raise ValueError("action_mask_reason_code shape must match action_returns.")
    action_quality_score = payload.get("action_quality_score")
    if action_quality_score is not None:
        action_quality_score = action_quality_score.float()
        if tuple(action_quality_score.shape) != tuple(action_returns.shape):
            raise ValueError("action_quality_score shape must match action_returns.")
        if bool(((action_quality_score < 0) | (action_quality_score > 1)).any().item()):
            raise ValueError("action_quality_score must be in [0, 1].")
    if tuple(entry_execution.shape) != tuple(action_returns.shape):
        raise ValueError("entry_execution_timestamps_ms shape must match action_returns.")
    if tuple(exit_execution.shape) != tuple(action_returns.shape):
        raise ValueError("exit_execution_timestamps_ms shape must match action_returns.")
    if action_features.shape[:2] != action_returns.shape:
        raise ValueError("action_features first two dimensions must match action_returns.")
    if bool((action_costs < 0).any().item()):
        raise ValueError("action_cost_bps must be non-negative.")
    if not bool(torch.isfinite(action_target_weights).all().item()):
        raise ValueError("action_target_weights must be finite.")
    if abs(float(action_target_weights[:, 0].abs().max().item())) > 1e-12:
        raise ValueError("CASH action target weight must be zero.")
    if not bool(action_valid[:, 0].all().item()):
        raise ValueError("CASH action must be valid for every row.")
    if not bool(label_valid[:, 0].all().item()):
        raise ValueError("CASH action must be label-valid for every row.")
    label_ready = action_valid & label_valid
    if not bool(torch.isfinite(action_returns[label_ready]).all().item()):
        raise ValueError("Label-valid action_returns must be finite.")
    if bool((action_returns[:, 0].abs() > 1e-12).any().item()):
        raise ValueError("CASH action return must be zero.")
    invalid_returns = action_returns[~label_ready]
    if invalid_returns.numel() and not bool(torch.isnan(invalid_returns).all().item()):
        raise ValueError("Non-label-valid action_returns must be NaN.")
    if entry_fill_observed is not None:
        missing_entry_labels = label_ready & ~entry_fill_observed
        if bool(missing_entry_labels.any().item()):
            raise ValueError("Label-valid actions must have observed entry fills.")
    if reward_exit_observed is not None:
        missing_exit_labels = label_ready & ~reward_exit_observed
        if bool(missing_exit_labels.any().item()):
            raise ValueError("Label-valid actions must have observed reward exits.")
    if abs(float(action_costs[:, 0].abs().max().item())) > 1e-12:
        raise ValueError("CASH action cost must be zero.")
    decision_quality_score = payload.get("decision_quality_score")
    if decision_quality_score is not None:
        decision_quality_score = decision_quality_score.float()
        if tuple(decision_quality_score.shape) != (rows,):
            raise ValueError("decision_quality_score must have shape [rows].")
        if bool(((decision_quality_score < 0) | (decision_quality_score > 1)).any().item()):
            raise ValueError("decision_quality_score must be in [0, 1].")
    force_cash_mask = payload.get("force_cash_mask")
    if force_cash_mask is not None and tuple(force_cash_mask.bool().shape) != (rows,):
        raise ValueError("force_cash_mask must have shape [rows].")
    valid_start_indices = payload.get("valid_start_indices")
    if valid_start_indices is not None:
        valid_start_indices = valid_start_indices.long()
        if valid_start_indices.ndim != 1:
            raise ValueError("valid_start_indices must be a 1D tensor.")
        if valid_start_indices.numel() and bool(((valid_start_indices < 0) | (valid_start_indices >= rows)).any().item()):
            raise ValueError("valid_start_indices contains out-of-range rows.")
    segment_ids = payload.get("segment_ids")
    if segment_ids is not None and tuple(segment_ids.long().shape) != (rows,):
        raise ValueError("segment_ids must have shape [rows].")
    session_ids = payload.get("session_ids")
    if session_ids is not None and len(list(session_ids)) != rows:
        raise ValueError("session_ids length must match rows.")
    for key in ("portfolio_state_available_timestamps_ms", "constraint_state_available_timestamps_ms"):
        values = payload.get(key)
        if values is None:
            continue
        values = values.long()
        if tuple(values.shape) != (rows,):
            raise ValueError(f"{key} must have shape [rows].")
        if bool((values > torch.tensor(decision_ms, dtype=torch.long)).any().item()):
            raise ValueError(f"{key} contains values unavailable at the decision timestamp.")
    non_cash_valid = label_ready.clone()
    non_cash_valid[:, 0] = False
    decision_ms_tensor = torch.tensor(decision_ms, dtype=torch.long).unsqueeze(1).expand_as(action_valid)
    next_ms_tensor = torch.tensor(next_ms, dtype=torch.long).unsqueeze(1).expand_as(action_valid)
    if bool((entry_execution[non_cash_valid] < decision_ms_tensor[non_cash_valid]).any().item()):
        raise ValueError("Valid non-CASH entry executions must be at or after the decision timestamp.")
    if bool((exit_execution[non_cash_valid] < next_ms_tensor[non_cash_valid]).any().item()):
        raise ValueError("Valid non-CASH exit executions must be at or after the reward end timestamp.")
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        execution_latency_ms = int(config.get("execution_latency_ms", 0) or 0)
        if execution_latency_ms > 0:
            entry_min = decision_ms_tensor + execution_latency_ms
            exit_min = next_ms_tensor + execution_latency_ms
            if bool((entry_execution[non_cash_valid] < entry_min[non_cash_valid]).any().item()):
                raise ValueError("Valid non-CASH entry executions must respect execution latency.")
            if bool((exit_execution[non_cash_valid] < exit_min[non_cash_valid]).any().item()):
                raise ValueError("Valid non-CASH exit executions must respect execution latency.")
    if any(not name for name in action_names):
        raise ValueError("action_names must be non-empty strings.")
    if str(action_names[0]).upper() != "CASH":
        raise ValueError("action_names must start with CASH.")
    manifest = payload["dataset_manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset_manifest must be a dictionary.")
    model_input_keys = set(payload.get("model_input_keys", manifest.get("model_input_keys", [])))
    forbidden_input_keys = set(payload.get("forbidden_model_input_keys", manifest.get("forbidden_model_input_keys", [])))
    if model_input_keys & forbidden_input_keys:
        raise ValueError("model_input_keys overlap forbidden_model_input_keys.")
    covariates_are_model_facing = bool(payload.get("action_features_augmented_with_covariates")) or "action_covariates" in model_input_keys
    forbidden_covariate_features = {
        "future_dividend_ex_date_unannounced",
        "future_split_effective_date_unannounced",
    }
    if covariates_are_model_facing and forbidden_covariate_features.intersection(action_covariate_feature_names):
        raise ValueError("Forbidden future-only covariate feature appears in model-facing inputs.")
    action_metadata = payload.get("action_metadata")
    if action_metadata is not None:
        actions = list(action_metadata.get("actions", [])) if isinstance(action_metadata, Mapping) else []
        if len(actions) != len(action_names):
            raise ValueError("action_metadata actions length must match action_names.")
    if manifest.get("source_download_complete") is False and manifest.get("reportable") is True:
        raise ValueError("Incomplete source downloads cannot produce reportable datasets.")
    reportability_errors = set(manifest.get("reportability_errors", []))
    if manifest.get("reportable") is True and (
        "future_universe_selection_timestamp" in reportability_errors
        or "universe_selection_timestamp_missing" in reportability_errors
        or "missing_intended_action_source_symbols" in reportability_errors
    ):
        raise ValueError("Universe/action-source reportability errors cannot produce reportable datasets.")


def save_second_context_payload(payload: Mapping[str, Any], path: Path) -> None:
    validate_second_context_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_payload = dict(payload)
    manifest = dict(payload.get("dataset_manifest", {}))
    report = payload.get("data_quality_report", {})
    split_manifest = payload.get("split_manifest")
    if split_manifest is not None:
        manifest["split_manifest_hash"] = stable_json_hash(split_manifest)
    saved_payload["dataset_manifest"] = manifest
    torch.save(saved_payload, path)
    (path.parent / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (path.parent / "data_quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if split_manifest is not None:
        (path.parent / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
    (path.parent / "action_metadata.json").write_text(
        json.dumps(payload.get("action_metadata", {}), indent=2, sort_keys=True, default=str) + "\n"
    )
    (path.parent / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": payload.get("schema_version"),
                "protocol_version": payload.get("protocol_version"),
                "decision_tensor_protocol_version": payload.get("decision_tensor_protocol_version"),
                "dataset_schema_version": payload.get("dataset_schema_version"),
                "feature_schema_hash": payload.get("feature_schema_hash"),
                "action_schema_hash": payload.get("action_schema_hash"),
                "constraint_schema_hash": payload.get("constraint_schema_hash"),
                "portfolio_state_schema_hash": payload.get("portfolio_state_schema_hash"),
                "execution_schema_hash": payload.get("execution_schema_hash"),
                "split_manifest_hash": manifest.get("split_manifest_hash"),
                "model_input_keys": payload.get("model_input_keys", []),
                "label_keys": payload.get("label_keys", []),
                "forbidden_model_input_keys": payload.get("forbidden_model_input_keys", []),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    (path.parent / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_set_id": manifest.get("feature_set_id", "stock_second_context_v001"),
                "feature_names": payload.get("feature_names", {}),
                "feature_schema_hash": payload.get("feature_schema_hash"),
                "input_dataset_manifest_hash": stable_json_hash(manifest),
                "created_at_utc": utc_now_iso(),
                "schema_version": payload.get("schema_version"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
