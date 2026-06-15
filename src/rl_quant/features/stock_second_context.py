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
    "valid_price_flag",
    "execution_delay_seconds",
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
    default_action_cost_bps: float = 1.0
    max_action_cost_bps: float = 25.0
    rth_only: bool = True
    include_extended_hours: bool = False
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
    exchange_tz: ZoneInfo = EASTERN,
) -> list[int]:
    interval_seconds = parse_duration_seconds(decision_interval)
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(exchange_tz)
    end_dt = datetime.fromisoformat(end_exclusive.replace("Z", "+00:00")).astimezone(exchange_tz)
    decisions: list[int] = []
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        if current_date.weekday() < 5:
            session_start = datetime.combine(current_date, RTH_START, tzinfo=exchange_tz)
            session_end = datetime.combine(current_date, RTH_END, tzinfo=exchange_tz)
            decision = session_start + timedelta(seconds=interval_seconds)
            while decision + timedelta(seconds=interval_seconds) <= session_end:
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
    return {
        "return": max(min(last_close / first_close - 1.0, 1.0), -1.0),
        "abs_return": abs(last_close / first_close - 1.0),
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


def _action_type_flags(symbol: str) -> tuple[float, float]:
    if symbol.upper() in ETF_SYMBOLS:
        return 1.0, 0.0
    return 0.0, 1.0


def _estimate_cost_bps(dollar_volume: float, config: StockSecondContextConfig) -> float:
    if dollar_volume <= 0:
        return min(config.max_action_cost_bps, config.default_action_cost_bps * 5.0)
    liquidity_discount = min(math.log1p(dollar_volume) / math.log1p(100_000_000.0), 1.0)
    extra = (1.0 - liquidity_discount) * config.default_action_cost_bps * 4.0
    return min(config.max_action_cost_bps, config.default_action_cost_bps + extra)


def _minutes_to_close_scaled(decision_ms: int) -> float:
    _is_pre, _is_regular, _is_post, _since_open, seconds_to_close = _session_flags(decision_ms)
    return max(0.0, min(seconds_to_close / (6.5 * 3600.0), 1.0))


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
    action_valid_mask: list[list[bool]] = []
    action_cost_bps: list[list[float]] = []
    execution_latency_ms = config.execution_latency_ms
    entry_execution_timestamps_ms: list[list[int]] = []
    exit_execution_timestamps_ms: list[list[int]] = []
    action_count_denom = max(len(action_names) - 1, 1)
    for decision_ms, next_ms, context_mask in zip(decision_timestamps_ms, next_timestamps_ms, market_mask):
        decision_action_features: list[list[float]] = []
        decision_returns: list[float] = []
        decision_valid: list[bool] = []
        decision_costs: list[float] = []
        decision_entry_ts: list[int] = []
        decision_exit_ts: list[int] = []
        valid_context_fraction = float(context_mask.float().mean().item())
        for action_index, action in enumerate(action_names):
            symbol = action.upper()
            action_index_scaled = action_index / action_count_denom
            if symbol == "CASH":
                decision_action_features.append([action_index_scaled, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                decision_returns.append(0.0)
                decision_valid.append(True)
                decision_costs.append(0.0)
                decision_entry_ts.append(int(decision_ms))
                decision_exit_ts.append(int(next_ms))
                continue
            lookup = action_lookups.get(symbol)
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
            valid = current is not None and future is not None and valid_context_fraction > 0.0
            if current is None:
                execution_delay = float(config.max_action_staleness_seconds + 1)
                last_dv = 0.0
                cost = config.max_action_cost_bps
                entry_ts = -1
            else:
                _close, entry_ts, last_dv = current
                execution_delay = max(0.0, (entry_ts - (decision_ms + execution_latency_ms)) / 1000.0)
                cost = _estimate_cost_bps(last_dv, config)
            exit_ts = -1 if future is None else int(future[1])
            is_etf, is_stock = _action_type_flags(symbol)
            decision_action_features.append(
                [
                    action_index_scaled,
                    0.0,
                    is_etf,
                    is_stock,
                    float(valid),
                    execution_delay,
                    math.log1p(max(last_dv, 0.0)),
                    cost,
                ]
            )
            decision_costs.append(cost)
            decision_valid.append(bool(valid))
            decision_entry_ts.append(int(entry_ts))
            decision_exit_ts.append(int(exit_ts))
            if valid and current is not None and future is not None:
                decision_returns.append(max(min(future[0] / current[0] - 1.0, 1.0), -1.0))
            else:
                decision_returns.append(math.nan)
        action_features.append(decision_action_features)
        action_returns.append(decision_returns)
        action_valid_mask.append(decision_valid)
        action_cost_bps.append(decision_costs)
        entry_execution_timestamps_ms.append(decision_entry_ts)
        exit_execution_timestamps_ms.append(decision_exit_ts)

    action_valid = torch.tensor(action_valid_mask, dtype=torch.bool)
    quality_by_row = market_mask.float().mean(dim=1).clamp(0.0, 1.0)
    valid_action_fraction = action_valid.float().mean(dim=1).clamp(0.0, 1.0)
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
    base_manifest = dict(dataset_manifest or {})
    report = dict(data_quality_report or {})
    source_download_complete = bool(
        base_manifest.get("source_download_complete", report.get("source_download_complete", False))
    )
    reportability_errors = list(report.get("reportability_errors", []))
    if not source_download_complete:
        reportability_errors.append("source_download_incomplete")
    if not report:
        reportability_errors.append("data_quality_report_missing")
    if config.min_active_symbols < max(1, len(stock_frames_by_symbol) // 2):
        reportability_errors.append("min_active_symbols_too_low_for_reportable_dataset")
    manifest = {
        **base_manifest,
        "schema_version": "stock_second_context_decision_v2",
        "feature_set_id": config.feature_set_id,
        "created_at_utc": utc_now_iso(),
        "source_bar_interval": config.source_bar_interval,
        "decision_interval": config.decision_interval,
        "context_seconds": config.context_seconds,
        "block_seconds": config.block_seconds,
        "lookback_blocks": config.lookback_blocks,
        "bar_latency_ms": config.bar_latency_ms,
        "ingestion_latency_ms": config.ingestion_latency_ms,
        "execution_latency_ms": config.execution_latency_ms,
        "source_download_complete": source_download_complete,
        "reportable": source_download_complete and not reportability_errors,
        "reportability_errors": list(dict.fromkeys(reportability_errors)),
        "known_limitations": [
            *config.known_limitations,
            "Top-stock second bars provide market context; action returns require separate tradable action bars.",
            "Sparse seconds are preserved through masks and active-second features.",
            "Action-return labels use first available action bar at or after the decision and reward timestamps.",
        ],
    }
    payload = {
        "schema_version": "stock_second_context_decision_v2",
        "decision_timestamps": [timestamp_ms_to_iso(value) for value in decision_timestamps_ms],
        "next_timestamps": [timestamp_ms_to_iso(value) for value in next_timestamps_ms],
        "decision_timestamps_ms": torch.tensor(decision_timestamps_ms, dtype=torch.long),
        "next_timestamps_ms": torch.tensor(next_timestamps_ms, dtype=torch.long),
        "market_context": market_context,
        "market_context_mask": market_mask,
        "market_context_available_timestamps_ms": available_ms,
        "action_features": torch.tensor(action_features, dtype=torch.float32),
        "action_returns": torch.tensor(action_returns, dtype=torch.float32),
        "action_valid_mask": action_valid,
        "action_cost_bps": torch.tensor(action_cost_bps, dtype=torch.float32),
        "entry_execution_timestamps_ms": torch.tensor(entry_execution_timestamps_ms, dtype=torch.long),
        "exit_execution_timestamps_ms": torch.tensor(exit_execution_timestamps_ms, dtype=torch.long),
        "entry_price_source": "first_action_close_at_or_after_decision_plus_execution_latency",
        "exit_price_source": "first_action_close_at_or_after_reward_end_plus_execution_latency",
        "portfolio_state": portfolio_state,
        "constraint_state": constraint_state,
        "feature_names": {
            "market_context": list(MARKET_CONTEXT_FEATURE_NAMES),
            "action_features": list(ACTION_FEATURE_NAMES),
            "portfolio_state": list(PORTFOLIO_STATE_FEATURE_NAMES),
            "constraint_state": list(CONSTRAINT_STATE_FEATURE_NAMES),
        },
        "action_names": list(action_names),
        "dataset_manifest": manifest,
        "data_quality_report": report,
        "config": config.to_dict(),
        "payload_hash": stable_json_hash(
            {
                "decision_timestamps": [timestamp_ms_to_iso(value) for value in decision_timestamps_ms],
                "action_names": list(action_names),
                "config": config.to_dict(),
            }
        ),
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
    action_costs = payload["action_cost_bps"].float()
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
    if tuple(action_returns.shape) != tuple(action_valid.shape):
        raise ValueError("action_valid_mask shape must match action_returns.")
    if tuple(action_costs.shape) != tuple(action_returns.shape):
        raise ValueError("action_cost_bps shape must match action_returns.")
    if tuple(entry_execution.shape) != tuple(action_returns.shape):
        raise ValueError("entry_execution_timestamps_ms shape must match action_returns.")
    if tuple(exit_execution.shape) != tuple(action_returns.shape):
        raise ValueError("exit_execution_timestamps_ms shape must match action_returns.")
    if action_features.shape[:2] != action_returns.shape:
        raise ValueError("action_features first two dimensions must match action_returns.")
    if bool((action_costs < 0).any().item()):
        raise ValueError("action_cost_bps must be non-negative.")
    if not bool(action_valid[:, 0].all().item()):
        raise ValueError("CASH action must be valid for every row.")
    if not bool(torch.isfinite(action_returns[action_valid]).all().item()):
        raise ValueError("Valid action_returns must be finite.")
    if bool((action_returns[:, 0].abs() > 1e-12).any().item()):
        raise ValueError("CASH action return must be zero.")
    invalid_returns = action_returns[~action_valid]
    if invalid_returns.numel() and not bool(torch.isnan(invalid_returns).all().item()):
        raise ValueError("Invalid action_returns must be NaN.")
    if abs(float(action_costs[:, 0].abs().max().item())) > 1e-12:
        raise ValueError("CASH action cost must be zero.")
    non_cash_valid = action_valid.clone()
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
    if any(not name for name in payload["action_names"]):
        raise ValueError("action_names must be non-empty strings.")
    if str(payload["action_names"][0]).upper() != "CASH":
        raise ValueError("action_names must start with CASH.")
    manifest = payload["dataset_manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset_manifest must be a dictionary.")
    if manifest.get("source_download_complete") is False and manifest.get("reportable") is True:
        raise ValueError("Incomplete source downloads cannot produce reportable datasets.")


def save_second_context_payload(payload: Mapping[str, Any], path: Path) -> None:
    validate_second_context_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    manifest = payload.get("dataset_manifest", {})
    report = payload.get("data_quality_report", {})
    (path.parent / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (path.parent / "data_quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    (path.parent / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_set_id": manifest.get("feature_set_id", "stock_second_context_v001"),
                "feature_names": payload.get("feature_names", {}),
                "created_at_utc": utc_now_iso(),
                "schema_version": payload.get("schema_version"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
