#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import csv
import json
import math
import statistics
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_hourly_transformer_dataset import (  # noqa: E402
    BarFeature,
    aggregate_stock_features,
    bar_file_map,
    clipped_simple_return,
    interval_minutes,
    load_exchange_times,
    load_symbol_features,
    parse_exchange_time,
    read_ranked_symbols,
    resolve_universe_selection_date,
)
from rl_quant.research_protocol import (  # noqa: E402
    DatasetManifest,
    hash_string_sequence,
    stable_json_hash,
    utc_now_iso,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


def default_derived_root() -> Path:
    shared_derived = PROJECT_ROOT.parent / "derived"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_derived.exists():
        return shared_derived
    return PROJECT_ROOT / "derived"


DATA_ROOT = default_data_root()
DERIVED_ROOT = default_derived_root()
DEFAULT_SOURCE_BAR_INTERVAL = "1m"
SECOND_SOURCE_BAR_INTERVAL = "1s"
SOURCE_BAR_INTERVAL = DEFAULT_SOURCE_BAR_INTERVAL
DEFAULT_DECISION_GRID_MINUTES = 60
DEFAULT_CONTEXT_MINUTES_PER_GRID = 60
DEFAULT_SECOND_CONTEXT_BARS_PER_GRID = 3600
DEFAULT_SECOND_BAR_LATENCY_MS = 1000
DEFAULT_DECISION_GRID_NAME = "hour"
POLYGON_SECOND_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"
POLYGON_TOP500_UNIVERSE = DATA_ROOT / "polygon" / "universes" / "top_500_s3_volume_common_stocks_2026-06-12.csv"


def validate_hourly_grid_args(args: argparse.Namespace) -> None:
    if args.decision_stride_minutes != DEFAULT_DECISION_GRID_MINUTES:
        raise ValueError("Subhour-source RL datasets use an hourly decision grid; decision-stride-minutes must be 60.")
    expected = expected_context_bars_per_grid(args.source_bar_interval)
    if args.minutes_per_hour != expected:
        raise ValueError(
            "Subhour-source hourly context must encode one hour of source bars; "
            f"{args.source_bar_interval} expects {expected} bars per hour."
        )


def source_interval_seconds(interval: str) -> int:
    text = interval.strip().lower()
    if text.endswith("s"):
        value = int(text[:-1])
    elif text.endswith("m"):
        value = int(text[:-1]) * 60
    else:
        raise ValueError(f"Unsupported source bar interval {interval!r}; expected values like 1s or 1m.")
    if value <= 0:
        raise ValueError("source bar interval must be positive.")
    return value


def expected_context_bars_per_grid(source_bar_interval: str) -> int:
    seconds = source_interval_seconds(source_bar_interval)
    hour_seconds = DEFAULT_DECISION_GRID_MINUTES * 60
    if hour_seconds % seconds != 0:
        raise ValueError("source bar interval must divide one hourly decision grid exactly.")
    return hour_seconds // seconds


def default_stock_bar_dir(source_bar_interval: str) -> Path:
    if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL:
        return POLYGON_SECOND_ROOT
    return DATA_ROOT / "minute_ohlcv" / "top_us_volume_stocks_nasdaq_1000_2026-06-14_1m_2026-05-25_2026-06-15"


def default_action_bar_dir(source_bar_interval: str) -> Path:
    if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL:
        return POLYGON_SECOND_ROOT
    return DATA_ROOT / "minute_ohlcv" / "top_us_volume_etfs_500_2026-06-14_1m_2026-05-25_2026-06-15"


def default_stock_universe(source_bar_interval: str) -> Path:
    if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL:
        return POLYGON_TOP500_UNIVERSE
    return DERIVED_ROOT / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv"


def default_action_universe(source_bar_interval: str) -> Path:
    if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL:
        return POLYGON_TOP500_UNIVERSE
    return DERIVED_ROOT / "universes" / "top_us_volume_etfs_500_2026-06-14.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an hourly-decision dataset with causal subhour context windows.",
    )
    parser.add_argument(
        "--source-bar-interval",
        default=DEFAULT_SOURCE_BAR_INTERVAL,
        help="Source bar spacing inside each hourly context window, for example 1m or 1s.",
    )
    parser.add_argument(
        "--stock-minute-dir",
        "--stock-bar-dir",
        type=Path,
        dest="stock_minute_dir",
        default=default_stock_bar_dir(DEFAULT_SOURCE_BAR_INTERVAL),
    )
    parser.add_argument(
        "--etf-minute-dir",
        "--action-bar-dir",
        type=Path,
        dest="etf_minute_dir",
        default=default_action_bar_dir(DEFAULT_SOURCE_BAR_INTERVAL),
    )
    parser.add_argument(
        "--stock-universe",
        type=Path,
        default=default_stock_universe(DEFAULT_SOURCE_BAR_INTERVAL),
    )
    parser.add_argument(
        "--etf-universe",
        "--action-universe",
        type=Path,
        dest="etf_universe",
        default=default_action_universe(DEFAULT_SOURCE_BAR_INTERVAL),
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "rl_hour_from_minute" / "top_volume_1m_recent")
    parser.add_argument("--dataset-file-name", default="hour_from_minute_dataset.pt")
    parser.add_argument("--start", default="2026-05-25T00:00:00+00:00")
    parser.add_argument("--end-exclusive", default="2026-06-15T00:00:00+00:00")
    parser.add_argument("--stock-limit", type=int, default=1000)
    parser.add_argument("--action-count", type=int, default=16)
    parser.add_argument("--actions", help="Comma-separated ETF action symbols. CASH is added automatically.")
    parser.add_argument(
        "--universe-selection-date",
        help="Optional ISO timestamp/date proving the universe was selected before the dataset starts.",
    )
    parser.add_argument("--min-active-stock-fraction", type=float, default=0.30)
    parser.add_argument("--hours-lookback", type=int, default=4)
    parser.add_argument(
        "--minutes-per-hour",
        "--context-bars-per-hour",
        dest="minutes_per_hour",
        type=int,
        default=DEFAULT_CONTEXT_MINUTES_PER_GRID,
        help="Number of source bars inside each hour context token; use 60 for 1m or 3600 for 1s.",
    )
    parser.add_argument(
        "--decision-stride-minutes",
        "--decision-grid-minutes",
        dest="decision_stride_minutes",
        type=int,
        default=DEFAULT_DECISION_GRID_MINUTES,
        help="Decision-grid spacing in minutes; fixed at 60 so minute data is consumed on an hourly grid.",
    )
    parser.add_argument("--min-context-valid-fraction", type=float, default=0.50)
    parser.add_argument(
        "--max-action-staleness-seconds",
        type=int,
        default=0,
        help="Use the latest action close at or before a decision within this many seconds; useful for sparse 1s bars.",
    )
    parser.add_argument(
        "--bar-latency-ms",
        type=int,
        default=0,
        help="Aggregate bar availability latency. Use 1000 for Polygon one-second bars.",
    )
    parser.add_argument(
        "--dense-hourly-grid",
        action="store_true",
        help="Generate hourly decision timestamps from exchange sessions instead of requiring exact source rows.",
    )
    parser.add_argument(
        "--allow-missing-action-context",
        action="store_true",
        help="Fill missing per-action context features with zeros instead of dropping that source timestamp.",
    )
    args = parser.parse_args(argv)
    if args.source_bar_interval == SECOND_SOURCE_BAR_INTERVAL:
        if args.stock_minute_dir == default_stock_bar_dir(DEFAULT_SOURCE_BAR_INTERVAL):
            args.stock_minute_dir = default_stock_bar_dir(SECOND_SOURCE_BAR_INTERVAL)
        if args.etf_minute_dir == default_action_bar_dir(DEFAULT_SOURCE_BAR_INTERVAL):
            args.etf_minute_dir = default_action_bar_dir(SECOND_SOURCE_BAR_INTERVAL)
        if args.stock_universe == default_stock_universe(DEFAULT_SOURCE_BAR_INTERVAL):
            args.stock_universe = default_stock_universe(SECOND_SOURCE_BAR_INTERVAL)
        if args.etf_universe == default_action_universe(DEFAULT_SOURCE_BAR_INTERVAL):
            args.etf_universe = default_action_universe(SECOND_SOURCE_BAR_INTERVAL)
        if args.output_dir == DATA_ROOT / "rl_hour_from_minute" / "top_volume_1m_recent":
            args.output_dir = DATA_ROOT / "rl_hour_from_second" / "top500_1s_recent"
        if args.dataset_file_name == "hour_from_minute_dataset.pt":
            args.dataset_file_name = "hour_from_second_dataset.pt"
        if args.minutes_per_hour == DEFAULT_CONTEXT_MINUTES_PER_GRID:
            args.minutes_per_hour = DEFAULT_SECOND_CONTEXT_BARS_PER_GRID
        if args.max_action_staleness_seconds == 0:
            args.max_action_staleness_seconds = 300
        if args.bar_latency_ms == 0:
            args.bar_latency_ms = DEFAULT_SECOND_BAR_LATENCY_MS
        args.dense_hourly_grid = True
        args.allow_missing_action_context = True
        args.min_context_valid_fraction = min(float(args.min_context_valid_fraction), 0.01)
    return args


def timestamp_add_minutes(value: str, minutes: int) -> str:
    return (datetime.fromisoformat(value) + timedelta(minutes=minutes)).isoformat()


def timestamp_add_seconds(value: str, seconds: int) -> str:
    return (datetime.fromisoformat(value) + timedelta(seconds=seconds)).isoformat()


def timestamp_add_milliseconds(value: str, milliseconds: int) -> str:
    return (datetime.fromisoformat(value) + timedelta(milliseconds=milliseconds)).isoformat()


def utc_iso_seconds(value: object) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def session_minutes(exchange_timestamp: str) -> int:
    dt = datetime.fromisoformat(exchange_timestamp)
    return dt.hour * 60 + dt.minute - (9 * 60 + 30)


def session_elapsed_seconds(exchange_timestamp: str) -> int:
    dt = datetime.fromisoformat(exchange_timestamp)
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) - (9 * 3600 + 30 * 60)


def parquet_symbol_from_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) >= 4:
        return relative.parts[0].upper()
    return path.stem.upper()


def bar_source_map(directory: Path, *, interval: str) -> dict[str, Path | list[Path]]:
    if interval.endswith("m"):
        flat = bar_file_map(directory, interval=interval)
        if flat:
            return flat
    out: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.parquet")):
        out.setdefault(path.stem.upper(), []).append(path)
    for path in sorted(directory.glob("*/*/*/*.parquet")):
        out.setdefault(parquet_symbol_from_path(directory, path), []).append(path)
    return {symbol: paths[0] if len(paths) == 1 else paths for symbol, paths in out.items()}


def load_parquet_bar_features(paths: list[Path], *, start: str, end_exclusive: str) -> dict[str, object]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas/pyarrow are required to read Polygon second-bar Parquet files in conda env ml1.") from exc

    start_dt = datetime.fromisoformat(start).astimezone(timezone.utc)
    end_dt = datetime.fromisoformat(end_exclusive).astimezone(timezone.utc)
    raw_rows: list[tuple[str, str, float, float, float, float, float]] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        if "timestamp_ms" in frame.columns:
            timestamps = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
        else:
            timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
        keep = (timestamps >= start_dt) & (timestamps < end_dt)
        if not bool(keep.any()):
            continue
        selected = frame.loc[keep].copy()
        normalized_timestamps = [value.to_pydatetime() for value in timestamps[keep]]
        for ts_value, row in zip(normalized_timestamps, selected.to_dict("records")):
            ts = utc_iso_seconds(ts_value)
            close = float(row["close"])
            open_value = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            volume = float(row["volume"])
            if close <= 0 or open_value <= 0:
                continue
            raw_rows.append((ts, str(row["timestamp_exchange"]), open_value, high, low, close, max(volume, 0.0)))
    raw_rows.sort(key=lambda item: item[0])

    out: dict[str, object] = {}
    previous_close: float | None = None
    for ts, exchange_ts, open_value, high, low, close, volume in raw_rows:
        if previous_close is None or previous_close <= 0:
            bar_return = 0.0
            bar_log_return = 0.0
        else:
            bar_return = close / previous_close - 1.0
            bar_log_return = math.log(close / previous_close)
        intraday = math.log(close / open_value) if open_value > 0 else 0.0
        scale = max(close, 1e-8)
        dollar_volume = close * volume
        out[ts] = BarFeature(
            close=close,
            bar_return=max(min(bar_return, 1.0), -1.0),
            bar_log_return=max(min(bar_log_return, 1.0), -1.0),
            intraday_ret=max(min(intraday, 1.0), -1.0),
            range_bps=max((high - low) / scale * 10_000.0, 0.0),
            log_volume=math.log1p(volume),
            log_dollar_volume=math.log1p(dollar_volume),
            dollar_volume=max(dollar_volume, 0.0),
        )
        out[f"__exchange__:{ts}"] = exchange_ts
        previous_close = close
    return out


def load_bar_features(source: Path | list[Path], *, start: str, end_exclusive: str) -> dict[str, object]:
    paths = source if isinstance(source, list) else [source]
    if paths and paths[0].suffix == ".parquet":
        return load_parquet_bar_features(paths, start=start, end_exclusive=end_exclusive)
    return load_symbol_features(paths[0], start=start, end_exclusive=end_exclusive)


def split_features_and_exchange(rows: dict[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    features = {key: value for key, value in rows.items() if not key.startswith("__exchange__:")}
    exchange = {key.removeprefix("__exchange__:"): str(value) for key, value in rows.items() if key.startswith("__exchange__:")}
    return features, exchange


def load_bar_exchange_times(source: Path | list[Path], *, start: str, end_exclusive: str) -> dict[str, str]:
    paths = source if isinstance(source, list) else [source]
    if paths and paths[0].suffix == ".parquet":
        _, exchange = split_features_and_exchange(load_parquet_bar_features(paths, start=start, end_exclusive=end_exclusive))
        return exchange
    return load_exchange_times(paths[0], start=start, end_exclusive=end_exclusive)


def make_action_lookup(features: dict[str, object]) -> tuple[list[str], list[float]]:
    timestamps = sorted(features)
    closes = [float(features[timestamp].close) for timestamp in timestamps]
    return timestamps, closes


def close_at_or_before(
    lookup: tuple[list[str], list[float]],
    timestamp: str,
    *,
    max_staleness_seconds: int,
) -> float | None:
    timestamps, closes = lookup
    pos = bisect.bisect_right(timestamps, timestamp) - 1
    if pos < 0:
        return None
    if max_staleness_seconds > 0:
        age = (datetime.fromisoformat(timestamp) - datetime.fromisoformat(timestamps[pos])).total_seconds()
        if age < 0 or age > max_staleness_seconds:
            return None
    elif timestamps[pos] != timestamp:
        return None
    return closes[pos]


def build_dense_hourly_decision_grid(exchange_times: dict[str, str], *, start: str, end_exclusive: str) -> list[str]:
    start_dt = datetime.fromisoformat(start).astimezone(timezone.utc)
    end_dt = datetime.fromisoformat(end_exclusive).astimezone(timezone.utc)
    date_tz: dict[str, timezone] = {}
    for exchange_timestamp in exchange_times.values():
        dt = datetime.fromisoformat(exchange_timestamp)
        if dt.tzinfo is not None:
            date_tz.setdefault(dt.date().isoformat(), dt.tzinfo)
    decisions: list[str] = []
    for date_text, tzinfo in sorted(date_tz.items()):
        session_start = datetime.combine(datetime.fromisoformat(date_text).date(), time(9, 30), tzinfo=tzinfo)
        session_end = datetime.combine(datetime.fromisoformat(date_text).date(), time(16, 0), tzinfo=tzinfo)
        decision = session_start + timedelta(minutes=DEFAULT_DECISION_GRID_MINUTES)
        while decision + timedelta(minutes=DEFAULT_DECISION_GRID_MINUTES) <= session_end:
            utc_decision = decision.astimezone(timezone.utc)
            if start_dt <= utc_decision < end_dt:
                decisions.append(utc_decision.replace(microsecond=0).isoformat())
            decision += timedelta(minutes=DEFAULT_DECISION_GRID_MINUTES)
    return decisions


def infer_exchange_timestamp(timestamp_utc: str, exchange_times: dict[str, str]) -> str | None:
    if timestamp_utc in exchange_times:
        return exchange_times[timestamp_utc]
    timestamp_dt = datetime.fromisoformat(timestamp_utc).astimezone(timezone.utc)
    date_tz: dict[str, timezone] = {}
    for exchange_timestamp in exchange_times.values():
        exchange_dt = datetime.fromisoformat(exchange_timestamp)
        if exchange_dt.tzinfo is not None:
            date_tz.setdefault(exchange_dt.date().isoformat(), exchange_dt.tzinfo)
    for date_text, tzinfo in date_tz.items():
        local_dt = timestamp_dt.astimezone(tzinfo)
        if local_dt.date().isoformat() == date_text:
            return local_dt.isoformat()
    return None


def infer_periods_per_year(decision_timestamps: list[str]) -> float:
    if not decision_timestamps:
        raise ValueError("decision_timestamps must not be empty.")
    counts = Counter(timestamp[:10] for timestamp in decision_timestamps)
    return 252.0 * float(statistics.median(counts.values()))


def write_action_returns(path: Path, action_names: list[str], timestamps: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["DecisionTimestamp", *action_names])
        for timestamp, values in zip(timestamps, rows):
            writer.writerow([timestamp, *[f"{value:.10f}" for value in values]])


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required. Use: conda run -n ml1 python scripts/build_hourly_from_minute_context_dataset.py") from exc

    source_bar_interval = args.source_bar_interval.strip().lower()
    source_bar_seconds = source_interval_seconds(source_bar_interval)
    if args.hours_lookback <= 0 or args.minutes_per_hour <= 0 or args.decision_stride_minutes <= 0:
        raise ValueError("hours-lookback, context-bars-per-hour, and decision-stride-minutes must be positive.")
    if args.bar_latency_ms < 0:
        raise ValueError("bar-latency-ms must be non-negative.")
    if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL and args.bar_latency_ms < DEFAULT_SECOND_BAR_LATENCY_MS:
        raise ValueError("Polygon one-second aggregate bars require bar-latency-ms >= 1000.")
    validate_hourly_grid_args(args)
    if interval_minutes("1m") != 1.0:
        raise ValueError("Internal interval check failed.")
    universe_selection_date = resolve_universe_selection_date(args)
    sparse_source = source_bar_seconds < 60
    dense_hourly_grid = bool(args.dense_hourly_grid or sparse_source)
    allow_missing_action_context = bool(args.allow_missing_action_context or sparse_source)

    stock_map = bar_source_map(args.stock_minute_dir, interval=source_bar_interval)
    etf_map = bar_source_map(args.etf_minute_dir, interval=source_bar_interval)
    ranked_stocks = [symbol for symbol in read_ranked_symbols(args.stock_universe) if symbol in stock_map]
    selected_stocks = ranked_stocks[: args.stock_limit]
    if not selected_stocks:
        raise ValueError("No stock source-bar files matched the selected universe.")

    if args.actions:
        etf_symbols = [symbol.strip().upper() for symbol in args.actions.split(",") if symbol.strip()]
    else:
        etf_symbols = [symbol for symbol in read_ranked_symbols(args.etf_universe) if symbol in etf_map][: args.action_count]
    etf_symbols = list(dict.fromkeys(symbol for symbol in etf_symbols if symbol in etf_map))
    if not etf_symbols:
        raise ValueError("No action source-bar files matched the selected universe.")
    action_names = ["CASH", *etf_symbols]

    print(f"Loading {len(selected_stocks)} stock {source_bar_interval} files for causal context...")
    stock_by_time = {}
    stock_exchange_times: dict[str, str] = {}
    for index, symbol in enumerate(selected_stocks, 1):
        rows, exchange = split_features_and_exchange(
            load_bar_features(stock_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        )
        for timestamp, feature in rows.items():
            stock_by_time.setdefault(timestamp, []).append(feature)
            if timestamp in exchange:
                stock_exchange_times.setdefault(timestamp, exchange[timestamp])
        if index % 100 == 0:
            print(f"  loaded {index}/{len(selected_stocks)} stocks", flush=True)

    print(f"Loading {len(etf_symbols)} action {source_bar_interval} series...")
    etf_features = {}
    etf_exchange_times = {}
    for symbol in etf_symbols:
        rows, exchange = split_features_and_exchange(
            load_bar_features(etf_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        )
        etf_features[symbol] = rows
        etf_exchange_times[symbol] = exchange
    exchange_times = etf_exchange_times[etf_symbols[0]]
    exact_action_common_times = sorted(set.intersection(*(set(rows) for rows in etf_features.values())))
    common_times = sorted(stock_by_time)
    min_active = max(1, int(len(selected_stocks) * args.min_active_stock_fraction))
    common_times = [
        timestamp
        for timestamp in common_times
        if len(stock_by_time.get(timestamp, [])) >= min_active
        and (timestamp in exchange_times or dense_hourly_grid or timestamp in exact_action_common_times)
    ]
    common_set = set(common_times)
    if len(common_times) < args.minutes_per_hour:
        raise ValueError("Too few aligned source rows after stock and action filtering.")
    action_price_lookup = {symbol: make_action_lookup(rows) for symbol, rows in etf_features.items()}

    stock_feature_names = [
        "stock_active_fraction",
        "stock_ret_ew",
        "stock_ret_dv",
        "stock_ret_std",
        "stock_up_fraction",
        "stock_ret_top_decile",
        "stock_ret_bottom_decile",
        "stock_intraday_ew",
        "stock_range_bps_ew",
        "stock_range_bps_std",
        "stock_log_dollar_volume_mean",
        "stock_log_dollar_volume_std",
        "stock_dollar_volume_concentration",
        "stock_abs_ret_dv",
    ]
    path_feature_names = [
        "source_cumulative_stock_ret_ew",
        "source_realized_stock_vol_so_far",
        "source_avg_log_dollar_volume_so_far",
    ]
    time_feature_names = [
        "session_progress_centered",
        "session_sin",
        "session_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    etf_feature_names: list[str] = []
    for symbol in etf_symbols:
        etf_feature_names.extend(
            [
                f"etf_{symbol}_ret_{source_bar_interval}",
                f"etf_{symbol}_intraday_ret",
                f"etf_{symbol}_range_bps",
                f"etf_{symbol}_log_dollar_volume",
            ]
        )
    minute_feature_names = [*stock_feature_names, *path_feature_names, *time_feature_names, *etf_feature_names]
    hour_feature_names = ["hour_valid_fraction", "hour_session_progress_centered"]

    minute_feature_by_time: dict[str, list[float]] = {}
    cumulative_by_date: dict[str, float] = {}
    returns_by_date: dict[str, list[float]] = {}
    volume_by_date: dict[str, float] = {}
    for timestamp in common_times:
        stock_features = aggregate_stock_features(stock_by_time[timestamp], total_symbols=len(selected_stocks))
        exchange_timestamp = stock_exchange_times.get(timestamp) or exchange_times.get(timestamp)
        if exchange_timestamp is None:
            continue
        date_key = exchange_timestamp[:10]
        stock_ret = stock_features[1]
        cumulative_by_date[date_key] = cumulative_by_date.get(date_key, 0.0) + stock_ret
        returns = returns_by_date.setdefault(date_key, [])
        returns.append(stock_ret)
        volume_by_date[date_key] = volume_by_date.get(date_key, 0.0) + stock_features[10]
        avg = sum(returns) / len(returns)
        realized_vol = (sum((value - avg) ** 2 for value in returns) / max(len(returns), 1)) ** 0.5
        avg_log_dollar_volume_so_far = min(volume_by_date[date_key] / max(len(returns), 1.0), 1e6)
        path_features = [cumulative_by_date[date_key], realized_vol, avg_log_dollar_volume_so_far]
        time_features = list(parse_exchange_time(exchange_timestamp))
        etf_row: list[float] = []
        missing = False
        for symbol in etf_symbols:
            current = etf_features[symbol].get(timestamp)
            if current is None:
                if allow_missing_action_context:
                    etf_row.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                missing = True
                break
            etf_row.extend([current.bar_return, current.intraday_ret, current.range_bps, current.log_dollar_volume])
        if not missing:
            minute_feature_by_time[timestamp] = [*stock_features, *path_features, *time_features, *etf_row]

    decision_timestamps: list[str] = []
    next_timestamps: list[str] = []
    minute_timestamp_grid: list[list[list[str]]] = []
    minute_feature_rows: list[list[list[list[float]]]] = []
    minute_mask_rows: list[list[list[bool]]] = []
    hour_feature_rows: list[list[list[float]]] = []
    action_return_rows: list[list[float]] = []
    feature_dim = len(minute_feature_names)
    decision_source_times = (
        build_dense_hourly_decision_grid(exchange_times, start=args.start, end_exclusive=args.end_exclusive)
        if dense_hourly_grid
        else common_times
    )
    decision_stride_seconds = int(args.decision_stride_minutes) * 60
    for decision_ts in decision_source_times:
        exchange_timestamp = (
            exchange_times.get(decision_ts)
            or stock_exchange_times.get(decision_ts)
            or infer_exchange_timestamp(decision_ts, exchange_times)
        )
        if exchange_timestamp is None:
            continue
        elapsed = session_elapsed_seconds(exchange_timestamp)
        if elapsed <= 0 or elapsed % decision_stride_seconds != 0:
            continue
        next_ts = timestamp_add_seconds(decision_ts, decision_stride_seconds)
        decision_context_ts = timestamp_add_milliseconds(decision_ts, -int(args.bar_latency_ms))
        next_context_ts = timestamp_add_milliseconds(next_ts, -int(args.bar_latency_ms))
        if not dense_hourly_grid and next_ts not in common_set:
            continue
        decision_date = exchange_timestamp[:10]
        minute_tensor: list[list[list[float]]] = []
        mask_tensor: list[list[bool]] = []
        timestamp_tensor: list[list[str]] = []
        hour_rows: list[list[float]] = []
        valid_count = 0
        decision_dt = datetime.fromisoformat(decision_context_ts)
        for hour_index in range(args.hours_lookback):
            offset_hours = args.hours_lookback - 1 - hour_index
            hour_end = decision_dt - timedelta(seconds=offset_hours * decision_stride_seconds)
            hour_start = hour_end - timedelta(seconds=(args.minutes_per_hour - 1) * source_bar_seconds)
            minute_rows: list[list[float]] = []
            mask_rows: list[bool] = []
            ts_rows: list[str] = []
            for minute_offset in range(args.minutes_per_hour):
                minute_ts = (hour_start + timedelta(seconds=minute_offset * source_bar_seconds)).isoformat()
                minute_exchange_timestamp = stock_exchange_times.get(minute_ts) or exchange_times.get(minute_ts)
                is_valid = (
                    minute_ts <= decision_context_ts
                    and minute_ts in minute_feature_by_time
                    and minute_exchange_timestamp is not None
                    and minute_exchange_timestamp[:10] == decision_date
                )
                ts_rows.append(minute_ts if is_valid else "")
                mask_rows.append(is_valid)
                minute_rows.append(minute_feature_by_time[minute_ts] if is_valid else [0.0] * feature_dim)
                valid_count += int(is_valid)
            valid_fraction = sum(mask_rows) / float(args.minutes_per_hour)
            hour_exchange_timestamp = (
                exchange_times.get(hour_end.isoformat())
                or stock_exchange_times.get(hour_end.isoformat())
                or infer_exchange_timestamp(hour_end.isoformat(), exchange_times)
                or exchange_timestamp
            )
            hour_rows.append([valid_fraction, parse_exchange_time(hour_exchange_timestamp)[0]])
            timestamp_tensor.append(ts_rows)
            mask_tensor.append(mask_rows)
            minute_tensor.append(minute_rows)
        if valid_count / float(args.hours_lookback * args.minutes_per_hour) < args.min_context_valid_fraction:
            continue
        action_returns = [0.0]
        missing_action = False
        for symbol in etf_symbols:
            current_close = close_at_or_before(
                action_price_lookup[symbol],
                decision_context_ts,
                max_staleness_seconds=args.max_action_staleness_seconds,
            )
            future_close = close_at_or_before(
                action_price_lookup[symbol],
                next_context_ts,
                max_staleness_seconds=args.max_action_staleness_seconds,
            )
            if current_close is None or future_close is None:
                missing_action = True
                break
            action_returns.append(clipped_simple_return(current_close, future_close))
        if missing_action:
            continue
        decision_timestamps.append(decision_ts)
        next_timestamps.append(next_ts)
        minute_timestamp_grid.append(timestamp_tensor)
        minute_feature_rows.append(minute_tensor)
        minute_mask_rows.append(mask_tensor)
        hour_feature_rows.append(hour_rows)
        action_return_rows.append(action_returns)

    if len(decision_timestamps) < 10:
        raise ValueError("Too few hourly decision rows after context/reward filtering.")

    minute_features = torch.tensor(minute_feature_rows, dtype=torch.float32)
    minute_mask = torch.tensor(minute_mask_rows, dtype=torch.bool)
    hour_features = torch.tensor(hour_feature_rows, dtype=torch.float32)
    action_returns = torch.tensor(action_return_rows, dtype=torch.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / args.dataset_file_name
    periods_per_year = infer_periods_per_year(decision_timestamps)
    median_decisions_per_day = periods_per_year / 252.0
    torch.save(
        {
            "decision_timestamps": decision_timestamps,
            "next_timestamps": next_timestamps,
            "minute_timestamp_grid": minute_timestamp_grid,
            "minute_feature_names": minute_feature_names,
            "hour_feature_names": hour_feature_names,
            "action_names": action_names,
            "minute_features": minute_features,
            "minute_mask": minute_mask,
            "hour_features": hour_features,
            "action_returns": action_returns,
            "hours_lookback": args.hours_lookback,
            "minutes_per_hour": args.minutes_per_hour,
            "context_bars_per_hour": args.minutes_per_hour,
            "source_bar_interval": source_bar_interval,
            "source_bar_seconds": source_bar_seconds,
            "bar_latency_ms": int(args.bar_latency_ms),
            "decision_grid": DEFAULT_DECISION_GRID_NAME,
            "decision_grid_minutes": DEFAULT_DECISION_GRID_MINUTES,
            "decision_stride_minutes": args.decision_stride_minutes,
            "periods_per_year": periods_per_year,
            "periods_per_year_formula": "252 * median_decisions_per_utc_day",
            "median_decisions_per_day": median_decisions_per_day,
            "source": {
                "stock_minute_dir": str(args.stock_minute_dir),
                "etf_minute_dir": str(args.etf_minute_dir),
                "stock_universe": str(args.stock_universe),
                "etf_universe": str(args.etf_universe),
                "stock_limit": len(selected_stocks),
                "action_symbols": etf_symbols,
                "min_active_stock_fraction": args.min_active_stock_fraction,
                "min_context_valid_fraction": args.min_context_valid_fraction,
                "max_action_staleness_seconds": args.max_action_staleness_seconds,
                "bar_latency_ms": int(args.bar_latency_ms),
                "dense_hourly_grid": dense_hourly_grid,
                "allow_missing_action_context": allow_missing_action_context,
                "start": args.start,
                "end_exclusive": args.end_exclusive,
                "universe_selection_date": universe_selection_date,
            },
        },
        dataset_path,
    )
    write_action_returns(args.output_dir / "action_returns.csv", action_names, decision_timestamps, action_return_rows)
    source_metadata = {
        "stock_minute_dir": str(args.stock_minute_dir),
        "etf_minute_dir": str(args.etf_minute_dir),
        "stock_universe": str(args.stock_universe),
        "etf_universe": str(args.etf_universe),
        "stock_limit": len(selected_stocks),
        "action_symbols": etf_symbols,
        "min_active_stock_fraction": args.min_active_stock_fraction,
        "min_context_valid_fraction": args.min_context_valid_fraction,
        "max_action_staleness_seconds": args.max_action_staleness_seconds,
        "bar_latency_ms": int(args.bar_latency_ms),
        "dense_hourly_grid": dense_hourly_grid,
        "allow_missing_action_context": allow_missing_action_context,
        "hours_lookback": args.hours_lookback,
        "minutes_per_hour": args.minutes_per_hour,
        "context_bars_per_hour": args.minutes_per_hour,
        "source_bar_interval": source_bar_interval,
        "source_bar_seconds": source_bar_seconds,
        "bar_latency_ms": int(args.bar_latency_ms),
        "decision_grid": DEFAULT_DECISION_GRID_NAME,
        "decision_grid_minutes": DEFAULT_DECISION_GRID_MINUTES,
        "decision_stride_minutes": args.decision_stride_minutes,
        "periods_per_year_formula": "252 * median_decisions_per_utc_day",
        "median_decisions_per_day": median_decisions_per_day,
        "start": args.start,
        "end_exclusive": args.end_exclusive,
        "universe_selection_date": universe_selection_date,
    }
    metadata = {
        "rows": len(decision_timestamps),
        "minute_shape": list(minute_features.shape),
        "hour_shape": list(hour_features.shape),
        "action_count": len(action_names),
        "action_names": action_names,
        "first_decision_timestamp": decision_timestamps[0],
        "last_decision_timestamp": decision_timestamps[-1],
        "first_next_timestamp": next_timestamps[0],
        "last_next_timestamp": next_timestamps[-1],
        "source_bar_interval": source_bar_interval,
        "source_bar_seconds": source_bar_seconds,
        "bar_latency_ms": int(args.bar_latency_ms),
        "context_bars_per_hour": args.minutes_per_hour,
        "decision_grid": DEFAULT_DECISION_GRID_NAME,
        "decision_grid_minutes": DEFAULT_DECISION_GRID_MINUTES,
        "periods_per_year": periods_per_year,
        "periods_per_year_formula": "252 * median_decisions_per_utc_day",
        "median_decisions_per_day": median_decisions_per_day,
        "dataset": str(dataset_path),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = DatasetManifest(
        dataset_id=f"{source_bar_interval}_to_hour_{hash_string_sequence(decision_timestamps)[:12]}",
        created_at_utc=utc_now_iso(),
        source_vendor="Polygon aggregates" if source_bar_interval == SECOND_SOURCE_BAR_INTERVAL else "Yahoo Finance chart",
        symbols=[*selected_stocks, *etf_symbols],
        universe_selection_date=universe_selection_date,
        bar_interval=f"1h decision / {source_bar_interval} context",
        timezone="UTC timestamps with exchange timestamp features",
        adjustment="Adjusted close when available, otherwise close",
        feature_names=[*minute_feature_names, *[f"hour_{name}" for name in hour_feature_names]],
        action_names=action_names,
        timestamps_hash=hash_string_sequence(decision_timestamps),
        next_timestamps_hash=hash_string_sequence(next_timestamps),
        first_timestamp=decision_timestamps[0],
        last_timestamp=decision_timestamps[-1],
        source_manifest_hash=stable_json_hash(source_metadata),
        known_limitations=[
            f"Source {source_bar_interval} bars may be sparse; missing context bars are masked.",
            "Universe selection date is validated to be no later than the first dataset timestamp.",
            "US regular-session timing uses simplified 9:30-16:00 assumptions.",
        ],
    )
    manifest.write_json(args.output_dir / "dataset_manifest.json")
    (args.output_dir / "README.md").write_text(
        f"""# Hourly Decisions From Subhour Context Dataset

Each row is an hourly allocation decision. State tensors contain only
`{source_bar_interval}` bars with timestamps less than or equal to the decision
timestamp. Action returns are close-to-close returns from the decision timestamp
to the next hourly decision timestamp, using as-of prices when configured.

- Rows: {len(decision_timestamps)}
- Context tensor: {list(minute_features.shape)}
- Hour tensor: {list(hour_features.shape)}
- Actions: {", ".join(action_names)}
- Source bar interval: {source_bar_interval}
- Bar latency: {int(args.bar_latency_ms)} ms
- Context bars per hour: {args.minutes_per_hour}
- Decision grid: {DEFAULT_DECISION_GRID_NAME} ({DEFAULT_DECISION_GRID_MINUTES} minutes)
- Periods per year: {periods_per_year:.1f}
- Median decisions per day: {median_decisions_per_day:.1f}
"""
    )
    print(f"Rows: {len(decision_timestamps)} | Context tensor: {tuple(minute_features.shape)} | Actions: {len(action_names)}")
    print(f"Dataset -> {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
