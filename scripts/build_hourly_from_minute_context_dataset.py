#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import statistics
import sys
from datetime import datetime, timedelta
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
    aggregate_stock_features,
    bar_file_map,
    clipped_simple_return,
    interval_minutes,
    load_exchange_times,
    load_symbol_features,
    parse_exchange_time,
    read_ranked_symbols,
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
SOURCE_BAR_INTERVAL = "1m"
DEFAULT_DECISION_GRID_MINUTES = 60
DEFAULT_CONTEXT_MINUTES_PER_GRID = 60
DEFAULT_DECISION_GRID_NAME = "hour"


def validate_hourly_grid_args(args: argparse.Namespace) -> None:
    if args.decision_stride_minutes != DEFAULT_DECISION_GRID_MINUTES:
        raise ValueError("Minute-source RL datasets use an hourly decision grid; decision-stride-minutes must be 60.")
    if args.minutes_per_hour != DEFAULT_CONTEXT_MINUTES_PER_GRID:
        raise ValueError("Minute-source hourly context uses 60 one-minute bars per hour; minutes-per-hour must be 60.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an hourly-decision dataset with causal minute-level context windows.",
    )
    parser.add_argument(
        "--stock-minute-dir",
        type=Path,
        default=DATA_ROOT / "minute_ohlcv" / "top_us_volume_stocks_nasdaq_1000_2026-06-14_1m_2026-05-25_2026-06-15",
    )
    parser.add_argument(
        "--etf-minute-dir",
        type=Path,
        default=DATA_ROOT / "minute_ohlcv" / "top_us_volume_etfs_500_2026-06-14_1m_2026-05-25_2026-06-15",
    )
    parser.add_argument(
        "--stock-universe",
        type=Path,
        default=DERIVED_ROOT / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv",
    )
    parser.add_argument(
        "--etf-universe",
        type=Path,
        default=DERIVED_ROOT / "universes" / "top_us_volume_etfs_500_2026-06-14.csv",
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
        type=int,
        default=DEFAULT_CONTEXT_MINUTES_PER_GRID,
        help="Number of 1m source bars inside each hour context token; fixed at 60 for the default hourly grid.",
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
    return parser.parse_args(argv)


def timestamp_add_minutes(value: str, minutes: int) -> str:
    return (datetime.fromisoformat(value) + timedelta(minutes=minutes)).isoformat()


def session_minutes(exchange_timestamp: str) -> int:
    dt = datetime.fromisoformat(exchange_timestamp)
    return dt.hour * 60 + dt.minute - (9 * 60 + 30)


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

    if args.hours_lookback <= 0 or args.minutes_per_hour <= 0 or args.decision_stride_minutes <= 0:
        raise ValueError("hours-lookback, minutes-per-hour, and decision-stride-minutes must be positive.")
    validate_hourly_grid_args(args)
    if interval_minutes(SOURCE_BAR_INTERVAL) != 1.0:
        raise ValueError("Internal interval check failed.")

    stock_map = bar_file_map(args.stock_minute_dir, interval=SOURCE_BAR_INTERVAL)
    etf_map = bar_file_map(args.etf_minute_dir, interval=SOURCE_BAR_INTERVAL)
    ranked_stocks = [symbol for symbol in read_ranked_symbols(args.stock_universe) if symbol in stock_map]
    selected_stocks = ranked_stocks[: args.stock_limit]
    if not selected_stocks:
        raise ValueError("No stock minute files matched the selected universe.")

    if args.actions:
        etf_symbols = [symbol.strip().upper() for symbol in args.actions.split(",") if symbol.strip()]
    else:
        etf_symbols = [symbol for symbol in read_ranked_symbols(args.etf_universe) if symbol in etf_map][: args.action_count]
    etf_symbols = list(dict.fromkeys(symbol for symbol in etf_symbols if symbol in etf_map))
    if not etf_symbols:
        raise ValueError("No ETF minute files matched the selected universe.")
    action_names = ["CASH", *etf_symbols]

    print(f"Loading {len(selected_stocks)} stock minute files for causal context...")
    stock_by_time = {}
    for index, symbol in enumerate(selected_stocks, 1):
        rows = load_symbol_features(stock_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        for timestamp, feature in rows.items():
            stock_by_time.setdefault(timestamp, []).append(feature)
        if index % 100 == 0:
            print(f"  loaded {index}/{len(selected_stocks)} stocks", flush=True)

    print(f"Loading {len(etf_symbols)} ETF action minute series...")
    etf_features = {
        symbol: load_symbol_features(etf_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        for symbol in etf_symbols
    }
    exchange_times = load_exchange_times(etf_map[etf_symbols[0]], start=args.start, end_exclusive=args.end_exclusive)
    common_times = sorted(set.intersection(*(set(rows) for rows in etf_features.values())))
    min_active = max(1, int(len(selected_stocks) * args.min_active_stock_fraction))
    common_times = [
        timestamp
        for timestamp in common_times
        if timestamp in exchange_times and len(stock_by_time.get(timestamp, [])) >= min_active
    ]
    common_set = set(common_times)
    if len(common_times) < args.minutes_per_hour:
        raise ValueError("Too few aligned minute rows after stock and ETF filtering.")

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
        "minute_cumulative_stock_ret_ew",
        "minute_realized_stock_vol_so_far",
        "minute_avg_log_dollar_volume_so_far",
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
                f"etf_{symbol}_ret_1m",
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
        exchange_timestamp = exchange_times[timestamp]
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
    for decision_ts in common_times:
        exchange_timestamp = exchange_times[decision_ts]
        elapsed = session_minutes(exchange_timestamp)
        if elapsed <= 0 or elapsed % args.decision_stride_minutes != 0:
            continue
        next_ts = timestamp_add_minutes(decision_ts, args.decision_stride_minutes)
        if next_ts not in common_set:
            continue
        decision_date = exchange_timestamp[:10]
        minute_tensor: list[list[list[float]]] = []
        mask_tensor: list[list[bool]] = []
        timestamp_tensor: list[list[str]] = []
        hour_rows: list[list[float]] = []
        valid_count = 0
        for hour_index in range(args.hours_lookback):
            offset_hours = args.hours_lookback - 1 - hour_index
            hour_end = datetime.fromisoformat(decision_ts) - timedelta(minutes=offset_hours * args.decision_stride_minutes)
            hour_start = hour_end - timedelta(minutes=args.minutes_per_hour - 1)
            minute_rows: list[list[float]] = []
            mask_rows: list[bool] = []
            ts_rows: list[str] = []
            for minute_offset in range(args.minutes_per_hour):
                minute_ts = (hour_start + timedelta(minutes=minute_offset)).isoformat()
                is_valid = (
                    minute_ts <= decision_ts
                    and minute_ts in minute_feature_by_time
                    and exchange_times.get(minute_ts, "")[:10] == decision_date
                )
                ts_rows.append(minute_ts if is_valid else "")
                mask_rows.append(is_valid)
                minute_rows.append(minute_feature_by_time[minute_ts] if is_valid else [0.0] * feature_dim)
                valid_count += int(is_valid)
            valid_fraction = sum(mask_rows) / float(args.minutes_per_hour)
            hour_exchange_timestamp = exchange_times.get(hour_end.isoformat(), exchange_timestamp)
            hour_rows.append([valid_fraction, parse_exchange_time(hour_exchange_timestamp)[0]])
            timestamp_tensor.append(ts_rows)
            mask_tensor.append(mask_rows)
            minute_tensor.append(minute_rows)
        if valid_count / float(args.hours_lookback * args.minutes_per_hour) < args.min_context_valid_fraction:
            continue
        action_returns = [0.0]
        missing_action = False
        for symbol in etf_symbols:
            current = etf_features[symbol].get(decision_ts)
            future = etf_features[symbol].get(next_ts)
            if current is None or future is None:
                missing_action = True
                break
            action_returns.append(clipped_simple_return(current.close, future.close))
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
            "source_bar_interval": SOURCE_BAR_INTERVAL,
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
                "start": args.start,
                "end_exclusive": args.end_exclusive,
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
        "hours_lookback": args.hours_lookback,
        "minutes_per_hour": args.minutes_per_hour,
        "source_bar_interval": SOURCE_BAR_INTERVAL,
        "decision_grid": DEFAULT_DECISION_GRID_NAME,
        "decision_grid_minutes": DEFAULT_DECISION_GRID_MINUTES,
        "decision_stride_minutes": args.decision_stride_minutes,
        "periods_per_year_formula": "252 * median_decisions_per_utc_day",
        "median_decisions_per_day": median_decisions_per_day,
        "start": args.start,
        "end_exclusive": args.end_exclusive,
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
        "source_bar_interval": SOURCE_BAR_INTERVAL,
        "decision_grid": DEFAULT_DECISION_GRID_NAME,
        "decision_grid_minutes": DEFAULT_DECISION_GRID_MINUTES,
        "periods_per_year": periods_per_year,
        "periods_per_year_formula": "252 * median_decisions_per_utc_day",
        "median_decisions_per_day": median_decisions_per_day,
        "dataset": str(dataset_path),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = DatasetManifest(
        dataset_id=f"minute_to_hour_{hash_string_sequence(decision_timestamps)[:12]}",
        created_at_utc=utc_now_iso(),
        source_vendor="Yahoo Finance chart",
        symbols=[*selected_stocks, *etf_symbols],
        universe_selection_date=args.universe_selection_date,
        bar_interval=f"1h decision / {SOURCE_BAR_INTERVAL} context",
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
            "Yahoo true 1m history is short and may contain missing bars.",
            "Universe selection is not point-in-time unless universe_selection_date is provided.",
            "US regular-session timing uses simplified 9:30-16:00 assumptions.",
        ],
    )
    manifest.write_json(args.output_dir / "dataset_manifest.json")
    (args.output_dir / "README.md").write_text(
        f"""# Hourly Decisions From Minute Context Dataset

Each row is an hourly allocation decision. State tensors contain only minute
bars with timestamps less than or equal to the decision timestamp. Action
returns are simple close-to-close ETF returns from the decision timestamp to the
next hourly decision timestamp.

- Rows: {len(decision_timestamps)}
- Minute tensor: {list(minute_features.shape)}
- Hour tensor: {list(hour_features.shape)}
- Actions: {", ".join(action_names)}
- Source bar interval: {SOURCE_BAR_INTERVAL}
- Decision grid: {DEFAULT_DECISION_GRID_NAME} ({DEFAULT_DECISION_GRID_MINUTES} minutes)
- Periods per year: {periods_per_year:.1f}
- Median decisions per day: {median_decisions_per_day:.1f}
"""
    )
    print(f"Rows: {len(decision_timestamps)} | Minute tensor: {tuple(minute_features.shape)} | Actions: {len(action_names)}")
    print(f"Dataset -> {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
