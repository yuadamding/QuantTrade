#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SYMBOL_DATE_RE = re.compile(r"^(?P<symbol>.+?)_\d{4}-\d{2}-\d{2}_")


@dataclass
class BarFeature:
    bar_return: float
    intraday_ret: float
    range_bps: float
    log_volume: float
    log_dollar_volume: float
    dollar_volume: float


def float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def symbol_from_bar_path(path: Path, *, interval: str) -> str:
    stem = path.name.removesuffix(f"_{interval}.csv")
    match = SYMBOL_DATE_RE.match(stem)
    if match:
        return match.group("symbol")
    return stem.removesuffix(f"_{interval}")


def read_ranked_symbols(path: Path, *, symbol_column: str = "yahoo_symbol") -> list[str]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        field = symbol_column if symbol_column in (reader.fieldnames or []) else "symbol"
        return [row[field].strip().upper() for row in reader if row.get(field)]


def bar_file_map(directory: Path, *, interval: str) -> dict[str, Path]:
    return {
        symbol_from_bar_path(path, interval=interval).upper(): path
        for path in sorted(directory.glob(f"*_{interval}.csv"))
    }


def interval_minutes(interval: str) -> float:
    text = interval.strip().lower()
    if text.endswith("m"):
        return float(text[:-1])
    if text.endswith("h"):
        return float(text[:-1]) * 60.0
    if text.endswith("d"):
        return 390.0
    raise ValueError(f"Unsupported bar interval {interval!r}; expected forms like 1m, 5m, 60m, or 1h.")


def periods_per_year_for_interval(interval: str) -> float:
    minutes = interval_minutes(interval)
    if minutes <= 0:
        raise ValueError("Bar interval must be positive.")
    return 252.0 * 390.0 / minutes


def interval_label(interval: str) -> str:
    text = interval.strip().lower()
    if text == "1m":
        return "minute"
    if text in {"1h", "60m"}:
        return "hourly"
    return text.replace("/", "_")


def parse_exchange_time(value: str) -> tuple[float, float, float, float, float]:
    dt = datetime.fromisoformat(value)
    minutes = dt.hour * 60 + dt.minute
    session_minutes = max(0.0, min(float(minutes - (9 * 60 + 30)), 390.0))
    session_progress = session_minutes / 390.0
    hour_angle = 2.0 * math.pi * session_progress
    dow_angle = 2.0 * math.pi * dt.weekday() / 5.0
    return (
        session_progress * 2.0 - 1.0,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(dow_angle),
        math.cos(dow_angle),
    )


def load_symbol_features(path: Path, *, start: str, end_exclusive: str) -> dict[str, BarFeature]:
    raw_rows: list[tuple[str, str, float, float, float, float, float]] = []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            ts = row["DatetimeUTC"]
            if ts < start or ts >= end_exclusive:
                continue
            open_value = float_or_none(row.get("Open"))
            high = float_or_none(row.get("High"))
            low = float_or_none(row.get("Low"))
            close = float_or_none(row.get("Adj Close")) or float_or_none(row.get("Close"))
            volume = float_or_none(row.get("Volume")) or 0.0
            if open_value is None or high is None or low is None or close is None or close <= 0:
                continue
            raw_rows.append((ts, row["DatetimeExchange"], open_value, high, low, close, max(volume, 0.0)))
    raw_rows.sort(key=lambda item: item[0])

    out: dict[str, BarFeature] = {}
    previous_close: float | None = None
    for ts, _exchange_ts, open_value, high, low, close, volume in raw_rows:
        if previous_close is None or previous_close <= 0:
            bar_return = 0.0
        else:
            bar_return = math.log(close / previous_close)
        intraday = math.log(close / open_value) if open_value > 0 else 0.0
        scale = max(close, 1e-8)
        dollar_volume = close * volume
        out[ts] = BarFeature(
            bar_return=max(min(bar_return, 1.0), -1.0),
            intraday_ret=max(min(intraday, 1.0), -1.0),
            range_bps=max((high - low) / scale * 10_000.0, 0.0),
            log_volume=math.log1p(volume),
            log_dollar_volume=math.log1p(dollar_volume),
            dollar_volume=max(dollar_volume, 0.0),
        )
        previous_close = close
    return out


def load_exchange_times(path: Path, *, start: str, end_exclusive: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            ts = row["DatetimeUTC"]
            if ts < start or ts >= end_exclusive:
                continue
            out[ts] = row["DatetimeExchange"]
    return out


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return sum(values) / max(len(values), 1)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def aggregate_stock_features(values: list[BarFeature], *, total_symbols: int) -> list[float]:
    if not values:
        return [0.0] * 14
    returns = [item.bar_return for item in values]
    abs_returns = [abs(item.bar_return) for item in values]
    intraday = [item.intraday_ret for item in values]
    ranges = [item.range_bps for item in values]
    log_dv = [item.log_dollar_volume for item in values]
    weights = [item.dollar_volume for item in values]
    sorted_returns = sorted(returns)
    bucket = max(1, len(sorted_returns) // 10)
    total_dv = sum(weights)
    concentration = max(weights) / total_dv if total_dv > 0 else 0.0
    return [
        len(values) / max(float(total_symbols), 1.0),
        sum(returns) / len(returns),
        weighted_mean(returns, weights),
        std(returns),
        sum(1.0 for value in returns if value > 0.0) / len(returns),
        sum(sorted_returns[-bucket:]) / bucket,
        sum(sorted_returns[:bucket]) / bucket,
        sum(intraday) / len(intraday),
        sum(ranges) / len(ranges),
        std(ranges),
        sum(log_dv) / len(log_dv),
        std(log_dv),
        concentration,
        weighted_mean(abs_returns, weights),
    ]


def write_numeric_csv(path: Path, index_name: str, names: list[str], index: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow([index_name, *names])
        for ts, values in zip(index, rows):
            writer.writerow([ts, *[f"{value:.10f}" for value in values]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a causal-transformer RL dataset from top-volume stock and ETF bars.",
    )
    parser.add_argument(
        "--stock-bar-dir",
        "--stock-hourly-dir",
        dest="stock_bar_dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "hourly_ohlcv" / "top_us_volume_stocks_nasdaq_1000_2026-06-14",
    )
    parser.add_argument(
        "--etf-bar-dir",
        "--etf-hourly-dir",
        dest="etf_bar_dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "hourly_ohlcv" / "top_us_volume_etfs_500_2026-06-14",
    )
    parser.add_argument(
        "--stock-universe",
        type=Path,
        default=PROJECT_ROOT / "derived" / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv",
    )
    parser.add_argument(
        "--etf-universe",
        type=Path,
        default=PROJECT_ROOT / "derived" / "universes" / "top_us_volume_etfs_500_2026-06-14.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "rl_hourly" / "top_volume_2026")
    parser.add_argument("--dataset-file-name", help="Dataset filename inside --output-dir.")
    parser.add_argument("--bar-interval", default="1h", help="Input bar interval suffix, for example 1h or 1m.")
    parser.add_argument("--start", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--end-exclusive", default="2026-06-15T00:00:00+00:00")
    parser.add_argument("--stock-limit", type=int, default=1000)
    parser.add_argument("--action-count", type=int, default=16)
    parser.add_argument("--actions", help="Comma-separated ETF action symbols. CASH is added automatically.")
    parser.add_argument("--min-active-stock-fraction", type=float, default=0.30)
    parser.add_argument(
        "--drop-session-gaps",
        action="store_true",
        help="Skip rewards whose next bar is on a different exchange date. Useful for minute bars.",
    )
    parser.add_argument(
        "--require-same-session-lookback",
        action="store_true",
        help="Mark datasets so training/evaluation only use state windows inside one exchange date.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required. Use: conda run -n ml1 python scripts/build_hourly_transformer_dataset.py") from exc

    bar_interval = args.bar_interval.strip().lower()
    periods_per_year = periods_per_year_for_interval(bar_interval)
    stock_map = bar_file_map(args.stock_bar_dir, interval=bar_interval)
    etf_map = bar_file_map(args.etf_bar_dir, interval=bar_interval)
    ranked_stocks = [symbol for symbol in read_ranked_symbols(args.stock_universe) if symbol in stock_map]
    selected_stocks = ranked_stocks[: args.stock_limit]
    if not selected_stocks:
        raise ValueError("No stock files matched the selected universe.")

    if args.actions:
        etf_symbols = [symbol.strip().upper() for symbol in args.actions.split(",") if symbol.strip()]
    else:
        etf_symbols = [symbol for symbol in read_ranked_symbols(args.etf_universe) if symbol in etf_map][: args.action_count]
    etf_symbols = list(dict.fromkeys(symbol for symbol in etf_symbols if symbol in etf_map))
    if not etf_symbols:
        raise ValueError("No ETF action files matched the selected universe.")
    action_names = ["CASH", *etf_symbols]

    print(f"Loading {len(selected_stocks)} stock files for causal market context...")
    stock_by_time: dict[str, list[BarFeature]] = {}
    for index, symbol in enumerate(selected_stocks, 1):
        rows = load_symbol_features(stock_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        for ts, feature in rows.items():
            stock_by_time.setdefault(ts, []).append(feature)
        if index % 100 == 0:
            print(f"  loaded {index}/{len(selected_stocks)} stocks", flush=True)

    print(f"Loading {len(etf_symbols)} ETF action series...")
    etf_features = {
        symbol: load_symbol_features(etf_map[symbol], start=args.start, end_exclusive=args.end_exclusive)
        for symbol in etf_symbols
    }
    exchange_times = load_exchange_times(etf_map[etf_symbols[0]], start=args.start, end_exclusive=args.end_exclusive)
    common_times = sorted(set.intersection(*(set(rows) for rows in etf_features.values())))
    min_active = max(1, int(len(selected_stocks) * args.min_active_stock_fraction))
    common_times = [
        ts
        for ts in common_times
        if len(stock_by_time.get(ts, [])) >= min_active and ts in exchange_times
    ]
    if len(common_times) < 10:
        raise ValueError("Too few aligned bar rows after stock and ETF filtering.")

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
                f"etf_{symbol}_ret_{bar_interval}",
                f"etf_{symbol}_intraday_ret",
                f"etf_{symbol}_range_bps",
                f"etf_{symbol}_log_dollar_volume",
            ]
        )
    feature_names = [*stock_feature_names, *time_feature_names, *etf_feature_names]

    timestamps: list[str] = []
    session_dates: list[str] = []
    feature_rows: list[list[float]] = []
    action_return_rows: list[list[float]] = []
    for pos, ts in enumerate(common_times[:-1]):
        next_ts = common_times[pos + 1]
        if args.drop_session_gaps and exchange_times[ts][:10] != exchange_times.get(next_ts, "")[:10]:
            continue
        stock_features = aggregate_stock_features(stock_by_time[ts], total_symbols=len(selected_stocks))
        time_features = list(parse_exchange_time(exchange_times[ts]))
        etf_row: list[float] = []
        action_returns = [0.0]
        missing = False
        for symbol in etf_symbols:
            current = etf_features[symbol].get(ts)
            next_bar = etf_features[symbol].get(next_ts)
            if current is None or next_bar is None:
                missing = True
                break
            etf_row.extend(
                [
                    current.bar_return,
                    current.intraday_ret,
                    current.range_bps,
                    current.log_dollar_volume,
                ]
            )
            action_returns.append(max(min(next_bar.bar_return, 1.0), -1.0))
        if missing:
            continue
        timestamps.append(ts)
        session_dates.append(exchange_times[ts][:10])
        feature_rows.append([*stock_features, *time_features, *etf_row])
        action_return_rows.append(action_returns)

    if len(timestamps) < 10:
        raise ValueError("Too few aligned reward rows after next-bar/session-gap filtering.")

    features = torch.tensor(feature_rows, dtype=torch.float32)
    action_returns = torch.tensor(action_return_rows, dtype=torch.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file_name = args.dataset_file_name or f"{interval_label(bar_interval)}_transformer_dataset.pt"
    dataset_path = args.output_dir / dataset_file_name
    torch.save(
        {
            "timestamps": timestamps,
            "feature_names": feature_names,
            "action_names": action_names,
            "features": features,
            "action_returns": action_returns,
            "bar_interval": bar_interval,
            "periods_per_year": periods_per_year,
            "session_dates": session_dates,
            "require_same_session_lookback": args.require_same_session_lookback,
            "source": {
                "stock_bar_dir": str(args.stock_bar_dir),
                "etf_bar_dir": str(args.etf_bar_dir),
                "stock_universe": str(args.stock_universe),
                "etf_universe": str(args.etf_universe),
                "stock_limit": len(selected_stocks),
                "action_symbols": etf_symbols,
                "bar_interval": bar_interval,
                "drop_session_gaps": args.drop_session_gaps,
                "require_same_session_lookback": args.require_same_session_lookback,
                "periods_per_year": periods_per_year,
                "start": args.start,
                "end_exclusive": args.end_exclusive,
            },
        },
        dataset_path,
    )
    write_numeric_csv(args.output_dir / "state_features.csv", "Timestamp", feature_names, timestamps, feature_rows)
    write_numeric_csv(args.output_dir / "action_returns.csv", "Timestamp", action_names, timestamps, action_return_rows)
    metadata = {
        "rows": len(timestamps),
        "feature_count": len(feature_names),
        "action_count": len(action_names),
        "action_names": action_names,
        "stock_symbols": len(selected_stocks),
        "min_active_stock_fraction": args.min_active_stock_fraction,
        "bar_interval": bar_interval,
        "periods_per_year": periods_per_year,
        "drop_session_gaps": args.drop_session_gaps,
        "require_same_session_lookback": args.require_same_session_lookback,
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "dataset": str(dataset_path),
        "state_features_csv": str(args.output_dir / "state_features.csv"),
        "action_returns_csv": str(args.output_dir / "action_returns.csv"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        f"""# {interval_label(bar_interval).title()} Causal Transformer RL Dataset

This dataset aligns the top-volume stock cross-section with tradable ETF
`{bar_interval}` bars. Each row is a decision point at bar `t`; action returns
are realized from bar `t` to the next aligned exchange bar.

The state is causal: stock and ETF features use only the current and previous
bars available by timestamp `t`. The transformer trainer applies an
upper-triangular attention mask across the lookback window.

- Rows: {len(timestamps)}
- Features: {len(feature_names)}
- Actions: {", ".join(action_names)}
- Frequency: Yahoo `{bar_interval}` exchange-session bars
- Window: {args.start} to {args.end_exclusive} exclusive
- Drop session gaps: {args.drop_session_gaps}
- Require same-session lookback: {args.require_same_session_lookback}
"""
    )
    print(f"Rows: {len(timestamps)} | Features: {len(feature_names)} | Actions: {len(action_names)}")
    print(f"Dataset -> {dataset_path}")
    print(f"State CSV -> {args.output_dir / 'state_features.csv'}")
    print(f"Action returns -> {args.output_dir / 'action_returns.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
