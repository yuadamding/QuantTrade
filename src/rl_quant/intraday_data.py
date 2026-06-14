from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import torch

from rl_quant.quote_utils import (
    NANOS_PER_MILLISECOND,
    NANOS_PER_SECOND,
    REGULAR_SESSION_START_NS,
    format_bucket_label,
    parse_time_to_ns,
)

FEATURE_NAMES = [
    "log_return_1s",
    "close_spread_bps",
    "avg_spread_bps",
    "close_imbalance",
    "avg_imbalance",
    "micro_gap_bps",
    "range_bps",
    "log_quote_updates",
    "depth_skew",
    "locked_ratio",
    "crossed_ratio",
    "session_progress",
    "session_sin",
    "session_cos",
]


@dataclass
class MarketDataSplit:
    name: str
    dates: list[str]
    times: list[str]
    feature_names: list[str]
    features: torch.Tensor
    close_mid: torch.Tensor
    best_bid: torch.Tensor
    best_ask: torch.Tensor
    half_spread: torch.Tensor
    day_ids: torch.Tensor
    day_starts: torch.Tensor
    day_ends: torch.Tensor
    valid_start_indices: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    lookback: int

    def to(self, device: torch.device | str) -> "MarketDataSplit":
        return replace(
            self,
            features=self.features.to(device),
            close_mid=self.close_mid.to(device),
            best_bid=self.best_bid.to(device),
            best_ask=self.best_ask.to(device),
            half_spread=self.half_spread.to(device),
            day_ids=self.day_ids.to(device),
            day_starts=self.day_starts.to(device),
            day_ends=self.day_ends.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
        )

    def state_windows(self, indices: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.lookback, device=indices.device, dtype=torch.long)
        window_indices = indices.unsqueeze(1) - (self.lookback - 1) + offsets.unsqueeze(0)
        return self.features[window_indices]


def feature_file_path(feature_dir: Path, date: str, bucket_seconds: int) -> Path:
    return feature_dir / f"{date}_nbbo_{format_bucket_label(bucket_seconds)}.csv"


def parse_date_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_raw_split(name: str, paths: Sequence[Path], lookback: int) -> dict[str, object]:
    dates: list[str] = []
    times: list[str] = []
    feature_rows: list[list[float]] = []
    close_mid: list[float] = []
    best_bid: list[float] = []
    best_ask: list[float] = []
    half_spread: list[float] = []
    day_ids: list[int] = []
    day_starts: list[int] = []
    day_ends: list[int] = []
    valid_indices: list[int] = []

    base_index = 0
    for day_id, path in enumerate(paths):
        date = path.name.split("_", 1)[0]
        dates.append(date)
        day_starts.append(base_index)

        previous_mid: float | None = None
        row_count = 0

        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                mid = float(row["close_mid"])
                bid = float(row["best_bid"])
                ask = float(row["best_ask"])
                bucket_start_ns = int(row["bucket_start_ns"]) if row.get("bucket_start_ns") else parse_time_to_ns(row["time"])
                bucket_seconds = float(row.get("bucket_seconds", 1.0))
                close_spread = float(row["close_spread"])
                avg_spread = float(row["avg_spread"])
                close_imbalance = float(row["close_imbalance"]) - 0.5
                avg_imbalance = float(row["avg_imbalance"]) - 0.5
                close_microprice = float(row["close_microprice"])
                high_mid = float(row["high_mid"])
                low_mid = float(row["low_mid"])
                quote_updates = int(row["quote_updates"])
                bid_depth = int(row["bid_depth_lots"])
                ask_depth = int(row["ask_depth_lots"])
                locked_quotes = int(row["locked_quotes"])
                crossed_quotes = int(row["crossed_quotes"])
                seconds_since_open = max((bucket_start_ns - REGULAR_SESSION_START_NS) / NANOS_PER_SECOND, 0.0)
                session_progress = min(max(seconds_since_open / 23_400.0, 0.0), 1.0)
                centered_progress = session_progress * 2.0 - 1.0

                if previous_mid is None or previous_mid <= 0.0 or mid <= 0.0:
                    log_return_1s = 0.0
                else:
                    log_return_1s = math.log(mid / previous_mid)
                previous_mid = mid

                scale = max(mid, 1e-6)
                quote_updates_safe = max(quote_updates, 1)
                updates_per_second = quote_updates / max(bucket_seconds, 1e-6)
                feature_rows.append(
                    [
                        log_return_1s,
                        close_spread / scale * 10_000.0,
                        avg_spread / scale * 10_000.0,
                        close_imbalance,
                        avg_imbalance,
                        (close_microprice - mid) / scale * 10_000.0,
                        (high_mid - low_mid) / scale * 10_000.0,
                        math.log1p(updates_per_second),
                        math.log1p(bid_depth) - math.log1p(ask_depth),
                        locked_quotes / quote_updates_safe,
                        crossed_quotes / quote_updates_safe,
                        centered_progress,
                        math.sin(2.0 * math.pi * session_progress),
                        math.cos(2.0 * math.pi * session_progress),
                    ]
                )

                close_mid.append(mid)
                best_bid.append(bid)
                best_ask.append(ask)
                half_spread.append(close_spread / 2.0)
                day_ids.append(day_id)
                times.append(row["time"])
                row_count += 1

        day_end = base_index + row_count
        day_ends.append(day_end)
        valid_start = base_index + lookback - 1
        valid_end = day_end - 1
        if valid_start < valid_end:
            valid_indices.extend(range(valid_start, valid_end))
        base_index = day_end

    return {
        "name": name,
        "dates": dates,
        "times": times,
        "feature_names": FEATURE_NAMES,
        "features": torch.tensor(feature_rows, dtype=torch.float32),
        "close_mid": torch.tensor(close_mid, dtype=torch.float32),
        "best_bid": torch.tensor(best_bid, dtype=torch.float32),
        "best_ask": torch.tensor(best_ask, dtype=torch.float32),
        "half_spread": torch.tensor(half_spread, dtype=torch.float32),
        "day_ids": torch.tensor(day_ids, dtype=torch.long),
        "day_starts": torch.tensor(day_starts, dtype=torch.long),
        "day_ends": torch.tensor(day_ends, dtype=torch.long),
        "valid_start_indices": torch.tensor(valid_indices, dtype=torch.long),
        "lookback": lookback,
    }


def _finalize_split(
    raw: dict[str, object],
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> MarketDataSplit:
    features = raw["features"]
    standardized = (features - feature_mean) / feature_std
    standardized = standardized.clamp_(-8.0, 8.0)

    return MarketDataSplit(
        name=str(raw["name"]),
        dates=list(raw["dates"]),
        times=list(raw["times"]),
        feature_names=list(raw["feature_names"]),
        features=standardized,
        close_mid=raw["close_mid"],
        best_bid=raw["best_bid"],
        best_ask=raw["best_ask"],
        half_spread=raw["half_spread"],
        day_ids=raw["day_ids"],
        day_starts=raw["day_starts"],
        day_ends=raw["day_ends"],
        valid_start_indices=raw["valid_start_indices"],
        feature_mean=feature_mean,
        feature_std=feature_std,
        lookback=int(raw["lookback"]),
    )


def build_splits(
    *,
    feature_dir: Path,
    train_dates: Sequence[str],
    val_dates: Sequence[str],
    test_dates: Sequence[str],
    lookback: int,
    bucket_ns: int | None = None,
    bucket_seconds: int | None = None,
) -> tuple[MarketDataSplit, MarketDataSplit, MarketDataSplit]:
    if bucket_ns is None:
        if bucket_seconds is None:
            raise ValueError("Either bucket_ns or bucket_seconds must be provided.")
        # Backward compatibility: older callers used bucket_seconds=1, while
        # newer code passes nanoseconds. Interpret small values as seconds.
        bucket_ns = bucket_seconds * NANOS_PER_SECOND if bucket_seconds < NANOS_PER_MILLISECOND else bucket_seconds

    train_paths = [feature_file_path(feature_dir, date, bucket_ns) for date in train_dates]
    val_paths = [feature_file_path(feature_dir, date, bucket_ns) for date in val_dates]
    test_paths = [feature_file_path(feature_dir, date, bucket_ns) for date in test_dates]

    for path in [*train_paths, *val_paths, *test_paths]:
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")

    raw_train = _load_raw_split("train", train_paths, lookback)
    raw_val = _load_raw_split("val", val_paths, lookback)
    raw_test = _load_raw_split("test", test_paths, lookback)

    feature_mean = raw_train["features"].mean(dim=0)
    feature_std = raw_train["features"].std(dim=0, unbiased=False).clamp_min(1e-6)

    train_split = _finalize_split(raw_train, feature_mean=feature_mean, feature_std=feature_std)
    val_split = _finalize_split(raw_val, feature_mean=feature_mean, feature_std=feature_std)
    test_split = _finalize_split(raw_test, feature_mean=feature_mean, feature_std=feature_std)
    return train_split, val_split, test_split
