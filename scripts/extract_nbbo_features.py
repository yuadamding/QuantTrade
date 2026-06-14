#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.quote_utils import (  # noqa: E402
    NANOS_PER_SECOND,
    NbboBuilder,
    NbboSnapshot,
    format_bucket_label,
    format_time_of_day_ns,
    in_session,
    list_date_files,
    parse_float,
    parse_int,
    parse_time_to_ns,
)


@dataclass
class FeatureBucket:
    date: str
    bucket_start_ns: int
    bucket_size_ns: int
    open_mid: float = 0.0
    high_mid: float = 0.0
    low_mid: float = 0.0
    close_mid: float = 0.0
    spread_sum: float = 0.0
    min_spread: float = 0.0
    max_spread: float = 0.0
    update_count: int = 0
    locked_quotes: int = 0
    crossed_quotes: int = 0
    imbalance_sum: float = 0.0
    microprice_sum: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth_lots: int = 0
    ask_depth_lots: int = 0
    close_imbalance: float = 0.5
    close_microprice: float = 0.0

    def apply(self, snapshot: NbboSnapshot) -> None:
        if self.update_count == 0:
            self.open_mid = snapshot.mid
            self.high_mid = snapshot.mid
            self.low_mid = snapshot.mid
            self.min_spread = snapshot.spread
            self.max_spread = snapshot.spread
        else:
            self.high_mid = max(self.high_mid, snapshot.mid)
            self.low_mid = min(self.low_mid, snapshot.mid)
            self.min_spread = min(self.min_spread, snapshot.spread)
            self.max_spread = max(self.max_spread, snapshot.spread)

        self.close_mid = snapshot.mid
        self.spread_sum += snapshot.spread
        self.update_count += 1
        self.locked_quotes += int(snapshot.locked)
        self.crossed_quotes += int(snapshot.crossed)
        self.imbalance_sum += snapshot.imbalance
        self.microprice_sum += snapshot.microprice
        self.best_bid = snapshot.best_bid
        self.best_ask = snapshot.best_ask
        self.bid_depth_lots = snapshot.bid_depth_lots
        self.ask_depth_lots = snapshot.ask_depth_lots
        self.close_imbalance = snapshot.imbalance
        self.close_microprice = snapshot.microprice

    def to_row(self) -> dict[str, str]:
        avg_spread = self.spread_sum / self.update_count
        avg_imbalance = self.imbalance_sum / self.update_count
        avg_microprice = self.microprice_sum / self.update_count
        bucket_start_second = self.bucket_start_ns // NANOS_PER_SECOND
        return {
            "date": self.date,
            "bucket_start_ns": str(self.bucket_start_ns),
            "bucket_start_second": str(bucket_start_second),
            "time": format_time_of_day_ns(self.bucket_start_ns),
            "bucket_seconds": f"{self.bucket_size_ns / NANOS_PER_SECOND:.9f}",
            "open_mid": f"{self.open_mid:.6f}",
            "high_mid": f"{self.high_mid:.6f}",
            "low_mid": f"{self.low_mid:.6f}",
            "close_mid": f"{self.close_mid:.6f}",
            "avg_spread": f"{avg_spread:.6f}",
            "close_spread": f"{(self.best_ask - self.best_bid):.6f}",
            "min_spread": f"{self.min_spread:.6f}",
            "max_spread": f"{self.max_spread:.6f}",
            "quote_updates": str(self.update_count),
            "locked_quotes": str(self.locked_quotes),
            "crossed_quotes": str(self.crossed_quotes),
            "avg_imbalance": f"{avg_imbalance:.6f}",
            "close_imbalance": f"{self.close_imbalance:.6f}",
            "avg_microprice": f"{avg_microprice:.6f}",
            "close_microprice": f"{self.close_microprice:.6f}",
            "best_bid": f"{self.best_bid:.6f}",
            "best_ask": f"{self.best_ask:.6f}",
            "bid_depth_lots": str(self.bid_depth_lots),
            "ask_depth_lots": str(self.ask_depth_lots),
        }


FIELDNAMES = [
    "date",
    "bucket_start_ns",
    "bucket_start_second",
    "time",
    "bucket_seconds",
    "open_mid",
    "high_mid",
    "low_mid",
    "close_mid",
    "avg_spread",
    "close_spread",
    "min_spread",
    "max_spread",
    "quote_updates",
    "locked_quotes",
    "crossed_quotes",
    "avg_imbalance",
    "close_imbalance",
    "avg_microprice",
    "close_microprice",
    "best_bid",
    "best_ask",
    "bid_depth_lots",
    "ask_depth_lots",
]


def build_output_path(output_dir: Path, input_file: Path, bucket_ns: int) -> Path:
    return output_dir / f"{input_file.stem}_nbbo_{format_bucket_label(bucket_ns)}.csv"


def process_file(
    input_file: Path,
    *,
    output_dir: Path,
    bucket_ns: int,
    session: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = build_output_path(output_dir, input_file, bucket_ns)
    builder = NbboBuilder()
    bucket: Optional[FeatureBucket] = None
    rows_read = 0
    session_rows = 0
    snapshots_emitted = 0
    bucket_count = 0

    with input_file.open(newline="") as source, output_file.open("w", newline="") as sink:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(sink, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in reader:
            rows_read += 1
            timestamp_ns = parse_time_to_ns(row["TIME_M"])
            if not in_session(timestamp_ns, session):
                continue

            session_rows += 1
            snapshot = builder.update(
                exchange=row["EX"],
                bid=parse_float(row["BID"]),
                bid_size_lots=parse_int(row["BIDSIZ"]),
                ask=parse_float(row["ASK"]),
                ask_size_lots=parse_int(row["ASKSIZ"]),
                timestamp_ns=timestamp_ns,
            )
            if snapshot is None:
                continue

            snapshots_emitted += 1
            bucket_start_ns = (timestamp_ns // bucket_ns) * bucket_ns

            if bucket is None or bucket.bucket_start_ns != bucket_start_ns:
                if bucket is not None and bucket.update_count:
                    writer.writerow(bucket.to_row())
                    bucket_count += 1
                bucket = FeatureBucket(
                    date=row["DATE"],
                    bucket_start_ns=bucket_start_ns,
                    bucket_size_ns=bucket_ns,
                )

            bucket.apply(snapshot)

        if bucket is not None and bucket.update_count:
            writer.writerow(bucket.to_row())
            bucket_count += 1

    return {
        "rows_read": rows_read,
        "session_rows": session_rows,
        "snapshots_emitted": snapshots_emitted,
        "buckets_written": bucket_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild NBBO-style per-bucket features from raw QQQ quote files.",
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "QQQ_2025")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "nbbo_features")
    parser.add_argument("--bucket-seconds", type=int)
    parser.add_argument("--bucket-ms", type=int)
    parser.add_argument("--session", choices=["regular", "all"], default="regular")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bucket_seconds is not None and args.bucket_ms is not None:
        raise SystemExit("Specify only one of --bucket-seconds or --bucket-ms.")
    if args.bucket_seconds is None and args.bucket_ms is None:
        bucket_ns = NANOS_PER_SECOND
    elif args.bucket_seconds is not None:
        if args.bucket_seconds <= 0:
            raise SystemExit("--bucket-seconds must be positive")
        bucket_ns = args.bucket_seconds * NANOS_PER_SECOND
    else:
        if args.bucket_ms <= 0:
            raise SystemExit("--bucket-ms must be positive")
        bucket_ns = args.bucket_ms * 1_000_000

    if args.input_file:
        input_files = [args.input_file]
    else:
        input_files = list_date_files(
            args.input_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if not input_files:
        raise SystemExit("No input files matched the requested date range.")

    total_rows = 0
    total_session_rows = 0
    total_snapshots = 0
    total_buckets = 0

    for input_file in input_files:
        stats = process_file(
            input_file,
            output_dir=args.output_dir,
            bucket_ns=bucket_ns,
            session=args.session,
        )
        total_rows += stats["rows_read"]
        total_session_rows += stats["session_rows"]
        total_snapshots += stats["snapshots_emitted"]
        total_buckets += stats["buckets_written"]
        print(
            f"{input_file.name}: rows={stats['rows_read']:,} "
            f"session_rows={stats['session_rows']:,} "
            f"snapshots={stats['snapshots_emitted']:,} "
            f"buckets={stats['buckets_written']:,}"
        )

    print(
        "TOTAL: "
        f"rows={total_rows:,} "
        f"session_rows={total_session_rows:,} "
        f"snapshots={total_snapshots:,} "
        f"buckets={total_buckets:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
