#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.data_sources.polygon_second_aggs import (  # noqa: E402
    PolygonSecondAggConfig,
    audit_symbol_day_files,
    iter_symbol_day_files,
    load_manifest,
    validate_manifest,
    write_coverage_csv,
    write_json,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_SECOND_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Polygon 1-second aggregate symbol-day Parquet files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_SECOND_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SECOND_ROOT / "manifest.csv")
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_SECOND_ROOT / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_SECOND_ROOT / "data_quality_report.json")
    parser.add_argument("--coverage-by-symbol", type=Path)
    parser.add_argument("--coverage-by-date", type=Path)
    parser.add_argument("--failed-symbol-days", type=Path)
    parser.add_argument(
        "--source-access",
        choices=["auto", "REST", "AWS S3"],
        default="auto",
        help="Override source_access in the audit output when the raw manifest was created by another downloader.",
    )
    parser.add_argument("--min-symbol-day-coverage", type=float, default=0.95)
    parser.add_argument("--max-failed-symbol-days", type=int, default=0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Parquet files to scan for row-level quality checks. 0 means manifest-only; negative means all files.",
    )
    return parser.parse_args(argv)


def write_failed_symbol_days(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [row for row in rows if row.get("status", "").lower() in {"failed", "error"}]
    fieldnames = ["symbol", "date", "status", "rows", "output_path", "error"]
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    config = PolygonSecondAggConfig(
        root=args.root,
        manifest_csv=args.manifest,
        dataset_manifest_json=args.dataset_manifest,
        min_symbol_day_coverage=args.min_symbol_day_coverage,
        max_failed_symbol_days=args.max_failed_symbol_days,
    )
    source_access = None if args.source_access == "auto" else args.source_access
    report = validate_manifest(rows, config, source_access=source_access).to_dict()
    if args.max_files != 0:
        max_files = None if args.max_files < 0 else args.max_files
        file_issues = audit_symbol_day_files(iter_symbol_day_files(rows), max_files=max_files)
        report["issue_counts"] = {**report.get("issue_counts", {}), **file_issues}
        if file_issues.get("bad_ohlc_rows", 0) or file_issues.get("bad_vwap_rows", 0):
            report["reportable"] = False
            report.setdefault("reportability_errors", []).append("row_level_price_quality_errors")
        if file_issues.get("bad_adjusted_rows", 0):
            report["reportable"] = False
            report.setdefault("reportability_errors", []).append("adjusted_flag_quality_errors")
    write_json(args.output, report)
    write_coverage_csv(args.coverage_by_symbol or args.output.with_name("coverage_by_symbol.csv"), rows, group_key="symbol")
    write_coverage_csv(args.coverage_by_date or args.output.with_name("coverage_by_date.csv"), rows, group_key="date")
    write_failed_symbol_days(args.failed_symbol_days or args.output.with_name("failed_symbol_days.csv"), rows)
    print(json.dumps({"reportable": report["reportable"], "errors": report["reportability_errors"]}, indent=2))
    print(f"Report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
