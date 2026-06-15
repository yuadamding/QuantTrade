#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SCRIPT_DIR = PACKAGE_ROOT / "scripts"
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.research_protocol import stable_json_hash, utc_now_iso  # noqa: E402

COMPLETED_STATUSES = {"downloaded", "exists", "empty"}
FILE_STATUSES = {"downloaded", "exists"}
PROGRESS_PATTERN = re.compile(r"\[(?P<done>\d+)/(?P<total>\d+)\]")
REQUIRED_GOLD_PROTOCOL_KEYS = {
    "decision_action_valid_mask",
    "label_valid_mask",
    "entry_fill_observed_mask",
    "reward_exit_observed_mask",
}


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_SOURCE_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"
DEFAULT_UNIVERSE = DATA_ROOT / "polygon" / "universes" / "top_500_s3_volume_common_stocks_2026-06-12_tickers.txt"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "protocol" / "polygon_second_top500_2025_to_2026-06-15"
DEFAULT_DOWNLOAD_LOG = DATA_ROOT / "polygon" / "second_aggs" / "logs" / "top500_second_aggs_download.log"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded Polygon 1-second top-stock bars into partitioned QuantTrade protocol datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end-exclusive", default="2026-06-15")
    parser.add_argument("--source-access", choices=["REST", "AWS S3"], default="REST")
    parser.add_argument("--stock-limit", type=int, default=500)
    parser.add_argument("--action-count", type=int, default=16)
    parser.add_argument("--actions", help="Comma-separated action symbols. Defaults to top symbols from --universe.")
    parser.add_argument("--decision-interval", choices=["5m", "15m", "30m", "60m"], default="15m")
    parser.add_argument("--context-seconds", type=int, default=3600)
    parser.add_argument("--block-seconds", type=int, default=300)
    parser.add_argument("--bar-latency-ms", type=int, default=1000)
    parser.add_argument("--execution-latency-ms", type=int, default=1000)
    parser.add_argument("--max-action-staleness-seconds", type=int, default=300)
    parser.add_argument("--min-active-symbols", type=int, default=250)
    parser.add_argument("--min-active-stock-fraction", type=float, default=0.01)
    parser.add_argument("--min-context-valid-fraction", type=float, default=0.005)
    parser.add_argument("--hourly-chunk-trading-days", type=int, default=3)
    parser.add_argument("--gold-chunk-trading-days", type=int, default=1)
    parser.add_argument("--build-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-hourly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gold-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--max-hourly-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--download-log", type=Path, default=DEFAULT_DOWNLOAD_LOG)
    parser.add_argument("--download-process-pattern", default="polygon_download_second_aggs_by_day.py")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    return datetime.fromisoformat(value[:10]).date()


def iso_start(day: date) -> str:
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat()


def iso_end_exclusive(day: date) -> str:
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(source)]


def read_universe_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        fields = {field.lower(): field for field in (reader.fieldnames or [])}
        column = fields.get("ticker") or fields.get("symbol") or (reader.fieldnames or [""])[0]
        return [str(row.get(column, "")).strip().upper() for row in rows if row.get(column)]
    return [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]


def symbol_has_files(source_root: Path, symbol: str) -> bool:
    symbol_root = source_root / symbol
    return symbol_root.exists() and any(symbol_root.glob("*/*/*.parquet"))


def resolve_actions(args: argparse.Namespace) -> list[str]:
    if args.actions:
        symbols = [symbol.strip().upper() for symbol in args.actions.split(",") if symbol.strip()]
    else:
        symbols = [symbol for symbol in read_universe_symbols(args.universe) if symbol_has_files(args.source_root, symbol)]
        symbols = symbols[: args.action_count]
    symbols = [symbol for symbol in dict.fromkeys(symbols) if symbol and symbol != "CASH"]
    return ["CASH", *symbols]


def manifest_summary(rows: list[dict[str, str]], dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "")).lower() for row in rows)
    symbols_seen = {str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}
    dates_seen = {str(row.get("date", "")) for row in rows if row.get("date")}
    symbols_expected = int(float(dataset_manifest.get("symbols", 0) or len(symbols_seen)))
    dates_expected = int(float(dataset_manifest.get("market_weekdays", 0) or len(dates_seen)))
    expected = max(symbols_expected * dates_expected, len(rows), 1)
    completed = sum(statuses.get(status, 0) for status in COMPLETED_STATUSES)
    failed = statuses.get("failed", 0) + statuses.get("error", 0)
    pending = max(expected - completed - failed, 0)
    output_bytes = 0
    source_rows = 0
    for row in rows:
        try:
            output_bytes += int(float(row.get("output_size", "") or 0))
            source_rows += int(float(row.get("rows", "") or 0))
        except ValueError:
            continue
    return {
        "expected_symbol_days": expected,
        "completed_symbol_days": completed,
        "failed_symbol_days": failed,
        "pending_symbol_days": pending,
        "coverage_ratio": min(completed / float(expected), 1.0),
        "status_counts": dict(statuses),
        "symbols_seen": len(symbols_seen),
        "dates_seen": len(dates_seen),
        "source_rows": source_rows,
        "output_bytes": output_bytes,
    }


def completed_manifest_dates(rows: list[dict[str, str]], *, start: str, end_exclusive: str) -> list[date]:
    start_day = parse_date(start)
    end_day = parse_date(end_exclusive)
    day_has_data: defaultdict[date, bool] = defaultdict(bool)
    for row in rows:
        if str(row.get("status", "")).lower() not in FILE_STATUSES or not row.get("date"):
            continue
        day = parse_date(str(row["date"]))
        try:
            source_rows = int(float(row.get("rows", "") or 0))
            output_size = int(float(row.get("output_size", "") or 0))
        except ValueError:
            source_rows = 0
            output_size = 0
        day_has_data[day] = bool(day_has_data[day] or source_rows > 0 or output_size > 12_000)
    days = {day for day, has_data in day_has_data.items() if has_data}
    return sorted(day for day in days if start_day <= day < end_day)


def chunk_dates(dates: list[date], chunk_trading_days: int) -> list[tuple[date, date]]:
    if chunk_trading_days <= 0:
        raise ValueError("chunk_trading_days must be positive.")
    chunks: list[tuple[date, date]] = []
    for start in range(0, len(dates), chunk_trading_days):
        group = dates[start : start + chunk_trading_days]
        if not group:
            continue
        chunks.append((group[0], group[-1] + timedelta(days=1)))
    return chunks


def parse_download_progress(path: Path) -> tuple[bool, int | None, int | None, str]:
    if not path.exists():
        return False, None, None, "download log missing"
    lines = path.read_text(errors="replace").splitlines()
    for line in reversed(lines[-200:]):
        if line.startswith("Done."):
            return True, None, None, line
        match = PROGRESS_PATTERN.search(line)
        if match:
            done = int(match.group("done"))
            total = int(match.group("total"))
            return done >= total, done, total, line
    return False, None, None, lines[-1] if lines else "download log empty"


def process_running(pattern: str) -> bool:
    result = subprocess.run(["pgrep", "-af", pattern], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False
    current_pid = str(subprocess.run(["bash", "-lc", "echo $$"], text=True, capture_output=True).stdout).strip()
    lines = [line for line in result.stdout.splitlines() if current_pid not in line]
    return bool(lines)


def wait_for_download(args: argparse.Namespace) -> None:
    last_message = ""
    while True:
        complete, done, total, message = parse_download_progress(args.download_log)
        if complete:
            print(f"Download complete according to log: {message}", flush=True)
            return
        running = process_running(args.download_process_pattern)
        status = f"Download not complete yet: {message}"
        if done is not None and total is not None:
            status = f"Download not complete yet: {done}/{total} ({done / max(total, 1):.2%})"
        if status != last_message:
            print(status, flush=True)
            last_message = status
        if not running:
            raise RuntimeError(f"Download process is not running and completion was not found in {args.download_log}.")
        time.sleep(max(int(args.poll_seconds), 1))


def normalize_source_manifest(
    *,
    source_manifest: Path,
    dataset_manifest_path: Path,
    output_path: Path,
    source_access: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_manifest(source_manifest)
    original = read_json(dataset_manifest_path)
    summary = manifest_summary(rows, original)
    payload = dict(original)
    payload.update(
        {
            "created_at_utc": original.get("created_at_utc") or utc_now_iso(),
            "normalized_at_utc": utc_now_iso(),
            "source": "Polygon REST aggregate range endpoint" if source_access == "REST" else "Polygon flat files / S3",
            "source_access": source_access,
            "provider": "polygon",
            "asset_class": "stocks",
            "bar_type": "second_aggregate",
            "adjusted": True,
            "timespan": "second",
            "multiplier": 1,
            "download_status": "complete" if summary["pending_symbol_days"] == 0 else "incomplete",
            "remaining_symbol_days": summary["pending_symbol_days"],
            "download_completed_at_utc": utc_now_iso() if summary["pending_symbol_days"] == 0 else None,
            "manifest": str(source_manifest),
            "conversion_manifest_summary": summary,
        }
    )
    write_json(output_path, payload)
    return payload, summary


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> tuple[str, int, float]:
    rendered = " ".join(command)
    if dry_run:
        print(f"DRY RUN: {rendered}", flush=True)
        return "dry_run", 0, 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a") as sink:
        sink.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n{rendered}\n")
        sink.flush()
        result = subprocess.run(command, cwd=cwd, stdout=sink, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.time() - started
    return ("ok" if result.returncode == 0 else "failed"), result.returncode, elapsed


def existing_gold_dataset_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        from rl_quant.features.stock_second_context import validate_second_context_payload

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not REQUIRED_GOLD_PROTOCOL_KEYS.issubset(payload):
            return False
        validate_second_context_payload(payload)
        manifest = payload.get("dataset_manifest", {})
        if "action_mask_semantics" not in manifest:
            return False
        if not torch.equal(payload["action_valid_mask"].bool(), payload["decision_action_valid_mask"].bool()):
            return False
        if bool((payload["label_valid_mask"].bool() & ~payload["decision_action_valid_mask"].bool()).any().item()):
            return False
    except Exception:
        return False
    return True


def build_gold_command(
    *,
    args: argparse.Namespace,
    source_manifest: Path,
    protocol_dataset_manifest: Path,
    actions: list[str],
    start_day: date,
    end_day: date,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_second_context_decision_dataset.py"),
        "--stock-second-root",
        str(args.source_root),
        "--stock-second-manifest",
        str(source_manifest),
        "--dataset-manifest",
        str(protocol_dataset_manifest),
        "--action-bar-root",
        str(args.source_root),
        "--output",
        str(output),
        "--start",
        iso_start(start_day),
        "--end-exclusive",
        iso_end_exclusive(end_day),
        "--decision-interval",
        args.decision_interval,
        "--context-seconds",
        str(args.context_seconds),
        "--block-seconds",
        str(args.block_seconds),
        "--bar-latency-ms",
        str(args.bar_latency_ms),
        "--execution-latency-ms",
        str(args.execution_latency_ms),
        "--max-action-staleness-seconds",
        str(args.max_action_staleness_seconds),
        "--min-active-symbols",
        str(args.min_active_symbols),
        "--symbol-limit",
        str(args.stock_limit),
        "--actions",
        ",".join(actions),
        "--source-access",
        args.source_access,
    ]


def build_hourly_command(
    *,
    args: argparse.Namespace,
    start_day: date,
    end_day: date,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_hourly_from_minute_context_dataset.py"),
        "--source-bar-interval",
        "1s",
        "--stock-bar-dir",
        str(args.source_root),
        "--action-bar-dir",
        str(args.source_root),
        "--stock-universe",
        str(args.universe),
        "--action-universe",
        str(args.universe),
        "--output-dir",
        str(output_dir),
        "--dataset-file-name",
        "hour_from_second_dataset.pt",
        "--start",
        iso_start(start_day),
        "--end-exclusive",
        iso_end_exclusive(end_day),
        "--stock-limit",
        str(args.stock_limit),
        "--action-count",
        str(args.action_count),
        "--context-bars-per-hour",
        "3600",
        "--min-active-stock-fraction",
        str(args.min_active_stock_fraction),
        "--min-context-valid-fraction",
        str(args.min_context_valid_fraction),
        "--max-action-staleness-seconds",
        str(args.max_action_staleness_seconds),
        "--bar-latency-ms",
        str(args.bar_latency_ms),
        "--dense-hourly-grid",
        "--allow-missing-action-context",
    ]


def convert_chunks(args: argparse.Namespace, protocol_dataset_manifest: Path, dates: list[date], actions: list[str]) -> list[dict[str, Any]]:
    source_manifest = args.source_manifest or args.source_root / "manifest.csv"
    log_path = args.output_root / "logs" / "protocol_conversion_commands.log"
    records: list[dict[str, Any]] = []
    if args.build_gold:
        chunks = chunk_dates(dates, args.gold_chunk_trading_days)
        if args.max_gold_chunks > 0:
            chunks = chunks[: args.max_gold_chunks]
        for start_day, end_day in chunks:
            label = f"{start_day.isoformat()}_to_{end_day.isoformat()}"
            output = args.output_root / f"second_context_gold_{args.decision_interval}" / "partitions" / label / "dataset.pt"
            if args.skip_existing and existing_gold_dataset_is_current(output):
                records.append(
                    {"kind": "second_context_gold", "chunk": label, "status": "skipped_existing_current", "output": str(output)}
                )
                continue
            command = build_gold_command(
                args=args,
                source_manifest=source_manifest,
                protocol_dataset_manifest=protocol_dataset_manifest,
                actions=actions,
                start_day=start_day,
                end_day=end_day,
                output=output,
            )
            status, return_code, elapsed = run_command(command, cwd=PACKAGE_ROOT, log_path=log_path, dry_run=args.dry_run)
            record = {
                "kind": "second_context_gold",
                "chunk": label,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 3),
                "output": str(output),
                "command": command,
            }
            records.append(record)
            if status == "failed" and not args.continue_on_error:
                raise RuntimeError(f"Gold conversion failed for {label}; see {log_path}")
    if args.build_hourly:
        chunks = chunk_dates(dates, args.hourly_chunk_trading_days)
        if args.max_hourly_chunks > 0:
            chunks = chunks[: args.max_hourly_chunks]
        for start_day, end_day in chunks:
            label = f"{start_day.isoformat()}_to_{end_day.isoformat()}"
            output_dir = args.output_root / "hour_from_second_1s" / "partitions" / label
            output = output_dir / "hour_from_second_dataset.pt"
            if args.skip_existing and output.exists():
                records.append({"kind": "hour_from_second", "chunk": label, "status": "skipped_existing", "output": str(output)})
                continue
            command = build_hourly_command(args=args, start_day=start_day, end_day=end_day, output_dir=output_dir)
            status, return_code, elapsed = run_command(command, cwd=PACKAGE_ROOT, log_path=log_path, dry_run=args.dry_run)
            record = {
                "kind": "hour_from_second",
                "chunk": label,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 3),
                "output": str(output),
                "command": command,
            }
            records.append(record)
            if status == "failed" and not args.continue_on_error:
                raise RuntimeError(f"Hourly conversion failed for {label}; see {log_path}")
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.source_manifest = args.source_manifest or args.source_root / "manifest.csv"
    args.dataset_manifest = args.dataset_manifest or args.source_root / "dataset_manifest.json"
    if args.wait_for_download:
        wait_for_download(args)
    rows = load_manifest(args.source_manifest)
    dates = completed_manifest_dates(rows, start=args.start, end_exclusive=args.end_exclusive)
    if not dates:
        raise ValueError("No completed source dates found for the requested range.")
    actions = resolve_actions(args)
    protocol_dataset_manifest = args.output_root / "source" / "dataset_manifest.protocol.json"
    source_payload, source_summary = normalize_source_manifest(
        source_manifest=args.source_manifest,
        dataset_manifest_path=args.dataset_manifest,
        output_path=protocol_dataset_manifest,
        source_access=args.source_access,
    )
    records = convert_chunks(args, protocol_dataset_manifest, dates, actions)
    status_counts = Counter(str(record.get("status", "")) for record in records)
    conversion_manifest = {
        "schema_version": "polygon_second_protocol_conversion_v1",
        "created_at_utc": utc_now_iso(),
        "source_root": str(args.source_root),
        "source_manifest": str(args.source_manifest),
        "protocol_dataset_manifest": str(protocol_dataset_manifest),
        "source_summary": source_summary,
        "source_payload_hash": stable_json_hash(source_payload),
        "output_root": str(args.output_root),
        "start": args.start,
        "end_exclusive": args.end_exclusive,
        "action_names": actions,
        "status_counts": dict(status_counts),
        "records": records,
    }
    write_json(args.output_root / "conversion_manifest.json", conversion_manifest)
    print(json.dumps({key: conversion_manifest[key] for key in ("output_root", "status_counts")}, indent=2, sort_keys=True))
    return 0 if status_counts.get("failed", 0) == 0 or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
