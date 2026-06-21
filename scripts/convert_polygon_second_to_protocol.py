#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
REQUIRED_HOURLY_PROTOCOL_KEYS = {
    "decision_action_valid_mask",
    "action_valid_mask",
    "label_valid_mask",
    "action_label_valid_mask",
    "action_mask_semantics",
    "model_input_keys",
    "forbidden_model_input_keys",
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
DEFAULT_COVARIATE_SILVER_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "silver" / "top500_2023_to_present"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded Polygon 1-second top-stock bars into partitioned QuantTrade protocol datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--universe-asof",
        help="Point-in-time universe as-of date. If omitted, inferred from the universe filename when possible.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end-exclusive", default="2026-06-15")
    parser.add_argument("--source-access", choices=["REST", "AWS S3"], default="REST")
    parser.add_argument("--stock-limit", type=int, default=500)
    parser.add_argument("--action-count", type=int, default=500)
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
    parser.add_argument(
        "--hourly-min-decision-rows",
        type=int,
        default=1,
        help="Minimum hourly decision rows per partition before the hourly builder fails.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of chunk conversion subprocesses to run concurrently.",
    )
    parser.add_argument("--build-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-hourly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero for non-reportable or partial conversion outputs.")
    parser.add_argument(
        "--allow-missing-action-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit dense hourly rows with missing action context. This is marked non-reportable in the conversion manifest.",
    )
    parser.add_argument(
        "--allow-non-reportable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit diagnostic/backfill conversions whose preflight checks are non-reportable.",
    )
    parser.add_argument(
        "--allow-fixed-survivor-universe-diagnostic",
        action="store_true",
        help=(
            "Forward the explicit fixed-survivor universe diagnostic override to gold second-context "
            "builders. Outputs are marked non-reportable when the universe is not point-in-time."
        ),
    )
    parser.add_argument("--max-gold-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--max-hourly-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--download-log", type=Path, default=DEFAULT_DOWNLOAD_LOG)
    parser.add_argument("--download-process-pattern", default="polygon_download_second_aggs_by_day.py")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--covariates-root", type=Path, default=DEFAULT_COVARIATE_SILVER_ROOT)
    parser.add_argument("--covariate-feature-schema", type=Path)
    parser.add_argument("--include-action-covariates", action="store_true")
    parser.add_argument("--include-market-covariates", action="store_true")
    parser.add_argument("--covariate-join-mode", choices=["latest_before_decision"], default="latest_before_decision")
    parser.add_argument("--covariate-max-age-days", type=int, default=0)
    parser.add_argument("--covariate-strict-coverage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    return datetime.fromisoformat(value[:10]).date()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_universe_asof(path: Path) -> str | None:
    matches = re.findall(r"20\d{2}-\d{2}-\d{2}", path.name)
    return matches[-1] if matches else None


def current_git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "converter_git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "converter_git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "converter_identity_hash": file_sha256(Path(__file__)),
    }


def conversion_config_payload(
    args: argparse.Namespace,
    *,
    actions: list[str],
    universe_asof: str,
) -> dict[str, Any]:
    keys = [
        "source_root",
        "source_manifest",
        "dataset_manifest",
        "universe",
        "output_root",
        "start",
        "end_exclusive",
        "source_access",
        "stock_limit",
        "action_count",
        "decision_interval",
        "context_seconds",
        "block_seconds",
        "bar_latency_ms",
        "execution_latency_ms",
        "max_action_staleness_seconds",
        "min_active_symbols",
        "min_active_stock_fraction",
        "min_context_valid_fraction",
        "hourly_chunk_trading_days",
        "gold_chunk_trading_days",
        "hourly_min_decision_rows",
        "build_gold",
        "build_hourly",
        "allow_missing_action_context",
        "allow_non_reportable",
        "allow_fixed_survivor_universe_diagnostic",
        "covariates_root",
        "covariate_feature_schema",
        "include_action_covariates",
        "include_market_covariates",
        "covariate_join_mode",
        "covariate_max_age_days",
        "covariate_strict_coverage",
    ]
    payload = {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key) for key in keys}
    if payload.get("include_action_covariates"):
        schema_path = args.covariate_feature_schema or args.covariates_root / "feature_schema.json"
        payload["covariate_silver_manifest_hash"] = file_sha256(args.covariates_root / "manifest.csv")
        payload["covariate_feature_schema_file_hash"] = file_sha256(schema_path)
    payload["universe_asof"] = universe_asof
    payload["actions"] = actions
    return payload


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
        symbols = read_universe_symbols(args.universe)[: args.action_count]
    symbols = [symbol for symbol in dict.fromkeys(symbols) if symbol and symbol != "CASH"]
    return ["CASH", *symbols]


def missing_action_source_symbols(source_root: Path, actions: list[str]) -> list[str]:
    return [symbol for symbol in actions if symbol != "CASH" and not symbol_has_files(source_root, symbol)]


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
    universe_path: Path,
    universe_asof: str,
    conversion_config: dict[str, Any],
    preflight_reportability_errors: list[str],
    missing_action_symbols: list[str],
    allow_non_reportable: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_manifest(source_manifest)
    original = read_json(dataset_manifest_path)
    summary = manifest_summary(rows, original)
    reportability_errors = list(preflight_reportability_errors)
    if summary["pending_symbol_days"] > 0:
        reportability_errors.append("source_download_incomplete")
    reportability_errors = list(dict.fromkeys(reportability_errors))
    git_metadata = current_git_metadata()
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
            "source_manifest_hash": file_sha256(source_manifest),
            "raw_dataset_manifest_hash": file_sha256(dataset_manifest_path),
            "universe_file": str(universe_path),
            "universe_file_hash": file_sha256(universe_path),
            "universe_asof": universe_asof,
            "conversion_config_hash": stable_json_hash(conversion_config),
            "missing_action_source_symbols": missing_action_symbols,
            "allow_non_reportable": bool(allow_non_reportable),
            **git_metadata,
            "conversion_manifest_summary": summary,
            "conversion_reportable": not reportability_errors,
            "conversion_reportability_errors": reportability_errors,
            "reportable": bool(original.get("reportable", True)) and not reportability_errors,
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


def safe_log_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def validate_cache_identity(existing_manifest: dict[str, Any], expected_identity: dict[str, Any] | None) -> list[str]:
    if not expected_identity:
        return []
    errors: list[str] = []
    for key, expected_value in expected_identity.items():
        if existing_manifest.get(key) != expected_value:
            errors.append(f"cache_identity_mismatch:{key}")
    return errors


def existing_gold_dataset_is_current(path: Path, expected_identity: dict[str, Any] | None = None) -> bool:
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
        if validate_cache_identity(dict(manifest), expected_identity):
            return False
        if "action_mask_semantics" not in manifest:
            return False
        if not torch.equal(payload["action_valid_mask"].bool(), payload["decision_action_valid_mask"].bool()):
            return False
        if bool((payload["label_valid_mask"].bool() & ~payload["decision_action_valid_mask"].bool()).any().item()):
            return False
    except Exception:
        return False
    return True


def existing_hourly_dataset_is_current(path: Path, expected_identity: dict[str, Any] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not REQUIRED_HOURLY_PROTOCOL_KEYS.issubset(payload):
            return False
        returns = payload["action_returns"].float()
        action_valid = payload["action_valid_mask"].bool()
        decision_valid = payload["decision_action_valid_mask"].bool()
        label_valid = payload["label_valid_mask"].bool()
        action_label_valid = payload["action_label_valid_mask"].bool()
        if tuple(action_valid.shape) != tuple(returns.shape):
            return False
        if tuple(decision_valid.shape) != tuple(returns.shape):
            return False
        if tuple(label_valid.shape) != tuple(returns.shape):
            return False
        if tuple(action_label_valid.shape) != tuple(returns.shape):
            return False
        if not torch.equal(action_valid, decision_valid):
            return False
        if not torch.equal(action_label_valid, label_valid):
            return False
        if bool((label_valid & ~decision_valid).any().item()):
            return False
        if returns.shape[1] > 0:
            if not bool(action_valid[:, 0].all().item() and label_valid[:, 0].all().item()):
                return False
            if not bool(torch.allclose(returns[:, 0], torch.zeros_like(returns[:, 0]), equal_nan=False)):
                return False
        if bool((label_valid & ~torch.isfinite(returns)).any().item()):
            return False
        non_cash_invalid = ~label_valid.clone()
        if non_cash_invalid.shape[1] > 0:
            non_cash_invalid[:, 0] = False
        if bool((non_cash_invalid & torch.isfinite(returns)).any().item()):
            return False
        if "label_valid_mask" in set(payload.get("model_input_keys", [])):
            return False
        if "label_valid_mask" not in set(payload.get("forbidden_model_input_keys", [])):
            return False
        manifest = payload.get("dataset_manifest", payload.get("source", payload))
        if validate_cache_identity(dict(manifest), expected_identity):
            return False
    except Exception:
        return False
    return True


def stamp_cache_identity(output: Path, expected_identity: dict[str, Any] | None) -> list[str]:
    if not expected_identity:
        return []
    if not output.exists():
        return ["cache_identity_stamp_failed:output_missing"]
    try:
        import torch

        payload = torch.load(output, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            return ["cache_identity_stamp_failed:payload_not_dict"]

        raw_manifest = payload.get("dataset_manifest")
        raw_source = payload.get("source")
        if isinstance(raw_manifest, dict):
            manifest = dict(raw_manifest)
        elif isinstance(raw_source, dict):
            manifest = dict(raw_source)
        else:
            manifest = {}
        if "reportable" not in manifest and "dataset_reportable" in payload:
            manifest["reportable"] = bool(payload["dataset_reportable"])
        if "reportability_errors" not in manifest and "dataset_reportability_errors" in payload:
            manifest["reportability_errors"] = list(payload["dataset_reportability_errors"])
        # Preserve the dataset's own self-derived action_schema_hash before stamping the run/cache
        # identity. The cache-identity action_schema_hash is a proxy (gold action list); the
        # partition's true action set is recorded separately for audit. Action-set drift driven by
        # the universe is still caught by universe_file_hash / conversion_config_hash.
        self_action_hash = manifest.get("action_schema_hash")
        if self_action_hash and self_action_hash != expected_identity.get("action_schema_hash"):
            manifest["dataset_action_schema_hash"] = self_action_hash
        manifest.update(expected_identity)
        payload["dataset_manifest"] = manifest

        if isinstance(raw_source, dict):
            source = dict(raw_source)
            source.update(expected_identity)
            payload["source"] = source
        for key, value in expected_identity.items():
            payload.setdefault(key, value)

        torch.save(payload, output)
        sidecar = output.parent / "dataset_manifest.json"
        sidecar_payload: dict[str, Any] = {}
        if sidecar.exists():
            try:
                loaded = json.loads(sidecar.read_text())
                if isinstance(loaded, dict):
                    sidecar_payload = loaded
            except json.JSONDecodeError:
                sidecar_payload = {}
        sidecar_payload.update(manifest)
        sidecar.write_text(json.dumps(sidecar_payload, indent=2, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        return [f"cache_identity_stamp_failed:{type(exc).__name__}"]
    return []


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
    command = [
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
    if args.include_action_covariates:
        command.extend(
            [
                "--include-action-covariates",
                "--covariates-root",
                str(args.covariates_root),
                "--covariate-join-mode",
                args.covariate_join_mode,
                "--covariate-max-age-days",
                str(args.covariate_max_age_days),
            ]
        )
        if args.covariate_feature_schema:
            command.extend(["--covariate-feature-schema", str(args.covariate_feature_schema)])
        if args.covariate_strict_coverage:
            command.append("--covariate-strict-coverage")
    if args.include_market_covariates:
        command.append("--include-market-covariates")
    if args.allow_fixed_survivor_universe_diagnostic:
        command.append("--allow-fixed-survivor-universe-diagnostic")
    return command


def build_hourly_command(
    *,
    args: argparse.Namespace,
    start_day: date,
    end_day: date,
    output_dir: Path,
) -> list[str]:
    command = [
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
        "--execution-latency-ms",
        str(args.execution_latency_ms),
        "--min-decision-rows",
        str(args.hourly_min_decision_rows),
        "--dense-hourly-grid",
    ]
    if args.allow_missing_action_context:
        command.append("--allow-missing-action-context")
    else:
        command.append("--no-allow-missing-action-context")
    return command


def run_conversion_tasks(
    *,
    args: argparse.Namespace,
    kind: str,
    tasks: list[dict[str, Any]],
    expected_cache_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    workers = max(int(args.workers), 1)
    total = len(tasks)
    log_root = args.output_root / "logs" / "protocol_conversion_commands" / kind

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        label = str(task["chunk"])
        log_path = log_root / f"{safe_log_label(label)}.log"
        status, return_code, elapsed = run_command(
            list(task["command"]),
            cwd=PACKAGE_ROOT,
            log_path=log_path,
            dry_run=args.dry_run,
        )
        stamp_errors: list[str] = []
        if status == "ok" and not args.dry_run:
            stamp_errors = stamp_cache_identity(Path(task["output"]), expected_cache_identity)
            if stamp_errors:
                status = "failed"
                return_code = return_code or 1
        return {
            **task,
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 3),
            "log_path": str(log_path),
            "cache_identity_errors": stamp_errors,
        }

    records: list[dict[str, Any]] = []
    print(f"Starting {kind} conversion tasks: {total} chunks with workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(run_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    **task,
                    "status": "failed",
                    "return_code": None,
                    "elapsed_seconds": None,
                    "error": repr(exc),
                }
            records.append(record)
            print(
                f"[{kind} {completed}/{total}] {record['status']} {record['chunk']} "
                f"elapsed={record.get('elapsed_seconds')}",
                flush=True,
            )
    records.sort(key=lambda record: str(record.get("chunk", "")))
    return records


def conversion_reportability_errors(
    *,
    args: argparse.Namespace,
    source_summary: dict[str, Any],
    status_counts: Counter[str],
    universe_asof_after_start: bool,
    missing_action_symbols: list[str],
) -> list[str]:
    errors: list[str] = []
    if universe_asof_after_start:
        errors.append("universe_asof_after_dataset_start")
    if int(source_summary.get("pending_symbol_days", 0) or 0) > 0:
        errors.append("source_download_incomplete")
    if status_counts.get("failed", 0) > 0 or status_counts.get("error", 0) > 0:
        errors.append("chunk_conversion_failed")
    if args.max_gold_chunks > 0 or args.max_hourly_chunks > 0:
        errors.append("conversion_chunk_limit_applied")
    if args.dry_run:
        errors.append("dry_run")
    if args.build_hourly and args.allow_missing_action_context:
        errors.append("missing_action_context_allowed")
    if missing_action_symbols:
        errors.append("missing_action_source_symbols")
    if args.include_action_covariates and not args.covariates_root.exists():
        errors.append("action_covariate_silver_root_missing")
    if args.include_action_covariates and not (args.covariates_root / "manifest.csv").exists():
        errors.append("action_covariate_silver_manifest_missing")
    covariate_schema = args.covariate_feature_schema or args.covariates_root / "feature_schema.json"
    if args.include_action_covariates and not covariate_schema.exists():
        errors.append("action_covariate_feature_schema_file_missing")
    return list(dict.fromkeys(errors))


def conversion_status(
    *,
    args: argparse.Namespace,
    source_summary: dict[str, Any],
    status_counts: Counter[str],
) -> str:
    if args.dry_run:
        return "dry_run"
    if (
        status_counts.get("failed", 0) > 0
        or status_counts.get("error", 0) > 0
        or int(source_summary.get("pending_symbol_days", 0) or 0) > 0
        or args.max_gold_chunks > 0
        or args.max_hourly_chunks > 0
    ):
        return "partial"
    return "complete"


def convert_chunks(
    args: argparse.Namespace,
    protocol_dataset_manifest: Path,
    dates: list[date],
    actions: list[str],
    *,
    expected_cache_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_manifest = args.source_manifest or args.source_root / "manifest.csv"
    records: list[dict[str, Any]] = []
    if args.build_gold:
        chunks = chunk_dates(dates, args.gold_chunk_trading_days)
        if args.max_gold_chunks > 0:
            chunks = chunks[: args.max_gold_chunks]
        tasks: list[dict[str, Any]] = []
        for start_day, end_day in chunks:
            label = f"{start_day.isoformat()}_to_{end_day.isoformat()}"
            output = args.output_root / f"second_context_gold_{args.decision_interval}" / "partitions" / label / "dataset.pt"
            if args.skip_existing and existing_gold_dataset_is_current(output, expected_cache_identity):
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
            tasks.append({"kind": "second_context_gold", "chunk": label, "output": str(output), "command": command})
        records.extend(
            run_conversion_tasks(
                args=args,
                kind="second_context_gold",
                tasks=tasks,
                expected_cache_identity=expected_cache_identity,
            )
        )
    if args.build_hourly:
        chunks = chunk_dates(dates, args.hourly_chunk_trading_days)
        if args.max_hourly_chunks > 0:
            chunks = chunks[: args.max_hourly_chunks]
        tasks = []
        for start_day, end_day in chunks:
            label = f"{start_day.isoformat()}_to_{end_day.isoformat()}"
            output_dir = args.output_root / "hour_from_second_1s" / "partitions" / label
            output = output_dir / "hour_from_second_dataset.pt"
            if args.skip_existing and existing_hourly_dataset_is_current(output, expected_cache_identity):
                records.append(
                    {"kind": "hour_from_second", "chunk": label, "status": "skipped_existing_current", "output": str(output)}
                )
                continue
            command = build_hourly_command(args=args, start_day=start_day, end_day=end_day, output_dir=output_dir)
            tasks.append({"kind": "hour_from_second", "chunk": label, "output": str(output), "command": command})
        records.extend(
            run_conversion_tasks(
                args=args,
                kind="hour_from_second",
                tasks=tasks,
                expected_cache_identity=expected_cache_identity,
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.include_market_covariates:
        raise ValueError("Market-level covariates are not implemented yet; use --include-action-covariates for v1.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.hourly_min_decision_rows <= 0:
        raise ValueError("--hourly-min-decision-rows must be positive.")
    args.source_manifest = args.source_manifest or args.source_root / "manifest.csv"
    args.dataset_manifest = args.dataset_manifest or args.source_root / "dataset_manifest.json"
    universe_asof = args.universe_asof or infer_universe_asof(args.universe)
    if not universe_asof:
        raise ValueError("--universe-asof is required when it cannot be inferred from --universe filename.")
    universe_asof_after_start = parse_date(universe_asof) > parse_date(args.start)
    if args.wait_for_download:
        wait_for_download(args)
    rows = load_manifest(args.source_manifest)
    dates = completed_manifest_dates(rows, start=args.start, end_exclusive=args.end_exclusive)
    if not dates:
        raise ValueError("No completed source dates found for the requested range.")
    actions = resolve_actions(args)
    missing_action_symbols = missing_action_source_symbols(args.source_root, actions)
    config_payload = conversion_config_payload(args, actions=actions, universe_asof=universe_asof)
    preflight_errors: list[str] = []
    if universe_asof_after_start:
        preflight_errors.append("universe_asof_after_dataset_start")
    if args.build_hourly and args.allow_missing_action_context:
        preflight_errors.append("missing_action_context_allowed")
    if missing_action_symbols:
        preflight_errors.append("missing_action_source_symbols")
    if args.include_action_covariates and not args.covariates_root.exists():
        preflight_errors.append("action_covariate_silver_root_missing")
    if args.include_action_covariates and not (args.covariates_root / "manifest.csv").exists():
        preflight_errors.append("action_covariate_silver_manifest_missing")
    covariate_schema = args.covariate_feature_schema or args.covariates_root / "feature_schema.json"
    if args.include_action_covariates and not covariate_schema.exists():
        preflight_errors.append("action_covariate_feature_schema_file_missing")
    preflight_errors = list(dict.fromkeys(preflight_errors))
    if preflight_errors and not args.allow_non_reportable:
        raise ValueError(
            "Non-reportable conversion preflight failed: "
            f"{', '.join(preflight_errors)}. "
            "Pass --allow-non-reportable for diagnostic/backfill conversion outputs."
        )
    protocol_dataset_manifest = args.output_root / "source" / "dataset_manifest.protocol.json"
    source_payload, source_summary = normalize_source_manifest(
        source_manifest=args.source_manifest,
        dataset_manifest_path=args.dataset_manifest,
        output_path=protocol_dataset_manifest,
        source_access=args.source_access,
        universe_path=args.universe,
        universe_asof=universe_asof,
        conversion_config=config_payload,
        preflight_reportability_errors=preflight_errors,
        missing_action_symbols=missing_action_symbols,
        allow_non_reportable=bool(args.allow_non_reportable),
    )
    expected_cache_identity = {
        key: source_payload.get(key)
        for key in (
            "source_manifest_hash",
            "universe_file_hash",
            "conversion_config_hash",
            "converter_identity_hash",
        )
    }
    expected_cache_identity["action_schema_hash"] = stable_json_hash(actions)
    records = convert_chunks(
        args,
        protocol_dataset_manifest,
        dates,
        actions,
        expected_cache_identity=expected_cache_identity,
    )
    status_counts = Counter(str(record.get("status", "")) for record in records)
    reportability_errors = conversion_reportability_errors(
        args=args,
        source_summary=source_summary,
        status_counts=status_counts,
        universe_asof_after_start=universe_asof_after_start,
        missing_action_symbols=missing_action_symbols,
    )
    reportable = not reportability_errors
    conversion_manifest = {
        "schema_version": "polygon_second_protocol_conversion_v1",
        "created_at_utc": utc_now_iso(),
        "conversion_status": conversion_status(args=args, source_summary=source_summary, status_counts=status_counts),
        "conversion_reportable": reportable,
        "dataset_reportable": reportable,
        "reportability_errors": {"conversion": reportability_errors, "dataset": []},
        "source_root": str(args.source_root),
        "source_manifest": str(args.source_manifest),
        "source_manifest_hash": file_sha256(args.source_manifest),
        "protocol_dataset_manifest": str(protocol_dataset_manifest),
        "source_summary": source_summary,
        "source_payload_hash": stable_json_hash(source_payload),
        "raw_dataset_manifest_hash": file_sha256(args.dataset_manifest),
        "universe": str(args.universe),
        "universe_file_hash": file_sha256(args.universe),
        "universe_asof": universe_asof,
        "universe_asof_after_start": universe_asof_after_start,
        "missing_action_source_symbols": missing_action_symbols,
        "output_root": str(args.output_root),
        "start": args.start,
        "end_exclusive": args.end_exclusive,
        "action_names": actions,
        "conversion_config": config_payload,
        "conversion_config_hash": stable_json_hash(config_payload),
        **current_git_metadata(),
        "continue_on_error": bool(args.continue_on_error),
        "strict": bool(args.strict),
        "allow_non_reportable": bool(args.allow_non_reportable),
        "allow_missing_action_context": bool(args.allow_missing_action_context),
        "status_counts": dict(status_counts),
        "records": records,
    }
    write_json(args.output_root / "conversion_manifest.json", conversion_manifest)
    print(
        json.dumps(
            {
                key: conversion_manifest[key]
                for key in ("output_root", "conversion_status", "conversion_reportable", "status_counts")
            },
            indent=2,
            sort_keys=True,
        )
    )
    failed = status_counts.get("failed", 0) > 0 or status_counts.get("error", 0) > 0
    if args.strict and reportability_errors:
        return 1
    return 0 if not failed or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
