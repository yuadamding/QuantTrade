#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.data_sources.polygon_second_aggs import (  # noqa: E402
    PolygonSecondAggConfig,
    build_source_manifest,
    iter_symbol_day_files,
    load_manifest,
    load_symbol_day,
    validate_manifest,
)
from rl_quant.features.stock_second_context import (  # noqa: E402
    StockSecondContextConfig,
    build_second_context_payload,
    regular_session_decision_grid_ms,
    save_second_context_payload,
)
from rl_quant.features.stock_covariates import (  # noqa: E402
    append_action_covariates_to_payload,
    build_action_covariate_tensor,
    load_silver_rows_by_symbol,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_SECOND_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"
DEFAULT_COVARIATE_SILVER_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "silver" / "top500_2023_to_present"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold RL decision datasets from Polygon stock second context.")
    parser.add_argument("--stock-second-root", type=Path, default=DEFAULT_SECOND_ROOT)
    parser.add_argument("--stock-second-manifest", type=Path, default=DEFAULT_SECOND_ROOT / "manifest.csv")
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_SECOND_ROOT / "dataset_manifest.json")
    parser.add_argument("--action-bar-root", type=Path, default=DEFAULT_SECOND_ROOT)
    parser.add_argument("--action-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "rl_decision_datasets" / "stock_second_context_15m_v001" / "dataset.pt")
    parser.add_argument("--start", default="2026-06-12T00:00:00+00:00")
    parser.add_argument("--end-exclusive", default="2026-06-13T00:00:00+00:00")
    parser.add_argument("--decision-interval", choices=["5m", "15m", "30m", "60m"], default="15m")
    parser.add_argument("--context-seconds", type=int, default=3600)
    parser.add_argument("--block-seconds", type=int, default=300)
    parser.add_argument("--bar-latency-ms", type=int, default=1000)
    parser.add_argument("--ingestion-latency-ms", type=int, default=0)
    parser.add_argument("--execution-latency-ms", type=int, default=1000)
    parser.add_argument("--allow-post-close-exit", action="store_true")
    parser.add_argument("--min-active-symbols", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use small smoke-test defaults such as min_active_symbols=10.")
    parser.add_argument("--symbol-limit", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=0, help="Limit stock source files for smoke builds. 0 means no limit.")
    parser.add_argument("--actions", default="CASH,QQQ,SPY", help="Comma-separated action symbols; CASH is added if absent.")
    parser.add_argument("--max-action-staleness-seconds", type=int, default=300)
    parser.add_argument("--include-extended-hours", action="store_true")
    parser.add_argument("--source-access", choices=["auto", "REST", "AWS S3"], default="auto")
    parser.add_argument("--covariates-root", type=Path, default=DEFAULT_COVARIATE_SILVER_ROOT)
    parser.add_argument("--covariate-feature-schema", type=Path)
    parser.add_argument("--include-action-covariates", action="store_true")
    parser.add_argument("--include-market-covariates", action="store_true")
    parser.add_argument("--covariate-join-mode", choices=["latest_before_decision"], default="latest_before_decision")
    parser.add_argument("--covariate-max-age-days", type=int, default=0, help="Reserved for stricter future filters; 0 means no age cutoff.")
    parser.add_argument("--covariate-strict-coverage", action="store_true")
    return parser.parse_args(argv)


def parquet_symbol_from_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0].upper() if len(relative.parts) >= 4 else path.stem.upper()


def parquet_files_by_symbol(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*/*/*/*.parquet")):
        out[parquet_symbol_from_path(root, path)].append(path)
    return dict(out)


def load_frames(paths: list[Path], *, include_extended_hours: bool) -> object:
    import pandas as pd

    frames = [
        load_symbol_day(
            path,
            rth_only=not include_extended_hours,
            include_extended_hours=include_extended_hours,
        )
        for path in paths
        if path.exists()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.include_market_covariates:
        raise ValueError("Market-level covariates are not implemented yet; use --include-action-covariates for v1.")
    min_active_symbols = args.min_active_symbols if args.min_active_symbols is not None else (10 if args.smoke else 250)
    rows = load_manifest(args.stock_second_manifest)
    source_access = None if args.source_access == "auto" else args.source_access
    source_config = PolygonSecondAggConfig(
        root=args.stock_second_root,
        manifest_csv=args.stock_second_manifest,
        dataset_manifest_json=args.dataset_manifest,
        rth_only=not args.include_extended_hours,
        include_extended_hours=args.include_extended_hours,
        bar_latency_ms=args.bar_latency_ms,
        ingestion_latency_ms=args.ingestion_latency_ms,
    )
    data_quality = validate_manifest(rows, source_config, source_access=source_access).to_dict()
    source_manifest = build_source_manifest(rows, source_config, source_access=source_access).to_dict()
    stock_files = [
        path
        for path in iter_symbol_day_files(rows)
        if args.start[:10] <= path.stem < args.end_exclusive[:10]
    ]
    if args.max_files > 0:
        stock_files = stock_files[: args.max_files]
    stock_by_symbol: dict[str, list[Path]] = defaultdict(list)
    for path in stock_files:
        stock_by_symbol[parquet_symbol_from_path(args.stock_second_root, path)].append(path)
    stock_frames = {
        symbol: load_frames(paths, include_extended_hours=args.include_extended_hours)
        for symbol, paths in list(stock_by_symbol.items())[: args.symbol_limit]
    }
    stock_frames = {symbol: frame for symbol, frame in stock_frames.items() if len(frame)}
    if not stock_frames:
        raise ValueError("No stock second-bar frames loaded for the requested decision dataset.")

    action_symbols = [symbol.strip().upper() for symbol in args.actions.split(",") if symbol.strip()]
    if "CASH" not in action_symbols:
        action_symbols.insert(0, "CASH")
    action_symbols = ["CASH", *[symbol for symbol in dict.fromkeys(action_symbols) if symbol != "CASH"]]
    if args.action_manifest and args.action_manifest.exists():
        action_rows = load_manifest(args.action_manifest)
        action_files_by_symbol: dict[str, list[Path]] = defaultdict(list)
        for path in iter_symbol_day_files(action_rows):
            if args.start[:10] <= path.stem < args.end_exclusive[:10]:
                action_files_by_symbol[parquet_symbol_from_path(args.action_bar_root, path)].append(path)
    else:
        action_files_by_symbol = parquet_files_by_symbol(args.action_bar_root)
        action_files_by_symbol = {
            symbol: [path for path in paths if args.start[:10] <= path.stem < args.end_exclusive[:10]]
            for symbol, paths in action_files_by_symbol.items()
        }
    action_frames = {
        symbol: load_frames(action_files_by_symbol.get(symbol, []), include_extended_hours=args.include_extended_hours)
        for symbol in action_symbols
        if symbol != "CASH" and action_files_by_symbol.get(symbol)
    }
    missing_actions = [symbol for symbol in action_symbols if symbol != "CASH" and symbol not in action_frames]
    if missing_actions:
        print(f"Skipping actions without bar files: {', '.join(missing_actions)}")
    action_names = ["CASH", *[symbol for symbol in action_symbols if symbol != "CASH" and symbol in action_frames]]
    if len(action_names) < 2:
        raise ValueError("At least one non-CASH action with second bars is required.")

    decision_ms = regular_session_decision_grid_ms(
        start=args.start,
        end_exclusive=args.end_exclusive,
        decision_interval=args.decision_interval,
        execution_latency_ms=args.execution_latency_ms,
        allow_post_close_exit=args.allow_post_close_exit,
    )
    config = StockSecondContextConfig(
        decision_interval=args.decision_interval,
        context_seconds=args.context_seconds,
        block_seconds=args.block_seconds,
        bar_latency_ms=args.bar_latency_ms,
        ingestion_latency_ms=args.ingestion_latency_ms,
        execution_latency_ms=args.execution_latency_ms,
        min_active_symbols=min_active_symbols,
        max_action_staleness_seconds=args.max_action_staleness_seconds,
        include_extended_hours=args.include_extended_hours,
        allow_post_close_exit=args.allow_post_close_exit,
        rth_only=not args.include_extended_hours,
    )
    payload = build_second_context_payload(
        stock_frames_by_symbol=stock_frames,
        action_frames_by_symbol=action_frames,
        action_names=action_names,
        decision_timestamps_ms=decision_ms,
        config=config,
        dataset_manifest=source_manifest,
        data_quality_report=data_quality,
    )
    if args.include_action_covariates:
        silver_rows = load_silver_rows_by_symbol(args.covariates_root, action_names)
        missing_covariate_actions = [
            symbol
            for symbol in action_names
            if symbol != "CASH" and not silver_rows.get(symbol)
        ]
        if missing_covariate_actions and args.covariate_strict_coverage:
            raise ValueError(f"Missing action covariate silver rows for: {', '.join(missing_covariate_actions)}")
        manifest_path = args.covariates_root / "manifest.csv"
        schema_path = args.covariate_feature_schema or args.covariates_root / "feature_schema.json"
        source_manifest_hash = file_sha256(manifest_path)
        schema_hash = file_sha256(schema_path)
        covariates = build_action_covariate_tensor(
            silver_rows_by_symbol=silver_rows,
            action_names=action_names,
            decision_timestamps_ms=payload["decision_timestamps_ms"],
            source_manifest_hash=source_manifest_hash,
            max_age_days=args.covariate_max_age_days,
        )
        if schema_hash is None:
            covariates["action_covariate_reportability_errors"] = list(
                dict.fromkeys(
                    [
                        *covariates.get("action_covariate_reportability_errors", []),
                        "action_covariate_feature_schema_file_missing",
                    ]
                )
            )
        if missing_covariate_actions:
            covariates["action_covariate_reportability_errors"] = list(
                dict.fromkeys(
                    [
                        *covariates.get("action_covariate_reportability_errors", []),
                        "action_covariate_silver_missing_for_selected_actions",
                    ]
                )
            )
            covariates["missing_action_covariate_symbols"] = missing_covariate_actions
        covariates["action_covariate_feature_schema_file_hash"] = schema_hash
        payload = append_action_covariates_to_payload(payload, covariates, append_to_action_features=True)
    save_second_context_payload(payload, args.output)
    metadata = {
        "rows": len(payload["decision_timestamps"]),
        "market_context_shape": list(payload["market_context"].shape),
        "actions": action_names,
        "stock_symbols": len(stock_frames),
        "dataset": str(args.output),
    }
    args.output.with_name("metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Rows: {metadata['rows']} | Context: {metadata['market_context_shape']} | Actions: {len(action_names)}")
    print(f"Gold dataset -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
