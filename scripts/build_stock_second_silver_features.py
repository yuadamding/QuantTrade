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
    iter_symbol_day_files,
    load_manifest,
    load_symbol_day,
    validate_manifest,
)
from rl_quant.features.stock_second_context import (  # noqa: E402
    MARKET_CONTEXT_FEATURE_NAMES,
    StockSecondContextConfig,
    build_market_context_from_frames,
    regular_session_decision_grid_ms,
)
from rl_quant.research_protocol import utc_now_iso  # noqa: E402


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_SECOND_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact silver market-context features from stock second bars.")
    parser.add_argument("--stock-second-root", type=Path, default=DEFAULT_SECOND_ROOT)
    parser.add_argument("--stock-second-manifest", type=Path, default=DEFAULT_SECOND_ROOT / "manifest.csv")
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_SECOND_ROOT / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "silver" / "polygon_second_features" / "top500_common_stocks_2025_to_2026-06-15" / "stock_second_context.csv")
    parser.add_argument("--start", default="2026-06-12T00:00:00+00:00")
    parser.add_argument("--end-exclusive", default="2026-06-13T00:00:00+00:00")
    parser.add_argument("--block-seconds", type=int, default=300)
    parser.add_argument("--min-active-symbols", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use small smoke-test defaults such as min_active_symbols=10.")
    parser.add_argument("--symbol-limit", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=0, help="Limit source files for smoke builds. 0 means no limit.")
    parser.add_argument("--include-extended-hours", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    min_active_symbols = args.min_active_symbols if args.min_active_symbols is not None else (10 if args.smoke else 250)
    rows = load_manifest(args.stock_second_manifest)
    config = PolygonSecondAggConfig(
        root=args.stock_second_root,
        manifest_csv=args.stock_second_manifest,
        dataset_manifest_json=args.dataset_manifest,
        rth_only=not args.include_extended_hours,
        include_extended_hours=args.include_extended_hours,
    )
    report = validate_manifest(rows, config).to_dict()
    files = [
        path
        for path in iter_symbol_day_files(rows)
        if args.start[:10] <= path.stem < args.end_exclusive[:10]
    ]
    if args.max_files > 0:
        files = files[: args.max_files]
    import pandas as pd

    # Group day files by SYMBOL (layout SYMBOL/YYYY/MM/DATE.parquet) and select the first
    # --symbol-limit SYMBOLS, then concatenate ALL of each symbol's days in the window. The prior
    # code applied --symbol-limit to day FILES and overwrote frames_by_symbol per symbol, so a
    # multi-day window silently kept only the last day per symbol and truncated symbols once the
    # file count hit the limit -- producing an incorrect market-context cross-section.
    files_by_symbol: dict[str, list[Path]] = {}
    for path in files:
        files_by_symbol.setdefault(path.parents[2].name.upper(), []).append(path)
    selected_symbols = (
        list(files_by_symbol)[: args.symbol_limit] if args.symbol_limit > 0 else list(files_by_symbol)
    )
    frames_by_symbol = {}
    for symbol in selected_symbols:
        symbol_frames = []
        for path in sorted(files_by_symbol[symbol], key=lambda item: item.stem):
            frame = load_symbol_day(
                path,
                rth_only=not args.include_extended_hours,
                include_extended_hours=args.include_extended_hours,
            )
            if len(frame):
                symbol_frames.append(frame)
        if symbol_frames:
            frames_by_symbol[symbol] = pd.concat(symbol_frames, ignore_index=True).sort_values(
                "timestamp_ms"
            ).reset_index(drop=True)
    if not frames_by_symbol:
        raise ValueError("No stock second-bar frames loaded for the requested window.")
    feature_config = StockSecondContextConfig(
        decision_interval=f"{args.block_seconds}s",
        context_seconds=args.block_seconds,
        block_seconds=args.block_seconds,
        min_active_symbols=min_active_symbols,
        include_extended_hours=args.include_extended_hours,
        rth_only=not args.include_extended_hours,
    )
    decision_ms = regular_session_decision_grid_ms(
        start=args.start,
        end_exclusive=args.end_exclusive,
        decision_interval=f"{args.block_seconds}s",
    )
    context, mask, available_ms = build_market_context_from_frames(
        frames_by_symbol,
        decision_timestamps_ms=decision_ms,
        config=feature_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["decision_timestamp", "available_timestamp_ms", "valid_context", *MARKET_CONTEXT_FEATURE_NAMES])
        for row_id, decision in enumerate(decision_ms):
            writer.writerow(
                [
                    decision,
                    int(available_ms[row_id, 0].item()),
                    int(mask[row_id, 0].item()),
                    *[f"{float(value):.10g}" for value in context[row_id, 0].tolist()],
                ]
            )
    feature_manifest = {
        "created_at_utc": utc_now_iso(),
        "source": str(args.stock_second_root),
        "rows": int(context.shape[0]),
        "symbols": sorted(frames_by_symbol),
        "feature_names": MARKET_CONTEXT_FEATURE_NAMES,
        "block_seconds": args.block_seconds,
        "data_quality_report": report,
    }
    args.output.with_name("feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    print(f"Rows: {context.shape[0]} | Symbols: {len(frames_by_symbol)}")
    print(f"Silver features -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
