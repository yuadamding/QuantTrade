"""`build_stock_second_silver_features` command: compact silver market-context features from second bars.

Migrated from scripts/build_stock_second_silver_features.py (now a thin wrapper). The feature logic lives in the
package (rl_quant.features.stock_second_context, rl_quant.data_sources.polygon_second_aggs); this is the
orchestration. The script's own ``default_data_root()`` (a position-dependent duplicate of
rl_quant.paths.default_data_root()) is replaced by the canonical helper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rl_quant.data_sources.polygon_second_aggs import (
    PolygonSecondAggConfig,
    iter_symbol_day_files,
    load_manifest,
    load_symbol_day,
    validate_manifest,
)
from rl_quant.evaluation.research_protocol import utc_now_iso
from rl_quant.features.stock_second_context import (
    MARKET_CONTEXT_FEATURE_NAMES,
    StockSecondContextConfig,
    build_market_context_from_frames,
    regular_session_decision_grid_ms,
    session_gating_method,
)
from rl_quant.paths import default_data_root

DATA_ROOT = default_data_root()
DEFAULT_SECOND_ROOT = DATA_ROOT / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"

# Bumped when the feature SEMANTICS change (so a stale feature_manifest is distinguishable); the schema version
# is bumped when the manifest's own SHAPE changes. Both are recorded for reproducibility/provenance.
BUILDER_VERSION = "stock_second_silver/1"
FEATURE_MANIFEST_SCHEMA_VERSION = "1"


def build_feature_manifest(
    *,
    source_root: Path,
    stock_second_manifest: Path,
    dataset_manifest: Path,
    manifest_content_hash: str,
    dataset_manifest_content_hash: str | None,
    session_gating_method: str,
    start: str,
    end_exclusive: str,
    block_seconds: int,
    min_active_symbols: int,
    include_extended_hours: bool,
    symbol_limit: int,
    max_files: int,
    selected_symbols: list[str],
    loaded_symbols: list[str],
    rows: int,
    files_considered: int,
    files_used: int,
    total_symbols_available: int,
    data_quality_report: dict,
    created_at_utc: str,
) -> dict:
    """Build the reproducibility manifest for a silver feature build (pure: no I/O, deterministic).

    Records the full set of inputs that determine the produced feature table -- the window, block size,
    active-symbol threshold, extended-hours flag, the symbol/file limits and whether they actually TRUNCATED,
    the deterministic (source-manifest-order) symbol selection, the session-gating path (real NYSE calendar vs
    weekend+RTH heuristic, which changes the emitted decision grid), and a content hash over exactly those
    determinative inputs (excluding ``created_at_utc`` and output diagnostics, so the hash is reproducible across
    rebuilds of the same inputs). Adds fields only -- every field the prior manifest carried
    (``source``/``rows``/``symbols``/``feature_names``/``block_seconds``/``data_quality_report``) is preserved.

    The input manifests are folded into the hash by their CONTENT digest (``manifest_content_hash`` /
    ``dataset_manifest_content_hash``, computed by the caller), NOT by path string -- a manifest edit (row
    add/remove/reorder, status flip, output_path change) or an upstream re-download changes the produced features
    and so must move the hash, while merely relocating the dataset to a different path must not.
    """
    files_truncated = max_files > 0 and files_considered > files_used
    symbols_truncated = symbol_limit > 0 and total_symbols_available > len(selected_symbols)
    # The determinative inputs (NOT created_at_utc, NOT output-derived rows/loaded symbols/quality report, NOT
    # the location PATHs): a change to any of these changes the produced features, so it must change the hash; a
    # pure rerun (and a relocation to a new path with identical content) must not.
    determinative = {
        "builder_version": BUILDER_VERSION,
        "manifest_content_hash": manifest_content_hash,
        "dataset_manifest_content_hash": dataset_manifest_content_hash,
        "session_gating_method": session_gating_method,
        "start": start,
        "end_exclusive": end_exclusive,
        "block_seconds": int(block_seconds),
        "min_active_symbols": int(min_active_symbols),
        "include_extended_hours": bool(include_extended_hours),
        "symbol_limit": int(symbol_limit),
        "max_files": int(max_files),
        "selected_symbols": list(selected_symbols),
        "feature_names": list(MARKET_CONTEXT_FEATURE_NAMES),
    }
    inputs_content_hash = hashlib.sha256(
        json.dumps(determinative, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "created_at_utc": created_at_utc,
        "builder_version": BUILDER_VERSION,
        "feature_manifest_schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "session_gating_method": session_gating_method,
        "source": str(source_root),
        "manifests": {
            "stock_second_manifest": str(stock_second_manifest),
            "dataset_manifest": str(dataset_manifest),
            "manifest_content_hash": manifest_content_hash,
            "dataset_manifest_content_hash": dataset_manifest_content_hash,
        },
        "window": {"start": start, "end_exclusive": end_exclusive},
        "block_seconds": int(block_seconds),
        "min_active_symbols": int(min_active_symbols),
        "include_extended_hours": bool(include_extended_hours),
        "rows": int(rows),
        "symbols": sorted(loaded_symbols),
        "feature_names": MARKET_CONTEXT_FEATURE_NAMES,
        "symbol_selection": {
            # source-manifest order, NOT sorted: --symbol-limit keeps the FIRST N symbols in manifest order, so
            # sorting here would change WHICH symbols survive the limit (a result-moving change). The order is
            # documented and the selection recorded for reproducibility instead.
            "order": "source_manifest_order",
            "limit": int(symbol_limit),
            "total_available": int(total_symbols_available),
            "selected_count": len(selected_symbols),
            # A selected symbol whose in-window days are all empty/extended-hours-only loads zero frames and is
            # dropped (the all-empty case raises upstream); record the gap so `symbols` shrinking below
            # selected_count is attested, not just derivable.
            "loaded_count": len(loaded_symbols),
            "dropped_at_load": len(selected_symbols) - len(loaded_symbols),
            "truncated": symbols_truncated,
        },
        "file_selection": {
            "max_files": int(max_files),
            "considered": int(files_considered),
            "used": int(files_used),
            "truncated": files_truncated,
        },
        "inputs_content_hash": inputs_content_hash,
        "data_quality_report": data_quality_report,
    }


def _file_content_hash(path: Path) -> str | None:
    """SHA-256 of a file's bytes, or None if it does not exist (the dataset_manifest may be absent)."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    files_considered = len(files)
    if args.max_files > 0:
        files = files[: args.max_files]
    files_used = len(files)
    import pandas as pd

    # Group day files by SYMBOL (layout SYMBOL/YYYY/MM/DATE.parquet) and select the first
    # --symbol-limit SYMBOLS, then concatenate ALL of each symbol's days in the window. The prior
    # code applied --symbol-limit to day FILES and overwrote frames_by_symbol per symbol, so a
    # multi-day window silently kept only the last day per symbol and truncated symbols once the
    # file count hit the limit -- producing an incorrect market-context cross-section.
    files_by_symbol: dict[str, list[Path]] = {}
    for path in files:
        files_by_symbol.setdefault(path.parents[2].name.upper(), []).append(path)
    total_symbols_available = len(files_by_symbol)
    # Selection is source-manifest order (the order iter_symbol_day_files yields, which follows the manifest
    # CSV): --symbol-limit keeps the FIRST N. Deterministic given the manifest; recorded in the feature manifest.
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
    feature_manifest = build_feature_manifest(
        source_root=args.stock_second_root,
        stock_second_manifest=args.stock_second_manifest,
        dataset_manifest=args.dataset_manifest,
        manifest_content_hash=_file_content_hash(args.stock_second_manifest),
        dataset_manifest_content_hash=_file_content_hash(args.dataset_manifest),
        session_gating_method=session_gating_method(),
        start=args.start,
        end_exclusive=args.end_exclusive,
        block_seconds=args.block_seconds,
        min_active_symbols=min_active_symbols,
        include_extended_hours=args.include_extended_hours,
        symbol_limit=args.symbol_limit,
        max_files=args.max_files,
        selected_symbols=selected_symbols,
        loaded_symbols=list(frames_by_symbol),
        rows=int(context.shape[0]),
        files_considered=files_considered,
        files_used=files_used,
        total_symbols_available=total_symbols_available,
        data_quality_report=report,
        created_at_utc=utc_now_iso(),
    )
    args.output.with_name("feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    print(f"Rows: {context.shape[0]} | Symbols: {len(frames_by_symbol)}")
    print(f"Silver features -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
