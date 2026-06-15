#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.data_sources.polygon_stock_covariates import (  # noqa: E402
    SUPPORTED_COVARIATE_DATASETS,
    covariate_source_coverage,
    load_raw_covariate_records_for_symbol,
)
from rl_quant.features.stock_covariates import build_symbol_silver_rows, write_silver_outputs  # noqa: E402


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_RAW_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "top500_2023_to_present"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "silver" / "top500_2023_to_present"
DEFAULT_UNIVERSE = DATA_ROOT / "polygon" / "universes" / "top_500_s3_volume_common_stocks_2026-06-12_tickers.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time silver features from raw Polygon stock covariate JSONL.")
    parser.add_argument("--raw-covariates-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all symbols in the universe file.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when a selected symbol is missing a raw source file.")
    return parser.parse_args(argv)


def read_universe_symbols(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        if text.lower() in {"symbol", "ticker", "yahoo_symbol"}:
            continue
        if "," in text:
            first = text.split(",", 1)[0].strip()
            if first.lower() in {"symbol", "ticker", "yahoo_symbol"}:
                continue
            text = first
        symbols.append(text.upper())
    return list(dict.fromkeys(symbols))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = read_universe_symbols(args.universe)
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    if not symbols:
        raise ValueError("No symbols found in the selected universe.")
    rows_by_symbol = {}
    coverage_by_symbol = {}
    strict_errors: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        coverage = covariate_source_coverage(args.raw_covariates_root, symbol)
        missing = [dataset for dataset in SUPPORTED_COVARIATE_DATASETS if not coverage.get(dataset, False)]
        if missing:
            strict_errors.append(f"{symbol}: missing {','.join(missing)}")
        records = load_raw_covariate_records_for_symbol(args.raw_covariates_root, symbol)
        rows_by_symbol[symbol] = build_symbol_silver_rows(records)
        coverage_by_symbol[symbol] = coverage
        if index % 50 == 0:
            print(f"built silver covariates for {index}/{len(symbols)} symbols", flush=True)
    if args.strict and strict_errors:
        preview = "; ".join(strict_errors[:10])
        raise SystemExit(f"Strict covariate build failed before writing outputs: {preview}")
    report = write_silver_outputs(
        rows_by_symbol=rows_by_symbol,
        output_root=args.output_root,
        coverage_by_symbol=coverage_by_symbol,
    )
    print(f"Silver covariate symbols: {report['symbols']} | rows: {report['total_silver_rows']}")
    print(f"Silver output -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
