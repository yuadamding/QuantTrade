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

from rl_quant.features.news_llm import (  # noqa: E402
    build_news_article_rows,
    discover_news_source_symbols,
    write_news_article_outputs,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_RAW_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "top500_2023_to_present"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_articles_v1" / "top500_2023_to_present"
DEFAULT_UNIVERSE = DATA_ROOT / "polygon" / "universes" / "top_500_s3_volume_common_stocks_2026-06-12_tickers.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated Polygon news article table without fetching live article URLs."
    )
    parser.add_argument("--raw-covariates-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all symbols in the universe file.")
    parser.add_argument("--strict", action="store_true", help="Fail if any selected symbol lacks usable news input.")
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
    rows, errors = build_news_article_rows(
        raw_root=args.raw_covariates_root,
        symbols=symbols,
        strict=args.strict,
    )
    source_symbols = discover_news_source_symbols(raw_root=args.raw_covariates_root, symbols=symbols)
    manifest = write_news_article_outputs(
        rows=rows,
        output_root=args.output_root,
        raw_root=args.raw_covariates_root,
        symbols=symbols,
        source_symbols=source_symbols,
        errors=errors,
    )
    print(f"News articles: {manifest['article_count']} | source symbols: {len(manifest['symbols_with_source_news'])}")
    print(f"News article output -> {args.output_root}")
    if errors:
        print(f"Warnings: {len(errors)} source errors recorded in manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
