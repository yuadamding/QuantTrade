#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT

SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
SOURCE_PAGE = "https://finance.yahoo.com/research-hub/screener/largest_market_cap/"
ALLOWED_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "ASE", "BTS"}

FIELDNAMES = [
    "rank",
    "symbol",
    "short_name",
    "long_name",
    "market_cap",
    "regular_market_price",
    "currency",
    "exchange",
    "full_exchange_name",
    "market",
    "quote_type",
    "source_screener",
    "retrieved_at_utc",
]


def fetch_json(start: int, count: int) -> dict:
    params = urlencode(
        {
            "scrIds": "largest_market_cap",
            "count": count,
            "start": start,
        }
    )
    request = Request(
        f"{SCREENER_URL}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; quant-system-data-fetch/1.0)",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def market_cap(row: dict) -> int:
    value = row.get("marketCap")
    if value is None:
        value = row.get("intradaymarketcap")
    return int(value or 0)


def fetch_screener_rows(*, page_size: int, pause_seconds: float) -> list[dict]:
    rows: list[dict] = []
    start = 0
    expected_total: int | None = None
    while True:
        payload = fetch_json(start=start, count=page_size)
        result = (payload.get("finance", {}).get("result") or [{}])[0]
        quotes = result.get("quotes") or []
        if expected_total is None:
            expected_total = int(result.get("total") or 0)
        if not quotes:
            break
        rows.extend(quotes)
        start += page_size
        if expected_total and start >= expected_total:
            break
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return rows


def clean_rows(rows: list[dict], *, exchange_listed_only: bool) -> list[dict]:
    best_by_symbol: dict[str, dict] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if row.get("quoteType") != "EQUITY":
            continue
        if row.get("market") != "us_market":
            continue
        if exchange_listed_only and row.get("exchange") not in ALLOWED_EXCHANGES:
            continue
        if market_cap(row) <= 0:
            continue
        current = best_by_symbol.get(symbol)
        if current is None or market_cap(row) > market_cap(current):
            best_by_symbol[symbol] = row
    return sorted(best_by_symbol.values(), key=market_cap, reverse=True)


def write_csv(path: Path, rows: list[dict], *, retrieved_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "symbol": str(row.get("symbol") or "").upper(),
                    "short_name": row.get("shortName") or "",
                    "long_name": row.get("longName") or "",
                    "market_cap": market_cap(row),
                    "regular_market_price": row.get("regularMarketPrice") or "",
                    "currency": row.get("currency") or "",
                    "exchange": row.get("exchange") or "",
                    "full_exchange_name": row.get("fullExchangeName") or "",
                    "market": row.get("market") or "",
                    "quote_type": row.get("quoteType") or "",
                    "source_screener": SOURCE_PAGE,
                    "retrieved_at_utc": retrieved_at,
                }
            )


def write_symbols(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{str(row.get('symbol') or '').upper()}\n" for row in rows))


def write_metadata(path: Path, *, rows: list[dict], raw_count: int, retrieved_at: str, args: argparse.Namespace) -> None:
    exchange_counts = Counter(row.get("fullExchangeName") or row.get("exchange") or "" for row in rows)
    payload = {
        "retrieved_at_utc": retrieved_at,
        "source_api": SCREENER_URL,
        "source_page": SOURCE_PAGE,
        "source_screener": "largest_market_cap",
        "rank_field": "marketCap/intradaymarketcap descending",
        "raw_rows_seen": raw_count,
        "universe_size": len(rows),
        "limit": args.limit,
        "exchange_listed_only": args.exchange_listed_only,
        "exchange_counts": dict(exchange_counts.most_common()),
        "top_symbol": rows[0].get("symbol") if rows else None,
        "bottom_symbol": rows[-1].get("symbol") if rows else None,
        "bottom_market_cap": market_cap(rows[-1]) if rows else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        json.dump(payload, sink, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a current Yahoo Finance top US-market equity universe by market cap.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--exchange-listed-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "universes")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).date().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.page_size <= 0 or args.page_size > 250:
        raise SystemExit("--page-size must be between 1 and 250")

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_rows = fetch_screener_rows(page_size=args.page_size, pause_seconds=args.pause_seconds)
    rows = clean_rows(raw_rows, exchange_listed_only=args.exchange_listed_only)
    rows = rows[: args.limit]
    if len(rows) < args.limit:
        print(f"warning: requested {args.limit} rows but only found {len(rows)} matching rows")

    stem = f"top_us_market_cap_{len(rows)}_{args.snapshot_date}"
    if args.exchange_listed_only:
        stem += "_exchange_listed"
    csv_path = args.output_dir / f"{stem}.csv"
    symbols_path = args.output_dir / f"{stem}_tickers.txt"
    metadata_path = args.output_dir / f"{stem}.json"

    write_csv(csv_path, rows, retrieved_at=retrieved_at)
    write_symbols(symbols_path, rows)
    write_metadata(metadata_path, rows=rows, raw_count=len(raw_rows), retrieved_at=retrieved_at, args=args)

    print(f"raw rows seen: {len(raw_rows)}")
    print(f"universe rows: {len(rows)}")
    print(f"csv -> {csv_path}")
    print(f"tickers -> {symbols_path}")
    print(f"metadata -> {metadata_path}")
    if rows:
        print(f"top: {rows[0].get('symbol')} market_cap={market_cap(rows[0])}")
        print(f"bottom: {rows[-1].get('symbol')} market_cap={market_cap(rows[-1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
