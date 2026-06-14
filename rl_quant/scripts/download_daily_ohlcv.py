#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

FIELDNAMES = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def unix_seconds(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def yahoo_chart_url(symbol: str, start: date, end_exclusive: date) -> str:
    params = urlencode(
        {
            "period1": unix_seconds(start),
            "period2": unix_seconds(end_exclusive),
            "interval": "1d",
            "events": "history|div|splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}?{params}"


def fetch_json(url: str, retries: int = 3) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; quant-system-data-fetch/1.0)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def parse_chart_payload(payload: dict, symbol: str) -> list[dict[str, str]]:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"No chart data returned for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []

    rows: list[dict[str, str]] = []
    for i, ts in enumerate(timestamps):
        rows.append(
            {
                "Date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                "Open": _fmt_float((quote.get("open") or [None])[i]),
                "High": _fmt_float((quote.get("high") or [None])[i]),
                "Low": _fmt_float((quote.get("low") or [None])[i]),
                "Close": _fmt_float((quote.get("close") or [None])[i]),
                "Adj Close": _fmt_float(adj[i] if i < len(adj) else None),
                "Volume": str((quote.get("volume") or [0])[i] or 0),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV bars from Yahoo Finance's chart endpoint.",
    )
    parser.add_argument("symbols", nargs="*", help="Ticker symbols, e.g. QQQ SPY")
    parser.add_argument("--symbols-file", type=Path, help="One ticker symbol per line.")
    parser.add_argument("--start", type=parse_date, default=parse_date("2025-01-01"))
    parser.add_argument(
        "--end-exclusive",
        type=parse_date,
        default=parse_date("2026-01-01"),
        help="Exclusive end date, YYYY-MM-DD.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "daily_ohlcv")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading remaining symbols if one symbol fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download files that already exist in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start >= args.end_exclusive:
        raise SystemExit("--start must be earlier than --end-exclusive")

    failed: list[str] = []
    symbols = list(args.symbols)
    if args.symbols_file:
        symbols.extend(
            line.strip()
            for line in args.symbols_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise SystemExit("Provide at least one symbol or --symbols-file.")

    for symbol in symbols:
        try:
            output_path = args.output_dir / f"{symbol.upper()}_{args.start}_{args.end_exclusive}_daily.csv"
            if args.skip_existing and output_path.exists():
                print(f"{symbol.upper()}: exists -> {output_path}")
                continue
            url = yahoo_chart_url(symbol, args.start, args.end_exclusive)
            payload = fetch_json(url)
            rows = parse_chart_payload(payload, symbol)
            write_csv(output_path, rows)
            print(f"{symbol.upper()}: wrote {len(rows)} rows -> {output_path}")
        except Exception as exc:
            if not args.keep_going:
                raise
            failed.append(symbol.upper())
            print(f"{symbol.upper()}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
    if failed:
        print(f"Failed symbols ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
