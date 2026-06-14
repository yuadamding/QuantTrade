#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

FIELDNAMES = [
    "DatetimeUTC",
    "DatetimeExchange",
    "DateExchange",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

MANIFEST_FIELDS = [
    "symbol",
    "status",
    "rows",
    "start",
    "end_exclusive",
    "first_datetime_utc",
    "last_datetime_utc",
    "output_path",
    "error",
]


def default_end_exclusive() -> date:
    return datetime.now(timezone.utc).date() + timedelta(days=1)


def default_start() -> date:
    return default_end_exclusive() - timedelta(days=730)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def unix_seconds(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "-").replace(".", "-")


def safe_filename_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace("^", "-").replace("=", "-")


def yahoo_chart_url(symbol: str, start: date, end_exclusive: date) -> str:
    params = urlencode(
        {
            "period1": unix_seconds(start),
            "period2": unix_seconds(end_exclusive),
            "interval": "1h",
            "events": "history|div|splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{normalize_symbol(symbol)}?{params}"


def fetch_json(url: str, retries: int = 3, pause_seconds: float = 1.0) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; quant-system-hourly-fetch/1.0)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(pause_seconds * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def _list_value(values: list | None, index: int) -> object:
    if not values or index >= len(values):
        return None
    return values[index]


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def parse_chart_payload(payload: dict, symbol: str) -> list[dict[str, str]]:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"No chart data returned for {symbol}")

    result = results[0]
    meta = result.get("meta", {})
    tz_name = meta.get("exchangeTimezoneName") or meta.get("timezone") or "America/New_York"
    try:
        exchange_tz = ZoneInfo(str(tz_name))
    except Exception:
        exchange_tz = ZoneInfo("America/New_York")

    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []

    rows: list[dict[str, str]] = []
    for i, ts in enumerate(timestamps):
        open_value = _list_value(quote.get("open"), i)
        high_value = _list_value(quote.get("high"), i)
        low_value = _list_value(quote.get("low"), i)
        close_value = _list_value(quote.get("close"), i)
        if open_value is None and high_value is None and low_value is None and close_value is None:
            continue
        dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        dt_exchange = dt_utc.astimezone(exchange_tz)
        adj_value = _list_value(adj, i)
        rows.append(
            {
                "DatetimeUTC": dt_utc.isoformat(),
                "DatetimeExchange": dt_exchange.isoformat(),
                "DateExchange": dt_exchange.date().isoformat(),
                "Open": _fmt_float(open_value),
                "High": _fmt_float(high_value),
                "Low": _fmt_float(low_value),
                "Close": _fmt_float(close_value),
                "Adj Close": _fmt_float(adj_value if adj_value is not None else close_value),
                "Volume": str(_list_value(quote.get("volume"), i) or 0),
            }
        )
    return rows


def read_symbols(args: argparse.Namespace) -> list[str]:
    symbols = [normalize_symbol(symbol) for symbol in args.symbols if symbol.strip()]
    if args.symbols_file:
        symbols.extend(
            normalize_symbol(line)
            for line in args.symbols_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.universe_csv:
        with args.universe_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            field = args.symbol_column
            if field not in (reader.fieldnames or []):
                if "yahoo_symbol" in (reader.fieldnames or []):
                    field = "yahoo_symbol"
                elif "symbol" in (reader.fieldnames or []):
                    field = "symbol"
                else:
                    raise ValueError(f"No symbol column found in {args.universe_csv}")
            symbols.extend(normalize_symbol(row[field]) for row in reader if row.get(field))
    return list(dict.fromkeys(symbol for symbol in symbols if symbol))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def existing_successes(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="") as source:
        return {
            row["symbol"]
            for row in csv.DictReader(source)
            if row.get("status") in {"downloaded", "exists"} and row.get("symbol")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 1-hour OHLCV bars from Yahoo Finance's chart endpoint.",
    )
    parser.add_argument("symbols", nargs="*", help="Ticker symbols, e.g. SPY QQQ AAPL")
    parser.add_argument("--symbols-file", type=Path, help="One ticker symbol per line.")
    parser.add_argument("--universe-csv", type=Path, help="CSV with a yahoo_symbol or symbol column.")
    parser.add_argument("--symbol-column", default="yahoo_symbol")
    parser.add_argument("--start", type=parse_date, default=default_start())
    parser.add_argument("--end-exclusive", type=parse_date, default=default_end_exclusive())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "hourly_ohlcv")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV path. Defaults inside output dir.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip symbols already marked successful in manifest.")
    parser.add_argument("--limit", type=int, help="Limit number of symbols from the combined symbol list.")
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start >= args.end_exclusive:
        raise SystemExit("--start must be earlier than --end-exclusive")
    if args.retries <= 0:
        raise SystemExit("--retries must be positive")

    symbols = read_symbols(args)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        symbols = symbols[: args.limit]
    if not symbols:
        raise SystemExit("Provide symbols, --symbols-file, or --universe-csv.")

    manifest_path = args.manifest or (args.output_dir / "manifest.csv")
    successful = existing_successes(manifest_path) if args.resume else set()
    failed: list[str] = []

    for index, symbol in enumerate(symbols, 1):
        output_path = args.output_dir / f"{safe_filename_symbol(symbol)}_{args.start}_{args.end_exclusive}_1h.csv"
        try:
            if symbol in successful:
                print(f"[{index}/{len(symbols)}] {symbol}: manifest success, skip")
                continue
            if args.skip_existing and output_path.exists():
                row_count = max(sum(1 for _ in output_path.open()) - 1, 0)
                append_manifest(
                    manifest_path,
                    {
                        "symbol": symbol,
                        "status": "exists",
                        "rows": row_count,
                        "start": args.start.isoformat(),
                        "end_exclusive": args.end_exclusive.isoformat(),
                        "first_datetime_utc": "",
                        "last_datetime_utc": "",
                        "output_path": str(output_path),
                        "error": "",
                    },
                )
                print(f"[{index}/{len(symbols)}] {symbol}: exists rows={row_count}")
                continue
            payload = fetch_json(
                yahoo_chart_url(symbol, args.start, args.end_exclusive),
                retries=args.retries,
                pause_seconds=max(args.pause_seconds, 0.1),
            )
            rows = parse_chart_payload(payload, symbol)
            if not rows:
                raise RuntimeError("No hourly rows after parsing chart payload.")
            write_csv(output_path, rows)
            append_manifest(
                manifest_path,
                {
                    "symbol": symbol,
                    "status": "downloaded",
                    "rows": len(rows),
                    "start": args.start.isoformat(),
                    "end_exclusive": args.end_exclusive.isoformat(),
                    "first_datetime_utc": rows[0]["DatetimeUTC"],
                    "last_datetime_utc": rows[-1]["DatetimeUTC"],
                    "output_path": str(output_path),
                    "error": "",
                },
            )
            print(f"[{index}/{len(symbols)}] {symbol}: wrote {len(rows)} rows")
        except Exception as exc:
            failed.append(symbol)
            append_manifest(
                manifest_path,
                {
                    "symbol": symbol,
                    "status": "failed",
                    "rows": 0,
                    "start": args.start.isoformat(),
                    "end_exclusive": args.end_exclusive.isoformat(),
                    "first_datetime_utc": "",
                    "last_datetime_utc": "",
                    "output_path": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"[{index}/{len(symbols)}] {symbol}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            if not args.keep_going:
                raise
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    print(f"symbols requested: {len(symbols)}")
    print(f"manifest -> {manifest_path}")
    if failed:
        print(f"failed symbols ({len(failed)}): {', '.join(failed[:50])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
