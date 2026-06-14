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
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT

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
    "interval",
    "rows",
    "chunks_requested",
    "chunks_succeeded",
    "start",
    "end_exclusive",
    "first_datetime_utc",
    "last_datetime_utc",
    "output_path",
    "error",
]

YAHOO_MAX_CHUNK_DAYS = {
    "1m": 7,
    "2m": 30,
    "5m": 30,
    "15m": 30,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
}


def default_end_exclusive() -> date:
    return datetime.now(timezone.utc).date() + timedelta(days=1)


def default_start() -> date:
    # Yahoo's 1m endpoint is currently limited to roughly the latest 30 days.
    return default_end_exclusive() - timedelta(days=21)


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


def yahoo_chart_url(symbol: str, start: date, end_exclusive: date, interval: str) -> str:
    params = urlencode(
        {
            "period1": unix_seconds(start),
            "period2": unix_seconds(end_exclusive),
            "interval": interval,
            "events": "history|div|splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{normalize_symbol(symbol)}?{params}"


def fetch_json(url: str, retries: int = 3, pause_seconds: float = 1.0) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; quant-system-intraday-fetch/1.0)",
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


def date_chunks(start: date, end_exclusive: date, chunk_days: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    current = start
    while current < end_exclusive:
        nxt = min(current + timedelta(days=chunk_days), end_exclusive)
        chunks.append((current, nxt))
        current = nxt
    return chunks


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
            if row.get("status") in {"downloaded", "partial", "exists"} and row.get("symbol")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download intraday OHLCV bars from Yahoo Finance's chart endpoint.",
    )
    parser.add_argument("symbols", nargs="*", help="Ticker symbols, e.g. SPY QQQ AAPL")
    parser.add_argument("--symbols-file", type=Path, help="One ticker symbol per line.")
    parser.add_argument("--universe-csv", type=Path, help="CSV with a yahoo_symbol or symbol column.")
    parser.add_argument("--symbol-column", default="yahoo_symbol")
    parser.add_argument("--interval", default="1m", choices=sorted(YAHOO_MAX_CHUNK_DAYS))
    parser.add_argument("--start", type=parse_date, default=default_start())
    parser.add_argument("--end-exclusive", type=parse_date, default=default_end_exclusive())
    parser.add_argument("--chunk-days", type=int, help="Request chunk width in calendar days.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "minute_ohlcv")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV path. Defaults inside output dir.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of symbols from the combined symbol list.")
    parser.add_argument("--pause-seconds", type=float, default=0.02)
    parser.add_argument("--chunk-pause-seconds", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--strict-chunks",
        action="store_true",
        help="Fail a symbol if any request chunk fails. By default, keep later valid chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start >= args.end_exclusive:
        raise SystemExit("--start must be earlier than --end-exclusive")
    if args.retries <= 0:
        raise SystemExit("--retries must be positive")
    chunk_days = args.chunk_days or YAHOO_MAX_CHUNK_DAYS[args.interval]
    if chunk_days <= 0:
        raise SystemExit("--chunk-days must be positive")

    symbols = read_symbols(args)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        symbols = symbols[: args.limit]
    if not symbols:
        raise SystemExit("Provide symbols, --symbols-file, or --universe-csv.")

    chunks = date_chunks(args.start, args.end_exclusive, chunk_days)
    manifest_path = args.manifest or (args.output_dir / "manifest.csv")
    successful = existing_successes(manifest_path) if args.resume else set()
    failed: list[str] = []

    for index, symbol in enumerate(symbols, 1):
        output_path = args.output_dir / (
            f"{safe_filename_symbol(symbol)}_{args.start}_{args.end_exclusive}_{args.interval}.csv"
        )
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
                        "interval": args.interval,
                        "rows": row_count,
                        "chunks_requested": len(chunks),
                        "chunks_succeeded": "",
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

            by_ts: dict[str, dict[str, str]] = {}
            chunks_succeeded = 0
            chunk_errors: list[str] = []
            for chunk_start, chunk_end in chunks:
                try:
                    payload = fetch_json(
                        yahoo_chart_url(symbol, chunk_start, chunk_end, args.interval),
                        retries=args.retries,
                        pause_seconds=max(args.pause_seconds, 0.1),
                    )
                    for row in parse_chart_payload(payload, symbol):
                        by_ts[row["DatetimeUTC"]] = row
                    chunks_succeeded += 1
                except Exception as exc:
                    chunk_error = f"{chunk_start}:{chunk_end} {type(exc).__name__}: {exc}"
                    chunk_errors.append(chunk_error)
                    if args.strict_chunks:
                        raise RuntimeError("; ".join(chunk_errors)) from exc
                if args.chunk_pause_seconds > 0:
                    time.sleep(args.chunk_pause_seconds)

            rows = [by_ts[ts] for ts in sorted(by_ts)]
            if not rows:
                error_text = "; ".join(chunk_errors) if chunk_errors else "No intraday rows after parsing chart payloads."
                raise RuntimeError(error_text)
            write_csv(output_path, rows)
            status = "partial" if chunk_errors else "downloaded"
            append_manifest(
                manifest_path,
                {
                    "symbol": symbol,
                    "status": status,
                    "interval": args.interval,
                    "rows": len(rows),
                    "chunks_requested": len(chunks),
                    "chunks_succeeded": chunks_succeeded,
                    "start": args.start.isoformat(),
                    "end_exclusive": args.end_exclusive.isoformat(),
                    "first_datetime_utc": rows[0]["DatetimeUTC"],
                    "last_datetime_utc": rows[-1]["DatetimeUTC"],
                    "output_path": str(output_path),
                    "error": "; ".join(chunk_errors),
                },
            )
            if chunk_errors:
                print(
                    f"[{index}/{len(symbols)}] {symbol}: wrote {len(rows)} rows "
                    f"from {chunks_succeeded}/{len(chunks)} chunks (partial)"
                )
            else:
                print(f"[{index}/{len(symbols)}] {symbol}: wrote {len(rows)} rows")
        except Exception as exc:
            failed.append(symbol)
            append_manifest(
                manifest_path,
                {
                    "symbol": symbol,
                    "status": "failed",
                    "interval": args.interval,
                    "rows": 0,
                    "chunks_requested": len(chunks),
                    "chunks_succeeded": "",
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
    print(f"chunks per symbol: {len(chunks)}")
    print(f"manifest -> {manifest_path}")
    if failed:
        print(f"failed symbols ({len(failed)}): {', '.join(failed[:50])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
