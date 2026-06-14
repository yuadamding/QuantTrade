#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT

NASDAQ_STOCKS_URL = "https://api.nasdaq.com/api/screener/stocks"
YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_ETF_SCREENER = "most_actives_etfs"
YAHOO_EQUITY_MOST_ACTIVE_SCREENER = "most_actives"
YAHOO_EQUITY_BROAD_SCREENER = "largest_market_cap"

STOCK_FIELDNAMES = [
    "rank",
    "symbol",
    "yahoo_symbol",
    "name",
    "volume",
    "last_sale",
    "market_cap",
    "country",
    "sector",
    "industry",
    "source_api",
    "retrieved_at_utc",
]

ETF_FIELDNAMES = [
    "rank",
    "symbol",
    "yahoo_symbol",
    "short_name",
    "long_name",
    "regular_market_volume",
    "average_daily_volume_3_month",
    "average_daily_volume_10_day",
    "regular_market_price",
    "currency",
    "exchange",
    "full_exchange_name",
    "market",
    "quote_type",
    "source_screener",
    "retrieved_at_utc",
]

NON_STOCK_NAME_PATTERNS = (
    " warrant",
    " warrants",
    " right",
    " rights",
    " unit",
    " units",
    " preferred",
    " preference",
    " depositary share",
    " notes due",
    " senior notes",
    " subordinated notes",
    " debenture",
    " bond",
)


def fetch_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout_seconds: float = 45.0,
) -> dict:
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        **(headers or {}),
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def int_value(value: object) -> int:
    if value is None:
        return 0
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text or text in {"N/A", "--"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def float_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("$", "").replace(",", "").strip()
    if not text or text in {"N/A", "--"}:
        return ""
    try:
        return f"{float(text):.6f}"
    except ValueError:
        return ""


def normalize_yahoo_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    symbol = symbol.replace("/", "-").replace(".", "-")
    return symbol


def usable_symbol(symbol: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", symbol))


def is_common_stock_like(name: str) -> bool:
    text = f" {name.lower()} "
    return not any(pattern in text for pattern in NON_STOCK_NAME_PATTERNS)


def fetch_nasdaq_stock_rows() -> list[dict]:
    params = urlencode({"tableonly": "true", "limit": 10000, "offset": 0, "download": "true"})
    payload = fetch_json(
        f"{NASDAQ_STOCKS_URL}?{params}",
        headers={
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
        },
        retries=3,
        timeout_seconds=90.0,
    )
    rows = payload.get("data", {}).get("rows") or []
    for row in rows:
        row["_source_api"] = NASDAQ_STOCKS_URL
    return rows


def clean_stock_rows(rows: list[dict], *, include_structured: bool) -> list[dict]:
    best_by_symbol: dict[str, dict] = {}
    for row in rows:
        symbol = normalize_yahoo_symbol(str(row.get("symbol") or ""))
        name = str(row.get("name") or row.get("shortName") or row.get("longName") or "").strip()
        volume = int_value(row.get("volume") or row.get("regularMarketVolume") or row.get("averageDailyVolume3Month"))
        if not symbol or not usable_symbol(symbol) or volume <= 0:
            continue
        if not include_structured and not is_common_stock_like(name):
            continue
        current = best_by_symbol.get(symbol)
        if current is None or volume > int_value(current.get("volume")):
            best_by_symbol[symbol] = {**row, "symbol": symbol, "volume": str(volume)}
    return sorted(best_by_symbol.values(), key=lambda row: int_value(row.get("volume")), reverse=True)


def fetch_yahoo_screener_rows(*, scr_id: str, page_size: int, pause_seconds: float) -> list[dict]:
    rows: list[dict] = []
    start = 0
    expected_total: int | None = None
    while True:
        params = urlencode({"scrIds": scr_id, "count": page_size, "start": start})
        payload = fetch_json(f"{YAHOO_SCREENER_URL}?{params}")
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
            import time

            time.sleep(pause_seconds)
    return rows


def fetch_yahoo_equity_rows(*, page_size: int, pause_seconds: float) -> list[dict]:
    rows_by_symbol: dict[str, dict] = {}
    for scr_id in (YAHOO_EQUITY_MOST_ACTIVE_SCREENER, YAHOO_EQUITY_BROAD_SCREENER):
        rows = fetch_yahoo_screener_rows(
            scr_id=scr_id,
            page_size=page_size,
            pause_seconds=pause_seconds,
        )
        for row in rows:
            if row.get("quoteType") != "EQUITY" or row.get("market") != "us_market":
                continue
            symbol = normalize_yahoo_symbol(str(row.get("symbol") or ""))
            if not symbol:
                continue
            current = rows_by_symbol.get(symbol)
            volume = int_value(row.get("regularMarketVolume") or row.get("averageDailyVolume3Month"))
            current_volume = int_value(
                (current or {}).get("regularMarketVolume") or (current or {}).get("averageDailyVolume3Month")
            )
            if current is None or volume > current_volume:
                rows_by_symbol[symbol] = {**row, "_source_screener": scr_id}
    mapped: list[dict] = []
    for row in rows_by_symbol.values():
        mapped.append(
            {
                **row,
                "name": row.get("shortName") or row.get("longName") or "",
                "volume": row.get("regularMarketVolume") or row.get("averageDailyVolume3Month") or 0,
                "lastsale": row.get("regularMarketPrice") or "",
                "marketCap": row.get("marketCap") or row.get("intradaymarketcap") or 0,
                "country": "",
                "sector": "",
                "industry": "",
                "_source_api": f"{YAHOO_SCREENER_URL}:{row.get('_source_screener') or 'unknown'}",
            }
        )
    return mapped


def clean_etf_rows(rows: list[dict]) -> list[dict]:
    best_by_symbol: dict[str, dict] = {}
    for row in rows:
        if row.get("quoteType") != "ETF" or row.get("market") != "us_market":
            continue
        symbol = normalize_yahoo_symbol(str(row.get("symbol") or ""))
        volume = int_value(row.get("regularMarketVolume"))
        avg_volume = int_value(row.get("averageDailyVolume3Month"))
        if not symbol or not usable_symbol(symbol) or max(volume, avg_volume) <= 0:
            continue
        current = best_by_symbol.get(symbol)
        if current is None or volume > int_value(current.get("regularMarketVolume")):
            best_by_symbol[symbol] = {**row, "symbol": symbol}
    return sorted(
        best_by_symbol.values(),
        key=lambda row: (
            int_value(row.get("regularMarketVolume")),
            int_value(row.get("averageDailyVolume3Month")),
        ),
        reverse=True,
    )


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_symbols(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{symbol}\n" for symbol in symbols))


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        json.dump(payload, sink, indent=2)


def build_stock_outputs(rows: list[dict], *, limit: int, retrieved_at: str) -> tuple[list[dict[str, object]], list[str]]:
    selected = rows[:limit]
    output_rows: list[dict[str, object]] = []
    symbols: list[str] = []
    for rank, row in enumerate(selected, 1):
        symbol = normalize_yahoo_symbol(str(row.get("symbol") or ""))
        symbols.append(symbol)
        output_rows.append(
            {
                "rank": rank,
                "symbol": symbol,
                "yahoo_symbol": symbol,
                "name": row.get("name") or "",
                "volume": int_value(row.get("volume")),
                "last_sale": float_text(row.get("lastsale")),
                "market_cap": int_value(row.get("marketCap")),
                "country": row.get("country") or "",
                "sector": row.get("sector") or "",
                "industry": row.get("industry") or "",
                "source_api": row.get("_source_api") or NASDAQ_STOCKS_URL,
                "retrieved_at_utc": retrieved_at,
            }
        )
    return output_rows, symbols


def build_etf_outputs(rows: list[dict], *, limit: int, retrieved_at: str) -> tuple[list[dict[str, object]], list[str]]:
    selected = rows[:limit]
    output_rows: list[dict[str, object]] = []
    symbols: list[str] = []
    for rank, row in enumerate(selected, 1):
        symbol = normalize_yahoo_symbol(str(row.get("symbol") or ""))
        symbols.append(symbol)
        output_rows.append(
            {
                "rank": rank,
                "symbol": symbol,
                "yahoo_symbol": symbol,
                "short_name": row.get("shortName") or "",
                "long_name": row.get("longName") or "",
                "regular_market_volume": int_value(row.get("regularMarketVolume")),
                "average_daily_volume_3_month": int_value(row.get("averageDailyVolume3Month")),
                "average_daily_volume_10_day": int_value(row.get("averageDailyVolume10Day")),
                "regular_market_price": row.get("regularMarketPrice") or "",
                "currency": row.get("currency") or "",
                "exchange": row.get("exchange") or "",
                "full_exchange_name": row.get("fullExchangeName") or "",
                "market": row.get("market") or "",
                "quote_type": row.get("quoteType") or "",
                "source_screener": YAHOO_ETF_SCREENER,
                "retrieved_at_utc": retrieved_at,
            }
        )
    return output_rows, symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch high-volume US stocks and ETFs for hourly OHLCV downloads.",
    )
    parser.add_argument("--stock-limit", type=int, default=1000)
    parser.add_argument("--etf-limit", type=int, default=500)
    parser.add_argument(
        "--stock-source",
        choices=("yahoo", "nasdaq", "auto"),
        default="auto",
        help="Yahoo broad equity screener, Nasdaq full stock screener, or Nasdaq with Yahoo fallback.",
    )
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "universes")
    parser.add_argument("--include-structured-stocks", action="store_true")
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stock_limit < 0:
        raise SystemExit("--stock-limit must be non-negative")
    if args.etf_limit < 0:
        raise SystemExit("--etf-limit must be non-negative")
    if args.stock_limit == 0 and args.etf_limit == 0:
        raise SystemExit("At least one of --stock-limit or --etf-limit must be positive")
    if args.page_size <= 0 or args.page_size > 250:
        raise SystemExit("--page-size must be between 1 and 250")

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stock_source_label = args.stock_source
    stock_raw: list[dict] = []
    etf_raw: list[dict] = []
    if args.stock_limit > 0:
        if args.stock_source == "nasdaq":
            stock_raw = fetch_nasdaq_stock_rows()
        elif args.stock_source == "auto":
            try:
                stock_raw = fetch_nasdaq_stock_rows()
                stock_source_label = "nasdaq"
            except Exception as exc:
                print(f"warning: Nasdaq stock screener failed; falling back to Yahoo ({type(exc).__name__}: {exc})")
                stock_raw = fetch_yahoo_equity_rows(page_size=args.page_size, pause_seconds=args.pause_seconds)
                stock_source_label = "yahoo"
        else:
            stock_raw = fetch_yahoo_equity_rows(page_size=args.page_size, pause_seconds=args.pause_seconds)
            stock_source_label = "yahoo"
    stock_clean = clean_stock_rows(stock_raw, include_structured=args.include_structured_stocks) if stock_raw else []
    if args.etf_limit > 0:
        etf_raw = fetch_yahoo_screener_rows(
            scr_id=YAHOO_ETF_SCREENER,
            page_size=args.page_size,
            pause_seconds=args.pause_seconds,
        )
    etf_clean = clean_etf_rows(etf_raw) if etf_raw else []

    stock_rows, stock_symbols = build_stock_outputs(stock_clean, limit=args.stock_limit, retrieved_at=retrieved_at)
    etf_rows, etf_symbols = build_etf_outputs(etf_clean, limit=args.etf_limit, retrieved_at=retrieved_at)

    stock_stem = f"top_us_volume_stocks_{stock_source_label}_{len(stock_rows)}_{args.snapshot_date}"
    etf_stem = f"top_us_volume_etfs_yahoo_{len(etf_rows)}_{args.snapshot_date}"

    if args.stock_limit > 0:
        stock_csv = args.output_dir / f"{stock_stem}.csv"
        stock_tickers = args.output_dir / f"{stock_stem}_tickers.txt"
        stock_meta = args.output_dir / f"{stock_stem}.json"
        write_rows(stock_csv, STOCK_FIELDNAMES, stock_rows)
        write_symbols(stock_tickers, stock_symbols)
        write_metadata(
            stock_meta,
            {
                "retrieved_at_utc": retrieved_at,
                "source_api": stock_source_label,
                "rank_field": "share volume descending",
                "raw_rows_seen": len(stock_raw),
                "clean_rows_seen": len(stock_clean),
                "universe_size": len(stock_rows),
                "include_structured_stocks": args.include_structured_stocks,
                "country_counts": dict(Counter(str(row.get("country") or "") for row in stock_rows).most_common()),
                "top_symbol": stock_rows[0]["symbol"] if stock_rows else None,
                "bottom_symbol": stock_rows[-1]["symbol"] if stock_rows else None,
                "bottom_volume": stock_rows[-1]["volume"] if stock_rows else None,
            },
        )
        print(f"stock raw rows: {len(stock_raw)} | clean: {len(stock_clean)} | selected: {len(stock_rows)}")
        print(f"stock csv -> {stock_csv}")
        print(f"stock tickers -> {stock_tickers}")
        print(f"stock metadata -> {stock_meta}")
        if stock_rows:
            print(f"stock top: {stock_rows[0]['symbol']} volume={stock_rows[0]['volume']}")
            print(f"stock bottom: {stock_rows[-1]['symbol']} volume={stock_rows[-1]['volume']}")

    if args.etf_limit > 0:
        etf_csv = args.output_dir / f"{etf_stem}.csv"
        etf_tickers = args.output_dir / f"{etf_stem}_tickers.txt"
        etf_meta = args.output_dir / f"{etf_stem}.json"
        write_rows(etf_csv, ETF_FIELDNAMES, etf_rows)
        write_symbols(etf_tickers, etf_symbols)
        write_metadata(
            etf_meta,
            {
                "retrieved_at_utc": retrieved_at,
                "source_api": YAHOO_SCREENER_URL,
                "source_screener": YAHOO_ETF_SCREENER,
                "rank_field": "regularMarketVolume descending, averageDailyVolume3Month descending",
                "raw_rows_seen": len(etf_raw),
                "clean_rows_seen": len(etf_clean),
                "universe_size": len(etf_rows),
                "exchange_counts": dict(Counter(str(row.get("exchange") or "") for row in etf_rows).most_common()),
                "top_symbol": etf_rows[0]["symbol"] if etf_rows else None,
                "bottom_symbol": etf_rows[-1]["symbol"] if etf_rows else None,
                "bottom_regular_market_volume": etf_rows[-1]["regular_market_volume"] if etf_rows else None,
            },
        )
        print(f"etf raw rows: {len(etf_raw)} | clean: {len(etf_clean)} | selected: {len(etf_rows)}")
        print(f"etf csv -> {etf_csv}")
        print(f"etf tickers -> {etf_tickers}")
        print(f"etf metadata -> {etf_meta}")
        if etf_rows:
            print(f"etf top: {etf_rows[0]['symbol']} volume={etf_rows[0]['regular_market_volume']}")
            print(f"etf bottom: {etf_rows[-1]['symbol']} volume={etf_rows[-1]['regular_market_volume']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
