from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from rl_quant.research_protocol import stable_json_hash


EASTERN = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
SUPPORTED_COVARIATE_DATASETS = (
    "overview_snapshots",
    "financials",
    "dividends",
    "splits",
    "news",
)


@dataclass(frozen=True)
class RawCovariateRecord:
    symbol: str
    source_dataset: str
    event_timestamp_ms: int
    available_timestamp_ms: int
    source_record_id: str
    source_record_hash: str
    payload: dict[str, Any]


def canonical_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def symbol_path_candidates(root: Path, symbol: str) -> list[Path]:
    symbol = canonical_symbol(symbol)
    variants = [symbol, symbol.replace(".", "-"), symbol.replace("-", ".")]
    return list(dict.fromkeys(root / variant for variant in variants))


def symbol_covariate_dir(root: Path, symbol: str) -> Path | None:
    for candidate in symbol_path_candidates(root, symbol):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def covariate_jsonl_path(root: Path, symbol: str, dataset: str) -> Path | None:
    directory = symbol_covariate_dir(root, symbol)
    if directory is None:
        return None
    path = directory / f"{dataset}.jsonl"
    return path if path.exists() else None


def parse_utc_timestamp_ms(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("timestamp value is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value[:10]).date()


def _next_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def regular_session_open_ms_on_or_after(date_value: str) -> int:
    day = _next_weekday(_parse_date(date_value))
    opened = datetime.combine(day, RTH_OPEN, tzinfo=EASTERN)
    return int(opened.astimezone(timezone.utc).timestamp() * 1000)


def regular_session_open_ms_after_date(date_value: str) -> int:
    day = _next_weekday(_parse_date(date_value) + timedelta(days=1))
    opened = datetime.combine(day, RTH_OPEN, tzinfo=EASTERN)
    return int(opened.astimezone(timezone.utc).timestamp() * 1000)


def _first_present(payload: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_covariate_timestamps_ms(source_dataset: str, payload: Mapping[str, Any]) -> tuple[int, int]:
    if source_dataset == "news":
        published = _first_present(payload, ("published_utc", "published_at", "created_at"))
        if published is None:
            raise ValueError("news record is missing published_utc")
        timestamp_ms = parse_utc_timestamp_ms(str(published))
        return timestamp_ms, timestamp_ms

    if source_dataset == "overview_snapshots":
        asof = _first_present(payload, ("asof_date", "date", "request_date"))
        if asof is None:
            raise ValueError("overview snapshot is missing asof_date")
        timestamp_ms = regular_session_open_ms_on_or_after(str(asof))
        return timestamp_ms, timestamp_ms

    if source_dataset == "financials":
        event_date = _first_present(payload, ("end_date", "fiscal_period_end", "period_of_report_date"))
        filing_date = _first_present(payload, ("filing_date", "filed_date", "acceptance_datetime"))
        if filing_date is None:
            raise ValueError("financial record is missing filing_date")
        if "T" in str(filing_date):
            available_ms = parse_utc_timestamp_ms(str(filing_date))
        else:
            available_ms = regular_session_open_ms_after_date(str(filing_date))
        event_ms = regular_session_open_ms_on_or_after(str(event_date or filing_date))
        return event_ms, available_ms

    if source_dataset == "dividends":
        event_date = _first_present(payload, ("ex_dividend_date", "pay_date", "record_date", "declaration_date"))
        declaration_date = _first_present(payload, ("declaration_date", "announcement_date", "declared_date"))
        if event_date is None:
            raise ValueError("dividend record is missing an event date")
        event_ms = regular_session_open_ms_on_or_after(str(event_date))
        if declaration_date is None:
            return event_ms, event_ms
        return event_ms, regular_session_open_ms_after_date(str(declaration_date))

    if source_dataset == "splits":
        event_date = _first_present(payload, ("execution_date", "split_date", "effective_date", "ex_date"))
        announcement_date = _first_present(payload, ("announcement_date", "declaration_date", "declared_date"))
        if event_date is None:
            raise ValueError("split record is missing an effective date")
        event_ms = regular_session_open_ms_on_or_after(str(event_date))
        if announcement_date is None:
            return event_ms, event_ms
        return event_ms, regular_session_open_ms_after_date(str(announcement_date))

    raise ValueError(f"Unsupported covariate dataset {source_dataset!r}")


def source_record_id(source_dataset: str, payload: Mapping[str, Any], *, line_number: int) -> str:
    explicit = _first_present(
        payload,
        (
            "id",
            "source_filing_url",
            "source_filing_file_url",
            "article_url",
            "ticker",
            "asof_date",
        ),
    )
    if explicit is not None:
        return str(explicit)
    return f"{source_dataset}:line-{line_number}"


def normalize_raw_covariate_record(
    *,
    symbol: str,
    source_dataset: str,
    payload: Mapping[str, Any],
    line_number: int = 0,
) -> RawCovariateRecord:
    event_ms, available_ms = infer_covariate_timestamps_ms(source_dataset, payload)
    record_payload = dict(payload)
    record_hash = stable_json_hash(record_payload)
    natural_id = source_record_id(source_dataset, record_payload, line_number=line_number)
    composite_id = f"{source_dataset}:{canonical_symbol(symbol)}:{natural_id}:{record_hash[:16]}"
    return RawCovariateRecord(
        symbol=canonical_symbol(symbol),
        source_dataset=source_dataset,
        event_timestamp_ms=int(event_ms),
        available_timestamp_ms=int(available_ms),
        source_record_id=composite_id,
        source_record_hash=record_hash,
        payload=record_payload,
    )


def iter_jsonl_payloads(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                yield line_number, payload


def load_raw_covariate_records_for_symbol(
    root: Path,
    symbol: str,
    *,
    datasets: Iterable[str] = SUPPORTED_COVARIATE_DATASETS,
) -> list[RawCovariateRecord]:
    records: list[RawCovariateRecord] = []
    for dataset in datasets:
        path = covariate_jsonl_path(root, symbol, dataset)
        if path is None:
            continue
        for line_number, payload in iter_jsonl_payloads(path):
            records.append(
                normalize_raw_covariate_record(
                    symbol=symbol,
                    source_dataset=dataset,
                    payload=payload,
                    line_number=line_number,
                )
            )
    return sorted(records, key=lambda record: (record.available_timestamp_ms, record.source_dataset))


def covariate_source_coverage(root: Path, symbol: str) -> dict[str, bool]:
    return {
        dataset: covariate_jsonl_path(root, symbol, dataset) is not None
        for dataset in SUPPORTED_COVARIATE_DATASETS
    }
