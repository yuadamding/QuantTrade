"""Whole-file, participant-time scan authority for finalized Massive trades."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
from io import TextIOWrapper
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_listing import (
    coverage_session_from_massive_trade_key,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    open_loaded_massive_source_stream,
)
from rl_quant.data_sources.massive.trade_canonicalization import (
    canonicalize_massive_flat_file_trade,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MassiveExtractedTradeRow,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256


EASTERN = ZoneInfo("America/New_York")
MASSIVE_DAILY_TRADE_FILE_SCAN_V0_SCHEMA = (
    "rl-quant.massive-daily-trade-file-scan-v0"
)
MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256 = semantic_sha256(
    {
        "dataset": MASSIVE_FLAT_TRADES_DATASET_ID,
        "format": "streaming-gzip-csv-exact-header",
        "row_scope": "every-source-row",
        "source_day": "participant-and-sip-eastern-calendar-date-equal-object-date",
        "economic_event_clock": "participant-timestamp",
        "correction_order": "sip-timestamp-sequence-exchange-trade-row",
        "regular_session_domain": "participant-timestamp-in-[open,close)",
        "after_close_corrections": "retained-for-finalized-replay",
        "row_provenance": "physical-line-raw-sha-canonical-receipt",
    }
)
MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveDailyTradeFileScanError(ValueError):
    """A finalized daily trade file cannot be certified in full."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveDailyTradeFileScanError(f"{name} must be a lowercase SHA-256")
    return value


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveDailyTradeFileScanError(f"{name} must be nonnegative")
    return value


def _timestamp_date(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=EASTERN).date().isoformat()


def _parse_conditions(value: str) -> tuple[int, ...]:
    import json

    stripped = value.strip()
    if not stripped:
        return ()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = [
            item
            for item in stripped.replace("[", "").replace("]", "").split(",")
            if item
        ]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise MassiveDailyTradeFileScanError("flat-file conditions are malformed")
    return tuple(int(item) for item in parsed)


def _parse_row(raw_line: str, *, source_row_number: int) -> list[str]:
    try:
        parsed = list(csv.reader((raw_line,)))
    except csv.Error as exc:
        raise MassiveDailyTradeFileScanError(
            f"flat-file row {source_row_number} is malformed"
        ) from exc
    if len(parsed) != 1 or len(parsed[0]) != len(MASSIVE_FLAT_TRADE_COLUMNS):
        raise MassiveDailyTradeFileScanError(
            f"flat-file row {source_row_number} has the wrong field count"
        )
    return parsed[0]


@dataclass(frozen=True, slots=True)
class MassiveDailyTradeFileScanEvidenceV0:
    source_session_date: str
    exchange: str
    regular_open_ns: int
    regular_close_ns: int
    source_row_count: int
    minimum_participant_timestamp_ns: int
    maximum_participant_timestamp_ns: int
    minimum_sip_timestamp_ns: int
    maximum_sip_timestamp_ns: int
    observed_participant_calendar_dates: tuple[str, ...]
    observed_sip_calendar_dates: tuple[str, ...]
    ticker_count: int
    all_row_canonical_inventory_sha256: str
    all_row_provenance_inventory_sha256: str
    regular_session_row_count: int
    premarket_row_count: int
    after_hours_row_count: int
    late_report_row_count: int
    post_close_correction_row_count: int
    correction_authority_receipt_sha256: str
    loaded_source_receipt_sha256: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    source_schema_sha256: str
    session_authority_receipt_sha256: str
    parser_spec_sha256: str
    parser_source_sha256: str
    compressed_bytes: int
    receipt_sha256: str
    schema: str = MASSIVE_DAILY_TRADE_FILE_SCAN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DAILY_TRADE_FILE_SCAN_V0_SCHEMA:
            raise MassiveDailyTradeFileScanError("daily scan schema drifted")
        if not self.source_session_date or not self.exchange:
            raise MassiveDailyTradeFileScanError("daily scan identity is absent")
        for name in (
            "source_row_count",
            "minimum_participant_timestamp_ns",
            "maximum_participant_timestamp_ns",
            "minimum_sip_timestamp_ns",
            "maximum_sip_timestamp_ns",
            "ticker_count",
            "regular_session_row_count",
            "premarket_row_count",
            "after_hours_row_count",
            "late_report_row_count",
            "post_close_correction_row_count",
            "compressed_bytes",
        ):
            _count(name, getattr(self, name))
        if self.source_row_count <= 0 or self.ticker_count <= 0:
            raise MassiveDailyTradeFileScanError("daily scan is empty")
        if self.minimum_participant_timestamp_ns > self.maximum_participant_timestamp_ns:
            raise MassiveDailyTradeFileScanError("participant timestamp range is inverted")
        if self.minimum_sip_timestamp_ns > self.maximum_sip_timestamp_ns:
            raise MassiveDailyTradeFileScanError("SIP timestamp range is inverted")
        if self.regular_close_ns <= self.regular_open_ns:
            raise MassiveDailyTradeFileScanError("daily scan session bounds are invalid")
        if self.source_row_count != (
            self.regular_session_row_count
            + self.premarket_row_count
            + self.after_hours_row_count
        ):
            raise MassiveDailyTradeFileScanError("daily scan event domains do not reconcile")
        expected_dates = (self.source_session_date,)
        if (
            self.observed_participant_calendar_dates != expected_dates
            or self.observed_sip_calendar_dates != expected_dates
        ):
            raise MassiveDailyTradeFileScanError(
                "daily scan contains rows outside the source calendar date"
            )
        for name in (
            "all_row_canonical_inventory_sha256",
            "all_row_provenance_inventory_sha256",
            "correction_authority_receipt_sha256",
            "loaded_source_receipt_sha256",
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "source_schema_sha256",
            "session_authority_receipt_sha256",
            "parser_spec_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.source_schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256:
            raise MassiveDailyTradeFileScanError("daily scan source schema drifted")
        if self.parser_spec_sha256 != MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256:
            raise MassiveDailyTradeFileScanError("daily scan specification drifted")
        if self.parser_source_sha256 != MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256:
            raise MassiveDailyTradeFileScanError("daily scan source implementation drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyTradeFileScanError("daily scan receipt differs")


def scan_massive_daily_trade_file_v0(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
    correction_authority: MassiveCorrectionAuthority,
) -> tuple[tuple[MassiveExtractedTradeRow, ...], MassiveDailyTradeFileScanEvidenceV0]:
    """Stream, canonicalize, and certify every row in one finalized trade file."""

    loaded_source.validate()
    session_authority.validate()
    session.validate()
    correction_authority.validate()
    source_date = coverage_session_from_massive_trade_key(
        loaded_source.receipt.source_object_key
    )
    if loaded_source.receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
        raise MassiveDailyTradeFileScanError("daily scan dataset differs")
    if loaded_source.receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256:
        raise MassiveDailyTradeFileScanError("daily scan source schema differs")
    if source_date != session.session_date:
        raise MassiveDailyTradeFileScanError("daily scan source and session dates differ")
    if session_authority.resolve(exchange=session.exchange, session_date=source_date) != session:
        raise MassiveDailyTradeFileScanError("daily scan session was not authority-resolved")

    rows: list[MassiveExtractedTradeRow] = []
    participant_dates: set[str] = set()
    sip_dates: set[str] = set()
    regular = premarket = after_hours = late_reports = post_close_corrections = 0
    with open_loaded_massive_source_stream(root=root, loaded_source=loaded_source) as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as decompressed:
            with TextIOWrapper(decompressed, encoding="utf-8", newline="") as text:
                header_line = text.readline()
                if not header_line:
                    raise MassiveDailyTradeFileScanError("daily trade source is empty")
                header_rows = list(csv.reader((header_line,)))
                header = tuple(header_rows[0]) if len(header_rows) == 1 else ()
                if header != MASSIVE_FLAT_TRADE_COLUMNS:
                    raise MassiveDailyTradeFileScanError("daily trade header differs")
                for source_row_number, raw_line in enumerate(text, start=2):
                    values = _parse_row(raw_line, source_row_number=source_row_number)
                    source: dict[str, object] = dict(
                        zip(MASSIVE_FLAT_TRADE_COLUMNS, values, strict=True)
                    )
                    source["conditions"] = _parse_conditions(str(source["conditions"]))
                    for nullable in ("trf_id", "trf_timestamp", "tape"):
                        if source[nullable] == "":
                            source[nullable] = None
                    raw_sha = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
                    canonical = canonicalize_massive_flat_file_trade(
                        source, raw_source_record_sha256=raw_sha
                    )
                    correction_kind = correction_authority.resolve(
                        0 if canonical.correction_code is None else canonical.correction_code
                    )
                    participant_date = _timestamp_date(canonical.participant_timestamp_ns)
                    sip_date = _timestamp_date(canonical.sip_timestamp_ns)
                    participant_dates.add(participant_date)
                    sip_dates.add(sip_date)
                    if participant_date != source_date or sip_date != source_date:
                        raise MassiveDailyTradeFileScanError(
                            f"row {source_row_number} lies outside source calendar date"
                        )
                    if canonical.participant_timestamp_ns < session.regular_open_ns:
                        premarket += 1
                    elif canonical.participant_timestamp_ns < session.regular_close_ns:
                        regular += 1
                    else:
                        after_hours += 1
                    if (
                        correction_kind == "late-report"
                        or (
                            session.is_regular(canonical.participant_timestamp_ns)
                            and canonical.sip_timestamp_ns >= session.regular_close_ns
                        )
                    ):
                        late_reports += 1
                    if (
                        correction_kind in {"replacement", "cancellation"}
                        and canonical.sip_timestamp_ns >= session.regular_close_ns
                    ):
                        post_close_corrections += 1
                    rows.append(
                        MassiveExtractedTradeRow.build(
                            source_row_number=source_row_number,
                            raw_row_sha256=raw_sha,
                            canonical_record=canonical,
                        )
                    )
    if not rows:
        raise MassiveDailyTradeFileScanError("daily trade source has no rows")
    source_rows = tuple(sorted(rows, key=lambda row: row.source_row_number))
    participant_times = tuple(row.canonical_record.participant_timestamp_ns for row in source_rows)
    sip_times = tuple(row.canonical_record.sip_timestamp_ns for row in source_rows)
    body: dict[str, object] = {
        "schema": MASSIVE_DAILY_TRADE_FILE_SCAN_V0_SCHEMA,
        "source_session_date": source_date,
        "exchange": session.exchange,
        "regular_open_ns": session.regular_open_ns,
        "regular_close_ns": session.regular_close_ns,
        "source_row_count": len(source_rows),
        "minimum_participant_timestamp_ns": min(participant_times),
        "maximum_participant_timestamp_ns": max(participant_times),
        "minimum_sip_timestamp_ns": min(sip_times),
        "maximum_sip_timestamp_ns": max(sip_times),
        "observed_participant_calendar_dates": tuple(sorted(participant_dates)),
        "observed_sip_calendar_dates": tuple(sorted(sip_dates)),
        "ticker_count": len({row.canonical_record.ticker for row in source_rows}),
        "all_row_canonical_inventory_sha256": semantic_sha256(
            tuple(row.canonical_record.receipt_sha256 for row in source_rows)
        ),
        "all_row_provenance_inventory_sha256": semantic_sha256(
            tuple(
                (row.source_row_number, row.raw_row_sha256, row.canonical_record.receipt_sha256)
                for row in source_rows
            )
        ),
        "regular_session_row_count": regular,
        "premarket_row_count": premarket,
        "after_hours_row_count": after_hours,
        "late_report_row_count": late_reports,
        "post_close_correction_row_count": post_close_corrections,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "source_object_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        "source_schema_sha256": loaded_source.receipt.schema_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "parser_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
        "compressed_bytes": loaded_source.receipt.content_length,
    }
    evidence = MassiveDailyTradeFileScanEvidenceV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    evidence.validate()
    return source_rows, evidence


__all__ = [
    "MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256",
    "MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256",
    "MASSIVE_DAILY_TRADE_FILE_SCAN_V0_SCHEMA",
    "MassiveDailyTradeFileScanError",
    "MassiveDailyTradeFileScanEvidenceV0",
    "scan_massive_daily_trade_file_v0",
]
