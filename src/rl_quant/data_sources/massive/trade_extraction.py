"""Streaming, source-bound extraction from committed Massive trade flat files."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    open_loaded_massive_source_stream,
)
from rl_quant.data_sources.massive.recorder_clock import MassiveRecorderClockAuthority
from rl_quant.data_sources.massive.trade_canonicalization import (
    MassiveCanonicalTradeSourceRecord,
    canonicalize_massive_flat_file_trade,
    canonicalize_massive_websocket_trade,
)
from rl_quant.data_sources.massive.trade_replay import MassiveResolvedSecurityIdentity
from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketCaptureAuthority,
    MassiveParsedWebSocketTradeMessage,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256


MASSIVE_FLAT_TRADES_DATASET_ID = "us_stocks_sip/trades_v1"
MASSIVE_FLAT_TRADE_COLUMNS = (
    "ticker",
    "conditions",
    "correction",
    "exchange",
    "id",
    "participant_timestamp",
    "price",
    "sequence_number",
    "sip_timestamp",
    "size",
    "tape",
    "trf_id",
    "trf_timestamp",
)
MASSIVE_FLAT_TRADE_SCHEMA_SHA256 = semantic_sha256(
    {
        "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
        "columns": MASSIVE_FLAT_TRADE_COLUMNS,
        "timestamp_unit": "nanoseconds",
        "compression": "gzip",
    }
)
MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256 = semantic_sha256(
    {
        "dataset": MASSIVE_FLAT_TRADES_DATASET_ID,
        "format": "streaming-gzip-csv-with-exact-header",
        "columns": MASSIVE_FLAT_TRADE_COLUMNS,
        "timestamp_unit": "nanoseconds",
        "selection": "exact-pit-ticker-and-eastern-sip-session-date",
        "row_provenance": "physical-line-number-and-raw-line-sha256",
    }
)
MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_EXTRACTED_TRADE_ROW_SCHEMA = "rl-quant.massive-extracted-trade-row-v1"
MASSIVE_EXTRACTED_WEBSOCKET_TRADE_ROW_SCHEMA = (
    "rl-quant.massive-extracted-websocket-trade-row-v3"
)
MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA = (
    "rl-quant.massive-trade-extraction-evidence-v2"
)


class MassiveTradeExtractionError(ValueError):
    """A committed flat file cannot be completely and causally extracted."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveTradeExtractionError(f"{name} is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveExtractedTradeRow:
    source_row_number: int
    raw_row_sha256: str
    canonical_record: MassiveCanonicalTradeSourceRecord
    receipt_sha256: str
    schema: str = MASSIVE_EXTRACTED_TRADE_ROW_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_row_number": self.source_row_number,
            "raw_row_sha256": self.raw_row_sha256,
            "canonical_record": asdict(self.canonical_record),
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_EXTRACTED_TRADE_ROW_SCHEMA:
            raise MassiveTradeExtractionError("extracted trade row schema drifted")
        if (
            isinstance(self.source_row_number, bool)
            or not isinstance(self.source_row_number, int)
            or self.source_row_number < 2
        ):
            raise MassiveTradeExtractionError(
                "source row number must include the header"
            )
        _digest("raw row SHA", self.raw_row_sha256)
        self.canonical_record.validate()
        if self.canonical_record.raw_source_record_sha256 != self.raw_row_sha256:
            raise MassiveTradeExtractionError("canonical row lost raw provenance")
        _digest("extracted row receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeExtractionError("extracted row receipt differs")

    @classmethod
    def build(
        cls,
        *,
        source_row_number: int,
        raw_row_sha256: str,
        canonical_record: MassiveCanonicalTradeSourceRecord,
    ) -> MassiveExtractedTradeRow:
        body = {
            "schema": MASSIVE_EXTRACTED_TRADE_ROW_SCHEMA,
            "source_row_number": source_row_number,
            "raw_row_sha256": raw_row_sha256,
            "canonical_record": asdict(canonical_record),
        }
        value = cls(
            source_row_number=source_row_number,
            raw_row_sha256=raw_row_sha256,
            canonical_record=canonical_record,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveExtractedWebSocketTradeRow:
    source_line_number: int
    server_batch_index: int
    message_index: int
    session_date: str
    local_received_at_ns: int
    canonical_received_at_ns: int
    raw_payload_sha256: str
    parsed_message: MassiveParsedWebSocketTradeMessage
    parsed_message_receipt_sha256: str
    canonical_record: MassiveCanonicalTradeSourceRecord
    parser_evidence_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_EXTRACTED_WEBSOCKET_TRADE_ROW_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_line_number": self.source_line_number,
            "server_batch_index": self.server_batch_index,
            "message_index": self.message_index,
            "session_date": self.session_date,
            "local_received_at_ns": self.local_received_at_ns,
            "canonical_received_at_ns": self.canonical_received_at_ns,
            "raw_payload_sha256": self.raw_payload_sha256,
            "parsed_message": asdict(self.parsed_message),
            "parsed_message_receipt_sha256": self.parsed_message_receipt_sha256,
            "canonical_record": asdict(self.canonical_record),
            "parser_evidence_receipt_sha256": self.parser_evidence_receipt_sha256,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_EXTRACTED_WEBSOCKET_TRADE_ROW_SCHEMA:
            raise MassiveTradeExtractionError(
                "extracted WebSocket trade schema drifted"
            )
        for name in (
            "source_line_number",
            "server_batch_index",
            "message_index",
            "local_received_at_ns",
            "canonical_received_at_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveTradeExtractionError(
                    f"extracted WebSocket {name} must be nonnegative"
                )
        if self.source_line_number <= 0 or self.server_batch_index <= 0:
            raise MassiveTradeExtractionError(
                "extracted WebSocket source location is absent"
            )
        if not self.session_date or self.session_date != self.session_date.strip():
            raise MassiveTradeExtractionError(
                "extracted WebSocket session date is absent"
            )
        for name in (
            "raw_payload_sha256",
            "parsed_message_receipt_sha256",
            "parser_evidence_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        self.parsed_message.validate()
        if (
            self.parsed_message.receipt_sha256 != self.parsed_message_receipt_sha256
            or self.parsed_message.source_line_number != self.source_line_number
            or self.parsed_message.server_batch_index != self.server_batch_index
            or self.parsed_message.message_index != self.message_index
            or self.parsed_message.event.session_date != self.session_date
            or self.parsed_message.event.received_at_ns != self.local_received_at_ns
            or self.parsed_message.event.payload_sha256 != self.raw_payload_sha256
        ):
            raise MassiveTradeExtractionError(
                "extracted WebSocket row differs from its parsed message"
            )
        self.canonical_record.validate()
        if self.canonical_record.source_kind != "delayed-websocket":
            raise MassiveTradeExtractionError(
                "extracted WebSocket row has another source kind"
            )
        if (
            self.canonical_record.raw_source_record_sha256 != self.raw_payload_sha256
            or self.canonical_record.local_received_at_ns != self.local_received_at_ns
            or self.canonical_record.canonical_received_at_ns
            != self.canonical_received_at_ns
        ):
            raise MassiveTradeExtractionError(
                "extracted WebSocket canonical provenance differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeExtractionError(
                "extracted WebSocket trade receipt differs"
            )

    @classmethod
    def build(
        cls,
        *,
        parsed_message: MassiveParsedWebSocketTradeMessage,
        parser_evidence_receipt_sha256: str,
        recorder_clock_authority: MassiveRecorderClockAuthority,
    ) -> MassiveExtractedWebSocketTradeRow:
        parsed_message.validate()
        recorder_clock_authority.validate()
        canonical_record = canonicalize_massive_websocket_trade(
            parsed_message.event,
            recorder_clock_authority=recorder_clock_authority,
        )
        if (
            canonical_record.local_received_at_ns is None
            or canonical_record.canonical_received_at_ns is None
        ):  # defensive against canonicalizer contract drift
            raise MassiveTradeExtractionError(
                "canonical WebSocket row lacks receive-time provenance"
            )
        body = {
            "schema": MASSIVE_EXTRACTED_WEBSOCKET_TRADE_ROW_SCHEMA,
            "source_line_number": parsed_message.source_line_number,
            "server_batch_index": parsed_message.server_batch_index,
            "message_index": parsed_message.message_index,
            "session_date": parsed_message.event.session_date,
            "local_received_at_ns": canonical_record.local_received_at_ns,
            "canonical_received_at_ns": canonical_record.canonical_received_at_ns,
            "raw_payload_sha256": parsed_message.event.payload_sha256,
            "parsed_message": asdict(parsed_message),
            "parsed_message_receipt_sha256": parsed_message.receipt_sha256,
            "canonical_record": asdict(canonical_record),
            "parser_evidence_receipt_sha256": parser_evidence_receipt_sha256,
        }
        value = cls(
            source_line_number=parsed_message.source_line_number,
            server_batch_index=parsed_message.server_batch_index,
            message_index=parsed_message.message_index,
            session_date=parsed_message.event.session_date,
            local_received_at_ns=canonical_record.local_received_at_ns,
            canonical_received_at_ns=canonical_record.canonical_received_at_ns,
            raw_payload_sha256=parsed_message.event.payload_sha256,
            parsed_message=parsed_message,
            parsed_message_receipt_sha256=parsed_message.receipt_sha256,
            canonical_record=canonical_record,
            parser_evidence_receipt_sha256=parser_evidence_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


def extract_massive_websocket_trade_rows(
    *,
    parsed_messages: Sequence[MassiveParsedWebSocketTradeMessage],
    capture: MassiveDelayedWebSocketCaptureAuthority,
    recorder_clock_authority: MassiveRecorderClockAuthority,
) -> tuple[MassiveExtractedWebSocketTradeRow, ...]:
    """Derive canonical rows only from the committed capture parser output."""

    capture.validate()
    recorder_clock_authority.validate()
    if (
        capture.lifecycle.recorder_clock_authority_receipt_sha256
        != recorder_clock_authority.receipt_sha256
    ):
        raise MassiveTradeExtractionError(
            "WebSocket extraction used another recorder clock"
        )
    parser_evidence = capture.parser_evidence
    if parser_evidence is None:
        raise MassiveTradeExtractionError(
            "WebSocket capture lacks committed parser evidence"
        )
    for message in parsed_messages:
        message.validate()
    if len(parsed_messages) != capture.event_count:
        raise MassiveTradeExtractionError(
            "parsed WebSocket message count differs from capture"
        )
    if (
        semantic_sha256(tuple(message.receipt_sha256 for message in parsed_messages))
        != parser_evidence.parsed_trade_transport_inventory_sha256
    ):
        raise MassiveTradeExtractionError(
            "parsed WebSocket transport inventory differs"
        )
    rows = tuple(
        MassiveExtractedWebSocketTradeRow.build(
            parsed_message=message,
            parser_evidence_receipt_sha256=parser_evidence.receipt_sha256,
            recorder_clock_authority=recorder_clock_authority,
        )
        for message in parsed_messages
    )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.source_line_number,
                row.server_batch_index,
                row.message_index,
            ),
        )
    )
    if (
        semantic_sha256(tuple(row.canonical_record.receipt_sha256 for row in ordered))
        != parser_evidence.parsed_trade_canonical_inventory_sha256
    ):
        raise MassiveTradeExtractionError(
            "parsed WebSocket canonical inventory differs"
        )
    return ordered


@dataclass(frozen=True, slots=True)
class MassiveTradeExtractionEvidence:
    loaded_source_receipt_sha256: str
    source_receipt_sha256: str
    source_commit_receipt_sha256: str
    dataset_id: str
    source_object_key: str
    source_schema_sha256: str
    parser_spec_sha256: str
    parser_source_sha256: str
    security_id: str
    source_ticker: str
    session_date: str
    source_row_count: int
    selected_row_count: int
    unselected_row_count: int
    rejected_row_count: int
    selected_canonical_record_inventory_sha256: str
    selected_raw_source_record_inventory_sha256: str
    selected_row_provenance_inventory_sha256: str
    complete_for_security_session: bool
    receipt_sha256: str
    schema: str = MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA:
            raise MassiveTradeExtractionError("trade extraction schema drifted")
        for name in (
            "loaded_source_receipt_sha256",
            "source_receipt_sha256",
            "source_commit_receipt_sha256",
            "source_schema_sha256",
            "parser_spec_sha256",
            "parser_source_sha256",
            "selected_canonical_record_inventory_sha256",
            "selected_raw_source_record_inventory_sha256",
            "selected_row_provenance_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveTradeExtractionError("flat-file dataset identity drifted")
        if self.source_schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256:
            raise MassiveTradeExtractionError("flat-file source schema drifted")
        if self.parser_spec_sha256 != MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256:
            raise MassiveTradeExtractionError("flat trade parser spec drifted")
        if self.parser_source_sha256 != MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256:
            raise MassiveTradeExtractionError("flat trade parser source drifted")
        expected_key = (
            f"{MASSIVE_FLAT_TRADES_DATASET_ID}/{self.session_date[:4]}/"
            f"{self.session_date[5:7]}/{self.session_date}.csv.gz"
        )
        if self.source_object_key != expected_key:
            raise MassiveTradeExtractionError("flat-file object key drifted")
        if not self.security_id or not self.source_ticker or not self.session_date:
            raise MassiveTradeExtractionError("trade extraction identity is absent")
        for name in (
            "source_row_count",
            "selected_row_count",
            "unselected_row_count",
            "rejected_row_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveTradeExtractionError(f"{name} must be nonnegative")
        if self.source_row_count != (
            self.selected_row_count
            + self.unselected_row_count
            + self.rejected_row_count
        ):
            raise MassiveTradeExtractionError("trade extraction rows do not reconcile")
        exact = self.selected_row_count > 0 and self.rejected_row_count == 0
        if self.complete_for_security_session is not exact:
            raise MassiveTradeExtractionError("trade extraction completeness drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeExtractionError("trade extraction receipt differs")


def _parse_conditions(value: str) -> tuple[int, ...]:
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
        raise MassiveTradeExtractionError("flat-file conditions are malformed")
    return tuple(int(item) for item in parsed)


def _parse_csv_line(raw_line: str, *, source_row_number: int) -> list[str]:
    try:
        rows = list(csv.reader((raw_line,)))
    except csv.Error as exc:
        raise MassiveTradeExtractionError(
            f"flat-file row {source_row_number} is malformed"
        ) from exc
    if len(rows) != 1 or len(rows[0]) != len(MASSIVE_FLAT_TRADE_COLUMNS):
        raise MassiveTradeExtractionError(
            f"flat-file row {source_row_number} has the wrong field count"
        )
    return rows[0]


def extract_massive_flat_file_security_session(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    identity_resolution: MassiveResolvedSecurityIdentity,
) -> tuple[
    tuple[MassiveExtractedTradeRow, ...],
    MassiveTradeExtractionEvidence,
]:
    """Stream every committed row and select one exact PIT ticker-session."""

    loaded_source.validate()
    identity_resolution.validate()
    expected_key = (
        f"{MASSIVE_FLAT_TRADES_DATASET_ID}/{identity_resolution.session_date[:4]}/"
        f"{identity_resolution.session_date[5:7]}/"
        f"{identity_resolution.session_date}.csv.gz"
    )
    if loaded_source.receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
        raise MassiveTradeExtractionError("flat-file dataset identity differs")
    if loaded_source.receipt.source_object_key != expected_key:
        raise MassiveTradeExtractionError("flat-file object key differs")
    if loaded_source.receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256:
        raise MassiveTradeExtractionError("flat-file schema authority differs")

    selected: list[MassiveExtractedTradeRow] = []
    source_rows = 0
    unselected = 0
    with open_loaded_massive_source_stream(
        root=root, loaded_source=loaded_source
    ) as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="rb") as decompressed:
            with TextIOWrapper(
                decompressed, encoding="utf-8", newline=""
            ) as text_stream:
                header_line = text_stream.readline()
                if not header_line:
                    raise MassiveTradeExtractionError("flat-file source is empty")
                header_rows = list(csv.reader((header_line,)))
                header = tuple(header_rows[0]) if len(header_rows) == 1 else ()
                if header != MASSIVE_FLAT_TRADE_COLUMNS:
                    raise MassiveTradeExtractionError("flat-file trade schema differs")
                for source_row_number, raw_line in enumerate(text_stream, start=2):
                    source_rows += 1
                    values = _parse_csv_line(
                        raw_line, source_row_number=source_row_number
                    )
                    row: dict[str, object] = dict(
                        zip(MASSIVE_FLAT_TRADE_COLUMNS, values, strict=True)
                    )
                    row["conditions"] = _parse_conditions(str(row["conditions"]))
                    for nullable in ("trf_id", "trf_timestamp", "tape"):
                        if row[nullable] == "":
                            row[nullable] = None
                    raw_row_sha256 = hashlib.sha256(
                        raw_line.encode("utf-8")
                    ).hexdigest()
                    canonical = canonicalize_massive_flat_file_trade(
                        row,
                        raw_source_record_sha256=raw_row_sha256,
                    )
                    observed_date = (
                        datetime.fromtimestamp(
                            canonical.sip_timestamp_ns / 1_000_000_000,
                            tz=ZoneInfo("America/New_York"),
                        )
                        .date()
                        .isoformat()
                    )
                    if (
                        canonical.ticker == identity_resolution.source_ticker
                        and observed_date == identity_resolution.session_date
                    ):
                        selected.append(
                            MassiveExtractedTradeRow.build(
                                source_row_number=source_row_number,
                                raw_row_sha256=raw_row_sha256,
                                canonical_record=canonical,
                            )
                        )
                    else:
                        unselected += 1
    if source_rows <= 0:
        raise MassiveTradeExtractionError("flat-file source has no data rows")
    ordered = tuple(
        sorted(
            selected,
            key=lambda row: (
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.canonical_record.exchange_id,
                row.canonical_record.trade_id,
                row.source_row_number,
            ),
        )
    )
    source_ordered = tuple(sorted(ordered, key=lambda row: row.source_row_number))
    canonical_inventory = semantic_sha256(
        tuple(row.canonical_record.receipt_sha256 for row in source_ordered)
    )
    raw_replay_inventory = semantic_sha256(
        tuple(
            (
                row.canonical_record.ticker,
                row.canonical_record.sequence_number,
                row.raw_row_sha256,
            )
            for row in source_ordered
        )
    )
    provenance_inventory = semantic_sha256(
        tuple(
            (
                row.source_row_number,
                row.raw_row_sha256,
                row.canonical_record.receipt_sha256,
            )
            for row in source_ordered
        )
    )
    body = {
        "schema": MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        "dataset_id": loaded_source.receipt.dataset_id,
        "source_object_key": loaded_source.receipt.source_object_key,
        "source_schema_sha256": loaded_source.receipt.schema_sha256,
        "parser_spec_sha256": MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256,
        "security_id": identity_resolution.security_id,
        "source_ticker": identity_resolution.source_ticker,
        "session_date": identity_resolution.session_date,
        "source_row_count": source_rows,
        "selected_row_count": len(ordered),
        "unselected_row_count": unselected,
        "rejected_row_count": 0,
        "selected_canonical_record_inventory_sha256": canonical_inventory,
        "selected_raw_source_record_inventory_sha256": raw_replay_inventory,
        "selected_row_provenance_inventory_sha256": provenance_inventory,
        "complete_for_security_session": bool(ordered),
    }
    evidence = MassiveTradeExtractionEvidence(
        loaded_source_receipt_sha256=loaded_source.receipt_sha256,
        source_receipt_sha256=loaded_source.receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded_source.commit.receipt_sha256,
        dataset_id=loaded_source.receipt.dataset_id,
        source_object_key=loaded_source.receipt.source_object_key,
        source_schema_sha256=loaded_source.receipt.schema_sha256,
        parser_spec_sha256=MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
        parser_source_sha256=MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256,
        security_id=identity_resolution.security_id,
        source_ticker=identity_resolution.source_ticker,
        session_date=identity_resolution.session_date,
        source_row_count=source_rows,
        selected_row_count=len(ordered),
        unselected_row_count=unselected,
        rejected_row_count=0,
        selected_canonical_record_inventory_sha256=canonical_inventory,
        selected_raw_source_record_inventory_sha256=raw_replay_inventory,
        selected_row_provenance_inventory_sha256=provenance_inventory,
        complete_for_security_session=bool(ordered),
        receipt_sha256=semantic_sha256(body),
    )
    evidence.validate()
    return ordered, evidence


__all__ = [
    "MASSIVE_EXTRACTED_TRADE_ROW_SCHEMA",
    "MASSIVE_EXTRACTED_WEBSOCKET_TRADE_ROW_SCHEMA",
    "MASSIVE_FLAT_TRADES_DATASET_ID",
    "MASSIVE_FLAT_TRADE_COLUMNS",
    "MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256",
    "MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256",
    "MASSIVE_FLAT_TRADE_SCHEMA_SHA256",
    "MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA",
    "MassiveExtractedTradeRow",
    "MassiveExtractedWebSocketTradeRow",
    "MassiveTradeExtractionError",
    "MassiveTradeExtractionEvidence",
    "extract_massive_flat_file_security_session",
    "extract_massive_websocket_trade_rows",
]
