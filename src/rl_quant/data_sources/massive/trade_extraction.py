"""Complete security-session extraction from committed Massive trade flat files."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
from io import StringIO
import json
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_canonicalization import (
    MassiveCanonicalTradeSourceRecord,
    canonicalize_massive_flat_file_trade,
)
from rl_quant.data_sources.massive.trade_replay import MassiveResolvedSecurityIdentity
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256 = semantic_sha256(
    {
        "dataset": "us_stocks_sip/trades_v1",
        "format": "gzip-or-plain-csv-with-header",
        "columns": (
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
        ),
        "timestamp_unit": "nanoseconds",
        "selection": "exact-pit-ticker-and-session-date",
    }
)
MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA = (
    "rl-quant.massive-trade-extraction-evidence-v1"
)


class MassiveTradeExtractionError(ValueError):
    """A committed flat file cannot be completely and causally extracted."""


@dataclass(frozen=True, slots=True)
class MassiveTradeExtractionEvidence:
    loaded_source_receipt_sha256: str
    source_receipt_sha256: str
    source_commit_receipt_sha256: str
    parser_spec_sha256: str
    security_id: str
    source_ticker: str
    session_date: str
    source_row_count: int
    selected_row_count: int
    unselected_row_count: int
    rejected_row_count: int
    selected_canonical_record_inventory_sha256: str
    selected_raw_source_record_inventory_sha256: str
    complete_for_security_session: bool
    receipt_sha256: str
    schema: str = MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA:
            raise MassiveTradeExtractionError("trade extraction schema drifted")
        for name in (
            "loaded_source_receipt_sha256",
            "source_receipt_sha256",
            "source_commit_receipt_sha256",
            "parser_spec_sha256",
            "selected_canonical_record_inventory_sha256",
            "selected_raw_source_record_inventory_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise MassiveTradeExtractionError(f"{name} is not a SHA-256")
        if self.parser_spec_sha256 != MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256:
            raise MassiveTradeExtractionError("flat trade parser spec drifted")
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
        parsed = [item for item in stripped.replace("[", "").replace("]", "").split(",") if item]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise MassiveTradeExtractionError("flat-file conditions are malformed")
    return tuple(int(item) for item in parsed)


def extract_massive_flat_file_security_session(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    identity_resolution: MassiveResolvedSecurityIdentity,
) -> tuple[
    tuple[MassiveCanonicalTradeSourceRecord, ...],
    MassiveTradeExtractionEvidence,
]:
    """Parse every committed row and select one exact PIT ticker-session."""

    loaded_source.validate()
    identity_resolution.validate()
    payload = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    if payload.startswith(b"\x1f\x8b"):
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise MassiveTradeExtractionError("flat-file gzip is invalid") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MassiveTradeExtractionError("flat-file CSV is not UTF-8") from exc
    reader = csv.DictReader(StringIO(text))
    required = {
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
    }
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        raise MassiveTradeExtractionError("flat-file trade schema differs")
    selected: list[MassiveCanonicalTradeSourceRecord] = []
    source_rows = 0
    unselected = 0
    for source_rows, row in enumerate(reader, start=1):
        normalized = dict(row)
        normalized["conditions"] = _parse_conditions(row["conditions"])
        for nullable in ("trf_id", "trf_timestamp"):
            if normalized[nullable] == "":
                normalized[nullable] = None
        canonical = canonicalize_massive_flat_file_trade(normalized)
        observed_date = datetime.fromtimestamp(
            canonical.sip_timestamp_ns / 1_000_000_000,
            tz=ZoneInfo("America/New_York"),
        ).date().isoformat()
        if (
            canonical.ticker == identity_resolution.source_ticker
            and observed_date == identity_resolution.session_date
        ):
            selected.append(canonical)
        else:
            unselected += 1
    if source_rows <= 0:
        raise MassiveTradeExtractionError("flat-file source has no data rows")
    ordered = tuple(
        sorted(
            selected,
            key=lambda row: (
                row.sip_timestamp_ns,
                row.sequence_number,
                row.exchange_id,
                row.trade_id,
            ),
        )
    )
    body = {
        "schema": MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        "parser_spec_sha256": MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
        "security_id": identity_resolution.security_id,
        "source_ticker": identity_resolution.source_ticker,
        "session_date": identity_resolution.session_date,
        "source_row_count": source_rows,
        "selected_row_count": len(ordered),
        "unselected_row_count": unselected,
        "rejected_row_count": 0,
        "selected_canonical_record_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "selected_raw_source_record_inventory_sha256": semantic_sha256(
            tuple(
                (row.ticker, row.sequence_number, row.raw_source_record_sha256)
                for row in ordered
            )
        ),
        "complete_for_security_session": bool(ordered),
    }
    evidence = MassiveTradeExtractionEvidence(
        loaded_source_receipt_sha256=loaded_source.receipt_sha256,
        source_receipt_sha256=loaded_source.receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded_source.commit.receipt_sha256,
        parser_spec_sha256=MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
        security_id=identity_resolution.security_id,
        source_ticker=identity_resolution.source_ticker,
        session_date=identity_resolution.session_date,
        source_row_count=source_rows,
        selected_row_count=len(ordered),
        unselected_row_count=unselected,
        rejected_row_count=0,
        selected_canonical_record_inventory_sha256=str(
            body["selected_canonical_record_inventory_sha256"]
        ),
        selected_raw_source_record_inventory_sha256=str(
            body["selected_raw_source_record_inventory_sha256"]
        ),
        complete_for_security_session=bool(ordered),
        receipt_sha256=semantic_sha256(body),
    )
    evidence.validate()
    return ordered, evidence


__all__ = [
    "MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256",
    "MASSIVE_TRADE_EXTRACTION_EVIDENCE_SCHEMA",
    "MassiveTradeExtractionError",
    "MassiveTradeExtractionEvidence",
    "extract_massive_flat_file_security_session",
]
