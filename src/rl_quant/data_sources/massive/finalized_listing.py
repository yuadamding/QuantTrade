"""Committed vendor-listing evidence for finalized Massive trade files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path, PurePosixPath
import re

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)


MASSIVE_FINALIZED_V0_SOURCE_ROLE = "finalized-trades-v1"
MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES = (MASSIVE_FINALIZED_V0_SOURCE_ROLE,)
MASSIVE_FLAT_FILE_LISTING_DATASET_ID = "massive-flat-file-listing-v0"
MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA = (
    "rl-quant.massive-flat-file-listing-capture-v0"
)
MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
        "format": "canonical-json",
        "timestamp_unit": "milliseconds-since-unix-epoch",
        "entry_fields": (
            "dataset_id",
            "source_object_key",
            "etag",
            "content_length",
            "last_modified_at_ms",
        ),
    }
)
MASSIVE_VENDOR_LISTING_ENTRY_V0_SCHEMA = "rl-quant.massive-vendor-listing-entry-v0"
MASSIVE_COMMITTED_FLAT_FILE_LISTING_V0_SCHEMA = (
    "rl-quant.massive-committed-flat-file-listing-v0"
)
MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256 = semantic_sha256(
    {
        "input_dataset_id": MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
        "input_schema_sha256": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
        "format": "exact-canonical-json",
        "entry_order": "source-object-key-ascending",
        "duplicate_keys": "reject",
        "supported_dataset": MASSIVE_FLAT_TRADES_DATASET_ID,
    }
)
MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256 = file_sha256(Path(__file__))

_TRADE_KEY_PATTERN = re.compile(
    r"^us_stocks_sip/trades_v1/"
    r"(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/"
    r"(?P<session>[0-9]{4}-[0-9]{2}-[0-9]{2})\.csv\.gz$"
)


class MassiveFinalizedListingError(ValueError):
    """Committed vendor-listing evidence is absent or inconsistent."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveFinalizedListingError(
            f"{name} must be a canonical nonempty string"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedListingError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveFinalizedListingError(f"{name} must be a nonnegative integer")
    return value


def canonical_massive_trade_object_key(session_date: str) -> str:
    """Return the only accepted flat-file key for one source session."""

    try:
        parsed = date.fromisoformat(session_date)
    except (TypeError, ValueError) as exc:
        raise MassiveFinalizedListingError("coverage session date is invalid") from exc
    return (
        f"{MASSIVE_FLAT_TRADES_DATASET_ID}/{parsed:%Y}/{parsed:%m}/"
        f"{parsed.isoformat()}.csv.gz"
    )


def coverage_session_from_massive_trade_key(source_object_key: str) -> str:
    """Derive and validate the source session encoded by a canonical key."""

    key = PurePosixPath(_text("source object key", source_object_key)).as_posix()
    match = _TRADE_KEY_PATTERN.fullmatch(key)
    if match is None:
        raise MassiveFinalizedListingError(
            "trade object key does not use the canonical dataset/date layout"
        )
    session_date = match.group("session")
    if key != canonical_massive_trade_object_key(session_date):
        raise MassiveFinalizedListingError("trade object key date components disagree")
    return session_date


@dataclass(frozen=True, slots=True)
class MassiveVendorListingEntryV0:
    source_role: str
    dataset_id: str
    source_object_key: str
    coverage_session_date: str
    etag: str
    content_length: int
    vendor_last_modified_at_ms: int
    listing_observed_at_ms: int
    loaded_listing_receipt_sha256: str
    parser_spec_sha256: str
    parser_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_VENDOR_LISTING_ENTRY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VENDOR_LISTING_ENTRY_V0_SCHEMA:
            raise MassiveFinalizedListingError("listing-entry schema drifted")
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveFinalizedListingError("listing-entry source role drifted")
        if self.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveFinalizedListingError("listing-entry dataset drifted")
        coverage = coverage_session_from_massive_trade_key(self.source_object_key)
        if self.coverage_session_date != coverage:
            raise MassiveFinalizedListingError(
                "listing-entry coverage differs from its object key"
            )
        _text("ETag", self.etag)
        _nonnegative_int("content length", self.content_length)
        modified = _nonnegative_int(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        observed = _nonnegative_int("listing observation", self.listing_observed_at_ms)
        if observed < modified:
            raise MassiveFinalizedListingError(
                "listing observation predates vendor LastModified"
            )
        for name in (
            "loaded_listing_receipt_sha256",
            "parser_spec_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.parser_spec_sha256 != MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256:
            raise MassiveFinalizedListingError("listing parser specification drifted")
        if self.parser_source_sha256 != MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256:
            raise MassiveFinalizedListingError("listing parser source drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedListingError("listing-entry receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveCommittedFlatFileListingV0:
    loaded_listing_receipt_sha256: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    listing_observed_at_ms: int
    entries: tuple[MassiveVendorListingEntryV0, ...]
    parser_spec_sha256: str
    parser_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_COMMITTED_FLAT_FILE_LISTING_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_COMMITTED_FLAT_FILE_LISTING_V0_SCHEMA:
            raise MassiveFinalizedListingError("committed-listing schema drifted")
        for name in (
            "loaded_listing_receipt_sha256",
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "parser_spec_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        _nonnegative_int("listing observation", self.listing_observed_at_ms)
        if self.parser_spec_sha256 != MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256:
            raise MassiveFinalizedListingError("listing parser specification drifted")
        if self.parser_source_sha256 != MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256:
            raise MassiveFinalizedListingError("listing parser source drifted")
        keys = tuple(entry.source_object_key for entry in self.entries)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveFinalizedListingError(
                "listing entries must be sorted and unique"
            )
        for entry in self.entries:
            entry.validate()
            if (
                entry.loaded_listing_receipt_sha256
                != self.loaded_listing_receipt_sha256
                or entry.listing_observed_at_ms != self.listing_observed_at_ms
            ):
                raise MassiveFinalizedListingError(
                    "listing entry differs from its committed listing"
                )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedListingError("committed-listing receipt differs")

    def resolve(self, *, source_object_key: str) -> MassiveVendorListingEntryV0:
        self.validate()
        for entry in self.entries:
            if entry.source_object_key == source_object_key:
                return entry
        raise MassiveFinalizedListingError(
            "source object is absent from the committed vendor listing"
        )


def parse_massive_flat_file_listing_v0(
    *,
    root: str | Path,
    loaded_listing: LoadedMassiveSourceObject,
) -> MassiveCommittedFlatFileListingV0:
    """Parse one exact canonical listing capture from committed bytes."""

    loaded_listing.validate()
    if loaded_listing.receipt.dataset_id != MASSIVE_FLAT_FILE_LISTING_DATASET_ID:
        raise MassiveFinalizedListingError("listing source dataset differs")
    if (
        loaded_listing.receipt.schema_sha256
        != MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256
    ):
        raise MassiveFinalizedListingError("listing source schema differs")
    payload_bytes = read_loaded_massive_source_bytes(
        root=root, loaded_source=loaded_listing
    )
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveFinalizedListingError("listing source is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or canonical_json_file_bytes(payload) != payload_bytes
    ):
        raise MassiveFinalizedListingError(
            "listing source must use exact canonical JSON bytes"
        )
    if set(payload) != {"schema", "observed_at_ms", "entries"}:
        raise MassiveFinalizedListingError("listing source field inventory drifted")
    if payload["schema"] != MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA:
        raise MassiveFinalizedListingError("listing capture schema drifted")
    observed = _nonnegative_int("listing observation", payload["observed_at_ms"])
    if observed != loaded_listing.receipt.downloaded_at_ms:
        raise MassiveFinalizedListingError(
            "listing observation differs from committed download completion"
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise MassiveFinalizedListingError("listing entry inventory is empty")
    entries: list[MassiveVendorListingEntryV0] = []
    expected_fields = {
        "dataset_id",
        "source_object_key",
        "etag",
        "content_length",
        "last_modified_at_ms",
    }
    for raw in raw_entries:
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise MassiveFinalizedListingError("listing entry fields drifted")
        dataset_id = _text("listing dataset ID", raw["dataset_id"])
        if dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveFinalizedListingError("listing contains an unknown dataset")
        key = _text("listing source object key", raw["source_object_key"])
        coverage = coverage_session_from_massive_trade_key(key)
        etag = _text("listing ETag", raw["etag"])
        content_length = _nonnegative_int(
            "listing content length", raw["content_length"]
        )
        last_modified = _nonnegative_int(
            "listing LastModified", raw["last_modified_at_ms"]
        )
        body = {
            "schema": MASSIVE_VENDOR_LISTING_ENTRY_V0_SCHEMA,
            "source_role": MASSIVE_FINALIZED_V0_SOURCE_ROLE,
            "dataset_id": dataset_id,
            "source_object_key": key,
            "coverage_session_date": coverage,
            "etag": etag,
            "content_length": content_length,
            "vendor_last_modified_at_ms": last_modified,
            "listing_observed_at_ms": observed,
            "loaded_listing_receipt_sha256": loaded_listing.receipt_sha256,
            "parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
            "parser_source_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256,
        }
        entry = MassiveVendorListingEntryV0(
            source_role=MASSIVE_FINALIZED_V0_SOURCE_ROLE,
            dataset_id=dataset_id,
            source_object_key=key,
            coverage_session_date=coverage,
            etag=etag,
            content_length=content_length,
            vendor_last_modified_at_ms=last_modified,
            listing_observed_at_ms=observed,
            loaded_listing_receipt_sha256=loaded_listing.receipt_sha256,
            parser_spec_sha256=MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
            parser_source_sha256=MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256,
            receipt_sha256=semantic_sha256(body),
        )
        entry.validate()
        entries.append(entry)
    ordered = tuple(sorted(entries, key=lambda entry: entry.source_object_key))
    body = {
        "schema": MASSIVE_COMMITTED_FLAT_FILE_LISTING_V0_SCHEMA,
        "loaded_listing_receipt_sha256": loaded_listing.receipt_sha256,
        "source_object_receipt_sha256": loaded_listing.receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_listing.commit.receipt_sha256,
        "listing_observed_at_ms": observed,
        "entries": [asdict(entry) for entry in ordered],
        "parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256,
    }
    listing = MassiveCommittedFlatFileListingV0(
        loaded_listing_receipt_sha256=loaded_listing.receipt_sha256,
        source_object_receipt_sha256=loaded_listing.receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded_listing.commit.receipt_sha256,
        listing_observed_at_ms=observed,
        entries=ordered,
        parser_spec_sha256=MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
        parser_source_sha256=MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256,
        receipt_sha256=semantic_sha256(body),
    )
    listing.validate()
    return listing


__all__ = [
    "MASSIVE_COMMITTED_FLAT_FILE_LISTING_V0_SCHEMA",
    "MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES",
    "MASSIVE_FINALIZED_V0_SOURCE_ROLE",
    "MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA",
    "MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256",
    "MASSIVE_FLAT_FILE_LISTING_DATASET_ID",
    "MASSIVE_FLAT_FILE_LISTING_PARSER_SOURCE_SHA256",
    "MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256",
    "MASSIVE_VENDOR_LISTING_ENTRY_V0_SCHEMA",
    "MassiveCommittedFlatFileListingV0",
    "MassiveFinalizedListingError",
    "MassiveVendorListingEntryV0",
    "canonical_massive_trade_object_key",
    "coverage_session_from_massive_trade_key",
    "parse_massive_flat_file_listing_v0",
]
