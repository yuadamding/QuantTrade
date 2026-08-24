"""Hardened source and decision-origin authority for finalized V0 research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
    MASSIVE_FINALIZED_V0_SOURCE_ROLE,
    MassiveCommittedFlatFileListingV0,
    MassiveVendorListingEntryV0,
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MassiveExtractedTradeRow,
    MassiveTradeExtractionEvidence,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_FINALIZED_PROCESSING_SPEC_V0_SCHEMA = (
    "rl-quant.massive-finalized-processing-spec-v0"
)
MASSIVE_FINALIZED_SOURCE_COVERAGE_V0_SCHEMA = (
    "rl-quant.massive-finalized-source-coverage-v0"
)
MASSIVE_FEATURE_INPUT_CUTOFF_EVIDENCE_V0_SCHEMA = (
    "rl-quant.massive-feature-input-cutoff-evidence-v0"
)
MASSIVE_VENDOR_METADATA_FROM_LISTING_V0_SCHEMA = (
    "rl-quant.massive-vendor-metadata-from-listing-v0"
)
MASSIVE_FINALIZED_DAILY_SOURCE_EVIDENCE_V0_SCHEMA = (
    "rl-quant.massive-finalized-daily-source-evidence-v0"
)
MASSIVE_FINALIZED_DECISION_ORIGIN_V0_SCHEMA = (
    "rl-quant.massive-finalized-decision-origin-v0"
)
MASSIVE_FINALIZED_SKIPPED_DECISION_V0_SCHEMA = (
    "rl-quant.massive-finalized-skipped-decision-v0"
)
MASSIVE_FINALIZED_DECISION_ORIGIN_PLAN_V0_SCHEMA = (
    "rl-quant.massive-finalized-decision-origin-plan-v0"
)

MASSIVE_FINALIZED_PUBLICATION_SAFETY_MARGIN_MS = 5 * 60 * 1_000
MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS = 55 * 60 * 1_000
MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS = 60 * 60 * 1_000
MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS = 3
MASSIVE_FEATURE_INPUT_CUTOFF_MATERIALIZER_SOURCE_SHA256 = file_sha256(Path(__file__))
EASTERN = ZoneInfo("America/New_York")


class MassiveFinalizedOriginError(ValueError):
    """Finalized V0 daily evidence or decision-origin mapping is invalid."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveFinalizedOriginError(f"{name} must be a canonical nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedOriginError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveFinalizedOriginError(f"{name} must be nonnegative")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value <= 0:
        raise MassiveFinalizedOriginError(f"{name} must be positive")
    return value


def _local_timestamp_ms(session_date: str, local_time: time) -> int:
    try:
        day = date.fromisoformat(session_date)
    except ValueError as exc:
        raise MassiveFinalizedOriginError("session date is invalid") from exc
    return int(datetime.combine(day, local_time, tzinfo=EASTERN).timestamp() * 1_000)


def _session_timestamp_ms(name: str, value_ns: int) -> int:
    if value_ns % 1_000_000:
        raise MassiveFinalizedOriginError(f"{name} is not millisecond aligned")
    return value_ns // 1_000_000


@dataclass(frozen=True, slots=True)
class MassiveFinalizedProcessingSpecV0:
    publication_safety_margin_ms: int
    processing_allowance_ms: int
    total_readiness_allowance_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_PROCESSING_SPEC_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_PROCESSING_SPEC_V0_SCHEMA:
            raise MassiveFinalizedOriginError("processing specification drifted")
        if (
            self.publication_safety_margin_ms
            != MASSIVE_FINALIZED_PUBLICATION_SAFETY_MARGIN_MS
            or self.processing_allowance_ms != MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS
            or self.total_readiness_allowance_ms
            != MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS
            or self.total_readiness_allowance_ms
            != self.publication_safety_margin_ms + self.processing_allowance_ms
        ):
            raise MassiveFinalizedOriginError("processing readiness budget drifted")
        _digest("processing specification receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError(
                "processing specification receipt differs"
            )

    @classmethod
    def canonical(cls) -> MassiveFinalizedProcessingSpecV0:
        body = {
            "schema": MASSIVE_FINALIZED_PROCESSING_SPEC_V0_SCHEMA,
            "publication_safety_margin_ms": MASSIVE_FINALIZED_PUBLICATION_SAFETY_MARGIN_MS,
            "processing_allowance_ms": MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
            "total_readiness_allowance_ms": MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS,
        }
        value = cls(
            publication_safety_margin_ms=MASSIVE_FINALIZED_PUBLICATION_SAFETY_MARGIN_MS,
            processing_allowance_ms=MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
            total_readiness_allowance_ms=MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


MASSIVE_FINALIZED_PROCESSING_SPEC_V0 = MassiveFinalizedProcessingSpecV0.canonical()


@dataclass(frozen=True, slots=True)
class MassiveVendorObjectMetadataFromListingV0:
    source_role: str
    dataset_id: str
    source_object_key: str
    coverage_session_date: str
    source_object_receipt_sha256: str
    loaded_source_receipt_sha256: str
    committed_listing_receipt_sha256: str
    listing_entry_receipt_sha256: str
    etag: str
    content_length: int
    vendor_last_modified_at_ms: int
    listing_observed_at_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_VENDOR_METADATA_FROM_LISTING_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VENDOR_METADATA_FROM_LISTING_V0_SCHEMA:
            raise MassiveFinalizedOriginError("listing-derived metadata schema drifted")
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveFinalizedOriginError("listing-derived source role drifted")
        if self.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveFinalizedOriginError("listing-derived dataset drifted")
        if self.source_object_key != canonical_massive_trade_object_key(
            self.coverage_session_date
        ):
            raise MassiveFinalizedOriginError("listing-derived coverage key drifted")
        _text("ETag", self.etag)
        _nonnegative_int("content length", self.content_length)
        modified = _nonnegative_int(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        observed = _nonnegative_int("listing observation", self.listing_observed_at_ms)
        if observed < modified:
            raise MassiveFinalizedOriginError(
                "listing observation predates LastModified"
            )
        for name in (
            "source_object_receipt_sha256",
            "loaded_source_receipt_sha256",
            "committed_listing_receipt_sha256",
            "listing_entry_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError(
                "listing-derived metadata receipt differs"
            )


def build_massive_vendor_object_metadata_from_listing_v0(
    *,
    committed_listing: MassiveCommittedFlatFileListingV0,
    listing_entry: MassiveVendorListingEntryV0,
    loaded_source: LoadedMassiveSourceObject,
) -> MassiveVendorObjectMetadataFromListingV0:
    """Bind a downloaded trade object to its exact committed listing row."""

    committed_listing.validate()
    listing_entry.validate()
    loaded_source.validate()
    if (
        committed_listing.resolve(source_object_key=listing_entry.source_object_key)
        != listing_entry
    ):
        raise MassiveFinalizedOriginError(
            "listing entry was not resolved by the committed listing"
        )
    receipt = loaded_source.receipt
    if (
        receipt.dataset_id != listing_entry.dataset_id
        or receipt.source_object_key != listing_entry.source_object_key
        or receipt.etag != listing_entry.etag
        or receipt.content_length != listing_entry.content_length
        or receipt.downloaded_at_ms < listing_entry.vendor_last_modified_at_ms
    ):
        raise MassiveFinalizedOriginError(
            "downloaded source differs from its committed listing entry"
        )
    body = {
        "schema": MASSIVE_VENDOR_METADATA_FROM_LISTING_V0_SCHEMA,
        "source_role": listing_entry.source_role,
        "dataset_id": listing_entry.dataset_id,
        "source_object_key": listing_entry.source_object_key,
        "coverage_session_date": listing_entry.coverage_session_date,
        "source_object_receipt_sha256": receipt.receipt_sha256,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "committed_listing_receipt_sha256": committed_listing.receipt_sha256,
        "listing_entry_receipt_sha256": listing_entry.receipt_sha256,
        "etag": listing_entry.etag,
        "content_length": listing_entry.content_length,
        "vendor_last_modified_at_ms": listing_entry.vendor_last_modified_at_ms,
        "listing_observed_at_ms": listing_entry.listing_observed_at_ms,
    }
    value = MassiveVendorObjectMetadataFromListingV0(
        source_role=listing_entry.source_role,
        dataset_id=listing_entry.dataset_id,
        source_object_key=listing_entry.source_object_key,
        coverage_session_date=listing_entry.coverage_session_date,
        source_object_receipt_sha256=receipt.receipt_sha256,
        loaded_source_receipt_sha256=loaded_source.receipt_sha256,
        committed_listing_receipt_sha256=committed_listing.receipt_sha256,
        listing_entry_receipt_sha256=listing_entry.receipt_sha256,
        etag=listing_entry.etag,
        content_length=listing_entry.content_length,
        vendor_last_modified_at_ms=listing_entry.vendor_last_modified_at_ms,
        listing_observed_at_ms=listing_entry.listing_observed_at_ms,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedSourceCoverageV0:
    source_role: str
    dataset_id: str
    source_object_key: str
    coverage_session_date: str
    source_schema_sha256: str
    loaded_source_receipt_sha256: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    parser_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_SOURCE_COVERAGE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_SOURCE_COVERAGE_V0_SCHEMA:
            raise MassiveFinalizedOriginError("source coverage schema drifted")
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveFinalizedOriginError("source coverage role drifted")
        if self.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveFinalizedOriginError("source coverage dataset drifted")
        if self.source_object_key != canonical_massive_trade_object_key(
            self.coverage_session_date
        ):
            raise MassiveFinalizedOriginError(
                "source coverage date differs from object key"
            )
        if self.source_schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256:
            raise MassiveFinalizedOriginError(
                "source coverage schema authority drifted"
            )
        for name in (
            "source_schema_sha256",
            "loaded_source_receipt_sha256",
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "parser_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("source coverage receipt differs")


def build_massive_finalized_source_coverage_v0(
    *,
    loaded_source: LoadedMassiveSourceObject,
    extraction_evidence: MassiveTradeExtractionEvidence,
) -> MassiveFinalizedSourceCoverageV0:
    """Derive source-date coverage from the committed trade parser evidence."""

    loaded_source.validate()
    extraction_evidence.validate()
    receipt = loaded_source.receipt
    if (
        extraction_evidence.loaded_source_receipt_sha256 != loaded_source.receipt_sha256
        or extraction_evidence.source_receipt_sha256 != receipt.receipt_sha256
        or extraction_evidence.source_commit_receipt_sha256
        != loaded_source.commit.receipt_sha256
        or extraction_evidence.dataset_id != receipt.dataset_id
        or extraction_evidence.source_object_key != receipt.source_object_key
        or extraction_evidence.source_schema_sha256 != receipt.schema_sha256
    ):
        raise MassiveFinalizedOriginError(
            "trade parser evidence differs from the committed source"
        )
    expected_key = canonical_massive_trade_object_key(extraction_evidence.session_date)
    if receipt.source_object_key != expected_key:
        raise MassiveFinalizedOriginError(
            "committed source object does not cover the parser session"
        )
    body = {
        "schema": MASSIVE_FINALIZED_SOURCE_COVERAGE_V0_SCHEMA,
        "source_role": MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        "dataset_id": receipt.dataset_id,
        "source_object_key": receipt.source_object_key,
        "coverage_session_date": extraction_evidence.session_date,
        "source_schema_sha256": receipt.schema_sha256,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "source_object_receipt_sha256": receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        "parser_receipt_sha256": extraction_evidence.receipt_sha256,
    }
    value = MassiveFinalizedSourceCoverageV0(
        source_role=MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        dataset_id=receipt.dataset_id,
        source_object_key=receipt.source_object_key,
        coverage_session_date=extraction_evidence.session_date,
        source_schema_sha256=receipt.schema_sha256,
        loaded_source_receipt_sha256=loaded_source.receipt_sha256,
        source_object_receipt_sha256=receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded_source.commit.receipt_sha256,
        parser_receipt_sha256=extraction_evidence.receipt_sha256,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class MassiveFeatureInputCutoffEvidenceV0:
    source_partition_receipts: tuple[str, ...]
    source_extraction_receipt_sha256: str
    feature_spec_receipt_sha256: str
    minimum_input_timestamp_ms: int
    maximum_input_timestamp_ms: int
    input_row_count: int
    input_row_inventory_sha256: str
    source_session_date: str
    materializer_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FEATURE_INPUT_CUTOFF_EVIDENCE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FEATURE_INPUT_CUTOFF_EVIDENCE_V0_SCHEMA:
            raise MassiveFinalizedOriginError("feature-input cutoff schema drifted")
        if (
            not self.source_partition_receipts
            or self.source_partition_receipts
            != tuple(sorted(set(self.source_partition_receipts)))
        ):
            raise MassiveFinalizedOriginError(
                "source partition receipts must be sorted and unique"
            )
        for name in (
            "source_extraction_receipt_sha256",
            "feature_spec_receipt_sha256",
            "input_row_inventory_sha256",
            "materializer_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for receipt in self.source_partition_receipts:
            _digest("source partition receipt", receipt)
        minimum = _nonnegative_int(
            "minimum input timestamp", self.minimum_input_timestamp_ms
        )
        maximum = _nonnegative_int(
            "maximum input timestamp", self.maximum_input_timestamp_ms
        )
        if maximum < minimum:
            raise MassiveFinalizedOriginError("feature input interval is inverted")
        _positive_int("feature input row count", self.input_row_count)
        _text("source session date", self.source_session_date)
        try:
            source_day = date.fromisoformat(self.source_session_date)
        except ValueError as exc:
            raise MassiveFinalizedOriginError(
                "feature input source session date is invalid"
            ) from exc
        if (
            datetime.fromtimestamp(minimum / 1_000, tz=EASTERN).date() != source_day
            or datetime.fromtimestamp(maximum / 1_000, tz=EASTERN).date() != source_day
        ):
            raise MassiveFinalizedOriginError(
                "feature input timestamps differ from the source session"
            )
        if (
            self.materializer_source_sha256
            != MASSIVE_FEATURE_INPUT_CUTOFF_MATERIALIZER_SOURCE_SHA256
        ):
            raise MassiveFinalizedOriginError("cutoff materializer source drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("feature-input cutoff receipt differs")


def build_massive_feature_input_cutoff_evidence_v0(
    *,
    extracted_rows: Sequence[MassiveExtractedTradeRow],
    extraction_evidence: MassiveTradeExtractionEvidence,
    source_partition_receipts: Sequence[str],
    feature_spec_receipt_sha256: str,
) -> MassiveFeatureInputCutoffEvidenceV0:
    """Derive the feature cutoff from exact committed parser-selected rows."""

    extraction_evidence.validate()
    rows = tuple(sorted(extracted_rows, key=lambda row: row.source_row_number))
    if not rows:
        raise MassiveFinalizedOriginError("feature input rows are empty")
    for row in rows:
        row.validate()
    if len(rows) != extraction_evidence.selected_row_count:
        raise MassiveFinalizedOriginError(
            "feature input rows differ from parser-selected row count"
        )
    canonical_inventory = semantic_sha256(
        tuple(row.canonical_record.receipt_sha256 for row in rows)
    )
    if (
        canonical_inventory
        != extraction_evidence.selected_canonical_record_inventory_sha256
    ):
        raise MassiveFinalizedOriginError(
            "feature inputs differ from parser canonical inventory"
        )
    timestamps_ns = tuple(row.canonical_record.sip_timestamp_ns for row in rows)
    minimum_ms = min(timestamps_ns) // 1_000_000
    maximum_ms = (max(timestamps_ns) + 999_999) // 1_000_000
    for timestamp_ns in timestamps_ns:
        observed_date = (
            datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=EASTERN)
            .date()
            .isoformat()
        )
        if observed_date != extraction_evidence.session_date:
            raise MassiveFinalizedOriginError(
                "feature input row lies outside the source session"
            )
    partitions = tuple(sorted(set(source_partition_receipts)))
    if not partitions or len(partitions) != len(tuple(source_partition_receipts)):
        raise MassiveFinalizedOriginError(
            "feature source partitions must be nonempty and unique"
        )
    for receipt in partitions:
        _digest("source partition receipt", receipt)
    feature_spec = _digest("feature specification receipt", feature_spec_receipt_sha256)
    inventory = semantic_sha256(
        tuple(
            (
                row.source_row_number,
                row.raw_row_sha256,
                row.canonical_record.receipt_sha256,
            )
            for row in rows
        )
    )
    body = {
        "schema": MASSIVE_FEATURE_INPUT_CUTOFF_EVIDENCE_V0_SCHEMA,
        "source_partition_receipts": list(partitions),
        "source_extraction_receipt_sha256": extraction_evidence.receipt_sha256,
        "feature_spec_receipt_sha256": feature_spec,
        "minimum_input_timestamp_ms": minimum_ms,
        "maximum_input_timestamp_ms": maximum_ms,
        "input_row_count": len(rows),
        "input_row_inventory_sha256": inventory,
        "source_session_date": extraction_evidence.session_date,
        "materializer_source_sha256": MASSIVE_FEATURE_INPUT_CUTOFF_MATERIALIZER_SOURCE_SHA256,
    }
    value = MassiveFeatureInputCutoffEvidenceV0(
        source_partition_receipts=partitions,
        source_extraction_receipt_sha256=extraction_evidence.receipt_sha256,
        feature_spec_receipt_sha256=feature_spec,
        minimum_input_timestamp_ms=minimum_ms,
        maximum_input_timestamp_ms=maximum_ms,
        input_row_count=len(rows),
        input_row_inventory_sha256=inventory,
        source_session_date=extraction_evidence.session_date,
        materializer_source_sha256=MASSIVE_FEATURE_INPUT_CUTOFF_MATERIALIZER_SOURCE_SHA256,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedDailySourceEvidenceV0:
    protocol_id: str
    protocol_receipt_sha256: str
    session_authority_receipt_sha256: str
    source_role: str
    dataset_id: str
    source_object_key: str
    source_session_date: str
    loaded_source_receipt_sha256: str
    committed_listing_receipt_sha256: str
    listing_entry_receipt_sha256: str
    metadata_receipt_sha256: str
    coverage_receipt_sha256: str
    feature_input_cutoff_receipt_sha256: str
    source_feature_cutoff_at_ms: int
    maximum_input_timestamp_ms: int
    vendor_last_modified_at_ms: int
    processing_spec_receipt_sha256: str
    processing_budget_ms: int
    feature_ready_at_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_DAILY_SOURCE_EVIDENCE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_DAILY_SOURCE_EVIDENCE_V0_SCHEMA:
            raise MassiveFinalizedOriginError("daily source evidence schema drifted")
        if self.protocol_id != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID or (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
        ):
            raise MassiveFinalizedOriginError("daily source protocol drifted")
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveFinalizedOriginError("daily source role drifted")
        if self.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID:
            raise MassiveFinalizedOriginError("daily source dataset drifted")
        if self.source_object_key != canonical_massive_trade_object_key(
            self.source_session_date
        ):
            raise MassiveFinalizedOriginError("daily source coverage key drifted")
        for name in (
            "protocol_receipt_sha256",
            "session_authority_receipt_sha256",
            "loaded_source_receipt_sha256",
            "committed_listing_receipt_sha256",
            "listing_entry_receipt_sha256",
            "metadata_receipt_sha256",
            "coverage_receipt_sha256",
            "feature_input_cutoff_receipt_sha256",
            "processing_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        cutoff = _nonnegative_int(
            "source feature cutoff", self.source_feature_cutoff_at_ms
        )
        maximum = _nonnegative_int(
            "maximum input timestamp", self.maximum_input_timestamp_ms
        )
        if maximum > cutoff:
            raise MassiveFinalizedOriginError(
                "feature inputs exceed the source-session cutoff"
            )
        try:
            source_day = date.fromisoformat(self.source_session_date)
        except ValueError as exc:
            raise MassiveFinalizedOriginError(
                "daily source session date is invalid"
            ) from exc
        if datetime.fromtimestamp(cutoff / 1_000, tz=EASTERN).date() != source_day:
            raise MassiveFinalizedOriginError(
                "daily feature cutoff differs from its source session"
            )
        modified = _nonnegative_int(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        if self.processing_spec_receipt_sha256 != (
            MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256
        ):
            raise MassiveFinalizedOriginError("daily processing specification drifted")
        if self.processing_budget_ms != MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS:
            raise MassiveFinalizedOriginError("daily processing budget drifted")
        if self.feature_ready_at_ms != modified + self.processing_budget_ms:
            raise MassiveFinalizedOriginError("feature readiness timestamp drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("daily source evidence receipt differs")


def build_massive_finalized_daily_source_evidence_v0(
    *,
    loaded_source: LoadedMassiveSourceObject,
    committed_listing: MassiveCommittedFlatFileListingV0,
    listing_entry: MassiveVendorListingEntryV0,
    metadata: MassiveVendorObjectMetadataFromListingV0,
    coverage: MassiveFinalizedSourceCoverageV0,
    feature_input_cutoff: MassiveFeatureInputCutoffEvidenceV0,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
) -> MassiveFinalizedDailySourceEvidenceV0:
    """Join all required source, listing, parser, cutoff, and readiness evidence."""

    loaded_source.validate()
    committed_listing.validate()
    listing_entry.validate()
    metadata.validate()
    coverage.validate()
    feature_input_cutoff.validate()
    session_authority.validate()
    source_session.validate()
    if (
        session_authority.resolve(
            exchange=source_session.exchange,
            session_date=source_session.session_date,
        )
        != source_session
    ):
        raise MassiveFinalizedOriginError(
            "source session was not resolved by its calendar authority"
        )
    identities = {
        loaded_source.receipt.source_object_key,
        listing_entry.source_object_key,
        metadata.source_object_key,
        coverage.source_object_key,
    }
    dates = {
        source_session.session_date,
        listing_entry.coverage_session_date,
        metadata.coverage_session_date,
        coverage.coverage_session_date,
        feature_input_cutoff.source_session_date,
    }
    if len(identities) != 1 or len(dates) != 1:
        raise MassiveFinalizedOriginError(
            "daily source evidence does not share one object and source session"
        )
    if (
        metadata.loaded_source_receipt_sha256 != loaded_source.receipt_sha256
        or metadata.committed_listing_receipt_sha256 != committed_listing.receipt_sha256
        or metadata.listing_entry_receipt_sha256 != listing_entry.receipt_sha256
        or coverage.loaded_source_receipt_sha256 != loaded_source.receipt_sha256
        or feature_input_cutoff.source_extraction_receipt_sha256
        != coverage.parser_receipt_sha256
    ):
        raise MassiveFinalizedOriginError("daily evidence receipt chain differs")
    source_cutoff = _session_timestamp_ms(
        "source-session close", source_session.regular_close_ns
    )
    if feature_input_cutoff.maximum_input_timestamp_ms > source_cutoff:
        raise MassiveFinalizedOriginError(
            "feature inputs extend beyond the source-session close"
        )
    if metadata.vendor_last_modified_at_ms < source_cutoff:
        raise MassiveFinalizedOriginError(
            "finalized vendor object predates its source-session close"
        )
    processing_spec = MASSIVE_FINALIZED_PROCESSING_SPEC_V0
    feature_ready = (
        metadata.vendor_last_modified_at_ms
        + processing_spec.total_readiness_allowance_ms
    )
    body = {
        "schema": MASSIVE_FINALIZED_DAILY_SOURCE_EVIDENCE_V0_SCHEMA,
        "protocol_id": MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "source_role": MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
        "source_object_key": loaded_source.receipt.source_object_key,
        "source_session_date": source_session.session_date,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "committed_listing_receipt_sha256": committed_listing.receipt_sha256,
        "listing_entry_receipt_sha256": listing_entry.receipt_sha256,
        "metadata_receipt_sha256": metadata.receipt_sha256,
        "coverage_receipt_sha256": coverage.receipt_sha256,
        "feature_input_cutoff_receipt_sha256": feature_input_cutoff.receipt_sha256,
        "source_feature_cutoff_at_ms": source_cutoff,
        "maximum_input_timestamp_ms": feature_input_cutoff.maximum_input_timestamp_ms,
        "vendor_last_modified_at_ms": metadata.vendor_last_modified_at_ms,
        "processing_spec_receipt_sha256": processing_spec.receipt_sha256,
        "processing_budget_ms": processing_spec.total_readiness_allowance_ms,
        "feature_ready_at_ms": feature_ready,
    }
    value = MassiveFinalizedDailySourceEvidenceV0(
        protocol_id=MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
        protocol_receipt_sha256=MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        source_role=MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        source_object_key=loaded_source.receipt.source_object_key,
        source_session_date=source_session.session_date,
        loaded_source_receipt_sha256=loaded_source.receipt_sha256,
        committed_listing_receipt_sha256=committed_listing.receipt_sha256,
        listing_entry_receipt_sha256=listing_entry.receipt_sha256,
        metadata_receipt_sha256=metadata.receipt_sha256,
        coverage_receipt_sha256=coverage.receipt_sha256,
        feature_input_cutoff_receipt_sha256=feature_input_cutoff.receipt_sha256,
        source_feature_cutoff_at_ms=source_cutoff,
        maximum_input_timestamp_ms=feature_input_cutoff.maximum_input_timestamp_ms,
        vendor_last_modified_at_ms=metadata.vendor_last_modified_at_ms,
        processing_spec_receipt_sha256=processing_spec.receipt_sha256,
        processing_budget_ms=processing_spec.total_readiness_allowance_ms,
        feature_ready_at_ms=feature_ready,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedDecisionOriginV0:
    decision_session_date: str
    source_session_date: str
    source_staleness_sessions: int
    required_source_roles: tuple[str, ...]
    daily_source_evidence_receipts: tuple[str, ...]
    feature_ready_at_ms: int
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_DECISION_ORIGIN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_DECISION_ORIGIN_V0_SCHEMA:
            raise MassiveFinalizedOriginError("decision-origin schema drifted")
        if (
            self.required_source_roles
            != MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES
        ):
            raise MassiveFinalizedOriginError("decision-origin source roles drifted")
        if len(self.daily_source_evidence_receipts) != len(
            self.required_source_roles
        ) or self.daily_source_evidence_receipts != tuple(
            sorted(set(self.daily_source_evidence_receipts))
        ):
            raise MassiveFinalizedOriginError(
                "decision origin lacks an exact source-role inventory"
            )
        for receipt in self.daily_source_evidence_receipts:
            _digest("daily source evidence receipt", receipt)
        staleness = _positive_int("source staleness", self.source_staleness_sessions)
        if staleness > MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS:
            raise MassiveFinalizedOriginError("decision origin uses a stale source")
        try:
            source_day = date.fromisoformat(self.source_session_date)
            decision_day = date.fromisoformat(self.decision_session_date)
        except ValueError as exc:
            raise MassiveFinalizedOriginError(
                "decision-origin session date is invalid"
            ) from exc
        if decision_day <= source_day:
            raise MassiveFinalizedOriginError(
                "decision origin does not use a prior source session"
            )
        for name in (
            "feature_ready_at_ms",
            "decision_at_ms",
            "fill_start_at_ms",
            "fill_end_at_ms",
        ):
            _nonnegative_int(name, getattr(self, name))
        if not (
            self.feature_ready_at_ms
            <= self.decision_at_ms
            < self.fill_start_at_ms
            < self.fill_end_at_ms
        ):
            raise MassiveFinalizedOriginError("decision-origin chronology drifted")
        if (
            self.decision_at_ms
            != _local_timestamp_ms(self.decision_session_date, time(12, 30))
            or self.fill_start_at_ms
            != _local_timestamp_ms(self.decision_session_date, time(15, 50))
            or self.fill_end_at_ms
            != _local_timestamp_ms(self.decision_session_date, time(16, 0))
        ):
            raise MassiveFinalizedOriginError(
                "decision origin differs from frozen local times"
            )
        _digest("decision-origin receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("decision-origin receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveFinalizedSkippedDecisionV0:
    decision_session_date: str
    reason: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_SKIPPED_DECISION_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_SKIPPED_DECISION_V0_SCHEMA:
            raise MassiveFinalizedOriginError("skipped-decision schema drifted")
        date.fromisoformat(self.decision_session_date)
        if self.reason not in {
            "decision-session-cannot-support-fill",
            "no-ready-source-within-staleness-bound",
        }:
            raise MassiveFinalizedOriginError("skipped-decision reason drifted")
        _digest("skipped-decision receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("skipped-decision receipt differs")


def _build_origin(
    *,
    decision_session: MassiveExchangeSession,
    source_session_index: int,
    decision_session_index: int,
    sources: tuple[MassiveFinalizedDailySourceEvidenceV0, ...],
) -> MassiveFinalizedDecisionOriginV0:
    decision_at = _local_timestamp_ms(decision_session.session_date, time(12, 30))
    fill_start = _local_timestamp_ms(decision_session.session_date, time(15, 50))
    fill_end = _local_timestamp_ms(decision_session.session_date, time(16, 0))
    feature_ready = max(source.feature_ready_at_ms for source in sources)
    receipts = tuple(sorted(source.receipt_sha256 for source in sources))
    staleness = decision_session_index - source_session_index
    body = {
        "schema": MASSIVE_FINALIZED_DECISION_ORIGIN_V0_SCHEMA,
        "decision_session_date": decision_session.session_date,
        "source_session_date": sources[0].source_session_date,
        "source_staleness_sessions": staleness,
        "required_source_roles": list(MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES),
        "daily_source_evidence_receipts": list(receipts),
        "feature_ready_at_ms": feature_ready,
        "decision_at_ms": decision_at,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
    }
    value = MassiveFinalizedDecisionOriginV0(
        decision_session_date=decision_session.session_date,
        source_session_date=sources[0].source_session_date,
        source_staleness_sessions=staleness,
        required_source_roles=MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        daily_source_evidence_receipts=receipts,
        feature_ready_at_ms=feature_ready,
        decision_at_ms=decision_at,
        fill_start_at_ms=fill_start,
        fill_end_at_ms=fill_end,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def _build_skip(
    *, decision_session_date: str, reason: str
) -> MassiveFinalizedSkippedDecisionV0:
    body = {
        "schema": MASSIVE_FINALIZED_SKIPPED_DECISION_V0_SCHEMA,
        "decision_session_date": decision_session_date,
        "reason": reason,
    }
    value = MassiveFinalizedSkippedDecisionV0(
        decision_session_date=decision_session_date,
        reason=reason,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def _derive_decision_rows(
    *,
    calendar_sessions: tuple[MassiveExchangeSession, ...],
    candidate_decision_dates: tuple[str, ...],
    daily_sources: tuple[MassiveFinalizedDailySourceEvidenceV0, ...],
) -> tuple[
    tuple[MassiveFinalizedDecisionOriginV0, ...],
    tuple[MassiveFinalizedSkippedDecisionV0, ...],
]:
    session_index = {
        session.session_date: index for index, session in enumerate(calendar_sessions)
    }
    sources_by_date: dict[str, tuple[MassiveFinalizedDailySourceEvidenceV0, ...]] = {}
    for source_date in sorted({source.source_session_date for source in daily_sources}):
        rows = tuple(
            sorted(
                (
                    source
                    for source in daily_sources
                    if source.source_session_date == source_date
                ),
                key=lambda source: source.source_role,
            )
        )
        if tuple(row.source_role for row in rows) != (
            MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES
        ):
            raise MassiveFinalizedOriginError(
                "source date does not have the exact required source-role inventory"
            )
        sources_by_date[source_date] = rows
    origins: list[MassiveFinalizedDecisionOriginV0] = []
    skipped: list[MassiveFinalizedSkippedDecisionV0] = []
    sessions_by_date = {session.session_date: session for session in calendar_sessions}
    for decision_date in candidate_decision_dates:
        decision_session = sessions_by_date[decision_date]
        close_ms = _session_timestamp_ms(
            "decision-session close", decision_session.regular_close_ns
        )
        fill_end = _local_timestamp_ms(decision_date, time(16, 0))
        if fill_end > close_ms:
            skipped.append(
                _build_skip(
                    decision_session_date=decision_date,
                    reason="decision-session-cannot-support-fill",
                )
            )
            continue
        decision_at = _local_timestamp_ms(decision_date, time(12, 30))
        decision_index = session_index[decision_date]
        eligible_dates: list[str] = []
        for source_date, sources in sources_by_date.items():
            source_index = session_index[source_date]
            staleness = decision_index - source_index
            if (
                1 <= staleness <= MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS
                and max(source.feature_ready_at_ms for source in sources) <= decision_at
            ):
                eligible_dates.append(source_date)
        if not eligible_dates:
            skipped.append(
                _build_skip(
                    decision_session_date=decision_date,
                    reason="no-ready-source-within-staleness-bound",
                )
            )
            continue
        source_date = max(eligible_dates, key=lambda value: session_index[value])
        origins.append(
            _build_origin(
                decision_session=decision_session,
                source_session_index=session_index[source_date],
                decision_session_index=decision_index,
                sources=sources_by_date[source_date],
            )
        )
    return tuple(origins), tuple(skipped)


@dataclass(frozen=True, slots=True)
class MassiveFinalizedDecisionOriginPlanV0:
    protocol_receipt_sha256: str
    session_authority_receipt_sha256: str
    exchange: str
    first_decision_session_date: str
    last_decision_session_date: str
    maximum_source_staleness_sessions: int
    required_daily_source_roles: tuple[str, ...]
    calendar_sessions: tuple[MassiveExchangeSession, ...]
    daily_sources: tuple[MassiveFinalizedDailySourceEvidenceV0, ...]
    candidate_decision_session_dates: tuple[str, ...]
    origins: tuple[MassiveFinalizedDecisionOriginV0, ...]
    skipped_decisions: tuple[MassiveFinalizedSkippedDecisionV0, ...]
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_DECISION_ORIGIN_PLAN_V0_SCHEMA

    @property
    def source_origin_authority_closed(self) -> bool:
        """The plan closes source roles, readiness, and decision chronology."""

        return True

    @property
    def panel_materialization_authorized(self) -> bool:
        """PIT identity and economic authorities remain a later data gate."""

        return False

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_DECISION_ORIGIN_PLAN_V0_SCHEMA:
            raise MassiveFinalizedOriginError("decision-origin plan schema drifted")
        if self.protocol_receipt_sha256 != (
            MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
        ):
            raise MassiveFinalizedOriginError("decision-origin protocol drifted")
        _digest("session authority receipt", self.session_authority_receipt_sha256)
        _text("exchange", self.exchange)
        try:
            first_day = date.fromisoformat(self.first_decision_session_date)
            last_day = date.fromisoformat(self.last_decision_session_date)
        except ValueError as exc:
            raise MassiveFinalizedOriginError(
                "decision-origin range date is invalid"
            ) from exc
        if last_day < first_day:
            raise MassiveFinalizedOriginError("decision-origin range is inverted")
        if (
            self.maximum_source_staleness_sessions
            != MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS
            or self.required_daily_source_roles
            != MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES
        ):
            raise MassiveFinalizedOriginError("decision-origin source policy drifted")
        session_dates = tuple(
            session.session_date for session in self.calendar_sessions
        )
        if not session_dates or session_dates != tuple(sorted(set(session_dates))):
            raise MassiveFinalizedOriginError(
                "calendar sessions must be sorted and unique"
            )
        for session in self.calendar_sessions:
            session.validate()
            if session.exchange != self.exchange:
                raise MassiveFinalizedOriginError("origin plan mixes exchanges")
        if (
            self.first_decision_session_date not in session_dates
            or self.last_decision_session_date not in session_dates
        ):
            raise MassiveFinalizedOriginError(
                "decision-origin range is absent from the calendar"
            )
        expected_candidates = tuple(
            session_date
            for session_date in session_dates
            if self.first_decision_session_date
            <= session_date
            <= self.last_decision_session_date
        )
        if self.candidate_decision_session_dates != expected_candidates:
            raise MassiveFinalizedOriginError(
                "decision origin plan omitted or inserted a calendar session"
            )
        if not self.candidate_decision_session_dates:
            raise MassiveFinalizedOriginError(
                "decision-origin candidate range is empty"
            )
        source_keys = tuple(
            (source.source_session_date, source.source_role)
            for source in self.daily_sources
        )
        if source_keys != tuple(sorted(set(source_keys))):
            raise MassiveFinalizedOriginError(
                "daily sources must be sorted and unique by date and role"
            )
        if not self.daily_sources:
            raise MassiveFinalizedOriginError("decision-origin daily sources are empty")
        for source in self.daily_sources:
            source.validate()
            if (
                source.session_authority_receipt_sha256
                != self.session_authority_receipt_sha256
                or source.source_session_date not in session_dates
            ):
                raise MassiveFinalizedOriginError(
                    "daily source differs from the plan calendar"
                )
        expected_origins, expected_skips = _derive_decision_rows(
            calendar_sessions=self.calendar_sessions,
            candidate_decision_dates=self.candidate_decision_session_dates,
            daily_sources=self.daily_sources,
        )
        if self.origins != expected_origins or self.skipped_decisions != expected_skips:
            raise MassiveFinalizedOriginError(
                "decision origins were not derived exhaustively from the calendar"
            )
        partition = tuple(
            sorted(
                [origin.decision_session_date for origin in self.origins]
                + [skip.decision_session_date for skip in self.skipped_decisions]
            )
        )
        if partition != self.candidate_decision_session_dates:
            raise MassiveFinalizedOriginError(
                "decision origin and skip rows do not partition candidate sessions"
            )
        _digest("decision-origin plan receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginError("decision-origin plan receipt differs")


def build_massive_finalized_decision_origin_plan_v0(
    *,
    session_authority: MassiveSessionAuthority,
    exchange: str,
    daily_sources: Sequence[MassiveFinalizedDailySourceEvidenceV0],
    first_decision_session_date: str,
    last_decision_session_date: str,
) -> MassiveFinalizedDecisionOriginPlanV0:
    """Build one exhaustive, decision-centric V0 origin plan."""

    session_authority.validate()
    exchange = _text("exchange", exchange)
    sessions = tuple(
        session
        for session in session_authority.sessions
        if session.exchange == exchange
    )
    if not sessions:
        raise MassiveFinalizedOriginError("exchange has no calendar sessions")
    available_dates = {session.session_date for session in sessions}
    if (
        first_decision_session_date not in available_dates
        or last_decision_session_date not in available_dates
        or first_decision_session_date > last_decision_session_date
    ):
        raise MassiveFinalizedOriginError("decision-origin range is absent or inverted")
    sources = tuple(
        sorted(
            daily_sources,
            key=lambda source: (source.source_session_date, source.source_role),
        )
    )
    if not sources:
        raise MassiveFinalizedOriginError("decision-origin source inventory is empty")
    for source in sources:
        source.validate()
        if (
            source.session_authority_receipt_sha256
            != session_authority.receipt_sha256
            or source.source_session_date not in available_dates
        ):
            raise MassiveFinalizedOriginError(
                "daily source uses another or absent session authority"
            )
    source_keys = tuple(
        (source.source_session_date, source.source_role) for source in sources
    )
    if source_keys != tuple(sorted(set(source_keys))):
        raise MassiveFinalizedOriginError(
            "daily sources must be sorted and unique by date and role"
        )
    candidate_dates = tuple(
        session.session_date
        for session in sessions
        if first_decision_session_date
        <= session.session_date
        <= last_decision_session_date
    )
    origins, skipped = _derive_decision_rows(
        calendar_sessions=sessions,
        candidate_decision_dates=candidate_dates,
        daily_sources=sources,
    )
    body = {
        "schema": MASSIVE_FINALIZED_DECISION_ORIGIN_PLAN_V0_SCHEMA,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "exchange": exchange,
        "first_decision_session_date": first_decision_session_date,
        "last_decision_session_date": last_decision_session_date,
        "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        "required_daily_source_roles": list(
            MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES
        ),
        "calendar_sessions": [asdict(session) for session in sessions],
        "daily_sources": [asdict(source) for source in sources],
        "candidate_decision_session_dates": list(candidate_dates),
        "origins": [asdict(origin) for origin in origins],
        "skipped_decisions": [asdict(skip) for skip in skipped],
    }
    value = MassiveFinalizedDecisionOriginPlanV0(
        protocol_receipt_sha256=MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        exchange=exchange,
        first_decision_session_date=first_decision_session_date,
        last_decision_session_date=last_decision_session_date,
        maximum_source_staleness_sessions=MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        required_daily_source_roles=MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        calendar_sessions=sessions,
        daily_sources=sources,
        candidate_decision_session_dates=candidate_dates,
        origins=origins,
        skipped_decisions=skipped,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


__all__ = [
    "MASSIVE_FEATURE_INPUT_CUTOFF_EVIDENCE_V0_SCHEMA",
    "MASSIVE_FINALIZED_DAILY_SOURCE_EVIDENCE_V0_SCHEMA",
    "MASSIVE_FINALIZED_DECISION_ORIGIN_PLAN_V0_SCHEMA",
    "MASSIVE_FINALIZED_DECISION_ORIGIN_V0_SCHEMA",
    "MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS",
    "MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS",
    "MASSIVE_FINALIZED_PROCESSING_SPEC_V0",
    "MASSIVE_FINALIZED_PROCESSING_SPEC_V0_SCHEMA",
    "MASSIVE_FINALIZED_PUBLICATION_SAFETY_MARGIN_MS",
    "MASSIVE_FINALIZED_SKIPPED_DECISION_V0_SCHEMA",
    "MASSIVE_FINALIZED_SOURCE_COVERAGE_V0_SCHEMA",
    "MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS",
    "MASSIVE_VENDOR_METADATA_FROM_LISTING_V0_SCHEMA",
    "MassiveFeatureInputCutoffEvidenceV0",
    "MassiveFinalizedDailySourceEvidenceV0",
    "MassiveFinalizedDecisionOriginPlanV0",
    "MassiveFinalizedDecisionOriginV0",
    "MassiveFinalizedOriginError",
    "MassiveFinalizedProcessingSpecV0",
    "MassiveFinalizedSkippedDecisionV0",
    "MassiveFinalizedSourceCoverageV0",
    "MassiveVendorObjectMetadataFromListingV0",
    "build_massive_feature_input_cutoff_evidence_v0",
    "build_massive_finalized_daily_source_evidence_v0",
    "build_massive_finalized_decision_origin_plan_v0",
    "build_massive_finalized_source_coverage_v0",
    "build_massive_vendor_object_metadata_from_listing_v0",
]
