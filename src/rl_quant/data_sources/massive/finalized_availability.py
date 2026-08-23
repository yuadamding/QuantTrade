"""Vendor-metadata chronology for finalized-file validation V0.

The authority deliberately uses vendor ``LastModified`` metadata rather than
the later time at which a researcher downloaded an object.  It evaluates one
candidate decision session at a time; callers may then select the first
eligible session without rewriting or backdating the source evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import PurePosixPath
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_VENDOR_OBJECT_METADATA_V0_SCHEMA = "rl-quant.massive-vendor-object-metadata-v0"
MASSIVE_FINALIZED_SOURCE_AVAILABILITY_V0_SCHEMA = (
    "rl-quant.massive-finalized-source-availability-v0"
)
MASSIVE_FINALIZED_ORIGIN_AVAILABILITY_V0_SCHEMA = (
    "rl-quant.massive-finalized-origin-availability-v0"
)
EASTERN = ZoneInfo("America/New_York")


class MassiveFinalizedAvailabilityError(ValueError):
    """Finalized source metadata or candidate chronology is inconsistent."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveFinalizedAvailabilityError(
            f"{name} must be a canonical nonempty string"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedAvailabilityError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _timestamp_ms(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveFinalizedAvailabilityError(
            f"{name} must be a nonnegative integer timestamp"
        )
    return value


def _safe_object_key(value: object) -> str:
    key = PurePosixPath(_text("source object key", value))
    if key.is_absolute() or any(part in {"", ".", ".."} for part in key.parts):
        raise MassiveFinalizedAvailabilityError(
            "source object key must be a safe relative path"
        )
    return key.as_posix()


def _local_timestamp_ms(session_date: str, local_time: time) -> int:
    try:
        day = date.fromisoformat(session_date)
    except ValueError as exc:
        raise MassiveFinalizedAvailabilityError("session date is invalid") from exc
    return int(datetime.combine(day, local_time, tzinfo=EASTERN).timestamp() * 1_000)


def _session_timestamp_ms(name: str, timestamp_ns: int) -> int:
    if timestamp_ns % 1_000_000:
        raise MassiveFinalizedAvailabilityError(
            f"{name} must be millisecond aligned for V0 chronology"
        )
    return timestamp_ns // 1_000_000


@dataclass(frozen=True, slots=True)
class MassiveVendorObjectMetadataV0:
    """One source object joined to vendor listing metadata."""

    dataset_id: str
    source_object_key: str
    source_object_receipt_sha256: str
    etag: str | None
    content_length: int
    vendor_last_modified_at_ms: int
    vendor_available_at_ms: int
    metadata_observed_at_ms: int
    listing_source_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_VENDOR_OBJECT_METADATA_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VENDOR_OBJECT_METADATA_V0_SCHEMA:
            raise MassiveFinalizedAvailabilityError(
                "vendor-object metadata schema drifted"
            )
        _text("dataset ID", self.dataset_id)
        _safe_object_key(self.source_object_key)
        _digest("source-object receipt", self.source_object_receipt_sha256)
        if self.etag is not None:
            _text("ETag", self.etag)
        if (
            isinstance(self.content_length, bool)
            or not isinstance(self.content_length, int)
            or self.content_length < 0
        ):
            raise MassiveFinalizedAvailabilityError(
                "vendor content length must be nonnegative"
            )
        last_modified = _timestamp_ms(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        available = _timestamp_ms("vendor availability", self.vendor_available_at_ms)
        observed = _timestamp_ms("metadata observation", self.metadata_observed_at_ms)
        if available != last_modified:
            raise MassiveFinalizedAvailabilityError(
                "V0 vendor availability must equal the observed LastModified timestamp"
            )
        if observed < available:
            raise MassiveFinalizedAvailabilityError(
                "vendor metadata was observed before its LastModified timestamp"
            )
        _digest("listing-source receipt", self.listing_source_receipt_sha256)
        _digest("vendor-object metadata receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedAvailabilityError(
                "vendor-object metadata receipt differs"
            )


def build_massive_vendor_object_metadata_v0(
    *,
    source_object_receipt: MassiveSourceObjectReceipt,
    vendor_last_modified_at_ms: int,
    metadata_observed_at_ms: int,
    listing_source_receipt_sha256: str,
) -> MassiveVendorObjectMetadataV0:
    """Bind one immutable source receipt to its vendor listing metadata."""

    source_object_receipt.validate()
    last_modified = _timestamp_ms("vendor LastModified", vendor_last_modified_at_ms)
    observed = _timestamp_ms("metadata observation", metadata_observed_at_ms)
    if source_object_receipt.downloaded_at_ms < last_modified:
        raise MassiveFinalizedAvailabilityError(
            "source payload was downloaded before vendor LastModified"
        )
    listing_receipt = _digest("listing-source receipt", listing_source_receipt_sha256)
    body = {
        "schema": MASSIVE_VENDOR_OBJECT_METADATA_V0_SCHEMA,
        "dataset_id": source_object_receipt.dataset_id,
        "source_object_key": source_object_receipt.source_object_key,
        "source_object_receipt_sha256": source_object_receipt.receipt_sha256,
        "etag": source_object_receipt.etag,
        "content_length": source_object_receipt.content_length,
        "vendor_last_modified_at_ms": last_modified,
        "vendor_available_at_ms": last_modified,
        "metadata_observed_at_ms": observed,
        "listing_source_receipt_sha256": listing_receipt,
    }
    value = MassiveVendorObjectMetadataV0(
        dataset_id=source_object_receipt.dataset_id,
        source_object_key=source_object_receipt.source_object_key,
        source_object_receipt_sha256=source_object_receipt.receipt_sha256,
        etag=source_object_receipt.etag,
        content_length=source_object_receipt.content_length,
        vendor_last_modified_at_ms=last_modified,
        vendor_available_at_ms=last_modified,
        metadata_observed_at_ms=observed,
        listing_source_receipt_sha256=listing_receipt,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedSourceAvailabilityAuthorityV0:
    protocol_id: str
    protocol_receipt_sha256: str
    session_authority_receipt_sha256: str
    source_object_receipt_sha256: str
    source_metadata_receipt_sha256: str
    dataset_id: str
    source_object_key: str
    exchange: str
    source_session_date: str
    decision_session_date: str
    source_regular_close_at_ms: int
    decision_regular_open_at_ms: int
    decision_regular_close_at_ms: int
    source_feature_cutoff_at_ms: int
    latest_input_observation_at_ms: int
    vendor_last_modified_at_ms: int
    vendor_available_at_ms: int
    availability_cutoff_at_ms: int
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    origin_eligible: bool
    ineligibility_reason: str | None
    availability_authority_receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_SOURCE_AVAILABILITY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "availability_authority_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_SOURCE_AVAILABILITY_V0_SCHEMA:
            raise MassiveFinalizedAvailabilityError(
                "finalized source-availability schema drifted"
            )
        if self.protocol_id != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID:
            raise MassiveFinalizedAvailabilityError(
                "source availability protocol ID drifted"
            )
        if (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
        ):
            raise MassiveFinalizedAvailabilityError(
                "source availability protocol receipt drifted"
            )
        for name in (
            "protocol_receipt_sha256",
            "session_authority_receipt_sha256",
            "source_object_receipt_sha256",
            "source_metadata_receipt_sha256",
            "availability_authority_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        _text("dataset ID", self.dataset_id)
        _safe_object_key(self.source_object_key)
        _text("exchange", self.exchange)
        try:
            source_day = date.fromisoformat(self.source_session_date)
            decision_day = date.fromisoformat(self.decision_session_date)
        except ValueError as exc:
            raise MassiveFinalizedAvailabilityError(
                "source or decision session date is invalid"
            ) from exc
        if decision_day <= source_day:
            raise MassiveFinalizedAvailabilityError(
                "decision session must follow the finalized source session"
            )
        for name in (
            "source_regular_close_at_ms",
            "decision_regular_open_at_ms",
            "decision_regular_close_at_ms",
            "source_feature_cutoff_at_ms",
            "latest_input_observation_at_ms",
            "vendor_last_modified_at_ms",
            "vendor_available_at_ms",
            "availability_cutoff_at_ms",
            "decision_at_ms",
            "fill_start_at_ms",
            "fill_end_at_ms",
        ):
            _timestamp_ms(name, getattr(self, name))
        if self.source_feature_cutoff_at_ms != self.source_regular_close_at_ms:
            raise MassiveFinalizedAvailabilityError(
                "feature cutoff differs from the finalized source-session close"
            )
        if (
            datetime.fromtimestamp(
                self.source_regular_close_at_ms / 1_000, tz=EASTERN
            ).date()
            != source_day
            or datetime.fromtimestamp(
                self.decision_regular_open_at_ms / 1_000, tz=EASTERN
            ).date()
            != decision_day
            or datetime.fromtimestamp(
                self.decision_regular_close_at_ms / 1_000, tz=EASTERN
            ).date()
            != decision_day
        ):
            raise MassiveFinalizedAvailabilityError(
                "session timestamps differ from their declared Eastern dates"
            )
        if self.latest_input_observation_at_ms > self.source_feature_cutoff_at_ms:
            raise MassiveFinalizedAvailabilityError(
                "decision-session observations cannot enter finalized V0 features"
            )
        if self.vendor_available_at_ms != self.vendor_last_modified_at_ms:
            raise MassiveFinalizedAvailabilityError(
                "vendor availability differs from LastModified"
            )
        if self.vendor_available_at_ms < self.source_regular_close_at_ms:
            raise MassiveFinalizedAvailabilityError(
                "finalized source predates its source-session close"
            )
        expected_availability_cutoff = _local_timestamp_ms(
            self.decision_session_date, time(11, 30)
        )
        expected_decision = _local_timestamp_ms(
            self.decision_session_date, time(12, 30)
        )
        expected_fill_start = _local_timestamp_ms(
            self.decision_session_date, time(15, 50)
        )
        expected_fill_end = _local_timestamp_ms(self.decision_session_date, time(16, 0))
        if (
            self.availability_cutoff_at_ms != expected_availability_cutoff
            or self.decision_at_ms != expected_decision
            or self.fill_start_at_ms != expected_fill_start
            or self.fill_end_at_ms != expected_fill_end
        ):
            raise MassiveFinalizedAvailabilityError(
                "candidate chronology differs from the frozen V0 local times"
            )
        if not (
            self.decision_regular_open_at_ms
            <= self.availability_cutoff_at_ms
            < self.decision_at_ms
            < self.fill_start_at_ms
            < self.fill_end_at_ms
            <= self.decision_regular_close_at_ms
        ):
            raise MassiveFinalizedAvailabilityError(
                "decision and fill chronology is invalid"
            )
        if not isinstance(self.origin_eligible, bool):
            raise MassiveFinalizedAvailabilityError(
                "origin eligibility must be Boolean"
            )
        expected_eligible = (
            self.vendor_available_at_ms <= self.availability_cutoff_at_ms
        )
        if self.origin_eligible != expected_eligible:
            raise MassiveFinalizedAvailabilityError(
                "origin eligibility differs from vendor availability"
            )
        expected_reason = None if expected_eligible else "vendor-available-after-cutoff"
        if self.ineligibility_reason is not None:
            _text("origin ineligibility reason", self.ineligibility_reason)
        if self.ineligibility_reason != expected_reason:
            raise MassiveFinalizedAvailabilityError(
                "origin ineligibility reason differs"
            )
        if self.availability_authority_receipt_sha256 != semantic_sha256(
            self.unsigned()
        ):
            raise MassiveFinalizedAvailabilityError(
                "source-availability authority receipt differs"
            )


def build_massive_finalized_source_availability_authority_v0(
    *,
    source_object_receipt: MassiveSourceObjectReceipt,
    source_metadata: MassiveVendorObjectMetadataV0,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    latest_input_observation_at_ms: int,
) -> MassiveFinalizedSourceAvailabilityAuthorityV0:
    """Evaluate one source object against one candidate V0 decision session."""

    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.validate()
    source_object_receipt.validate()
    source_metadata.validate()
    session_authority.validate()
    source_session.validate()
    decision_session.validate()
    if (
        session_authority.resolve(
            exchange=source_session.exchange,
            session_date=source_session.session_date,
        )
        != source_session
        or session_authority.resolve(
            exchange=decision_session.exchange,
            session_date=decision_session.session_date,
        )
        != decision_session
    ):
        raise MassiveFinalizedAvailabilityError(
            "source or decision session was not resolved by its authority"
        )
    if source_session.exchange != decision_session.exchange:
        raise MassiveFinalizedAvailabilityError(
            "source and decision sessions use different exchanges"
        )
    if (
        source_metadata.source_object_receipt_sha256
        != source_object_receipt.receipt_sha256
        or source_metadata.dataset_id != source_object_receipt.dataset_id
        or source_metadata.source_object_key != source_object_receipt.source_object_key
        or source_metadata.etag != source_object_receipt.etag
        or source_metadata.content_length != source_object_receipt.content_length
    ):
        raise MassiveFinalizedAvailabilityError(
            "vendor metadata does not describe the exact source object"
        )
    source_cutoff = _session_timestamp_ms(
        "source-session close", source_session.regular_close_ns
    )
    latest_observation = _timestamp_ms(
        "latest input observation", latest_input_observation_at_ms
    )
    if latest_observation > source_cutoff:
        raise MassiveFinalizedAvailabilityError(
            "decision-session observations cannot enter finalized V0 features"
        )
    if source_metadata.vendor_available_at_ms < source_cutoff:
        raise MassiveFinalizedAvailabilityError(
            "finalized source was marked available before its source session closed"
        )
    availability_cutoff = _local_timestamp_ms(
        decision_session.session_date, time(11, 30)
    )
    decision_at = _local_timestamp_ms(decision_session.session_date, time(12, 30))
    fill_start = _local_timestamp_ms(decision_session.session_date, time(15, 50))
    fill_end = _local_timestamp_ms(decision_session.session_date, time(16, 0))
    regular_open = _session_timestamp_ms(
        "decision-session open", decision_session.regular_open_ns
    )
    regular_close = _session_timestamp_ms(
        "decision-session close", decision_session.regular_close_ns
    )
    if not (
        regular_open <= availability_cutoff < decision_at < fill_start < fill_end
        and fill_end <= regular_close
    ):
        raise MassiveFinalizedAvailabilityError(
            "candidate decision session cannot support the frozen V0 decision and fill"
        )
    eligible = source_metadata.vendor_available_at_ms <= availability_cutoff
    body = {
        "schema": MASSIVE_FINALIZED_SOURCE_AVAILABILITY_V0_SCHEMA,
        "protocol_id": MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "source_object_receipt_sha256": source_object_receipt.receipt_sha256,
        "source_metadata_receipt_sha256": source_metadata.receipt_sha256,
        "dataset_id": source_object_receipt.dataset_id,
        "source_object_key": source_object_receipt.source_object_key,
        "exchange": source_session.exchange,
        "source_session_date": source_session.session_date,
        "decision_session_date": decision_session.session_date,
        "source_regular_close_at_ms": source_cutoff,
        "decision_regular_open_at_ms": regular_open,
        "decision_regular_close_at_ms": regular_close,
        "source_feature_cutoff_at_ms": source_cutoff,
        "latest_input_observation_at_ms": latest_observation,
        "vendor_last_modified_at_ms": source_metadata.vendor_last_modified_at_ms,
        "vendor_available_at_ms": source_metadata.vendor_available_at_ms,
        "availability_cutoff_at_ms": availability_cutoff,
        "decision_at_ms": decision_at,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
        "origin_eligible": eligible,
        "ineligibility_reason": (None if eligible else "vendor-available-after-cutoff"),
    }
    authority = MassiveFinalizedSourceAvailabilityAuthorityV0(
        protocol_id=MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
        protocol_receipt_sha256=MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        source_object_receipt_sha256=source_object_receipt.receipt_sha256,
        source_metadata_receipt_sha256=source_metadata.receipt_sha256,
        dataset_id=source_object_receipt.dataset_id,
        source_object_key=source_object_receipt.source_object_key,
        exchange=source_session.exchange,
        source_session_date=source_session.session_date,
        decision_session_date=decision_session.session_date,
        source_regular_close_at_ms=source_cutoff,
        decision_regular_open_at_ms=regular_open,
        decision_regular_close_at_ms=regular_close,
        source_feature_cutoff_at_ms=source_cutoff,
        latest_input_observation_at_ms=latest_observation,
        vendor_last_modified_at_ms=source_metadata.vendor_last_modified_at_ms,
        vendor_available_at_ms=source_metadata.vendor_available_at_ms,
        availability_cutoff_at_ms=availability_cutoff,
        decision_at_ms=decision_at,
        fill_start_at_ms=fill_start,
        fill_end_at_ms=fill_end,
        origin_eligible=eligible,
        ineligibility_reason=(None if eligible else "vendor-available-after-cutoff"),
        availability_authority_receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginAvailabilityAuthorityV0:
    protocol_receipt_sha256: str
    source_session_date: str
    decision_session_date: str
    source_availability_authority_receipts: tuple[str, ...]
    origin_eligible: bool
    ineligibility_reasons: tuple[str, ...]
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_AVAILABILITY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_AVAILABILITY_V0_SCHEMA:
            raise MassiveFinalizedAvailabilityError(
                "finalized origin-availability schema drifted"
            )
        if (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
        ):
            raise MassiveFinalizedAvailabilityError(
                "origin availability protocol receipt drifted"
            )
        receipts = self.source_availability_authority_receipts
        if not receipts or receipts != tuple(sorted(set(receipts))):
            raise MassiveFinalizedAvailabilityError(
                "source availability receipts must be sorted and unique"
            )
        for receipt in receipts:
            _digest("source availability authority receipt", receipt)
        if not isinstance(self.origin_eligible, bool):
            raise MassiveFinalizedAvailabilityError(
                "origin eligibility must be Boolean"
            )
        expected_eligible = not self.ineligibility_reasons
        if self.origin_eligible != expected_eligible:
            raise MassiveFinalizedAvailabilityError(
                "origin eligibility differs from source reasons"
            )
        if self.ineligibility_reasons != tuple(sorted(set(self.ineligibility_reasons))):
            raise MassiveFinalizedAvailabilityError(
                "origin ineligibility reasons must be sorted and unique"
            )
        _digest("origin availability receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedAvailabilityError(
                "origin-availability receipt differs"
            )


def build_massive_finalized_origin_availability_authority_v0(
    sources: Sequence[MassiveFinalizedSourceAvailabilityAuthorityV0],
) -> MassiveFinalizedOriginAvailabilityAuthorityV0:
    """Require every finalized input object to pass one common origin."""

    rows = tuple(sources)
    if not rows:
        raise MassiveFinalizedAvailabilityError(
            "a finalized origin requires at least one source object"
        )
    for row in rows:
        row.validate()
    source_dates = {row.source_session_date for row in rows}
    decision_dates = {row.decision_session_date for row in rows}
    if len(source_dates) != 1 or len(decision_dates) != 1:
        raise MassiveFinalizedAvailabilityError(
            "all finalized inputs must describe one source and decision session"
        )
    receipts = tuple(sorted(row.availability_authority_receipt_sha256 for row in rows))
    if len(receipts) != len(set(receipts)):
        raise MassiveFinalizedAvailabilityError(
            "duplicate finalized source availability authority"
        )
    reasons = tuple(
        sorted(
            {
                row.ineligibility_reason
                for row in rows
                if row.ineligibility_reason is not None
            }
        )
    )
    body = {
        "schema": MASSIVE_FINALIZED_ORIGIN_AVAILABILITY_V0_SCHEMA,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "source_session_date": next(iter(source_dates)),
        "decision_session_date": next(iter(decision_dates)),
        "source_availability_authority_receipts": list(receipts),
        "origin_eligible": not reasons,
        "ineligibility_reasons": list(reasons),
    }
    authority = MassiveFinalizedOriginAvailabilityAuthorityV0(
        protocol_receipt_sha256=MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        source_session_date=next(iter(source_dates)),
        decision_session_date=next(iter(decision_dates)),
        source_availability_authority_receipts=receipts,
        origin_eligible=not reasons,
        ineligibility_reasons=reasons,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


def select_first_eligible_massive_finalized_origin_v0(
    candidates: Sequence[MassiveFinalizedOriginAvailabilityAuthorityV0],
) -> MassiveFinalizedOriginAvailabilityAuthorityV0:
    """Return the first eligible candidate without backdating vendor metadata."""

    rows = tuple(candidates)
    if not rows:
        raise MassiveFinalizedAvailabilityError("candidate origin inventory is empty")
    for row in rows:
        row.validate()
    source_dates = {row.source_session_date for row in rows}
    if len(source_dates) != 1:
        raise MassiveFinalizedAvailabilityError(
            "candidate origins must share one source session"
        )
    ordered = tuple(sorted(rows, key=lambda row: row.decision_session_date))
    if ordered != rows or len({row.decision_session_date for row in rows}) != len(rows):
        raise MassiveFinalizedAvailabilityError(
            "candidate origins must be sorted and unique by decision session"
        )
    for row in rows:
        if row.origin_eligible:
            return row
    raise MassiveFinalizedAvailabilityError(
        "no eligible finalized decision session was supplied"
    )


__all__ = [
    "MASSIVE_FINALIZED_ORIGIN_AVAILABILITY_V0_SCHEMA",
    "MASSIVE_FINALIZED_SOURCE_AVAILABILITY_V0_SCHEMA",
    "MASSIVE_VENDOR_OBJECT_METADATA_V0_SCHEMA",
    "MassiveFinalizedAvailabilityError",
    "MassiveFinalizedOriginAvailabilityAuthorityV0",
    "MassiveFinalizedSourceAvailabilityAuthorityV0",
    "MassiveVendorObjectMetadataV0",
    "build_massive_finalized_origin_availability_authority_v0",
    "build_massive_finalized_source_availability_authority_v0",
    "build_massive_vendor_object_metadata_v0",
    "select_first_eligible_massive_finalized_origin_v0",
]
