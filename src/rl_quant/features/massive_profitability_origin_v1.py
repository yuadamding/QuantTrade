"""Acquisition-bound P0 origins with a frozen monthly PIT-500 schedule."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from rl_quant.alpha.pit_universe import (
    HistoricalMembershipRecord,
    PITSecurityUniverseAuthority,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0,
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-profitability-acquired-source-evidence-v1"
)
MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_DATASET = (
    "massive-profitability-acquired-source-evidence-v1"
)
MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_OBJECT_PREFIX = (
    "massive-profitability-p0/acquired-source-evidence-v1/"
)
MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA,
            "listing": "authenticated-captured-flat-file-listing-v0",
            "source": "exact-committed-flat-trade-object",
            "publication": "create-only-canonical-json",
            "generic_reload": "nonauthorizing-until-physical-rederivation",
        }
    )
)
MASSIVE_PROFITABILITY_MONTHLY_MEMBERSHIP_SCHEDULE_V1_SCHEMA = (
    "rl-quant.massive-profitability-monthly-membership-schedule-v1"
)
MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA = (
    "rl-quant.massive-profitability-decision-origin-v1"
)
MASSIVE_PROFITABILITY_SKIPPED_DECISION_V1_SCHEMA = (
    "rl-quant.massive-profitability-skipped-decision-v1"
)
MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA = (
    "rl-quant.massive-profitability-origin-plan-v1"
)

MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS = 18 * 60 * 60 * 1_000
MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source_session": "decision-minus-two-XNYS-sessions",
        "source_evidence": "persisted-and-rederived-captured-listing-v0",
        "membership_schedule": "first-XNYS-session-of-every-calendar-month",
        "membership_activation": "complete-group-available-by-scheduled-open",
        "membership_selection": "scheduled-group-for-decision-calendar-month",
        "origin_identity": "selected-membership-and-origin-available-identity-only",
        "full_identity": "audit-only",
        "decision": "12:30:00-America/New_York",
        "fill": "[15:50:00,16:00:00)-America/New_York",
        "feature_cutoff": "source-session-close",
        "authorizations": "all-performance-authorizations-false",
    }
)

MASSIVE_PROFITABILITY_ORIGIN_V1_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V1_LOCKBOX_ACCESS_AUTHORIZED = False

_EASTERN = ZoneInfo("America/New_York")
_SKIP_REASONS = {
    "decision-session-cannot-support-frozen-clock",
    "insufficient-prior-session-history",
    "missing-acquired-source-evidence",
    "vendor-object-predates-source-close",
    "vendor-lead-below-18-hours",
    "missing-scheduled-monthly-membership",
    "scheduled-membership-unavailable-at-activation",
}


class MassiveProfitabilityOriginV1Error(ValueError):
    """Acquired source evidence or monthly membership is not authoritative."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveProfitabilityOriginV1Error(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOriginV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveProfitabilityOriginV1Error(f"{name} must be nonnegative")
    return value


def _canonical_date(name: str, value: object) -> str:
    raw = _text(name, value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveProfitabilityOriginV1Error(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != raw:
        raise MassiveProfitabilityOriginV1Error(f"{name} is not canonical")
    return raw


def _artifact_id(value: object) -> str:
    result = _text("source evidence artifact ID", value)
    if any(not (character.isalnum() or character in "-_") for character in result):
        raise MassiveProfitabilityOriginV1Error("artifact ID is not path safe")
    return result


def _local_ms(session_date: str, value: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(session_date), value, tzinfo=_EASTERN
        ).timestamp()
        * 1_000
    )


def _session_ms(value_ns: int) -> int:
    value = _nonnegative_int("session timestamp", value_ns)
    if value % 1_000_000:
        raise MassiveProfitabilityOriginV1Error(
            "session timestamp is not millisecond aligned"
        )
    return value // 1_000_000


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveProfitabilityOriginV1Error(f"{name} fields differ")


def _captured_listing_semantic_receipt(
    captured: MassiveCapturedFlatFileListingV0,
) -> str:
    return semantic_sha256(
        {
            "acquisition_evidence_receipt_sha256": (
                captured.acquisition_evidence.receipt_sha256
            ),
            "loaded_acquisition_receipt_sha256": (
                captured.loaded_acquisition.receipt_sha256
            ),
            "loaded_listing_receipt_sha256": captured.loaded_listing.receipt_sha256,
            "committed_listing_receipt_sha256": (
                captured.committed_listing.receipt_sha256
            ),
            "provider_request_ids": captured.acquisition_evidence.provider_request_ids,
        }
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAcquiredSourceEvidenceV1:
    source_session_date: str
    source_object_key: str
    vendor_last_modified_at_ms: int
    listing_observed_at_ms: int
    acquisition_requested_at_ms: int
    acquisition_completed_at_ms: int
    research_downloaded_at_ms: int
    research_verified_at_ms: int
    content_length: int
    etag: str
    listing_entry_receipt_sha256: str
    captured_listing_semantic_receipt_sha256: str
    acquisition_evidence_receipt_sha256: str
    loaded_acquisition_receipt_sha256: str
    committed_listing_receipt_sha256: str
    loaded_listing_receipt_sha256: str
    provider_request_inventory_sha256: str
    loaded_source_receipt_sha256: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        source_date = _canonical_date("source session", self.source_session_date)
        if self.source_object_key != canonical_massive_trade_object_key(source_date):
            raise MassiveProfitabilityOriginV1Error("source object key differs")
        modified = _nonnegative_int(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        observed = _nonnegative_int("listing observation", self.listing_observed_at_ms)
        requested = _nonnegative_int(
            "listing acquisition request", self.acquisition_requested_at_ms
        )
        completed = _nonnegative_int(
            "listing acquisition completion", self.acquisition_completed_at_ms
        )
        downloaded = _nonnegative_int(
            "research download", self.research_downloaded_at_ms
        )
        verified = _nonnegative_int(
            "research verification", self.research_verified_at_ms
        )
        if (
            modified > observed
            or requested > completed
            or completed != observed
            or downloaded < modified
            or verified < downloaded
        ):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source chronology differs"
            )
        if _nonnegative_int("source content length", self.content_length) <= 0:
            raise MassiveProfitabilityOriginV1Error("source object is empty")
        _text("source ETag", self.etag)
        for name in (
            "listing_entry_receipt_sha256",
            "captured_listing_semantic_receipt_sha256",
            "acquisition_evidence_receipt_sha256",
            "loaded_acquisition_receipt_sha256",
            "committed_listing_receipt_sha256",
            "loaded_listing_receipt_sha256",
            "provider_request_inventory_sha256",
            "loaded_source_receipt_sha256",
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source evidence receipt differs"
            )


def _captured_listing_inventory(
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                captured.acquisition_evidence.prefix,
                captured.acquisition_evidence.receipt_sha256,
                captured.committed_listing.receipt_sha256,
                _captured_listing_semantic_receipt(captured),
            )
            for captured in captured_listings
        )
    )


def _derive_acquired_source_rows(
    *,
    root: str | Path,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    loaded_trade_sources: Sequence[LoadedMassiveSourceObject],
) -> tuple[
    tuple[MassiveProfitabilityAcquiredSourceEvidenceV1, ...],
    str,
    str,
]:
    if not captured_listings or not loaded_trade_sources:
        raise MassiveProfitabilityOriginV1Error(
            "acquired source evidence requires listings and trade sources"
        )
    by_key: dict[str, tuple[MassiveCapturedFlatFileListingV0, Any]] = {}
    prefixes: set[str] = set()
    for captured in captured_listings:
        validate_massive_captured_flat_file_listing_v0(
            root=root, captured_listing=captured
        )
        prefix = captured.acquisition_evidence.prefix
        if prefix in prefixes:
            raise MassiveProfitabilityOriginV1Error(
                "captured listing months are duplicated"
            )
        prefixes.add(prefix)
        for entry in captured.committed_listing.entries:
            if entry.source_object_key in by_key:
                raise MassiveProfitabilityOriginV1Error(
                    "captured listing entries overlap"
                )
            by_key[entry.source_object_key] = (captured, entry)

    rows: list[MassiveProfitabilityAcquiredSourceEvidenceV1] = []
    source_keys: set[str] = set()
    for loaded_source in loaded_trade_sources:
        loaded_source.validate()
        receipt = loaded_source.receipt
        if (
            receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID
            or receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source is not a Massive flat trade file"
            )
        if receipt.source_object_key in source_keys:
            raise MassiveProfitabilityOriginV1Error(
                "acquired trade sources are duplicated"
            )
        source_keys.add(receipt.source_object_key)
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
        match = by_key.get(receipt.source_object_key)
        if match is None:
            raise MassiveProfitabilityOriginV1Error(
                "trade source is absent from authenticated captured listings"
            )
        captured, entry = match
        if (
            receipt.content_length != entry.content_length
            or receipt.etag != entry.etag
            or receipt.source_object_key != entry.source_object_key
        ):
            raise MassiveProfitabilityOriginV1Error(
                "trade source and captured listing entry differ"
            )
        acquisition = captured.acquisition_evidence
        body: dict[str, object] = {
            "source_session_date": entry.coverage_session_date,
            "source_object_key": entry.source_object_key,
            "vendor_last_modified_at_ms": entry.vendor_last_modified_at_ms,
            "listing_observed_at_ms": entry.listing_observed_at_ms,
            "acquisition_requested_at_ms": acquisition.requested_at_ms,
            "acquisition_completed_at_ms": acquisition.completed_at_ms,
            "research_downloaded_at_ms": receipt.downloaded_at_ms,
            "research_verified_at_ms": loaded_source.verified_at_ms,
            "content_length": receipt.content_length,
            "etag": entry.etag,
            "listing_entry_receipt_sha256": entry.receipt_sha256,
            "captured_listing_semantic_receipt_sha256": (
                _captured_listing_semantic_receipt(captured)
            ),
            "acquisition_evidence_receipt_sha256": acquisition.receipt_sha256,
            "loaded_acquisition_receipt_sha256": (
                captured.loaded_acquisition.receipt_sha256
            ),
            "committed_listing_receipt_sha256": (
                captured.committed_listing.receipt_sha256
            ),
            "loaded_listing_receipt_sha256": (captured.loaded_listing.receipt_sha256),
            "provider_request_inventory_sha256": semantic_sha256(
                acquisition.provider_request_ids
            ),
            "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            "source_object_receipt_sha256": receipt.receipt_sha256,
            "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        }
        row = MassiveProfitabilityAcquiredSourceEvidenceV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    ordered = tuple(sorted(rows, key=lambda row: row.source_session_date))
    if tuple(row.source_session_date for row in ordered) != tuple(
        sorted({row.source_session_date for row in ordered})
    ):
        raise MassiveProfitabilityOriginV1Error("acquired source dates are not unique")
    captured_inventory = semantic_sha256(_captured_listing_inventory(captured_listings))
    source_inventory = semantic_sha256(
        tuple((row.source_session_date, row.receipt_sha256) for row in ordered)
    )
    return ordered, captured_inventory, source_inventory


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAcquiredSourceEvidenceArtifactV1:
    rows: tuple[MassiveProfitabilityAcquiredSourceEvidenceV1, ...]
    captured_listing_inventory_sha256: str
    source_inventory_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    acquisition_qualified: bool = False
    schema: str = MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rows": [asdict(row) for row in self.rows],
            "captured_listing_inventory_sha256": (
                self.captured_listing_inventory_sha256
            ),
            "source_inventory_sha256": self.source_inventory_sha256,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA:
            raise MassiveProfitabilityOriginV1Error(
                "acquired source artifact schema differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source artifact transaction differs"
            )
        if self.acquisition_qualified is not False:
            raise MassiveProfitabilityOriginV1Error(
                "generic acquired source artifacts must remain nonauthorizing"
            )
        source_dates = tuple(row.source_session_date for row in self.rows)
        if not self.rows or source_dates != tuple(sorted(set(source_dates))):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source artifact rows differ"
            )
        for row in self.rows:
            row.validate()
        for name in (
            "captured_listing_inventory_sha256",
            "source_inventory_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.source_inventory_sha256 != semantic_sha256(
            tuple((row.source_session_date, row.receipt_sha256) for row in self.rows)
        ):
            raise MassiveProfitabilityOriginV1Error("acquired source inventory differs")
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginV1Error(
                "acquired source semantic receipt differs"
            )


def _source_artifact_payload(
    *,
    rows: tuple[MassiveProfitabilityAcquiredSourceEvidenceV1, ...],
    captured_listing_inventory_sha256: str,
    source_inventory_sha256: str,
    semantic_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA,
        "rows": [asdict(row) for row in rows],
        "captured_listing_inventory_sha256": captured_listing_inventory_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "semantic_receipt_sha256": semantic_receipt_sha256,
    }


def materialize_massive_profitability_acquired_source_evidence_v1(
    *,
    root: str | Path,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    loaded_trade_sources: Sequence[LoadedMassiveSourceObject],
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityAcquiredSourceEvidenceArtifactV1:
    """Persist source availability derived only from authenticated listings."""

    rows, captured_inventory, source_inventory = _derive_acquired_source_rows(
        root=root,
        captured_listings=captured_listings,
        loaded_trade_sources=loaded_trade_sources,
    )
    semantic_unsigned = {
        "schema": MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA,
        "rows": [asdict(row) for row in rows],
        "captured_listing_inventory_sha256": captured_inventory,
        "source_inventory_sha256": source_inventory,
    }
    semantic_receipt = semantic_sha256(semantic_unsigned)
    payload = _source_artifact_payload(
        rows=rows,
        captured_listing_inventory_sha256=captured_inventory,
        source_inventory_sha256=source_inventory,
        semantic_receipt_sha256=semantic_receipt,
    )
    identifier = _artifact_id(artifact_id)
    relative = (
        f"{MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_OBJECT_PREFIX}"
        f"{identifier}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=_digest(
            "source evidence entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-ACQUIRED-SOURCE-EVIDENCE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    result = parse_massive_profitability_acquired_source_evidence_v1(
        root=root, loaded_source=loaded
    )
    expected = replace(result, loaded_source=loaded)
    if (
        expected.rows != rows
        or expected.captured_listing_inventory_sha256 != captured_inventory
        or expected.source_inventory_sha256 != source_inventory
        or expected.semantic_receipt_sha256 != semantic_receipt
    ):
        raise MassiveProfitabilityOriginV1Error(
            "persisted acquired source evidence differs from derivation"
        )
    return result


def parse_massive_profitability_acquired_source_evidence_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityAcquiredSourceEvidenceArtifactV1:
    """Reload evidence bytes; reloading alone never authorizes their acquisition."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityOriginV1Error(
            "acquired source artifact is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityOriginV1Error(
            "acquired source artifact is not canonical JSON"
        )
    _exact_keys(
        payload,
        {
            "schema",
            "rows",
            "captured_listing_inventory_sha256",
            "source_inventory_sha256",
            "semantic_receipt_sha256",
        },
        name="acquired source artifact",
    )
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, dict) for row in raw_rows
    ):
        raise MassiveProfitabilityOriginV1Error("acquired source rows are malformed")
    try:
        rows = tuple(
            MassiveProfitabilityAcquiredSourceEvidenceV1(**row) for row in raw_rows
        )
        result = MassiveProfitabilityAcquiredSourceEvidenceArtifactV1(
            schema=payload["schema"],
            rows=rows,
            captured_listing_inventory_sha256=payload[
                "captured_listing_inventory_sha256"
            ],
            source_inventory_sha256=payload["source_inventory_sha256"],
            semantic_receipt_sha256=payload["semantic_receipt_sha256"],
            loaded_source=loaded_source,
            acquisition_qualified=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityOriginV1Error(
            "acquired source artifact values are malformed"
        ) from exc
    regenerated = _source_artifact_payload(
        rows=result.rows,
        captured_listing_inventory_sha256=(result.captured_listing_inventory_sha256),
        source_inventory_sha256=result.source_inventory_sha256,
        semantic_receipt_sha256=result.semantic_receipt_sha256,
    )
    if raw != canonical_json_file_bytes(regenerated):
        raise MassiveProfitabilityOriginV1Error(
            "acquired source artifact regenerated bytes differ"
        )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityMonthlyMembershipScheduleRowV1:
    calendar_month: str
    scheduled_rebalance_session_date: str
    scheduled_effective_at_ms: int
    membership_group_present: bool
    membership_group_activation_qualified: bool
    membership_group_available_at_ms: int | None
    member_security_ids: tuple[str, ...]
    member_universe_ranks: tuple[int, ...]
    membership_group_semantic_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            len(self.calendar_month) != 7
            or self.calendar_month[4] != "-"
            or _canonical_date(
                "scheduled rebalance session", self.scheduled_rebalance_session_date
            )[:7]
            != self.calendar_month
        ):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership schedule identity differs"
            )
        effective = _nonnegative_int(
            "scheduled membership effective time", self.scheduled_effective_at_ms
        )
        if not isinstance(self.membership_group_present, bool) or not isinstance(
            self.membership_group_activation_qualified, bool
        ):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership state must be Boolean"
            )
        optional_values = (
            self.membership_group_available_at_ms,
            self.membership_group_semantic_receipt_sha256,
        )
        if self.membership_group_present is not all(
            value is not None for value in optional_values
        ):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership evidence shape differs"
            )
        if not self.membership_group_present:
            if (
                self.membership_group_activation_qualified
                or self.member_security_ids
                or self.member_universe_ranks
            ):
                raise MassiveProfitabilityOriginV1Error(
                    "missing monthly membership carries selected members"
                )
        else:
            available = _nonnegative_int(
                "membership group availability", self.membership_group_available_at_ms
            )
            _digest(
                "membership group semantic receipt",
                self.membership_group_semantic_receipt_sha256,
            )
            if self.membership_group_activation_qualified is not (
                available <= effective
            ):
                raise MassiveProfitabilityOriginV1Error(
                    "monthly membership activation state differs"
                )
            if (
                not self.member_security_ids
                or len(self.member_security_ids) != len(self.member_universe_ranks)
                or len(set(self.member_security_ids)) != len(self.member_security_ids)
                or self.member_universe_ranks
                != tuple(sorted(set(self.member_universe_ranks)))
                or len(self.member_security_ids)
                > MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.target_size
            ):
                raise MassiveProfitabilityOriginV1Error(
                    "monthly membership member inventory differs"
                )
            for security_id in self.member_security_ids:
                _text("monthly member security ID", security_id)
            if any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
                for rank in self.member_universe_ranks
            ):
                raise MassiveProfitabilityOriginV1Error(
                    "monthly membership ranks are invalid"
                )
        _digest("monthly membership row receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityMonthlyMembershipScheduleV1:
    first_candidate_decision_session_date: str
    last_candidate_decision_session_date: str
    rows: tuple[MassiveProfitabilityMonthlyMembershipScheduleRowV1, ...]
    schedule_semantic_receipt_sha256: str
    identity_authority_audit_receipt_sha256: str
    audit_receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_MONTHLY_MEMBERSHIP_SCHEDULE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "first_candidate_decision_session_date": (
                self.first_candidate_decision_session_date
            ),
            "last_candidate_decision_session_date": (
                self.last_candidate_decision_session_date
            ),
            "rows": [asdict(row) for row in self.rows],
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_MONTHLY_MEMBERSHIP_SCHEDULE_V1_SCHEMA:
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership schedule schema differs"
            )
        first = _canonical_date(
            "first candidate decision", self.first_candidate_decision_session_date
        )
        last = _canonical_date(
            "last candidate decision", self.last_candidate_decision_session_date
        )
        if last < first or not self.rows:
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership schedule interval differs"
            )
        for row in self.rows:
            row.validate()
        months = tuple(row.calendar_month for row in self.rows)
        if months != tuple(sorted(set(months))):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership schedule rows are not unique"
            )
        for name in (
            "schedule_semantic_receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.schedule_semantic_receipt_sha256 != semantic_sha256(
            self.semantic_unsigned()
        ):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "schedule_semantic_receipt_sha256": (
                    self.schedule_semantic_receipt_sha256
                ),
                "identity_authority_audit_receipt_sha256": (
                    self.identity_authority_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityOriginV1Error(
                "monthly membership audit receipt differs"
            )


def _scheduled_session_rows(
    *,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> tuple[MassiveProfitabilityMonthlyMembershipScheduleRowV1, ...]:
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    by_month: defaultdict[str, list[MassiveExchangeSession]] = defaultdict(list)
    for session in sessions:
        by_month[session.session_date[:7]].append(session)
    candidate_months = tuple(
        sorted(
            {
                session.session_date[:7]
                for session in sessions
                if first_candidate_decision_session_date
                <= session.session_date
                <= last_candidate_decision_session_date
            }
        )
    )
    if not candidate_months:
        raise MassiveProfitabilityOriginV1Error(
            "monthly membership candidate interval is empty"
        )
    membership_by_effective: defaultdict[int, list[HistoricalMembershipRecord]] = (
        defaultdict(list)
    )
    for row in identity_authority.membership_events:
        membership_by_effective[row.effective_at_ms].append(row)
    output: list[MassiveProfitabilityMonthlyMembershipScheduleRowV1] = []
    for month in candidate_months:
        scheduled = min(by_month[month], key=lambda row: row.session_date)
        effective = _session_ms(scheduled.regular_open_ns)
        group = tuple(
            sorted(
                membership_by_effective.get(effective, ()),
                key=lambda row: row.security_id,
            )
        )
        if not group:
            body: dict[str, object] = {
                "calendar_month": month,
                "scheduled_rebalance_session_date": scheduled.session_date,
                "scheduled_effective_at_ms": effective,
                "membership_group_present": False,
                "membership_group_activation_qualified": False,
                "membership_group_available_at_ms": None,
                "member_security_ids": (),
                "member_universe_ranks": (),
                "membership_group_semantic_receipt_sha256": None,
            }
        else:
            available = max(row.available_at_ms for row in group)
            members = tuple(
                sorted(
                    (row for row in group if row.is_member),
                    key=lambda row: (row.universe_rank or 10**9, row.security_id),
                )
            )
            body = {
                "calendar_month": month,
                "scheduled_rebalance_session_date": scheduled.session_date,
                "scheduled_effective_at_ms": effective,
                "membership_group_present": True,
                "membership_group_activation_qualified": (
                    available <= effective
                    and all(row.observation_end_ms <= effective for row in group)
                ),
                "membership_group_available_at_ms": available,
                "member_security_ids": tuple(row.security_id for row in members),
                "member_universe_ranks": tuple(
                    int(row.universe_rank or 0) for row in members
                ),
                "membership_group_semantic_receipt_sha256": semantic_sha256(
                    tuple(asdict(row) for row in group)
                ),
            }
        schedule_row = MassiveProfitabilityMonthlyMembershipScheduleRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        schedule_row.validate()
        output.append(schedule_row)
    return tuple(output)


def build_massive_profitability_monthly_membership_schedule_v1(
    *,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveProfitabilityMonthlyMembershipScheduleV1:
    """Enumerate one first-session PIT membership requirement per month."""

    session_authority.validate()
    identity_authority.validate()
    if (
        identity_authority.rule.rebalance_frequency != "monthly"
        or identity_authority.rule.receipt_sha256
        != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
    ):
        raise MassiveProfitabilityOriginV1Error(
            "P0 membership schedule requires the frozen monthly universe rule"
        )
    rows = _scheduled_session_rows(
        session_authority=session_authority,
        identity_authority=identity_authority,
        first_candidate_decision_session_date=first_candidate_decision_session_date,
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    )
    semantic_unsigned = {
        "schema": MASSIVE_PROFITABILITY_MONTHLY_MEMBERSHIP_SCHEDULE_V1_SCHEMA,
        "first_candidate_decision_session_date": (
            first_candidate_decision_session_date
        ),
        "last_candidate_decision_session_date": last_candidate_decision_session_date,
        "rows": [asdict(row) for row in rows],
    }
    semantic_receipt = semantic_sha256(semantic_unsigned)
    audit_receipt = semantic_sha256(
        {
            "schedule_semantic_receipt_sha256": semantic_receipt,
            "identity_authority_audit_receipt_sha256": (
                identity_authority.receipt_sha256
            ),
        }
    )
    result = MassiveProfitabilityMonthlyMembershipScheduleV1(
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
        rows=rows,
        schedule_semantic_receipt_sha256=semantic_receipt,
        identity_authority_audit_receipt_sha256=identity_authority.receipt_sha256,
        audit_receipt_sha256=audit_receipt,
    )
    result.validate()
    return result


def _origin_available_identity_receipt(
    *,
    identity_authority: PITSecurityUniverseAuthority,
    group: tuple[HistoricalMembershipRecord, ...],
    member_security_ids: tuple[str, ...],
    decision_at_ms: int,
) -> str:
    masters = {row.security_id: row for row in identity_authority.security_master}
    member_identity: list[dict[str, object]] = []
    for security_id in member_security_ids:
        master = masters[security_id]
        ticker_candidates = tuple(
            row
            for row in identity_authority.ticker_history
            if row.security_id == security_id
            and row.valid_from_ms <= decision_at_ms
            and (row.valid_to_ms is None or decision_at_ms < row.valid_to_ms)
            and row.available_at_ms <= decision_at_ms
        )
        if len(ticker_candidates) != 1:
            raise MassiveProfitabilityOriginV1Error(
                "scheduled member lacks one origin-available ticker identity"
            )
        ticker = ticker_candidates[0]
        known_delisting = next(
            (
                row
                for row in identity_authority.delisting_events
                if row.security_id == security_id
                and row.available_at_ms <= decision_at_ms
            ),
            None,
        )
        member_identity.append(
            {
                "security_id": security_id,
                "issuer_id": master.issuer_id,
                "primary_exchange": master.primary_exchange,
                "share_class": master.share_class,
                "security_type": master.security_type,
                "listing_at_ms": master.listing_at_ms,
                "current_ticker": ticker.ticker,
                "ticker_valid_from_ms": ticker.valid_from_ms,
                "ticker_available_at_ms": ticker.available_at_ms,
                "ticker_source_receipt_sha256": ticker.source_receipt_sha256,
                "known_delisting": None
                if known_delisting is None
                else asdict(known_delisting),
            }
        )
    return semantic_sha256(
        {
            "membership_group": tuple(asdict(row) for row in group),
            "member_identity": tuple(member_identity),
        }
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDecisionOriginV1:
    source_session_date: str
    decision_session_date: str
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    feature_cutoff_at_ms: int
    source_staleness_sessions: int
    vendor_last_modified_at_ms: int
    vendor_lead_time_ms: int
    source_object_key: str
    source_evidence_receipt_sha256: str
    source_evidence_artifact_semantic_receipt_sha256: str
    scheduled_rebalance_session_date: str
    membership_age_sessions: int
    membership_effective_at_ms: int
    decision_member_security_ids: tuple[str, ...]
    decision_member_universe_ranks: tuple[int, ...]
    membership_group_semantic_receipt_sha256: str
    membership_schedule_semantic_receipt_sha256: str
    origin_available_identity_receipt_sha256: str
    session_authority_receipt_sha256: str
    protocol_receipt_sha256: str
    origin_spec_receipt_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    identity_authority_audit_receipt_sha256: str
    audit_receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        excluded = {
            "receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "audit_receipt_sha256",
        }
        return {
            key: value for key, value in asdict(self).items() if key not in excluded
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA
            or self.source_staleness_sessions
            != MASSIVE_FINALIZED_PROFITABILITY_P0.source_staleness_sessions
            or self.source_object_key
            != canonical_massive_trade_object_key(self.source_session_date)
        ):
            raise MassiveProfitabilityOriginV1Error("decision origin identity differs")
        _canonical_date("origin source session", self.source_session_date)
        _canonical_date("origin decision session", self.decision_session_date)
        _canonical_date(
            "scheduled rebalance session", self.scheduled_rebalance_session_date
        )
        if (
            self.decision_at_ms != _local_ms(self.decision_session_date, time(12, 30))
            or self.fill_start_at_ms
            != _local_ms(self.decision_session_date, time(15, 50))
            or self.fill_end_at_ms != _local_ms(self.decision_session_date, time(16, 0))
            or self.vendor_lead_time_ms
            != self.decision_at_ms - self.vendor_last_modified_at_ms
            or self.vendor_lead_time_ms < MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS
        ):
            raise MassiveProfitabilityOriginV1Error(
                "decision origin chronology differs"
            )
        if _nonnegative_int("membership age", self.membership_age_sessions) < 0:
            raise MassiveProfitabilityOriginV1Error("membership age differs")
        if (
            not self.decision_member_security_ids
            or len(self.decision_member_security_ids)
            != len(self.decision_member_universe_ranks)
            or self.decision_member_universe_ranks
            != tuple(sorted(set(self.decision_member_universe_ranks)))
        ):
            raise MassiveProfitabilityOriginV1Error(
                "decision membership inventory differs"
            )
        for name in (
            "source_evidence_receipt_sha256",
            "source_evidence_artifact_semantic_receipt_sha256",
            "membership_group_semantic_receipt_sha256",
            "membership_schedule_semantic_receipt_sha256",
            "origin_available_identity_receipt_sha256",
            "session_authority_receipt_sha256",
            "protocol_receipt_sha256",
            "origin_spec_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.origin_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityOriginV1Error(
                "decision origin semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.receipt_sha256,
                "identity_authority_audit_receipt_sha256": (
                    self.identity_authority_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityOriginV1Error(
                "decision origin audit receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySkippedDecisionV1:
    decision_session_date: str
    source_session_date: str | None
    source_evidence_receipt_sha256: str | None
    scheduled_rebalance_session_date: str | None
    reason: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_SKIPPED_DECISION_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_SKIPPED_DECISION_V1_SCHEMA
            or self.reason not in _SKIP_REASONS
        ):
            raise MassiveProfitabilityOriginV1Error("skipped decision identity differs")
        _canonical_date("skipped decision", self.decision_session_date)
        if self.source_session_date is not None:
            _canonical_date("skipped source session", self.source_session_date)
        if self.source_evidence_receipt_sha256 is not None:
            _digest("skipped source evidence", self.source_evidence_receipt_sha256)
        if self.scheduled_rebalance_session_date is not None:
            _canonical_date(
                "skipped scheduled rebalance", self.scheduled_rebalance_session_date
            )
        _digest("skipped decision receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginV1Error("skipped decision receipt differs")


def _skip(
    *,
    decision_session_date: str,
    source_session_date: str | None,
    source_evidence_receipt_sha256: str | None,
    scheduled_rebalance_session_date: str | None,
    reason: str,
) -> MassiveProfitabilitySkippedDecisionV1:
    body: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_SKIPPED_DECISION_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "source_session_date": source_session_date,
        "source_evidence_receipt_sha256": source_evidence_receipt_sha256,
        "scheduled_rebalance_session_date": scheduled_rebalance_session_date,
        "reason": reason,
    }
    result = MassiveProfitabilitySkippedDecisionV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _group_for_schedule_row(
    *,
    identity_authority: PITSecurityUniverseAuthority,
    schedule_row: MassiveProfitabilityMonthlyMembershipScheduleRowV1,
) -> tuple[HistoricalMembershipRecord, ...]:
    return tuple(
        sorted(
            (
                row
                for row in identity_authority.membership_events
                if row.effective_at_ms == schedule_row.scheduled_effective_at_ms
            ),
            key=lambda row: row.security_id,
        )
    )


def _derive_origin_rows(
    *,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    membership_schedule: MassiveProfitabilityMonthlyMembershipScheduleV1,
    candidate_dates: tuple[str, ...],
) -> tuple[
    tuple[MassiveProfitabilityDecisionOriginV1, ...],
    tuple[MassiveProfitabilitySkippedDecisionV1, ...],
]:
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    positions = {row.session_date: index for index, row in enumerate(sessions)}
    by_date = {row.session_date: row for row in sessions}
    sources = {row.source_session_date: row for row in source_artifact.rows}
    schedule_by_month = {row.calendar_month: row for row in membership_schedule.rows}
    origins: list[MassiveProfitabilityDecisionOriginV1] = []
    skips: list[MassiveProfitabilitySkippedDecisionV1] = []
    for decision_date in candidate_dates:
        decision_position = positions[decision_date]
        schedule_row = schedule_by_month[decision_date[:7]]
        if decision_position < 2:
            skips.append(
                _skip(
                    decision_session_date=decision_date,
                    source_session_date=None,
                    source_evidence_receipt_sha256=None,
                    scheduled_rebalance_session_date=(
                        schedule_row.scheduled_rebalance_session_date
                    ),
                    reason="insufficient-prior-session-history",
                )
            )
            continue
        source_session = sessions[decision_position - 2]
        source = sources.get(source_session.session_date)
        if source is None:
            skips.append(
                _skip(
                    decision_session_date=decision_date,
                    source_session_date=source_session.session_date,
                    source_evidence_receipt_sha256=None,
                    scheduled_rebalance_session_date=(
                        schedule_row.scheduled_rebalance_session_date
                    ),
                    reason="missing-acquired-source-evidence",
                )
            )
            continue
        decision_session = by_date[decision_date]
        decision_at = _local_ms(decision_date, time(12, 30))
        fill_start = _local_ms(decision_date, time(15, 50))
        fill_end = _local_ms(decision_date, time(16, 0))
        regular_open = _session_ms(decision_session.regular_open_ns)
        regular_close = _session_ms(decision_session.regular_close_ns)
        if not regular_open <= decision_at < fill_start < fill_end <= regular_close:
            skips.append(
                _skip(
                    decision_session_date=decision_date,
                    source_session_date=source_session.session_date,
                    source_evidence_receipt_sha256=source.receipt_sha256,
                    scheduled_rebalance_session_date=(
                        schedule_row.scheduled_rebalance_session_date
                    ),
                    reason="decision-session-cannot-support-frozen-clock",
                )
            )
            continue
        source_close = _session_ms(source_session.regular_close_ns)
        if source.vendor_last_modified_at_ms < source_close:
            reason = "vendor-object-predates-source-close"
        elif (
            decision_at - source.vendor_last_modified_at_ms
            < MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS
        ):
            reason = "vendor-lead-below-18-hours"
        elif not schedule_row.membership_group_present:
            reason = "missing-scheduled-monthly-membership"
        elif not schedule_row.membership_group_activation_qualified:
            reason = "scheduled-membership-unavailable-at-activation"
        else:
            reason = None
        if reason is not None:
            skips.append(
                _skip(
                    decision_session_date=decision_date,
                    source_session_date=source_session.session_date,
                    source_evidence_receipt_sha256=source.receipt_sha256,
                    scheduled_rebalance_session_date=(
                        schedule_row.scheduled_rebalance_session_date
                    ),
                    reason=reason,
                )
            )
            continue
        group = _group_for_schedule_row(
            identity_authority=identity_authority, schedule_row=schedule_row
        )
        group_receipt = semantic_sha256(tuple(asdict(row) for row in group))
        if group_receipt != schedule_row.membership_group_semantic_receipt_sha256:
            raise MassiveProfitabilityOriginV1Error(
                "scheduled membership group differs from identity authority"
            )
        identity_semantic = _origin_available_identity_receipt(
            identity_authority=identity_authority,
            group=group,
            member_security_ids=schedule_row.member_security_ids,
            decision_at_ms=decision_at,
        )
        membership_age = (
            decision_position - positions[schedule_row.scheduled_rebalance_session_date]
        )
        semantic_body: dict[str, object] = {
            "schema": MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA,
            "source_session_date": source_session.session_date,
            "decision_session_date": decision_date,
            "decision_at_ms": decision_at,
            "fill_start_at_ms": fill_start,
            "fill_end_at_ms": fill_end,
            "feature_cutoff_at_ms": source_close,
            "source_staleness_sessions": 2,
            "vendor_last_modified_at_ms": source.vendor_last_modified_at_ms,
            "vendor_lead_time_ms": decision_at - source.vendor_last_modified_at_ms,
            "source_object_key": source.source_object_key,
            "source_evidence_receipt_sha256": source.receipt_sha256,
            "source_evidence_artifact_semantic_receipt_sha256": (
                source_artifact.semantic_receipt_sha256
            ),
            "scheduled_rebalance_session_date": (
                schedule_row.scheduled_rebalance_session_date
            ),
            "membership_age_sessions": membership_age,
            "membership_effective_at_ms": schedule_row.scheduled_effective_at_ms,
            "decision_member_security_ids": schedule_row.member_security_ids,
            "decision_member_universe_ranks": schedule_row.member_universe_ranks,
            "membership_group_semantic_receipt_sha256": group_receipt,
            "membership_schedule_semantic_receipt_sha256": (
                membership_schedule.schedule_semantic_receipt_sha256
            ),
            "origin_available_identity_receipt_sha256": identity_semantic,
            "session_authority_receipt_sha256": session_authority.receipt_sha256,
            "protocol_receipt_sha256": (
                MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            ),
            "origin_spec_receipt_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
            "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
        }
        semantic_receipt = semantic_sha256(semantic_body)
        audit_receipt = semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "identity_authority_audit_receipt_sha256": (
                    identity_authority.receipt_sha256
                ),
            }
        )
        origin = MassiveProfitabilityDecisionOriginV1(
            **semantic_body,  # type: ignore[arg-type]
            receipt_sha256=semantic_receipt,
            identity_authority_audit_receipt_sha256=(identity_authority.receipt_sha256),
            audit_receipt_sha256=audit_receipt,
        )
        origin.validate()
        origins.append(origin)
    return tuple(origins), tuple(skips)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDecisionOriginPlanV1:
    first_candidate_decision_session_date: str
    last_candidate_decision_session_date: str
    candidate_decision_session_dates: tuple[str, ...]
    origins: tuple[MassiveProfitabilityDecisionOriginV1, ...]
    skipped_decisions: tuple[MassiveProfitabilitySkippedDecisionV1, ...]
    source_evidence_artifact_semantic_receipt_sha256: str
    membership_schedule_semantic_receipt_sha256: str
    session_authority_receipt_sha256: str
    protocol_receipt_sha256: str
    origin_spec_receipt_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    identity_authority_audit_receipt_sha256: str
    source_evidence_artifact_audit_receipt_sha256: str
    audit_receipt_sha256: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        excluded = {
            "semantic_receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "source_evidence_artifact_audit_receipt_sha256",
            "audit_receipt_sha256",
        }
        result = {
            key: value for key, value in asdict(self).items() if key not in excluded
        }
        result["origins"] = tuple(
            row.semantic_unsigned() | {"receipt_sha256": row.receipt_sha256}
            for row in self.origins
        )
        result["skipped_decisions"] = tuple(
            asdict(row) for row in self.skipped_decisions
        )
        return result

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.origin_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityOriginV1Error("origin plan identity differs")
        first = _canonical_date(
            "first candidate decision", self.first_candidate_decision_session_date
        )
        last = _canonical_date(
            "last candidate decision", self.last_candidate_decision_session_date
        )
        if last < first or not self.candidate_decision_session_dates:
            raise MassiveProfitabilityOriginV1Error("origin plan interval differs")
        if self.candidate_decision_session_dates != tuple(
            sorted(set(self.candidate_decision_session_dates))
        ):
            raise MassiveProfitabilityOriginV1Error(
                "origin candidate dates are not sorted and unique"
            )
        for origin in self.origins:
            origin.validate()
        for skipped in self.skipped_decisions:
            skipped.validate()
        partition = tuple(
            sorted(
                tuple(row.decision_session_date for row in self.origins)
                + tuple(row.decision_session_date for row in self.skipped_decisions)
            )
        )
        if partition != self.candidate_decision_session_dates:
            raise MassiveProfitabilityOriginV1Error(
                "origin rows do not partition candidate decisions"
            )
        for name in (
            "source_evidence_artifact_semantic_receipt_sha256",
            "membership_schedule_semantic_receipt_sha256",
            "session_authority_receipt_sha256",
            "protocol_receipt_sha256",
            "origin_spec_receipt_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "source_evidence_artifact_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if any(
            (
                self.panel_materialization_authorized,
                self.predictive_training_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
            )
        ):
            raise MassiveProfitabilityOriginV1Error(
                "origin plan performance authorization differs"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginV1Error(
                "origin plan semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "identity_authority_audit_receipt_sha256": (
                    self.identity_authority_audit_receipt_sha256
                ),
                "source_evidence_artifact_audit_receipt_sha256": (
                    self.source_evidence_artifact_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityOriginV1Error("origin plan audit receipt differs")


def _artifact_audit_receipt(
    artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
) -> str:
    return semantic_sha256(
        {
            "semantic_receipt_sha256": artifact.semantic_receipt_sha256,
            "loaded_source_receipt_sha256": artifact.loaded_source.receipt_sha256,
            "source_commit_receipt_sha256": (
                artifact.loaded_source.commit.receipt_sha256
            ),
        }
    )


def _build_massive_profitability_decision_origin_plan_v1(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_evidence_artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    loaded_trade_sources: Sequence[LoadedMassiveSourceObject],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveProfitabilityDecisionOriginPlanV1:
    """Build P0 origins only after physically rederiving acquisition evidence."""

    session_authority.validate()
    identity_authority.validate()
    parsed_artifact = parse_massive_profitability_acquired_source_evidence_v1(
        root=root, loaded_source=source_evidence_artifact.loaded_source
    )
    if parsed_artifact != source_evidence_artifact:
        raise MassiveProfitabilityOriginV1Error(
            "source evidence artifact differs from committed bytes"
        )
    rows, captured_inventory, source_inventory = _derive_acquired_source_rows(
        root=root,
        captured_listings=captured_listings,
        loaded_trade_sources=loaded_trade_sources,
    )
    if (
        rows != source_evidence_artifact.rows
        or captured_inventory
        != source_evidence_artifact.captured_listing_inventory_sha256
        or source_inventory != source_evidence_artifact.source_inventory_sha256
    ):
        raise MassiveProfitabilityOriginV1Error(
            "source evidence artifact was not rederived from captured listings"
        )
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    dates = tuple(row.session_date for row in sessions)
    if (
        first_candidate_decision_session_date not in dates
        or last_candidate_decision_session_date not in dates
        or first_candidate_decision_session_date > last_candidate_decision_session_date
    ):
        raise MassiveProfitabilityOriginV1Error(
            "origin candidate interval is absent or inverted"
        )
    candidates = tuple(
        value
        for value in dates
        if first_candidate_decision_session_date
        <= value
        <= last_candidate_decision_session_date
    )
    positions = {value: index for index, value in enumerate(dates)}
    required_source_dates = {
        dates[positions[value] - 2] for value in candidates if positions[value] >= 2
    }
    if {
        row.source_session_date for row in source_evidence_artifact.rows
    } - required_source_dates:
        raise MassiveProfitabilityOriginV1Error(
            "origin plan contains irrelevant acquired source evidence"
        )
    schedule = build_massive_profitability_monthly_membership_schedule_v1(
        session_authority=session_authority,
        identity_authority=identity_authority,
        first_candidate_decision_session_date=first_candidate_decision_session_date,
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    )
    origins, skips = _derive_origin_rows(
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_artifact=source_evidence_artifact,
        membership_schedule=schedule,
        candidate_dates=candidates,
    )
    artifact_audit = _artifact_audit_receipt(source_evidence_artifact)
    semantic_body: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA,
        "first_candidate_decision_session_date": (
            first_candidate_decision_session_date
        ),
        "last_candidate_decision_session_date": last_candidate_decision_session_date,
        "candidate_decision_session_dates": candidates,
        "origins": origins,
        "skipped_decisions": skips,
        "source_evidence_artifact_semantic_receipt_sha256": (
            source_evidence_artifact.semantic_receipt_sha256
        ),
        "membership_schedule_semantic_receipt_sha256": (
            schedule.schedule_semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "origin_spec_receipt_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(
        semantic_body
        | {
            "origins": tuple(
                row.semantic_unsigned() | {"receipt_sha256": row.receipt_sha256}
                for row in origins
            ),
            "skipped_decisions": tuple(asdict(row) for row in skips),
        }
    )
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "identity_authority_audit_receipt_sha256": (
                identity_authority.receipt_sha256
            ),
            "source_evidence_artifact_audit_receipt_sha256": artifact_audit,
        }
    )
    result = MassiveProfitabilityDecisionOriginPlanV1(
        **semantic_body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_receipt,
        identity_authority_audit_receipt_sha256=identity_authority.receipt_sha256,
        source_evidence_artifact_audit_receipt_sha256=artifact_audit,
        audit_receipt_sha256=audit_receipt,
    )
    result.validate()
    return result


def build_massive_profitability_decision_origin_plan_v1(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_evidence_artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    loaded_trade_sources: Sequence[LoadedMassiveSourceObject],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveProfitabilityDecisionOriginPlanV1:
    """Build and independently rederive the acquired-source monthly P0 plan."""

    result = _build_massive_profitability_decision_origin_plan_v1(
        root=root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidence_artifact=source_evidence_artifact,
        captured_listings=captured_listings,
        loaded_trade_sources=loaded_trade_sources,
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    )
    validate_massive_profitability_decision_origin_plan_v1(
        root=root,
        plan=result,
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidence_artifact=source_evidence_artifact,
        captured_listings=captured_listings,
        loaded_trade_sources=loaded_trade_sources,
    )
    return result


def validate_massive_profitability_decision_origin_plan_v1(
    *,
    root: str | Path,
    plan: MassiveProfitabilityDecisionOriginPlanV1,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_evidence_artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    loaded_trade_sources: Sequence[LoadedMassiveSourceObject],
) -> None:
    """Independently reconstruct the complete acquired-source monthly plan."""

    plan.validate()
    expected = _build_massive_profitability_decision_origin_plan_v1(
        root=root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidence_artifact=source_evidence_artifact,
        captured_listings=captured_listings,
        loaded_trade_sources=loaded_trade_sources,
        first_candidate_decision_session_date=(
            plan.first_candidate_decision_session_date
        ),
        last_candidate_decision_session_date=plan.last_candidate_decision_session_date,
    )
    if plan != expected:
        raise MassiveProfitabilityOriginV1Error(
            "origin plan was not independently rederived"
        )


MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))


__all__ = [
    "MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_DATASET",
    "MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_OBJECT_PREFIX",
    "MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_ACQUIRED_SOURCE_EVIDENCE_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256",
    "MassiveProfitabilityAcquiredSourceEvidenceArtifactV1",
    "MassiveProfitabilityAcquiredSourceEvidenceV1",
    "MassiveProfitabilityDecisionOriginPlanV1",
    "MassiveProfitabilityDecisionOriginV1",
    "MassiveProfitabilityMonthlyMembershipScheduleRowV1",
    "MassiveProfitabilityMonthlyMembershipScheduleV1",
    "MassiveProfitabilityOriginV1Error",
    "MassiveProfitabilitySkippedDecisionV1",
    "build_massive_profitability_decision_origin_plan_v1",
    "build_massive_profitability_monthly_membership_schedule_v1",
    "materialize_massive_profitability_acquired_source_evidence_v1",
    "parse_massive_profitability_acquired_source_evidence_v1",
    "validate_massive_profitability_decision_origin_plan_v1",
]
