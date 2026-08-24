"""Whole-file-qualified, decision-centric origin authority for finalized V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_SOURCE_ROLE,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
    MASSIVE_FLAT_FILE_ENDPOINT,
    MassiveCapturedFlatFileListingV0,
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
    build_massive_vendor_object_metadata_from_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V1,
    MassiveFinalizedOriginPolicyV1,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
    MassiveFinalizedFeatureDomainSpecV0,
)
from rl_quant.data_sources.massive.finalized_readiness import (
    MassiveFinalizedReadinessCapabilityV0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.protocol.canonical_artifact import semantic_sha256


EASTERN = ZoneInfo("America/New_York")
MASSIVE_QUALIFIED_FINALIZED_DAILY_SOURCE_V0_SCHEMA = (
    "rl-quant.massive-acquired-qualified-finalized-daily-source-v1"
)
MASSIVE_QUALIFIED_FINALIZED_DECISION_ORIGIN_V0_SCHEMA = (
    "rl-quant.massive-acquired-qualified-finalized-decision-origin-v1"
)
MASSIVE_QUALIFIED_FINALIZED_SKIPPED_DECISION_V0_SCHEMA = (
    "rl-quant.massive-acquired-qualified-finalized-skipped-decision-v1"
)
MASSIVE_QUALIFIED_FINALIZED_ORIGIN_PLAN_V0_SCHEMA = (
    "rl-quant.massive-acquired-qualified-finalized-origin-plan-v1"
)


class MassiveQualifiedFinalizedOriginError(ValueError):
    """A whole-file-qualified V0 origin cannot be established."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveQualifiedFinalizedOriginError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _nonnegative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveQualifiedFinalizedOriginError(f"{name} must be nonnegative")
    return value


def _local_ms(session_date: str, local_time: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(session_date), local_time, tzinfo=EASTERN).timestamp()
        * 1_000
    )


@dataclass(frozen=True, slots=True)
class MassiveQualifiedFinalizedDailySourceV0:
    source_role: str
    source_session_date: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    listing_acquisition_receipt_sha256: str
    listing_acquisition_source_receipt_sha256: str
    listing_endpoint: str
    listing_bucket: str
    listing_prefix: str
    provider_request_inventory_sha256: str
    listing_object_inventory_sha256: str
    committed_listing_receipt_sha256: str
    listing_entry_receipt_sha256: str
    vendor_metadata_receipt_sha256: str
    vendor_last_modified_at_ms: int
    whole_file_scan_receipt_sha256: str
    partition_manifest_receipt_sha256: str
    feature_domain_spec_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    readiness_capability_receipt_sha256: str
    origin_policy_receipt_sha256: str
    session_authority_receipt_sha256: str
    exchange: str
    regular_open_at_ms: int
    regular_close_at_ms: int
    publication_safety_margin_ms: int
    readiness_maximum_runtime_ms: int
    measured_feature_ready_upper_bound_at_ms: int
    source_row_count: int
    partitioned_row_count: int
    receipt_sha256: str
    schema: str = MASSIVE_QUALIFIED_FINALIZED_DAILY_SOURCE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_QUALIFIED_FINALIZED_DAILY_SOURCE_V0_SCHEMA:
            raise MassiveQualifiedFinalizedOriginError("qualified daily source schema drifted")
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveQualifiedFinalizedOriginError("qualified source role drifted")
        if not self.source_session_date or not self.exchange:
            raise MassiveQualifiedFinalizedOriginError("qualified daily source identity is absent")
        for name in (
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "listing_acquisition_receipt_sha256",
            "listing_acquisition_source_receipt_sha256",
            "provider_request_inventory_sha256",
            "listing_object_inventory_sha256",
            "committed_listing_receipt_sha256",
            "listing_entry_receipt_sha256",
            "vendor_metadata_receipt_sha256",
            "whole_file_scan_receipt_sha256",
            "partition_manifest_receipt_sha256",
            "feature_domain_spec_receipt_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "readiness_capability_receipt_sha256",
            "origin_policy_receipt_sha256",
            "session_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "vendor_last_modified_at_ms",
            "regular_open_at_ms",
            "regular_close_at_ms",
            "publication_safety_margin_ms",
            "readiness_maximum_runtime_ms",
            "measured_feature_ready_upper_bound_at_ms",
            "source_row_count",
            "partitioned_row_count",
        ):
            _nonnegative(name, getattr(self, name))
        if self.regular_close_at_ms <= self.regular_open_at_ms:
            raise MassiveQualifiedFinalizedOriginError("source session bounds are invalid")
        if self.source_row_count <= 0 or self.partitioned_row_count != self.source_row_count:
            raise MassiveQualifiedFinalizedOriginError("qualified source rows do not reconcile")
        if (
            self.listing_endpoint != MASSIVE_FLAT_FILE_ENDPOINT
            or self.listing_bucket != MASSIVE_FLAT_FILE_BUCKET
            or not self.listing_prefix
        ):
            raise MassiveQualifiedFinalizedOriginError(
                "qualified listing acquisition identity drifted"
            )
        if (
            self.publication_safety_margin_ms
            != MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms
            or self.readiness_maximum_runtime_ms <= 0
            or self.measured_feature_ready_upper_bound_at_ms
            != self.vendor_last_modified_at_ms
            + self.publication_safety_margin_ms
            + self.readiness_maximum_runtime_ms
        ):
            raise MassiveQualifiedFinalizedOriginError(
                "qualified measured readiness differs"
            )
        if self.origin_policy_receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256:
            raise MassiveQualifiedFinalizedOriginError("qualified source policy drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveQualifiedFinalizedOriginError("qualified source receipt differs")

    @property
    def assumed_feature_ready_upper_bound_at_ms(self) -> int:
        """Compatibility view; the bound is now measured, not assumed."""

        return self.measured_feature_ready_upper_bound_at_ms

    @property
    def processing_capability_receipt_sha256(self) -> str:
        """Compatibility view over the full readiness capability receipt."""

        return self.readiness_capability_receipt_sha256


def build_massive_qualified_finalized_daily_source_v0(
    *,
    listing_root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    captured_listing: MassiveCapturedFlatFileListingV0,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    partition_manifest: MassiveDailyTradePartitionManifestV0,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    readiness_capability: MassiveFinalizedReadinessCapabilityV0,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    origin_policy: MassiveFinalizedOriginPolicyV1 = MASSIVE_FINALIZED_ORIGIN_POLICY_V1,
) -> MassiveQualifiedFinalizedDailySourceV0:
    if not isinstance(captured_listing, MassiveCapturedFlatFileListingV0):
        raise MassiveQualifiedFinalizedOriginError(
            "qualified source requires an authenticated captured listing"
        )
    if not isinstance(
        readiness_capability, MassiveFinalizedReadinessCapabilityV0
    ):
        raise MassiveQualifiedFinalizedOriginError(
            "qualified source requires full feature-to-order readiness evidence"
        )
    loaded_source.validate()
    validate_massive_captured_flat_file_listing_v0(
        root=listing_root, captured_listing=captured_listing
    )
    committed_listing = captured_listing.committed_listing
    listing_entry = committed_listing.resolve(
        source_object_key=loaded_source.receipt.source_object_key
    )
    metadata = build_massive_vendor_object_metadata_from_listing_v0(
        committed_listing=committed_listing,
        listing_entry=listing_entry,
        loaded_source=loaded_source,
    )
    scan_evidence.validate()
    partition_manifest.validate()
    feature_domain_spec.validate()
    readiness_capability.validate()
    session_authority.validate()
    source_session.validate()
    origin_policy.validate()
    links = {
        "source object": loaded_source.receipt.receipt_sha256,
        "scan source object": scan_evidence.source_object_receipt_sha256,
        "metadata source object": metadata.source_object_receipt_sha256,
    }
    if len(set(links.values())) != 1:
        raise MassiveQualifiedFinalizedOriginError("qualified source object links differ")
    if (
        listing_entry.receipt_sha256 != metadata.listing_entry_receipt_sha256
        or committed_listing.receipt_sha256 != metadata.committed_listing_receipt_sha256
        or partition_manifest.source_file_scan_receipt_sha256 != scan_evidence.receipt_sha256
        or partition_manifest.feature_domain_spec_receipt_sha256 != feature_domain_spec.receipt_sha256
        or partition_manifest.identity_authority_receipt_sha256 == "0" * 64
        or partition_manifest.condition_authority_receipt_sha256
        != feature_domain_spec.condition_authority_receipt_sha256
        or partition_manifest.correction_authority_receipt_sha256
        != feature_domain_spec.correction_authority_receipt_sha256
        or scan_evidence.session_authority_receipt_sha256 != session_authority.receipt_sha256
        or scan_evidence.source_session_date != source_session.session_date
    ):
        raise MassiveQualifiedFinalizedOriginError("qualified source authorities differ")
    if session_authority.resolve(
        exchange=source_session.exchange,
        session_date=source_session.session_date,
    ) != source_session:
        raise MassiveQualifiedFinalizedOriginError("source session was not authority-resolved")
    if not readiness_capability.covers(scan_evidence):
        raise MassiveQualifiedFinalizedOriginError(
            "full readiness capability does not cover this source"
        )
    acquisition = captured_listing.acquisition_evidence
    measured_ready_at = (
        metadata.vendor_last_modified_at_ms
        + MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms
        + readiness_capability.maximum_runtime_ms
    )
    body: dict[str, object] = {
        "schema": MASSIVE_QUALIFIED_FINALIZED_DAILY_SOURCE_V0_SCHEMA,
        "source_role": MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        "source_session_date": source_session.session_date,
        "source_object_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
        "listing_acquisition_receipt_sha256": acquisition.receipt_sha256,
        "listing_acquisition_source_receipt_sha256": (
            captured_listing.loaded_acquisition.receipt.receipt_sha256
        ),
        "listing_endpoint": acquisition.endpoint_url,
        "listing_bucket": acquisition.bucket,
        "listing_prefix": acquisition.prefix,
        "provider_request_inventory_sha256": semantic_sha256(
            acquisition.provider_request_ids
        ),
        "listing_object_inventory_sha256": acquisition.object_inventory_sha256,
        "committed_listing_receipt_sha256": committed_listing.receipt_sha256,
        "listing_entry_receipt_sha256": listing_entry.receipt_sha256,
        "vendor_metadata_receipt_sha256": metadata.receipt_sha256,
        "vendor_last_modified_at_ms": metadata.vendor_last_modified_at_ms,
        "whole_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "partition_manifest_receipt_sha256": partition_manifest.receipt_sha256,
        "feature_domain_spec_receipt_sha256": feature_domain_spec.receipt_sha256,
        "identity_authority_receipt_sha256": partition_manifest.identity_authority_receipt_sha256,
        "condition_authority_receipt_sha256": partition_manifest.condition_authority_receipt_sha256,
        "correction_authority_receipt_sha256": partition_manifest.correction_authority_receipt_sha256,
        "readiness_capability_receipt_sha256": readiness_capability.receipt_sha256,
        "origin_policy_receipt_sha256": origin_policy.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "exchange": source_session.exchange,
        "regular_open_at_ms": source_session.regular_open_ns // 1_000_000,
        "regular_close_at_ms": source_session.regular_close_ns // 1_000_000,
        "publication_safety_margin_ms": (
            MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms
        ),
        "readiness_maximum_runtime_ms": readiness_capability.maximum_runtime_ms,
        "measured_feature_ready_upper_bound_at_ms": measured_ready_at,
        "source_row_count": scan_evidence.source_row_count,
        "partitioned_row_count": partition_manifest.partitioned_row_count,
    }
    result = MassiveQualifiedFinalizedDailySourceV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveQualifiedFinalizedDecisionOriginV0:
    decision_session_date: str
    source_session_date: str
    source_staleness_sessions: int
    source_staleness_context_value: int
    origin_policy_receipt_sha256: str
    session_authority_receipt_sha256: str
    exchange: str
    regular_open_at_ms: int
    regular_close_at_ms: int
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    source_receipt_sha256: str
    partition_manifest_receipt_sha256: str
    feature_domain_spec_receipt_sha256: str
    processing_capability_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_QUALIFIED_FINALIZED_DECISION_ORIGIN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_QUALIFIED_FINALIZED_DECISION_ORIGIN_V0_SCHEMA:
            raise MassiveQualifiedFinalizedOriginError("qualified origin schema drifted")
        if not self.decision_session_date or not self.source_session_date or not self.exchange:
            raise MassiveQualifiedFinalizedOriginError("qualified origin identity is absent")
        if not 1 <= self.source_staleness_sessions <= MASSIVE_FINALIZED_ORIGIN_POLICY_V1.maximum_source_staleness_sessions:
            raise MassiveQualifiedFinalizedOriginError("qualified source staleness is invalid")
        if self.source_staleness_context_value != self.source_staleness_sessions:
            raise MassiveQualifiedFinalizedOriginError("staleness context was not preserved")
        if not (
            self.regular_open_at_ms
            <= self.decision_at_ms
            < self.fill_start_at_ms
            < self.fill_end_at_ms
            <= self.regular_close_at_ms
        ):
            raise MassiveQualifiedFinalizedOriginError("decision/fill session bounds are invalid")
        for name in (
            "origin_policy_receipt_sha256",
            "session_authority_receipt_sha256",
            "source_receipt_sha256",
            "partition_manifest_receipt_sha256",
            "feature_domain_spec_receipt_sha256",
            "processing_capability_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.origin_policy_receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256:
            raise MassiveQualifiedFinalizedOriginError("qualified origin policy drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveQualifiedFinalizedOriginError("qualified origin receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveQualifiedFinalizedSkippedDecisionV0:
    decision_session_date: str
    reason: str
    session_authority_receipt_sha256: str
    origin_policy_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_QUALIFIED_FINALIZED_SKIPPED_DECISION_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_QUALIFIED_FINALIZED_SKIPPED_DECISION_V0_SCHEMA:
            raise MassiveQualifiedFinalizedOriginError("qualified skip schema drifted")
        if not self.decision_session_date or self.reason not in {
            "session-does-not-support-decision-and-fill",
            "no-ready-source-within-staleness-bound",
        }:
            raise MassiveQualifiedFinalizedOriginError("qualified skip reason is invalid")
        _digest("session authority", self.session_authority_receipt_sha256)
        _digest("origin policy", self.origin_policy_receipt_sha256)
        _digest("skip receipt", self.receipt_sha256)
        if self.origin_policy_receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256:
            raise MassiveQualifiedFinalizedOriginError("qualified skip policy drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveQualifiedFinalizedOriginError("qualified skip receipt differs")


def _build_origin(
    *,
    source: MassiveQualifiedFinalizedDailySourceV0,
    decision_session: MassiveExchangeSession,
    session_authority: MassiveSessionAuthority,
    staleness: int,
) -> MassiveQualifiedFinalizedDecisionOriginV0:
    decision = _local_ms(decision_session.session_date, time(12, 30))
    fill_start = _local_ms(decision_session.session_date, time(15, 50))
    fill_end = _local_ms(decision_session.session_date, time(16, 0))
    body: dict[str, object] = {
        "schema": MASSIVE_QUALIFIED_FINALIZED_DECISION_ORIGIN_V0_SCHEMA,
        "decision_session_date": decision_session.session_date,
        "source_session_date": source.source_session_date,
        "source_staleness_sessions": staleness,
        "source_staleness_context_value": staleness,
        "origin_policy_receipt_sha256": MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "exchange": decision_session.exchange,
        "regular_open_at_ms": decision_session.regular_open_ns // 1_000_000,
        "regular_close_at_ms": decision_session.regular_close_ns // 1_000_000,
        "decision_at_ms": decision,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
        "source_receipt_sha256": source.receipt_sha256,
        "partition_manifest_receipt_sha256": source.partition_manifest_receipt_sha256,
        "feature_domain_spec_receipt_sha256": source.feature_domain_spec_receipt_sha256,
        "processing_capability_receipt_sha256": source.processing_capability_receipt_sha256,
    }
    result = MassiveQualifiedFinalizedDecisionOriginV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _build_skip(
    *,
    decision_session_date: str,
    reason: str,
    session_authority: MassiveSessionAuthority,
) -> MassiveQualifiedFinalizedSkippedDecisionV0:
    body: dict[str, object] = {
        "schema": MASSIVE_QUALIFIED_FINALIZED_SKIPPED_DECISION_V0_SCHEMA,
        "decision_session_date": decision_session_date,
        "reason": reason,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "origin_policy_receipt_sha256": MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256,
    }
    result = MassiveQualifiedFinalizedSkippedDecisionV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _derive_decision_rows(
    *,
    session_authority: MassiveSessionAuthority,
    exchange: str,
    sources: tuple[MassiveQualifiedFinalizedDailySourceV0, ...],
    candidates: tuple[MassiveExchangeSession, ...],
    origin_policy: MassiveFinalizedOriginPolicyV1,
) -> tuple[
    tuple[MassiveQualifiedFinalizedDecisionOriginV0, ...],
    tuple[MassiveQualifiedFinalizedSkippedDecisionV0, ...],
]:
    calendar = tuple(row for row in session_authority.sessions if row.exchange == exchange)
    calendar_index = {row.session_date: index for index, row in enumerate(calendar)}
    origins: list[MassiveQualifiedFinalizedDecisionOriginV0] = []
    skips: list[MassiveQualifiedFinalizedSkippedDecisionV0] = []
    for decision_session in candidates:
        decision_ms = _local_ms(decision_session.session_date, time(12, 30))
        fill_start_ms = _local_ms(decision_session.session_date, time(15, 50))
        fill_end_ms = _local_ms(decision_session.session_date, time(16, 0))
        open_ms = decision_session.regular_open_ns // 1_000_000
        close_ms = decision_session.regular_close_ns // 1_000_000
        if not (open_ms <= decision_ms < fill_start_ms < fill_end_ms <= close_ms):
            skips.append(
                _build_skip(
                    decision_session_date=decision_session.session_date,
                    reason="session-does-not-support-decision-and-fill",
                    session_authority=session_authority,
                )
            )
            continue
        ready: list[tuple[int, MassiveQualifiedFinalizedDailySourceV0]] = []
        decision_index = calendar_index[decision_session.session_date]
        for source in sources:
            source_index = calendar_index.get(source.source_session_date)
            if source_index is None or source_index >= decision_index:
                continue
            staleness = decision_index - source_index
            if (
                1 <= staleness <= origin_policy.maximum_source_staleness_sessions
                and source.assumed_feature_ready_upper_bound_at_ms <= decision_ms
            ):
                ready.append((staleness, source))
        if not ready:
            skips.append(
                _build_skip(
                    decision_session_date=decision_session.session_date,
                    reason="no-ready-source-within-staleness-bound",
                    session_authority=session_authority,
                )
            )
            continue
        staleness, source = min(ready, key=lambda item: item[0])
        origins.append(
            _build_origin(
                source=source,
                decision_session=decision_session,
                session_authority=session_authority,
                staleness=staleness,
            )
        )
    return tuple(origins), tuple(skips)


@dataclass(frozen=True, slots=True)
class MassiveQualifiedFinalizedOriginPlanV0:
    exchange: str
    first_decision_session_date: str
    last_decision_session_date: str
    candidate_decision_session_dates: tuple[str, ...]
    session_authority_receipt_sha256: str
    origin_policy: MassiveFinalizedOriginPolicyV1
    origin_policy_receipt_sha256: str
    feature_domain_spec_receipt_sha256: str
    processing_capability_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    calendar_sessions: tuple[MassiveExchangeSession, ...]
    daily_sources: tuple[MassiveQualifiedFinalizedDailySourceV0, ...]
    origins: tuple[MassiveQualifiedFinalizedDecisionOriginV0, ...]
    skipped_decisions: tuple[MassiveQualifiedFinalizedSkippedDecisionV0, ...]
    panel_materialization_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_QUALIFIED_FINALIZED_ORIGIN_PLAN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_QUALIFIED_FINALIZED_ORIGIN_PLAN_V0_SCHEMA:
            raise MassiveQualifiedFinalizedOriginError("qualified plan schema drifted")
        self.origin_policy.validate()
        if self.origin_policy_receipt_sha256 != self.origin_policy.receipt_sha256:
            raise MassiveQualifiedFinalizedOriginError("qualified plan policy differs")
        for calendar_row in self.calendar_sessions:
            calendar_row.validate()
        for source_row in self.daily_sources:
            source_row.validate()
        for origin_row in self.origins:
            origin_row.validate()
        for skipped_row in self.skipped_decisions:
            skipped_row.validate()
        if not self.calendar_sessions:
            raise MassiveQualifiedFinalizedOriginError("qualified calendar is empty")
        calendar_source_receipts = {
            row.calendar_source_receipt_sha256 for row in self.calendar_sessions
        }
        if len(calendar_source_receipts) != 1:
            raise MassiveQualifiedFinalizedOriginError("calendar source receipts differ")
        reconstructed_session_authority = MassiveSessionAuthority(
            sessions=self.calendar_sessions,
            calendar_source_receipt_sha256=next(iter(calendar_source_receipts)),
            receipt_sha256=self.session_authority_receipt_sha256,
        )
        reconstructed_session_authority.validate()
        candidate_keys = self.candidate_decision_session_dates
        expected_candidates = tuple(
            row.session_date
            for row in self.calendar_sessions
            if row.exchange == self.exchange
            and self.first_decision_session_date <= row.session_date <= self.last_decision_session_date
        )
        if candidate_keys != expected_candidates or candidate_keys != tuple(sorted(set(candidate_keys))):
            raise MassiveQualifiedFinalizedOriginError("qualified plan omitted calendar sessions")
        observed = tuple(sorted(
            [row.decision_session_date for row in self.origins]
            + [row.decision_session_date for row in self.skipped_decisions]
        ))
        if observed != candidate_keys:
            raise MassiveQualifiedFinalizedOriginError("qualified decisions do not partition calendar")
        if any(row.feature_domain_spec_receipt_sha256 != self.feature_domain_spec_receipt_sha256 for row in self.daily_sources):
            raise MassiveQualifiedFinalizedOriginError("daily feature-domain specs differ")
        if any(row.processing_capability_receipt_sha256 != self.processing_capability_receipt_sha256 for row in self.daily_sources):
            raise MassiveQualifiedFinalizedOriginError("daily processing capabilities differ")
        for field_name in (
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
        ):
            _digest(field_name, getattr(self, field_name))
            if any(
                getattr(row, field_name) != getattr(self, field_name)
                for row in self.daily_sources
            ):
                raise MassiveQualifiedFinalizedOriginError(
                    f"daily {field_name} values differ"
                )
        expected_origins, expected_skips = _derive_decision_rows(
            session_authority=reconstructed_session_authority,
            exchange=self.exchange,
            sources=self.daily_sources,
            candidates=tuple(
                row
                for row in self.calendar_sessions
                if row.exchange == self.exchange
                and self.first_decision_session_date
                <= row.session_date
                <= self.last_decision_session_date
            ),
            origin_policy=self.origin_policy,
        )
        if self.origins != expected_origins or self.skipped_decisions != expected_skips:
            raise MassiveQualifiedFinalizedOriginError(
                "qualified decision rows were not independently derived"
            )
        if self.panel_materialization_authorized is not False:
            raise MassiveQualifiedFinalizedOriginError("origin plan cannot authorize the PIT panel")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveQualifiedFinalizedOriginError("qualified plan receipt differs")


def build_massive_qualified_finalized_origin_plan_v0(
    *,
    session_authority: MassiveSessionAuthority,
    exchange: str,
    daily_sources: Sequence[MassiveQualifiedFinalizedDailySourceV0],
    first_decision_session_date: str,
    last_decision_session_date: str,
    origin_policy: MassiveFinalizedOriginPolicyV1 = MASSIVE_FINALIZED_ORIGIN_POLICY_V1,
) -> MassiveQualifiedFinalizedOriginPlanV0:
    session_authority.validate()
    origin_policy.validate()
    sources = tuple(sorted(daily_sources, key=lambda row: row.source_session_date))
    if not sources or tuple(row.source_session_date for row in sources) != tuple(sorted(set(row.source_session_date for row in sources))):
        raise MassiveQualifiedFinalizedOriginError("qualified daily sources are empty or duplicate")
    for row in sources:
        row.validate()
        if row.exchange != exchange or row.origin_policy_receipt_sha256 != origin_policy.receipt_sha256:
            raise MassiveQualifiedFinalizedOriginError("qualified source policy/exchange differs")
    feature_specs = {row.feature_domain_spec_receipt_sha256 for row in sources}
    capabilities = {row.processing_capability_receipt_sha256 for row in sources}
    identity_authorities = {row.identity_authority_receipt_sha256 for row in sources}
    condition_authorities = {row.condition_authority_receipt_sha256 for row in sources}
    correction_authorities = {row.correction_authority_receipt_sha256 for row in sources}
    if len(feature_specs) != 1:
        raise MassiveQualifiedFinalizedOriginError("daily feature-domain specs differ")
    if len(capabilities) != 1:
        raise MassiveQualifiedFinalizedOriginError("daily processing capabilities differ")
    if (
        len(identity_authorities) != 1
        or len(condition_authorities) != 1
        or len(correction_authorities) != 1
    ):
        raise MassiveQualifiedFinalizedOriginError(
            "daily identity/condition/correction authorities differ"
        )
    calendar = session_authority.sessions
    candidates = tuple(
        row for row in calendar
        if row.exchange == exchange
        and first_decision_session_date <= row.session_date <= last_decision_session_date
    )
    if not candidates:
        raise MassiveQualifiedFinalizedOriginError("qualified decision interval is empty")
    origins, skips = _derive_decision_rows(
        session_authority=session_authority,
        exchange=exchange,
        sources=sources,
        candidates=candidates,
        origin_policy=origin_policy,
    )
    body: dict[str, object] = {
        "schema": MASSIVE_QUALIFIED_FINALIZED_ORIGIN_PLAN_V0_SCHEMA,
        "exchange": exchange,
        "first_decision_session_date": first_decision_session_date,
        "last_decision_session_date": last_decision_session_date,
        "candidate_decision_session_dates": tuple(row.session_date for row in candidates),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "origin_policy": origin_policy,
        "origin_policy_receipt_sha256": origin_policy.receipt_sha256,
        "feature_domain_spec_receipt_sha256": next(iter(feature_specs)),
        "processing_capability_receipt_sha256": next(iter(capabilities)),
        "identity_authority_receipt_sha256": next(iter(identity_authorities)),
        "condition_authority_receipt_sha256": next(iter(condition_authorities)),
        "correction_authority_receipt_sha256": next(iter(correction_authorities)),
        "calendar_sessions": calendar,
        "daily_sources": sources,
        "origins": origins,
        "skipped_decisions": skips,
        "panel_materialization_authorized": False,
    }
    provisional = MassiveQualifiedFinalizedOriginPlanV0(
        **body, receipt_sha256="0" * 64  # type: ignore[arg-type]
    )
    result = MassiveQualifiedFinalizedOriginPlanV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_QUALIFIED_FINALIZED_DAILY_SOURCE_V0_SCHEMA",
    "MASSIVE_QUALIFIED_FINALIZED_DECISION_ORIGIN_V0_SCHEMA",
    "MASSIVE_QUALIFIED_FINALIZED_ORIGIN_PLAN_V0_SCHEMA",
    "MASSIVE_QUALIFIED_FINALIZED_SKIPPED_DECISION_V0_SCHEMA",
    "MassiveQualifiedFinalizedDailySourceV0",
    "MassiveQualifiedFinalizedDecisionOriginV0",
    "MassiveQualifiedFinalizedOriginError",
    "MassiveQualifiedFinalizedOriginPlanV0",
    "MassiveQualifiedFinalizedSkippedDecisionV0",
    "build_massive_qualified_finalized_daily_source_v0",
    "build_massive_qualified_finalized_origin_plan_v0",
]
