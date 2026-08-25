"""Feature-layer P0 origins, security support, and experiment coverage."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from rl_quant.alpha.pit_universe import (
    HistoricalMembershipRecord,
    PITSecurityUniverseAuthority,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MassiveCommittedFlatFileListingV0,
    MassiveVendorListingEntryV0,
    canonical_massive_trade_object_key,
    parse_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0,
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

if TYPE_CHECKING:
    from rl_quant.features.massive_economic_coverage_v8 import (
        MassiveEconomicCoverageScopeV8,
    )

MASSIVE_PROFITABILITY_SOURCE_EVIDENCE_P0_SCHEMA = (
    "rl-quant.massive-profitability-source-evidence-p0"
)
MASSIVE_PROFITABILITY_DECISION_ORIGIN_P0_SCHEMA = (
    "rl-quant.massive-profitability-decision-origin-p0"
)
MASSIVE_PROFITABILITY_SKIPPED_DECISION_P0_SCHEMA = (
    "rl-quant.massive-profitability-skipped-decision-p0"
)
MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SCHEMA = (
    "rl-quant.massive-profitability-origin-plan-p0"
)
MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SCHEMA = (
    "rl-quant.massive-profitability-security-support-v1"
)
MASSIVE_PROFITABILITY_PHASE_INTERVAL_V1_SCHEMA = (
    "rl-quant.massive-profitability-phase-interval-v1"
)
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SCHEMA = (
    "rl-quant.massive-profitability-experiment-coverage-v1"
)

MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS = max(
    row.end_offset_sessions for row in MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.horizons
)
MASSIVE_PROFITABILITY_MAXIMUM_TARGET_HORIZON_SESSIONS = max(
    row.end_offset_sessions for row in MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.horizons
)
MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS = 18 * 60 * 60 * 1_000
MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS = (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.historical_lockbox_sessions
)
MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS = (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.outer_fold_count
    * MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.outer_fold_sessions
)
MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS = (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.minimum_initial_training_sessions
    + MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.inner_purge_sessions
    + MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.inner_validation_sessions
    + MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.target_overlap_purge_sessions
    + MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS
    + MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS
)

MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "exchange": "XNYS",
        "candidate_inventory": "every-session-in-inclusive-requested-range",
        "source_session": "exactly-two-exchange-sessions-before-decision",
        "vendor_lead": "at-least-18-hours-before-decision",
        "decision": "12:30:00-America/New_York",
        "fill": "[15:50:00,16:00:00)-America/New_York",
        "feature_cutoff": "source-session-close",
        "membership": "latest-complete-effective-group-known-at-decision",
        "partition": "exactly-one-origin-or-typed-skip-per-candidate",
    }
)
MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "decision_support": "union-of-every-origin-PIT-member-inventory",
        "accounting_support": (
            "undirected-successor-and-corporate-action-chain-closure"
        ),
        "outside_support": "audited-but-not-panel-invalidating",
    }
)
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "feature_start": "earliest-source-minus-63-exchange-sessions",
        "target_end": "latest-decision-plus-63-exchange-sessions",
        "development": "all-eligible-nonlockbox-origins",
        "confirmation": "last-four-126-origin-outer-tests-before-lockbox",
        "confirmation_history": "same-dates-as-subset-of-development",
        "lockbox": "final-252-eligible-origins",
        "minimum_eligible_origins": MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS,
        "interval_selection": "derived-only-from-origin-plan-and-session-authority",
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
)

MASSIVE_PROFITABILITY_P0_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_P0_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_P0_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_P0_LOCKBOX_ACCESS_AUTHORIZED = False

_EASTERN = ZoneInfo("America/New_York")
_SKIP_REASONS = {
    "decision-session-cannot-support-frozen-clock",
    "insufficient-prior-session-history",
    "missing-source-object-evidence",
    "vendor-object-predates-source-close",
    "vendor-lead-below-18-hours",
    "no-complete-pit-membership-known-at-decision",
}


class MassiveProfitabilityOriginP0Error(ValueError):
    """P0 chronology, support, or experiment coverage is not authority-derived."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveProfitabilityOriginP0Error(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOriginP0Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveProfitabilityOriginP0Error(f"{name} must be nonnegative")
    return value


def _canonical_date(name: str, value: object) -> str:
    raw = _text(name, value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveProfitabilityOriginP0Error(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != raw:
        raise MassiveProfitabilityOriginP0Error(f"{name} is not canonical")
    return raw


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
        raise MassiveProfitabilityOriginP0Error(
            "session timestamp is not millisecond aligned"
        )
    return value // 1_000_000


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySourceEvidenceP0:
    source_session_date: str
    source_object_key: str
    vendor_last_modified_at_ms: int
    listing_observed_at_ms: int
    research_downloaded_at_ms: int
    research_verified_at_ms: int
    content_length: int
    etag: str
    listing_entry_receipt_sha256: str
    committed_listing_receipt_sha256: str
    loaded_listing_receipt_sha256: str
    loaded_source_receipt_sha256: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_SOURCE_EVIDENCE_P0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_SOURCE_EVIDENCE_P0_SCHEMA:
            raise MassiveProfitabilityOriginP0Error("source evidence schema differs")
        source_date = _canonical_date("source session", self.source_session_date)
        if self.source_object_key != canonical_massive_trade_object_key(source_date):
            raise MassiveProfitabilityOriginP0Error("source object key differs")
        modified = _nonnegative_int(
            "vendor LastModified", self.vendor_last_modified_at_ms
        )
        observed = _nonnegative_int("listing observation", self.listing_observed_at_ms)
        downloaded = _nonnegative_int(
            "research download", self.research_downloaded_at_ms
        )
        verified = _nonnegative_int(
            "research verification", self.research_verified_at_ms
        )
        if observed < modified or downloaded < modified or verified < downloaded:
            raise MassiveProfitabilityOriginP0Error(
                "source evidence chronology differs"
            )
        content_length = _nonnegative_int("source content length", self.content_length)
        if content_length <= 0:
            raise MassiveProfitabilityOriginP0Error("source object is empty")
        _text("source ETag", self.etag)
        for name in (
            "listing_entry_receipt_sha256",
            "committed_listing_receipt_sha256",
            "loaded_listing_receipt_sha256",
            "loaded_source_receipt_sha256",
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginP0Error("source evidence receipt differs")


def build_massive_profitability_source_evidence_p0(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    loaded_listing: LoadedMassiveSourceObject,
    committed_listing: MassiveCommittedFlatFileListingV0,
    listing_entry: MassiveVendorListingEntryV0,
) -> MassiveProfitabilitySourceEvidenceP0:
    """Join and physically recheck one research source with its vendor listing."""

    loaded_source.validate()
    loaded_listing.validate()
    committed_listing.validate()
    listing_entry.validate()
    read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    parsed_listing = parse_massive_flat_file_listing_v0(
        root=root, loaded_listing=loaded_listing
    )
    if parsed_listing != committed_listing:
        raise MassiveProfitabilityOriginP0Error(
            "committed listing differs from its immutable bytes"
        )
    expected_entry = committed_listing.resolve(
        source_object_key=listing_entry.source_object_key
    )
    if expected_entry != listing_entry:
        raise MassiveProfitabilityOriginP0Error(
            "listing entry differs from committed listing"
        )
    receipt = loaded_source.receipt
    if (
        receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID
        or receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256
        or receipt.source_object_key != listing_entry.source_object_key
        or receipt.content_length != listing_entry.content_length
        or receipt.etag != listing_entry.etag
        or committed_listing.loaded_listing_receipt_sha256
        != loaded_listing.receipt_sha256
    ):
        raise MassiveProfitabilityOriginP0Error(
            "research source and vendor listing identities differ"
        )
    body = {
        "schema": MASSIVE_PROFITABILITY_SOURCE_EVIDENCE_P0_SCHEMA,
        "source_session_date": listing_entry.coverage_session_date,
        "source_object_key": listing_entry.source_object_key,
        "vendor_last_modified_at_ms": listing_entry.vendor_last_modified_at_ms,
        "listing_observed_at_ms": listing_entry.listing_observed_at_ms,
        "research_downloaded_at_ms": receipt.downloaded_at_ms,
        "research_verified_at_ms": loaded_source.verified_at_ms,
        "content_length": receipt.content_length,
        "etag": listing_entry.etag,
        "listing_entry_receipt_sha256": listing_entry.receipt_sha256,
        "committed_listing_receipt_sha256": committed_listing.receipt_sha256,
        "loaded_listing_receipt_sha256": loaded_listing.receipt_sha256,
        "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
        "source_object_receipt_sha256": receipt.receipt_sha256,
        "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
    }
    result = MassiveProfitabilitySourceEvidenceP0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDecisionOriginP0:
    source_session_date: str
    decision_session_date: str
    exchange: str
    source_staleness_sessions: int
    decision_regular_open_at_ms: int
    decision_regular_close_at_ms: int
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    feature_maximum_session_date: str
    feature_cutoff_at_ms: int
    vendor_last_modified_at_ms: int
    vendor_lead_time_ms: int
    source_object_key: str
    source_evidence_receipt_sha256: str
    loaded_source_receipt_sha256: str
    source_object_receipt_sha256: str
    loaded_listing_receipt_sha256: str
    listing_entry_receipt_sha256: str
    committed_listing_receipt_sha256: str
    membership_effective_at_ms: int
    decision_member_security_ids: tuple[str, ...]
    decision_member_universe_ranks: tuple[int, ...]
    membership_group_receipt_sha256: str
    identity_authority_receipt_sha256: str
    universe_rule_receipt_sha256: str
    session_authority_receipt_sha256: str
    protocol_receipt_sha256: str
    origin_spec_receipt_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_DECISION_ORIGIN_P0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_DECISION_ORIGIN_P0_SCHEMA
            or self.exchange != "XNYS"
            or self.source_staleness_sessions
            != MASSIVE_FINALIZED_PROFITABILITY_P0.source_staleness_sessions
            or self.feature_maximum_session_date != self.source_session_date
            or self.source_object_key
            != canonical_massive_trade_object_key(self.source_session_date)
        ):
            raise MassiveProfitabilityOriginP0Error("decision origin identity differs")
        _canonical_date("origin source session", self.source_session_date)
        _canonical_date("origin decision session", self.decision_session_date)
        for name in (
            "decision_regular_open_at_ms",
            "decision_regular_close_at_ms",
            "decision_at_ms",
            "fill_start_at_ms",
            "fill_end_at_ms",
            "feature_cutoff_at_ms",
            "vendor_last_modified_at_ms",
            "vendor_lead_time_ms",
            "membership_effective_at_ms",
        ):
            _nonnegative_int(name, getattr(self, name))
        if not (
            self.decision_regular_open_at_ms
            <= self.decision_at_ms
            < self.fill_start_at_ms
            < self.fill_end_at_ms
            <= self.decision_regular_close_at_ms
        ):
            raise MassiveProfitabilityOriginP0Error("decision chronology differs")
        if (
            self.decision_at_ms != _local_ms(self.decision_session_date, time(12, 30))
            or self.fill_start_at_ms
            != _local_ms(self.decision_session_date, time(15, 50))
            or self.fill_end_at_ms != _local_ms(self.decision_session_date, time(16, 0))
            or self.vendor_lead_time_ms
            != self.decision_at_ms - self.vendor_last_modified_at_ms
            or self.vendor_lead_time_ms < MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS
        ):
            raise MassiveProfitabilityOriginP0Error(
                "decision clock or vendor lead differs"
            )
        if (
            not self.decision_member_security_ids
            or len(self.decision_member_security_ids)
            != len(self.decision_member_universe_ranks)
            or len(set(self.decision_member_security_ids))
            != len(self.decision_member_security_ids)
            or self.decision_member_universe_ranks
            != tuple(sorted(set(self.decision_member_universe_ranks)))
            or len(self.decision_member_security_ids)
            > MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.target_size
        ):
            raise MassiveProfitabilityOriginP0Error(
                "decision membership inventory differs"
            )
        for security_id in self.decision_member_security_ids:
            _text("decision member security ID", security_id)
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            for rank in self.decision_member_universe_ranks
        ):
            raise MassiveProfitabilityOriginP0Error(
                "decision membership ranks are invalid"
            )
        if self.membership_effective_at_ms > self.decision_at_ms:
            raise MassiveProfitabilityOriginP0Error(
                "decision membership became effective after the decision"
            )
        for name in (
            "source_evidence_receipt_sha256",
            "loaded_source_receipt_sha256",
            "source_object_receipt_sha256",
            "loaded_listing_receipt_sha256",
            "listing_entry_receipt_sha256",
            "committed_listing_receipt_sha256",
            "membership_group_receipt_sha256",
            "identity_authority_receipt_sha256",
            "universe_rule_receipt_sha256",
            "session_authority_receipt_sha256",
            "protocol_receipt_sha256",
            "origin_spec_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.origin_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256
            or self.universe_rule_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityOriginP0Error("decision origin receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySkippedDecisionP0:
    decision_session_date: str
    decision_at_ms: int
    source_session_date: str | None
    source_evidence_receipt_sha256: str | None
    reason: str
    protocol_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_SKIPPED_DECISION_P0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_SKIPPED_DECISION_P0_SCHEMA
            or self.reason not in _SKIP_REASONS
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
        ):
            raise MassiveProfitabilityOriginP0Error("skipped decision identity differs")
        _canonical_date("skipped decision session", self.decision_session_date)
        if self.decision_at_ms != _local_ms(self.decision_session_date, time(12, 30)):
            raise MassiveProfitabilityOriginP0Error("skipped decision clock differs")
        expected_source_shape = {
            "insufficient-prior-session-history": (True, True),
            "missing-source-object-evidence": (False, True),
        }.get(self.reason, (False, False))
        observed_source_shape = (
            self.source_session_date is None,
            self.source_evidence_receipt_sha256 is None,
        )
        if observed_source_shape != expected_source_shape:
            raise MassiveProfitabilityOriginP0Error(
                "skipped source identity is incomplete"
            )
        if self.source_session_date is not None:
            _canonical_date("skipped source session", self.source_session_date)
        if self.source_evidence_receipt_sha256 is not None:
            _digest("skipped source evidence", self.source_evidence_receipt_sha256)
        _digest("skipped decision receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginP0Error("skipped decision receipt differs")


def _build_skip(
    *,
    decision_session_date: str,
    source_session_date: str | None,
    source_evidence_receipt_sha256: str | None,
    reason: str,
) -> MassiveProfitabilitySkippedDecisionP0:
    provisional = MassiveProfitabilitySkippedDecisionP0(
        decision_session_date=decision_session_date,
        decision_at_ms=_local_ms(decision_session_date, time(12, 30)),
        source_session_date=source_session_date,
        source_evidence_receipt_sha256=source_evidence_receipt_sha256,
        reason=reason,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _latest_membership_group(
    *, identity_authority: PITSecurityUniverseAuthority, decision_at_ms: int
) -> tuple[int, tuple[HistoricalMembershipRecord, ...]] | None:
    grouped: defaultdict[int, list[HistoricalMembershipRecord]] = defaultdict(list)
    for row in identity_authority.membership_events:
        if row.effective_at_ms <= decision_at_ms:
            grouped[row.effective_at_ms].append(row)
    candidates = tuple(
        effective
        for effective, rows in grouped.items()
        if all(
            row.available_at_ms <= decision_at_ms
            and row.observation_end_ms <= decision_at_ms
            for row in rows
        )
    )
    if not candidates:
        return None
    effective = max(candidates)
    rows = tuple(sorted(grouped[effective], key=lambda row: row.security_id))
    return effective, rows


def _origin_from_authorities(
    *,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    source_evidence: MassiveProfitabilitySourceEvidenceP0,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveProfitabilityDecisionOriginP0 | MassiveProfitabilitySkippedDecisionP0:
    decision_at = _local_ms(decision_session.session_date, time(12, 30))
    fill_start = _local_ms(decision_session.session_date, time(15, 50))
    fill_end = _local_ms(decision_session.session_date, time(16, 0))
    regular_open = _session_ms(decision_session.regular_open_ns)
    regular_close = _session_ms(decision_session.regular_close_ns)
    if not (regular_open <= decision_at < fill_start < fill_end <= regular_close):
        return _build_skip(
            decision_session_date=decision_session.session_date,
            source_session_date=source_session.session_date,
            source_evidence_receipt_sha256=source_evidence.receipt_sha256,
            reason="decision-session-cannot-support-frozen-clock",
        )
    source_close = _session_ms(source_session.regular_close_ns)
    if source_evidence.vendor_last_modified_at_ms < source_close:
        return _build_skip(
            decision_session_date=decision_session.session_date,
            source_session_date=source_session.session_date,
            source_evidence_receipt_sha256=source_evidence.receipt_sha256,
            reason="vendor-object-predates-source-close",
        )
    lead = decision_at - source_evidence.vendor_last_modified_at_ms
    if lead < MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS:
        return _build_skip(
            decision_session_date=decision_session.session_date,
            source_session_date=source_session.session_date,
            source_evidence_receipt_sha256=source_evidence.receipt_sha256,
            reason="vendor-lead-below-18-hours",
        )
    selected_group = _latest_membership_group(
        identity_authority=identity_authority, decision_at_ms=decision_at
    )
    if selected_group is None:
        return _build_skip(
            decision_session_date=decision_session.session_date,
            source_session_date=source_session.session_date,
            source_evidence_receipt_sha256=source_evidence.receipt_sha256,
            reason="no-complete-pit-membership-known-at-decision",
        )
    effective, group = selected_group
    members = tuple(
        sorted(
            (row for row in group if row.is_member),
            key=lambda row: (row.universe_rank or 10**9, row.security_id),
        )
    )
    if not members:
        return _build_skip(
            decision_session_date=decision_session.session_date,
            source_session_date=source_session.session_date,
            source_evidence_receipt_sha256=source_evidence.receipt_sha256,
            reason="no-complete-pit-membership-known-at-decision",
        )
    body = {
        "schema": MASSIVE_PROFITABILITY_DECISION_ORIGIN_P0_SCHEMA,
        "source_session_date": source_session.session_date,
        "decision_session_date": decision_session.session_date,
        "exchange": "XNYS",
        "source_staleness_sessions": 2,
        "decision_regular_open_at_ms": regular_open,
        "decision_regular_close_at_ms": regular_close,
        "decision_at_ms": decision_at,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
        "feature_maximum_session_date": source_session.session_date,
        "feature_cutoff_at_ms": source_close,
        "vendor_last_modified_at_ms": source_evidence.vendor_last_modified_at_ms,
        "vendor_lead_time_ms": lead,
        "source_object_key": source_evidence.source_object_key,
        "source_evidence_receipt_sha256": source_evidence.receipt_sha256,
        "loaded_source_receipt_sha256": source_evidence.loaded_source_receipt_sha256,
        "source_object_receipt_sha256": (source_evidence.source_object_receipt_sha256),
        "loaded_listing_receipt_sha256": (
            source_evidence.loaded_listing_receipt_sha256
        ),
        "listing_entry_receipt_sha256": (source_evidence.listing_entry_receipt_sha256),
        "committed_listing_receipt_sha256": (
            source_evidence.committed_listing_receipt_sha256
        ),
        "membership_effective_at_ms": effective,
        "decision_member_security_ids": tuple(row.security_id for row in members),
        "decision_member_universe_ranks": tuple(
            int(row.universe_rank or 0) for row in members
        ),
        "membership_group_receipt_sha256": semantic_sha256(
            tuple(asdict(row) for row in group)
        ),
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "universe_rule_receipt_sha256": identity_authority.rule.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "origin_spec_receipt_sha256": (
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256
        ),
        "implementation_source_sha256": (MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256),
    }
    result = MassiveProfitabilityDecisionOriginP0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _derive_origin_rows(
    *,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_evidences: tuple[MassiveProfitabilitySourceEvidenceP0, ...],
    candidate_dates: tuple[str, ...],
) -> tuple[
    tuple[MassiveProfitabilityDecisionOriginP0, ...],
    tuple[MassiveProfitabilitySkippedDecisionP0, ...],
]:
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    by_date = {row.session_date: row for row in sessions}
    positions = {row.session_date: index for index, row in enumerate(sessions)}
    sources = {row.source_session_date: row for row in source_evidences}
    origins: list[MassiveProfitabilityDecisionOriginP0] = []
    skips: list[MassiveProfitabilitySkippedDecisionP0] = []
    for decision_date in candidate_dates:
        decision_position = positions[decision_date]
        if decision_position < 2:
            skips.append(
                _build_skip(
                    decision_session_date=decision_date,
                    source_session_date=None,
                    source_evidence_receipt_sha256=None,
                    reason="insufficient-prior-session-history",
                )
            )
            continue
        source_session = sessions[decision_position - 2]
        source = sources.get(source_session.session_date)
        if source is None:
            skips.append(
                _build_skip(
                    decision_session_date=decision_date,
                    source_session_date=source_session.session_date,
                    source_evidence_receipt_sha256=None,
                    reason="missing-source-object-evidence",
                )
            )
            continue
        row = _origin_from_authorities(
            source_session=source_session,
            decision_session=by_date[decision_date],
            source_evidence=source,
            session_authority=session_authority,
            identity_authority=identity_authority,
        )
        if isinstance(row, MassiveProfitabilityDecisionOriginP0):
            origins.append(row)
        else:
            skips.append(row)
    return tuple(origins), tuple(skips)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDecisionOriginPlanP0:
    session_authority: MassiveSessionAuthority
    identity_authority: PITSecurityUniverseAuthority
    source_evidences: tuple[MassiveProfitabilitySourceEvidenceP0, ...]
    first_candidate_decision_session_date: str
    last_candidate_decision_session_date: str
    candidate_decision_session_dates: tuple[str, ...]
    origins: tuple[MassiveProfitabilityDecisionOriginP0, ...]
    skipped_decisions: tuple[MassiveProfitabilitySkippedDecisionP0, ...]
    origin_spec_receipt_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SCHEMA
            or self.origin_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256
        ):
            raise MassiveProfitabilityOriginP0Error("origin plan identity differs")
        self.session_authority.validate()
        self.identity_authority.validate()
        if self.identity_authority.rule.receipt_sha256 != (
            MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
        ):
            raise MassiveProfitabilityOriginP0Error(
                "origin plan does not use the frozen PIT-500 rule"
            )
        sessions = tuple(
            row for row in self.session_authority.sessions if row.exchange == "XNYS"
        )
        dates = tuple(row.session_date for row in sessions)
        if (
            self.first_candidate_decision_session_date not in dates
            or self.last_candidate_decision_session_date not in dates
            or self.first_candidate_decision_session_date
            > self.last_candidate_decision_session_date
        ):
            raise MassiveProfitabilityOriginP0Error(
                "origin candidate interval is absent or inverted"
            )
        expected_candidates = tuple(
            value
            for value in dates
            if self.first_candidate_decision_session_date
            <= value
            <= self.last_candidate_decision_session_date
        )
        if self.candidate_decision_session_dates != expected_candidates:
            raise MassiveProfitabilityOriginP0Error(
                "origin plan omitted or inserted a candidate session"
            )
        if not expected_candidates:
            raise MassiveProfitabilityOriginP0Error(
                "origin candidate interval is empty"
            )
        source_dates = tuple(row.source_session_date for row in self.source_evidences)
        if source_dates != tuple(sorted(set(source_dates))):
            raise MassiveProfitabilityOriginP0Error(
                "source evidence dates are not sorted and unique"
            )
        positions = {value: index for index, value in enumerate(dates)}
        required_source_dates = {
            dates[positions[value] - 2]
            for value in expected_candidates
            if positions[value] >= 2
        }
        if set(source_dates) - required_source_dates:
            raise MassiveProfitabilityOriginP0Error(
                "origin plan contains irrelevant source evidence"
            )
        for row in self.source_evidences:
            row.validate()
        expected_origins, expected_skips = _derive_origin_rows(
            session_authority=self.session_authority,
            identity_authority=self.identity_authority,
            source_evidences=self.source_evidences,
            candidate_dates=expected_candidates,
        )
        if self.origins != expected_origins or self.skipped_decisions != expected_skips:
            raise MassiveProfitabilityOriginP0Error(
                "origin plan rows were not independently rederived"
            )
        partition = tuple(
            sorted(
                tuple(row.decision_session_date for row in self.origins)
                + tuple(row.decision_session_date for row in self.skipped_decisions)
            )
        )
        if partition != expected_candidates:
            raise MassiveProfitabilityOriginP0Error(
                "origins and skips do not partition candidate decisions"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginP0Error("origin plan receipt differs")


def build_massive_profitability_decision_origin_plan_p0(
    *,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    source_evidences: Sequence[MassiveProfitabilitySourceEvidenceP0],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveProfitabilityDecisionOriginPlanP0:
    """Exhaustively derive one eligible origin or explicit skip per session."""

    session_authority.validate()
    identity_authority.validate()
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    dates = tuple(row.session_date for row in sessions)
    if (
        first_candidate_decision_session_date not in dates
        or last_candidate_decision_session_date not in dates
        or first_candidate_decision_session_date > last_candidate_decision_session_date
    ):
        raise MassiveProfitabilityOriginP0Error(
            "origin candidate interval is absent or inverted"
        )
    candidates = tuple(
        value
        for value in dates
        if first_candidate_decision_session_date
        <= value
        <= last_candidate_decision_session_date
    )
    sources = tuple(sorted(source_evidences, key=lambda row: row.source_session_date))
    origins, skips = _derive_origin_rows(
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidences=sources,
        candidate_dates=candidates,
    )
    provisional = MassiveProfitabilityDecisionOriginPlanP0(
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidences=sources,
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
        candidate_decision_session_dates=candidates,
        origins=origins,
        skipped_decisions=skips,
        origin_spec_receipt_sha256=MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256,
        implementation_source_sha256=(MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _security_support_ids(
    *, plan: MassiveProfitabilityDecisionOriginPlanP0
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    decision_members = {
        security_id
        for origin in plan.origins
        for security_id in origin.decision_member_security_ids
    }
    masters = {row.security_id: row for row in plan.identity_authority.security_master}
    adjacency: dict[str, set[str]] = {security_id: set() for security_id in masters}
    chains: defaultdict[str, set[str]] = defaultdict(set)
    for row in masters.values():
        if row.successor_security_id is not None:
            adjacency[row.security_id].add(row.successor_security_id)
            adjacency[row.successor_security_id].add(row.security_id)
        if row.corporate_action_chain_id is not None:
            chains[row.corporate_action_chain_id].add(row.security_id)
    for members in chains.values():
        for security_id in members:
            adjacency[security_id].update(members - {security_id})
    supported = set(decision_members)
    queue = deque(sorted(decision_members))
    while queue:
        current = queue.popleft()
        for linked in sorted(adjacency[current]):
            if linked not in supported:
                supported.add(linked)
                queue.append(linked)
    accounting = supported - decision_members
    return (
        tuple(sorted(decision_members)),
        tuple(sorted(accounting)),
        tuple(sorted(supported)),
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySecuritySupportV1:
    decision_member_security_ids: tuple[str, ...]
    accounting_chain_security_ids: tuple[str, ...]
    all_supported_security_ids: tuple[str, ...]
    decision_origin_member_inventory_sha256: str
    origin_plan_receipt_sha256: str
    identity_authority_receipt_sha256: str
    support_spec_receipt_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SCHEMA:
            raise MassiveProfitabilityOriginP0Error("security support schema differs")
        for inventory in (
            self.decision_member_security_ids,
            self.accounting_chain_security_ids,
            self.all_supported_security_ids,
        ):
            if inventory != tuple(sorted(set(inventory))):
                raise MassiveProfitabilityOriginP0Error(
                    "security support inventory is not canonical"
                )
        if (
            not self.decision_member_security_ids
            or set(self.decision_member_security_ids)
            & set(self.accounting_chain_security_ids)
            or set(self.all_supported_security_ids)
            != set(self.decision_member_security_ids)
            | set(self.accounting_chain_security_ids)
        ):
            raise MassiveProfitabilityOriginP0Error(
                "security support inventories do not reconcile"
            )
        for name in (
            "decision_origin_member_inventory_sha256",
            "origin_plan_receipt_sha256",
            "identity_authority_receipt_sha256",
            "support_spec_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.support_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityOriginP0Error("security support receipt differs")


def build_massive_profitability_security_support_v1(
    *, plan: MassiveProfitabilityDecisionOriginPlanP0
) -> MassiveProfitabilitySecuritySupportV1:
    """Close decision members over predecessor/successor accounting chains."""

    plan.validate()
    decision, accounting, supported = _security_support_ids(plan=plan)
    member_inventory = semantic_sha256(
        tuple(
            (
                row.decision_session_date,
                row.decision_member_security_ids,
                row.decision_member_universe_ranks,
                row.membership_group_receipt_sha256,
            )
            for row in plan.origins
        )
    )
    provisional = MassiveProfitabilitySecuritySupportV1(
        decision_member_security_ids=decision,
        accounting_chain_security_ids=accounting,
        all_supported_security_ids=supported,
        decision_origin_member_inventory_sha256=member_inventory,
        origin_plan_receipt_sha256=plan.receipt_sha256,
        identity_authority_receipt_sha256=plan.identity_authority.receipt_sha256,
        support_spec_receipt_sha256=(
            MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SPEC_SHA256
        ),
        implementation_source_sha256=(MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    validate_massive_profitability_security_support_v1(plan=plan, support=result)
    return result


def validate_massive_profitability_security_support_v1(
    *,
    plan: MassiveProfitabilityDecisionOriginPlanP0,
    support: MassiveProfitabilitySecuritySupportV1,
) -> None:
    plan.validate()
    support.validate()
    decision, accounting, supported = _security_support_ids(plan=plan)
    expected_member_inventory = semantic_sha256(
        tuple(
            (
                row.decision_session_date,
                row.decision_member_security_ids,
                row.decision_member_universe_ranks,
                row.membership_group_receipt_sha256,
            )
            for row in plan.origins
        )
    )
    if (
        support.decision_member_security_ids != decision
        or support.accounting_chain_security_ids != accounting
        or support.all_supported_security_ids != supported
        or support.origin_plan_receipt_sha256 != plan.receipt_sha256
        or support.identity_authority_receipt_sha256
        != plan.identity_authority.receipt_sha256
        or support.decision_origin_member_inventory_sha256 != expected_member_inventory
    ):
        raise MassiveProfitabilityOriginP0Error(
            "security support was not independently rederived"
        )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPhaseIntervalV1:
    phase_id: str
    first_decision_session_date: str
    last_decision_session_date: str
    origin_count: int
    origin_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_PHASE_INTERVAL_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_PHASE_INTERVAL_V1_SCHEMA
            or self.phase_id not in {"development", "confirmation", "lockbox"}
        ):
            raise MassiveProfitabilityOriginP0Error("phase interval identity differs")
        if _nonnegative_int("phase origin count", self.origin_count) <= 0:
            raise MassiveProfitabilityOriginP0Error("phase interval is empty")
        first = _canonical_date(
            "phase first decision", self.first_decision_session_date
        )
        last = _canonical_date("phase last decision", self.last_decision_session_date)
        if last < first:
            raise MassiveProfitabilityOriginP0Error("phase interval is inverted")
        _digest("phase origin inventory", self.origin_inventory_sha256)
        _digest("phase receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginP0Error("phase receipt differs")


def _phase_interval(
    *, phase_id: str, origins: tuple[MassiveProfitabilityDecisionOriginP0, ...]
) -> MassiveProfitabilityPhaseIntervalV1:
    provisional = MassiveProfitabilityPhaseIntervalV1(
        phase_id=phase_id,
        first_decision_session_date=origins[0].decision_session_date,
        last_decision_session_date=origins[-1].decision_session_date,
        origin_count=len(origins),
        origin_inventory_sha256=semantic_sha256(
            tuple((row.decision_session_date, row.receipt_sha256) for row in origins)
        ),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityExperimentCoverageV1:
    earliest_feature_base_session_date: str
    earliest_source_session_date: str
    latest_source_session_date: str
    earliest_fill_session_date: str
    latest_fill_session_date: str
    latest_h63_endpoint_session_date: str
    economic_coverage_start_date: str
    economic_coverage_end_date: str
    development_interval: MassiveProfitabilityPhaseIntervalV1
    confirmation_interval: MassiveProfitabilityPhaseIntervalV1
    lockbox_interval: MassiveProfitabilityPhaseIntervalV1
    expected_decision_origin_inventory_sha256: str
    eligible_decision_origin_inventory_sha256: str
    origin_plan_receipt_sha256: str
    security_support_receipt_sha256: str
    session_authority_receipt_sha256: str
    coverage_spec_receipt_sha256: str
    implementation_source_sha256: str
    coverage_complete: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SCHEMA:
            raise MassiveProfitabilityOriginP0Error(
                "experiment coverage schema differs"
            )
        dates = tuple(
            _canonical_date(name, getattr(self, name))
            for name in (
                "earliest_feature_base_session_date",
                "earliest_source_session_date",
                "latest_source_session_date",
                "earliest_fill_session_date",
                "latest_fill_session_date",
                "latest_h63_endpoint_session_date",
                "economic_coverage_start_date",
                "economic_coverage_end_date",
            )
        )
        if (
            self.economic_coverage_start_date != self.earliest_feature_base_session_date
            or self.economic_coverage_end_date != self.latest_h63_endpoint_session_date
            or dates[0] > dates[1]
            or dates[1] > dates[2]
            or dates[3] > dates[4]
            or dates[4] > dates[5]
        ):
            raise MassiveProfitabilityOriginP0Error(
                "experiment coverage chronology differs"
            )
        for interval in (
            self.development_interval,
            self.confirmation_interval,
            self.lockbox_interval,
        ):
            interval.validate()
        if (
            self.development_interval.phase_id != "development"
            or self.confirmation_interval.phase_id != "confirmation"
            or self.lockbox_interval.phase_id != "lockbox"
            or self.confirmation_interval.origin_count
            != MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS
            or self.lockbox_interval.origin_count
            != MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS
            or self.coverage_complete is not True
            or any(
                (
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityOriginP0Error(
                "experiment phase or authorization policy differs"
            )
        for name in (
            "expected_decision_origin_inventory_sha256",
            "eligible_decision_origin_inventory_sha256",
            "origin_plan_receipt_sha256",
            "security_support_receipt_sha256",
            "session_authority_receipt_sha256",
            "coverage_spec_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.coverage_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityOriginP0Error(
                "experiment coverage receipt differs"
            )


def _coverage_components(
    *,
    plan: MassiveProfitabilityDecisionOriginPlanP0,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[MassiveProfitabilityDecisionOriginP0, ...],
    tuple[MassiveProfitabilityDecisionOriginP0, ...],
    tuple[MassiveProfitabilityDecisionOriginP0, ...],
]:
    origins = tuple(sorted(plan.origins, key=lambda row: row.decision_session_date))
    if len(origins) < MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS:
        raise MassiveProfitabilityOriginP0Error(
            "experiment lacks the frozen minimum eligible-origin history"
        )
    sessions = tuple(
        row for row in plan.session_authority.sessions if row.exchange == "XNYS"
    )
    positions = {row.session_date: index for index, row in enumerate(sessions)}
    earliest_source = origins[0].source_session_date
    latest_source = origins[-1].source_session_date
    feature_base_position = (
        positions[earliest_source] - MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS
    )
    endpoint_position = (
        positions[origins[-1].decision_session_date]
        + MASSIVE_PROFITABILITY_MAXIMUM_TARGET_HORIZON_SESSIONS
    )
    if feature_base_position < 0 or endpoint_position >= len(sessions):
        raise MassiveProfitabilityOriginP0Error(
            "session authority does not cover feature and target boundaries"
        )
    development = origins[:-MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS]
    confirmation = development[-MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS:]
    lockbox = origins[-MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS:]
    return (
        sessions[feature_base_position].session_date,
        earliest_source,
        latest_source,
        origins[0].decision_session_date,
        origins[-1].decision_session_date,
        sessions[endpoint_position].session_date,
        development,
        confirmation,
        lockbox,
    )


def build_massive_profitability_experiment_coverage_v1(
    *,
    plan: MassiveProfitabilityDecisionOriginPlanP0,
    support: MassiveProfitabilitySecuritySupportV1,
) -> MassiveProfitabilityExperimentCoverageV1:
    """Derive feature, development, confirmation, target, and lockbox coverage."""

    plan.validate()
    validate_massive_profitability_security_support_v1(plan=plan, support=support)
    (
        feature_base,
        earliest_source,
        latest_source,
        earliest_fill,
        latest_fill,
        h63_endpoint,
        development_origins,
        confirmation_origins,
        lockbox_origins,
    ) = _coverage_components(plan=plan)
    expected_inventory = semantic_sha256(
        tuple(
            sorted(
                tuple(
                    (row.decision_session_date, "origin", row.receipt_sha256)
                    for row in plan.origins
                )
                + tuple(
                    (row.decision_session_date, "skip", row.receipt_sha256)
                    for row in plan.skipped_decisions
                )
            )
        )
    )
    eligible_inventory = semantic_sha256(
        tuple((row.decision_session_date, row.receipt_sha256) for row in plan.origins)
    )
    provisional = MassiveProfitabilityExperimentCoverageV1(
        earliest_feature_base_session_date=feature_base,
        earliest_source_session_date=earliest_source,
        latest_source_session_date=latest_source,
        earliest_fill_session_date=earliest_fill,
        latest_fill_session_date=latest_fill,
        latest_h63_endpoint_session_date=h63_endpoint,
        economic_coverage_start_date=feature_base,
        economic_coverage_end_date=h63_endpoint,
        development_interval=_phase_interval(
            phase_id="development", origins=development_origins
        ),
        confirmation_interval=_phase_interval(
            phase_id="confirmation", origins=confirmation_origins
        ),
        lockbox_interval=_phase_interval(phase_id="lockbox", origins=lockbox_origins),
        expected_decision_origin_inventory_sha256=expected_inventory,
        eligible_decision_origin_inventory_sha256=eligible_inventory,
        origin_plan_receipt_sha256=plan.receipt_sha256,
        security_support_receipt_sha256=support.receipt_sha256,
        session_authority_receipt_sha256=plan.session_authority.receipt_sha256,
        coverage_spec_receipt_sha256=(
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SPEC_SHA256
        ),
        implementation_source_sha256=(MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256),
        coverage_complete=True,
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    validate_massive_profitability_experiment_coverage_v1(
        plan=plan, support=support, coverage=result
    )
    return result


def validate_massive_profitability_experiment_coverage_v1(
    *,
    plan: MassiveProfitabilityDecisionOriginPlanP0,
    support: MassiveProfitabilitySecuritySupportV1,
    coverage: MassiveProfitabilityExperimentCoverageV1,
) -> None:
    plan.validate()
    validate_massive_profitability_security_support_v1(plan=plan, support=support)
    coverage.validate()
    (
        feature_base,
        earliest_source,
        latest_source,
        earliest_fill,
        latest_fill,
        h63_endpoint,
        development_origins,
        confirmation_origins,
        lockbox_origins,
    ) = _coverage_components(plan=plan)
    expected_inventory = semantic_sha256(
        tuple(
            sorted(
                tuple(
                    (row.decision_session_date, "origin", row.receipt_sha256)
                    for row in plan.origins
                )
                + tuple(
                    (row.decision_session_date, "skip", row.receipt_sha256)
                    for row in plan.skipped_decisions
                )
            )
        )
    )
    eligible_inventory = semantic_sha256(
        tuple((row.decision_session_date, row.receipt_sha256) for row in plan.origins)
    )
    if (
        coverage.earliest_feature_base_session_date != feature_base
        or coverage.earliest_source_session_date != earliest_source
        or coverage.latest_source_session_date != latest_source
        or coverage.earliest_fill_session_date != earliest_fill
        or coverage.latest_fill_session_date != latest_fill
        or coverage.latest_h63_endpoint_session_date != h63_endpoint
        or coverage.development_interval
        != _phase_interval(phase_id="development", origins=development_origins)
        or coverage.confirmation_interval
        != _phase_interval(phase_id="confirmation", origins=confirmation_origins)
        or coverage.lockbox_interval
        != _phase_interval(phase_id="lockbox", origins=lockbox_origins)
        or coverage.origin_plan_receipt_sha256 != plan.receipt_sha256
        or coverage.security_support_receipt_sha256 != support.receipt_sha256
        or coverage.session_authority_receipt_sha256
        != plan.session_authority.receipt_sha256
        or coverage.expected_decision_origin_inventory_sha256 != expected_inventory
        or coverage.eligible_decision_origin_inventory_sha256 != eligible_inventory
    ):
        raise MassiveProfitabilityOriginP0Error(
            "experiment coverage was not independently rederived"
        )


def build_massive_economic_coverage_scope_from_profitability_experiment_v8(
    *,
    plan: MassiveProfitabilityDecisionOriginPlanP0,
    support: MassiveProfitabilitySecuritySupportV1,
    coverage: MassiveProfitabilityExperimentCoverageV1,
) -> MassiveEconomicCoverageScopeV8:
    """Create V8 capture scope only from a rederived P0 experiment interval."""

    from rl_quant.features.massive_economic_coverage_v8 import (
        MassiveEconomicCoverageScopeV8,
    )

    validate_massive_profitability_experiment_coverage_v1(
        plan=plan, support=support, coverage=coverage
    )
    return MassiveEconomicCoverageScopeV8.build(
        coverage_start_date=coverage.economic_coverage_start_date,
        coverage_end_date=coverage.economic_coverage_end_date,
    )


MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256 = file_sha256(Path(__file__))


__all__ = [
    "MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V1_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS",
    "MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS",
    "MASSIVE_PROFITABILITY_MAXIMUM_TARGET_HORIZON_SESSIONS",
    "MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS",
    "MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS",
    "MASSIVE_PROFITABILITY_ORIGIN_P0_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_P0_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_P0_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_P0_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_P0_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V1_SPEC_SHA256",
    "MassiveProfitabilityDecisionOriginP0",
    "MassiveProfitabilityDecisionOriginPlanP0",
    "MassiveProfitabilityExperimentCoverageV1",
    "MassiveProfitabilityOriginP0Error",
    "MassiveProfitabilityPhaseIntervalV1",
    "MassiveProfitabilitySecuritySupportV1",
    "MassiveProfitabilitySkippedDecisionP0",
    "MassiveProfitabilitySourceEvidenceP0",
    "build_massive_economic_coverage_scope_from_profitability_experiment_v8",
    "build_massive_profitability_decision_origin_plan_p0",
    "build_massive_profitability_experiment_coverage_v1",
    "build_massive_profitability_security_support_v1",
    "build_massive_profitability_source_evidence_p0",
    "validate_massive_profitability_experiment_coverage_v1",
    "validate_massive_profitability_security_support_v1",
]
