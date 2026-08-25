"""Immutable archive and candidate-date freeze for Massive P0 profitability.

This generation freezes dates before model outcomes exist.  Its candidate
interval is derived from a complete source archive, the exchange calendar,
the acquired monthly rank inventory, and target maturity at one data-freeze
timestamp.  Callers cannot supply first or last candidate dates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
    coverage_session_from_massive_trade_key,
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
from rl_quant.features.massive_monthly_rank_bar_authority_v1 import (
    MassiveMonthlyRankBarAuthorityV1,
)
from rl_quant.features.massive_monthly_rank_input_authority_v2 import (
    MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_BINDING_SHA256,
    MassiveMonthlyRankInputAuthorityV2,
)
from rl_quant.features.massive_profitability_frozen_authorities_v1 import (
    MassiveProfitabilityFrozenAuthorityArtifactV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityProductionAcquisitionV2,
    validate_massive_profitability_production_acquisition_v2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA = (
    "rl-quant.massive-profitability-archive-freeze-v1"
)
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_DATASET = (
    "massive-profitability-archive-freeze-v1"
)
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))

_PROTOCOL = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
_LOOKBACK = max(row.end_offset_sessions for row in _PROTOCOL.horizons)
_MAX_HORIZON = _LOOKBACK
_OUTER_COUNT = _PROTOCOL.outer_fold_count
_OUTER_SESSIONS = _PROTOCOL.outer_fold_sessions
_LOCKBOX_SESSIONS = _PROTOCOL.historical_lockbox_sessions
_MINIMUM_CANDIDATES = (
    _PROTOCOL.minimum_initial_training_sessions
    + _PROTOCOL.inner_purge_sessions
    + _PROTOCOL.inner_validation_sessions
    + _PROTOCOL.target_overlap_purge_sessions
    + _OUTER_COUNT * _OUTER_SESSIONS
    + _LOCKBOX_SESSIONS
)
_MINIMUM_VENDOR_LEAD_MS = 18 * 60 * 60 * 1_000
_EASTERN = ZoneInfo("America/New_York")

MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source_archive": "every-XNYS-session-between-derived-boundaries",
        "source_evidence": "production-listing-and-authenticated-object-get-v2",
        "rank_inventory": MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_BINDING_SHA256,
        "rank_bar_inventory": "authenticated-partition-rederived-rank-bars-v1",
        "candidate_start": (
            "first-calendar-origin-with-source-minus-63-history-valid-d-minus-2-"
            "source-and-scheduled-monthly-rank"
        ),
        "candidate_end": "last-origin-with-H63-close-mature-by-data-freeze",
        "candidate_clock": "12:30-decision-and-15:50-16:00-fill-supported",
        "outer_tests": (4, 126, "fixed-candidate-dates"),
        "lockbox": (252, "fixed-candidate-dates"),
        "minimum_candidate_dates": _MINIMUM_CANDIDATES,
        "caller_dates": "prohibited",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityArchiveFreezeV1Error(ValueError):
    """Archive evidence or derived phase dates differ from the frozen rule."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveProfitabilityArchiveFreezeV1Error(f"{name} must be canonical text")
    return value


def _canonical_date(name: str, value: object) -> str:
    raw = _text(name, value)
    try:
        observed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            f"{name} must be an ISO date"
        ) from exc
    if observed.isoformat() != raw:
        raise MassiveProfitabilityArchiveFreezeV1Error(f"{name} must be canonical")
    return raw


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveProfitabilityArchiveFreezeV1Error(f"{name} must be nonnegative")
    return value


def _artifact_id(value: object) -> str:
    result = _text("archive freeze artifact ID", value)
    if any(not (character.isalnum() or character in "-_") for character in result):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze artifact ID is not path safe"
        )
    return result


def _session_ms(value_ns: int) -> int:
    value = _nonnegative_int("session timestamp", value_ns)
    if value % 1_000_000:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "session timestamp is not millisecond aligned"
        )
    return value // 1_000_000


def _local_ms(session_date: str, value: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(session_date), value, tzinfo=_EASTERN
        ).timestamp()
        * 1_000
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityArchiveSourceSessionV1:
    source_session_date: str
    source_object_key: str
    vendor_last_modified_at_ms: int
    authenticated_get_completed_at_ms: int
    listing_entry_receipt_sha256: str
    authenticated_download_receipt_sha256: str
    loaded_source_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        session_date = _canonical_date(
            "archive source session", self.source_session_date
        )
        if self.source_object_key != canonical_massive_trade_object_key(session_date):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive source object key differs"
            )
        vendor = _nonnegative_int(
            "archive vendor availability", self.vendor_last_modified_at_ms
        )
        completed = _nonnegative_int(
            "archive authenticated GET completion",
            self.authenticated_get_completed_at_ms,
        )
        if completed < vendor:
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive GET completed before provider availability"
            )
        for name in (
            "listing_entry_receipt_sha256",
            "authenticated_download_receipt_sha256",
            "loaded_source_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive source row receipt differs"
            )

    @classmethod
    def build_for_test(
        cls,
        *,
        source_session_date: str,
        vendor_last_modified_at_ms: int,
        authenticated_get_completed_at_ms: int,
    ) -> MassiveProfitabilityArchiveSourceSessionV1:
        """Build nonauthorizing fixture semantics without transport claims."""

        body: dict[str, object] = {
            "source_session_date": source_session_date,
            "source_object_key": canonical_massive_trade_object_key(
                source_session_date
            ),
            "vendor_last_modified_at_ms": vendor_last_modified_at_ms,
            "authenticated_get_completed_at_ms": authenticated_get_completed_at_ms,
            "listing_entry_receipt_sha256": semantic_sha256(
                ("test-listing-entry", source_session_date)
            ),
            "authenticated_download_receipt_sha256": semantic_sha256(
                ("test-authenticated-download", source_session_date)
            ),
            "loaded_source_receipt_sha256": semantic_sha256(
                ("test-loaded-source", source_session_date)
            ),
        }
        result = cls(**body, receipt_sha256=semantic_sha256(body))  # type: ignore[arg-type]
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityArchiveMonthlyRankV1:
    calendar_month: str
    scheduled_rebalance_session_date: str
    activated_at_ms: int
    maximum_input_available_at_ms: int
    rank_input_group_receipt_sha256: str
    daily_bar_inventory_sha256: str
    authenticated_source_inventory_sha256: str
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
                "archive rank rebalance", self.scheduled_rebalance_session_date
            )[:7]
            != self.calendar_month
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive monthly rank identity differs"
            )
        activated = _nonnegative_int("archive rank activation", self.activated_at_ms)
        available = _nonnegative_int(
            "archive rank availability", self.maximum_input_available_at_ms
        )
        if available > activated:
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive rank activated before its inputs were available"
            )
        for name in (
            "rank_input_group_receipt_sha256",
            "daily_bar_inventory_sha256",
            "authenticated_source_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive monthly rank receipt differs"
            )

    @classmethod
    def build_for_test(
        cls,
        *,
        calendar_month: str,
        scheduled_rebalance_session_date: str,
        activated_at_ms: int,
        maximum_input_available_at_ms: int,
    ) -> MassiveProfitabilityArchiveMonthlyRankV1:
        body: dict[str, object] = {
            "calendar_month": calendar_month,
            "scheduled_rebalance_session_date": scheduled_rebalance_session_date,
            "activated_at_ms": activated_at_ms,
            "maximum_input_available_at_ms": maximum_input_available_at_ms,
            "rank_input_group_receipt_sha256": semantic_sha256(
                ("test-rank-group", calendar_month)
            ),
            "daily_bar_inventory_sha256": semantic_sha256(
                ("test-rank-bars", calendar_month)
            ),
            "authenticated_source_inventory_sha256": semantic_sha256(
                ("test-rank-sources", calendar_month)
            ),
        }
        result = cls(**body, receipt_sha256=semantic_sha256(body))  # type: ignore[arg-type]
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityArchiveFreezeV1:
    data_freeze_at_ms: int
    source_archive_start_session_date: str
    source_archive_end_session_date: str
    earliest_feature_base_session_date: str
    earliest_eligible_decision_session_date: str
    latest_h63_mature_decision_session_date: str
    latest_h63_endpoint_session_date: str
    fixed_candidate_session_dates: tuple[str, ...]
    fixed_outer_test_session_inventories: tuple[tuple[str, ...], ...]
    fixed_lockbox_session_dates: tuple[str, ...]
    source_sessions: tuple[MassiveProfitabilityArchiveSourceSessionV1, ...]
    monthly_ranks: tuple[MassiveProfitabilityArchiveMonthlyRankV1, ...]
    candidate_inventory_sha256: str
    captured_listing_inventory_sha256: str
    authenticated_download_inventory_sha256: str
    monthly_rank_authority_inventory_sha256: str
    monthly_rank_bar_session_inventory_sha256: str
    session_authority_receipt_sha256: str
    acquisition_semantic_receipt_sha256: str
    monthly_rank_semantic_receipt_sha256: str
    monthly_rank_bar_semantic_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_acquisition_audit_receipt_sha256: str
    monthly_rank_audit_receipt_sha256: str
    monthly_rank_bar_audit_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    source_transport_qualified: bool
    rank_bar_data_qualified: bool
    calendar_geometry_complete: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "data_freeze_at_ms": self.data_freeze_at_ms,
            "source_archive_start_session_date": (
                self.source_archive_start_session_date
            ),
            "source_archive_end_session_date": self.source_archive_end_session_date,
            "earliest_feature_base_session_date": (
                self.earliest_feature_base_session_date
            ),
            "earliest_eligible_decision_session_date": (
                self.earliest_eligible_decision_session_date
            ),
            "latest_h63_mature_decision_session_date": (
                self.latest_h63_mature_decision_session_date
            ),
            "latest_h63_endpoint_session_date": (self.latest_h63_endpoint_session_date),
            "fixed_candidate_session_dates": self.fixed_candidate_session_dates,
            "fixed_outer_test_session_inventories": (
                self.fixed_outer_test_session_inventories
            ),
            "fixed_lockbox_session_dates": self.fixed_lockbox_session_dates,
            "source_sessions": tuple(asdict(row) for row in self.source_sessions),
            "monthly_ranks": tuple(asdict(row) for row in self.monthly_ranks),
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "captured_listing_inventory_sha256": (
                self.captured_listing_inventory_sha256
            ),
            "authenticated_download_inventory_sha256": (
                self.authenticated_download_inventory_sha256
            ),
            "monthly_rank_authority_inventory_sha256": (
                self.monthly_rank_authority_inventory_sha256
            ),
            "monthly_rank_bar_session_inventory_sha256": (
                self.monthly_rank_bar_session_inventory_sha256
            ),
            "session_authority_receipt_sha256": (self.session_authority_receipt_sha256),
            "acquisition_semantic_receipt_sha256": (
                self.acquisition_semantic_receipt_sha256
            ),
            "monthly_rank_semantic_receipt_sha256": (
                self.monthly_rank_semantic_receipt_sha256
            ),
            "monthly_rank_bar_semantic_receipt_sha256": (
                self.monthly_rank_bar_semantic_receipt_sha256
            ),
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "calendar_geometry_complete": self.calendar_geometry_complete,
            "panel_materialization_authorized": self.panel_materialization_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
            "profitability_reporting_authorized": self.profitability_reporting_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze identity differs"
            )
        _nonnegative_int("data freeze", self.data_freeze_at_ms)
        dates = tuple(
            _canonical_date(name, getattr(self, name))
            for name in (
                "source_archive_start_session_date",
                "source_archive_end_session_date",
                "earliest_feature_base_session_date",
                "earliest_eligible_decision_session_date",
                "latest_h63_mature_decision_session_date",
                "latest_h63_endpoint_session_date",
            )
        )
        if (
            dates[0] > dates[1]
            or dates[2] > dates[3]
            or dates[3] > dates[4]
            or dates[4] > dates[5]
            or not self.fixed_candidate_session_dates
            or self.fixed_candidate_session_dates
            != tuple(sorted(set(self.fixed_candidate_session_dates)))
            or self.fixed_candidate_session_dates[0] != dates[3]
            or self.fixed_candidate_session_dates[-1] != dates[4]
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze chronology differs"
            )
        if len(self.fixed_candidate_session_dates) < _MINIMUM_CANDIDATES:
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze lacks the frozen minimum candidate history"
            )
        if (
            len(self.fixed_outer_test_session_inventories) != _OUTER_COUNT
            or any(
                len(inventory) != _OUTER_SESSIONS
                for inventory in self.fixed_outer_test_session_inventories
            )
            or len(self.fixed_lockbox_session_dates) != _LOCKBOX_SESSIONS
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze confirmation or lockbox geometry differs"
            )
        expected_tail = (
            tuple(
                value
                for inventory in self.fixed_outer_test_session_inventories
                for value in inventory
            )
            + self.fixed_lockbox_session_dates
        )
        if expected_tail != self.fixed_candidate_session_dates[-len(expected_tail) :]:
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive phase dates are not the fixed candidate tail"
            )
        source_dates = tuple(row.source_session_date for row in self.source_sessions)
        rank_months = tuple(row.calendar_month for row in self.monthly_ranks)
        if (
            not source_dates
            or source_dates != tuple(sorted(set(source_dates)))
            or source_dates[0] != dates[0]
            or source_dates[-1] != dates[1]
            or not rank_months
            or rank_months != tuple(sorted(set(rank_months)))
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive source or rank inventories differ"
            )
        for source_row in self.source_sessions:
            source_row.validate()
        for rank_row in self.monthly_ranks:
            rank_row.validate()
        if (
            not isinstance(self.source_transport_qualified, bool)
            or not isinstance(self.rank_bar_data_qualified, bool)
            or self.rank_bar_data_qualified
            and not self.source_transport_qualified
            or self.calendar_geometry_complete is not True
            or any(
                (
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive qualification or authorization state differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze source transaction differs"
            )
        for name in (
            "candidate_inventory_sha256",
            "captured_listing_inventory_sha256",
            "authenticated_download_inventory_sha256",
            "monthly_rank_authority_inventory_sha256",
            "monthly_rank_bar_session_inventory_sha256",
            "session_authority_receipt_sha256",
            "acquisition_semantic_receipt_sha256",
            "monthly_rank_semantic_receipt_sha256",
            "monthly_rank_bar_semantic_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "source_acquisition_audit_receipt_sha256",
            "monthly_rank_audit_receipt_sha256",
            "monthly_rank_bar_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.candidate_inventory_sha256 != semantic_sha256(
            self.fixed_candidate_session_dates
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive candidate inventory differs"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "source_acquisition_audit_receipt_sha256": (
                    self.source_acquisition_audit_receipt_sha256
                ),
                "monthly_rank_audit_receipt_sha256": (
                    self.monthly_rank_audit_receipt_sha256
                ),
                "monthly_rank_bar_audit_receipt_sha256": (
                    self.monthly_rank_bar_audit_receipt_sha256
                ),
                "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            }
        ):
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive freeze audit receipt differs"
            )


def _source_rows_from_acquisition(
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
) -> tuple[MassiveProfitabilityArchiveSourceSessionV1, ...]:
    listings = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    output = []
    for download in acquisition.authenticated_downloads:
        captured = listings.get(download.listing_acquisition_receipt_sha256)
        if captured is None:
            raise MassiveProfitabilityArchiveFreezeV1Error(
                "archive download is absent from its listing capture"
            )
        entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        body: dict[str, object] = {
            "source_session_date": coverage_session_from_massive_trade_key(
                download.source_object_key
            ),
            "source_object_key": download.source_object_key,
            "vendor_last_modified_at_ms": entry.vendor_last_modified_at_ms,
            "authenticated_get_completed_at_ms": download.completed_at_ms,
            "listing_entry_receipt_sha256": entry.receipt_sha256,
            "authenticated_download_receipt_sha256": download.receipt_sha256,
            "loaded_source_receipt_sha256": download.loaded_source.receipt_sha256,
        }
        row = MassiveProfitabilityArchiveSourceSessionV1(
            source_session_date=cast(str, body["source_session_date"]),
            source_object_key=cast(str, body["source_object_key"]),
            vendor_last_modified_at_ms=cast(int, body["vendor_last_modified_at_ms"]),
            authenticated_get_completed_at_ms=cast(
                int, body["authenticated_get_completed_at_ms"]
            ),
            listing_entry_receipt_sha256=cast(
                str, body["listing_entry_receipt_sha256"]
            ),
            authenticated_download_receipt_sha256=cast(
                str, body["authenticated_download_receipt_sha256"]
            ),
            loaded_source_receipt_sha256=cast(
                str, body["loaded_source_receipt_sha256"]
            ),
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        output.append(row)
    ordered = tuple(sorted(output, key=lambda row: row.source_session_date))
    if tuple(row.source_session_date for row in ordered) != tuple(
        sorted({row.source_session_date for row in ordered})
    ):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive acquisition duplicates source sessions"
        )
    return ordered


def _rank_rows_from_authority(
    authority: MassiveMonthlyRankInputAuthorityV2,
) -> tuple[MassiveProfitabilityArchiveMonthlyRankV1, ...]:
    authority.validate()
    output = []
    for group in authority.groups:
        body: dict[str, object] = {
            "calendar_month": group.calendar_month,
            "scheduled_rebalance_session_date": (
                group.scheduled_rebalance_session_date
            ),
            "activated_at_ms": group.scheduled_effective_at_ms,
            "maximum_input_available_at_ms": (group.maximum_vendor_available_at_ms),
            "rank_input_group_receipt_sha256": (group.rank_input_group_receipt_sha256),
            "daily_bar_inventory_sha256": group.daily_bar_inventory_sha256,
            "authenticated_source_inventory_sha256": (
                group.authenticated_source_inventory_sha256
            ),
        }
        row = MassiveProfitabilityArchiveMonthlyRankV1(
            calendar_month=cast(str, body["calendar_month"]),
            scheduled_rebalance_session_date=cast(
                str, body["scheduled_rebalance_session_date"]
            ),
            activated_at_ms=cast(int, body["activated_at_ms"]),
            maximum_input_available_at_ms=cast(
                int, body["maximum_input_available_at_ms"]
            ),
            rank_input_group_receipt_sha256=cast(
                str, body["rank_input_group_receipt_sha256"]
            ),
            daily_bar_inventory_sha256=cast(str, body["daily_bar_inventory_sha256"]),
            authenticated_source_inventory_sha256=cast(
                str, body["authenticated_source_inventory_sha256"]
            ),
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        output.append(row)
    return tuple(output)


def _calendar_components(
    *,
    session_authority: MassiveSessionAuthority,
    source_rows: Sequence[MassiveProfitabilityArchiveSourceSessionV1],
    rank_rows: Sequence[MassiveProfitabilityArchiveMonthlyRankV1],
    data_freeze_at_ms: int,
) -> dict[str, object]:
    session_authority.validate()
    freeze_at = _nonnegative_int("data freeze", data_freeze_at_ms)
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    dates = tuple(row.session_date for row in sessions)
    positions = {value: index for index, value in enumerate(dates)}
    ordered_sources = tuple(
        sorted(source_rows, key=lambda row: row.source_session_date)
    )
    for row in ordered_sources:
        row.validate()
    source_dates = tuple(row.source_session_date for row in ordered_sources)
    if (
        not source_dates
        or source_dates[0] not in positions
        or source_dates[-1] not in positions
    ):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive source range is absent from the session authority"
        )
    start_position = positions[source_dates[0]]
    end_position = positions[source_dates[-1]]
    expected_source_dates = dates[start_position : end_position + 1]
    if source_dates != expected_source_dates:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive source inventory is not complete for every XNYS session"
        )
    if (
        max(row.authenticated_get_completed_at_ms for row in ordered_sources)
        > freeze_at
    ):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "data freeze predates an authenticated archive download"
        )
    source_by_date = {row.source_session_date: row for row in ordered_sources}

    by_month: defaultdict[str, list[tuple[int, MassiveExchangeSession]]] = defaultdict(
        list
    )
    for index, session in enumerate(sessions):
        by_month[session.session_date[:7]].append((index, session))
    expected_rank_months = tuple(
        month
        for month in sorted(by_month)
        if (
            start_position
            <= min(by_month[month], key=lambda value: value[1].session_date)[0]
            - _PROTOCOL.universe_rule.ranking_lag_sessions
            - _PROTOCOL.universe_rule.ranking_lookback_sessions
            + 1
            and min(by_month[month], key=lambda value: value[1].session_date)[0]
            <= end_position
        )
    )
    ordered_ranks = tuple(sorted(rank_rows, key=lambda row: row.calendar_month))
    for rank_row in ordered_ranks:
        rank_row.validate()
    if tuple(row.calendar_month for row in ordered_ranks) != expected_rank_months:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "monthly rank authority does not cover every archive-supported month"
        )
    rank_by_month = {row.calendar_month: row for row in ordered_ranks}

    earliest_position: int | None = None
    for index, decision_session in enumerate(sessions):
        source_index = index - 2
        feature_base_index = source_index - _LOOKBACK
        if (
            feature_base_index < start_position
            or source_index < start_position
            or decision_session.session_date[:7] not in rank_by_month
            or decision_session.session_date not in source_by_date
        ):
            continue
        source = source_by_date[sessions[source_index].session_date]
        decision_at = _local_ms(decision_session.session_date, time(12, 30))
        fill_start = _local_ms(decision_session.session_date, time(15, 50))
        fill_end = _local_ms(decision_session.session_date, time(16, 0))
        if not (
            _session_ms(decision_session.regular_open_ns)
            <= decision_at
            < fill_start
            < fill_end
            <= _session_ms(decision_session.regular_close_ns)
        ):
            continue
        if (
            source.vendor_last_modified_at_ms
            < _session_ms(sessions[source_index].regular_close_ns)
            or decision_at - source.vendor_last_modified_at_ms < _MINIMUM_VENDOR_LEAD_MS
        ):
            continue
        earliest_position = index
        break
    if earliest_position is None:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive has no calendar-supported first P0 decision"
        )

    latest_position: int | None = None
    for index in range(end_position - _MAX_HORIZON, earliest_position - 1, -1):
        decision_session = sessions[index]
        endpoint_index = index + _MAX_HORIZON
        if (
            endpoint_index > end_position
            or _session_ms(sessions[endpoint_index].regular_close_ns) > freeze_at
            or decision_session.session_date[:7] not in rank_by_month
        ):
            continue
        fill_start = _local_ms(decision_session.session_date, time(15, 50))
        fill_end = _local_ms(decision_session.session_date, time(16, 0))
        if not (
            _session_ms(decision_session.regular_open_ns)
            <= _local_ms(decision_session.session_date, time(12, 30))
            < fill_start
            < fill_end
            <= _session_ms(decision_session.regular_close_ns)
        ):
            continue
        latest_position = index
        break
    if latest_position is None or latest_position < earliest_position:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive has no H63-mature P0 decision interval"
        )
    candidates = tuple(
        session.session_date
        for session in sessions[earliest_position : latest_position + 1]
        if _local_ms(session.session_date, time(16, 0))
        <= _session_ms(session.regular_close_ns)
    )
    if len(candidates) < _MINIMUM_CANDIDATES:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive lacks the frozen minimum candidate-date history"
        )
    lockbox = candidates[-_LOCKBOX_SESSIONS:]
    outer_flat = candidates[
        -(_LOCKBOX_SESSIONS + _OUTER_COUNT * _OUTER_SESSIONS) : -_LOCKBOX_SESSIONS
    ]
    outer = tuple(
        outer_flat[index * _OUTER_SESSIONS : (index + 1) * _OUTER_SESSIONS]
        for index in range(_OUTER_COUNT)
    )
    earliest_source_index = positions[candidates[0]] - 2
    latest_endpoint_index = positions[candidates[-1]] + _MAX_HORIZON
    return {
        "source_archive_start_session_date": source_dates[0],
        "source_archive_end_session_date": source_dates[-1],
        "earliest_feature_base_session_date": sessions[
            earliest_source_index - _LOOKBACK
        ].session_date,
        "earliest_eligible_decision_session_date": candidates[0],
        "latest_h63_mature_decision_session_date": candidates[-1],
        "latest_h63_endpoint_session_date": sessions[
            latest_endpoint_index
        ].session_date,
        "fixed_candidate_session_dates": candidates,
        "fixed_outer_test_session_inventories": outer,
        "fixed_lockbox_session_dates": lockbox,
        "source_sessions": ordered_sources,
        "monthly_ranks": ordered_ranks,
    }


def _semantic_payload(
    *,
    data_freeze_at_ms: int,
    components: Mapping[str, object],
    captured_listing_inventory_sha256: str,
    authenticated_download_inventory_sha256: str,
    monthly_rank_authority_inventory_sha256: str,
    monthly_rank_bar_session_inventory_sha256: str,
    session_authority_receipt_sha256: str,
    acquisition_semantic_receipt_sha256: str,
    monthly_rank_semantic_receipt_sha256: str,
    monthly_rank_bar_semantic_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
        "data_freeze_at_ms": data_freeze_at_ms,
        **{
            key: value
            for key, value in components.items()
            if key not in {"source_sessions", "monthly_ranks"}
        },
        "source_sessions": tuple(
            asdict(row)
            for row in cast(
                Sequence[MassiveProfitabilityArchiveSourceSessionV1],
                components["source_sessions"],
            )
        ),
        "monthly_ranks": tuple(
            asdict(row)
            for row in cast(
                Sequence[MassiveProfitabilityArchiveMonthlyRankV1],
                components["monthly_ranks"],
            )
        ),
        "candidate_inventory_sha256": semantic_sha256(
            components["fixed_candidate_session_dates"]
        ),
        "captured_listing_inventory_sha256": captured_listing_inventory_sha256,
        "authenticated_download_inventory_sha256": (
            authenticated_download_inventory_sha256
        ),
        "monthly_rank_authority_inventory_sha256": (
            monthly_rank_authority_inventory_sha256
        ),
        "monthly_rank_bar_session_inventory_sha256": (
            monthly_rank_bar_session_inventory_sha256
        ),
        "session_authority_receipt_sha256": session_authority_receipt_sha256,
        "acquisition_semantic_receipt_sha256": (acquisition_semantic_receipt_sha256),
        "monthly_rank_semantic_receipt_sha256": monthly_rank_semantic_receipt_sha256,
        "monthly_rank_bar_semantic_receipt_sha256": (
            monthly_rank_bar_semantic_receipt_sha256
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SHA256
        ),
        "calendar_geometry_complete": True,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }


def _materialize(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    source_rows: Sequence[MassiveProfitabilityArchiveSourceSessionV1],
    rank_rows: Sequence[MassiveProfitabilityArchiveMonthlyRankV1],
    data_freeze_at_ms: int,
    captured_listing_inventory_sha256: str,
    authenticated_download_inventory_sha256: str,
    monthly_rank_authority_inventory_sha256: str,
    monthly_rank_bar_session_inventory_sha256: str,
    acquisition_semantic_receipt_sha256: str,
    monthly_rank_semantic_receipt_sha256: str,
    monthly_rank_bar_semantic_receipt_sha256: str,
    source_acquisition_audit_receipt_sha256: str,
    monthly_rank_audit_receipt_sha256: str,
    monthly_rank_bar_audit_receipt_sha256: str,
    source_transport_qualified: bool,
    rank_bar_data_qualified: bool,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityArchiveFreezeV1:
    if committed_at_ms < data_freeze_at_ms:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive artifact cannot be committed before its data freeze"
        )
    components = _calendar_components(
        session_authority=session_authority,
        source_rows=source_rows,
        rank_rows=rank_rows,
        data_freeze_at_ms=data_freeze_at_ms,
    )
    semantic = _semantic_payload(
        data_freeze_at_ms=data_freeze_at_ms,
        components=components,
        captured_listing_inventory_sha256=captured_listing_inventory_sha256,
        authenticated_download_inventory_sha256=(
            authenticated_download_inventory_sha256
        ),
        monthly_rank_authority_inventory_sha256=(
            monthly_rank_authority_inventory_sha256
        ),
        monthly_rank_bar_session_inventory_sha256=(
            monthly_rank_bar_session_inventory_sha256
        ),
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        acquisition_semantic_receipt_sha256=acquisition_semantic_receipt_sha256,
        monthly_rank_semantic_receipt_sha256=monthly_rank_semantic_receipt_sha256,
        monthly_rank_bar_semantic_receipt_sha256=(
            monthly_rank_bar_semantic_receipt_sha256
        ),
    )
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "semantic_receipt_sha256": semantic_receipt,
        "source_acquisition_audit_receipt_sha256": (
            source_acquisition_audit_receipt_sha256
        ),
        "monthly_rank_audit_receipt_sha256": monthly_rank_audit_receipt_sha256,
        "monthly_rank_bar_audit_receipt_sha256": (
            monthly_rank_bar_audit_receipt_sha256
        ),
    }
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/archive-freeze-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "archive freeze entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-ARCHIVE-FREEZE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_archive_freeze_v1(
        root=root, loaded_source=loaded
    )
    result = replace(
        parsed,
        source_transport_qualified=source_transport_qualified,
        rank_bar_data_qualified=rank_bar_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_profitability_archive_freeze_v1(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    monthly_rank_authority: MassiveMonthlyRankInputAuthorityV2,
    monthly_rank_bar_authority: MassiveMonthlyRankBarAuthorityV1,
    monthly_rank_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    monthly_rank_bar_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    data_freeze_at_ms: int,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityArchiveFreezeV1:
    """Freeze the complete production-acquired archive without caller dates."""

    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    monthly_rank_authority.validate()
    monthly_rank_bar_authority.validate()
    monthly_rank_artifact.validate()
    monthly_rank_bar_artifact.validate()
    if (
        not monthly_rank_bar_authority.source_transport_qualified
        or not monthly_rank_bar_authority.rank_bar_data_qualified
        or monthly_rank_authority.acquisition_receipt_sha256
        != acquisition.receipt_sha256
        or monthly_rank_bar_authority.production_acquisition_receipt_sha256
        != acquisition.receipt_sha256
        or monthly_rank_bar_authority.rank_input_authority_semantic_receipt_sha256
        != monthly_rank_authority.semantic_receipt_sha256
        or monthly_rank_bar_authority.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or monthly_rank_artifact.component_id != "monthly-rank-input-v2"
        or monthly_rank_artifact.authority_semantic_receipt_sha256
        != monthly_rank_authority.semantic_receipt_sha256
        or monthly_rank_bar_artifact.component_id != "monthly-rank-bar-v1"
        or monthly_rank_bar_artifact.authority_semantic_receipt_sha256
        != monthly_rank_bar_authority.semantic_receipt_sha256
        or not monthly_rank_artifact.runtime_qualified
        or not monthly_rank_bar_artifact.runtime_qualified
    ):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze requires acquired monthly rank inputs"
        )
    source_rows = _source_rows_from_acquisition(acquisition)
    rank_rows = _rank_rows_from_authority(monthly_rank_authority)
    return _materialize(
        root=root,
        session_authority=session_authority,
        source_rows=source_rows,
        rank_rows=rank_rows,
        data_freeze_at_ms=data_freeze_at_ms,
        captured_listing_inventory_sha256=acquisition.listing_inventory_sha256,
        authenticated_download_inventory_sha256=acquisition.download_inventory_sha256,
        monthly_rank_authority_inventory_sha256=semantic_sha256(
            tuple((row.calendar_month, row.receipt_sha256) for row in rank_rows)
        ),
        monthly_rank_bar_session_inventory_sha256=(
            monthly_rank_bar_authority.session_inventory_sha256
        ),
        acquisition_semantic_receipt_sha256=acquisition.receipt_sha256,
        monthly_rank_semantic_receipt_sha256=(
            monthly_rank_authority.semantic_receipt_sha256
        ),
        monthly_rank_bar_semantic_receipt_sha256=(
            monthly_rank_bar_authority.semantic_receipt_sha256
        ),
        source_acquisition_audit_receipt_sha256=semantic_sha256(
            (
                acquisition.receipt_sha256,
                acquisition.fixed_runtime_capture_receipt_sha256,
            )
        ),
        monthly_rank_audit_receipt_sha256=monthly_rank_authority.audit_receipt_sha256,
        monthly_rank_bar_audit_receipt_sha256=(
            monthly_rank_bar_authority.audit_receipt_sha256
        ),
        source_transport_qualified=True,
        rank_bar_data_qualified=True,
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )


def materialize_massive_profitability_archive_freeze_for_test_v1(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    source_rows: Sequence[MassiveProfitabilityArchiveSourceSessionV1],
    rank_rows: Sequence[MassiveProfitabilityArchiveMonthlyRankV1],
    data_freeze_at_ms: int,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityArchiveFreezeV1:
    """Publish deterministic geometry canaries; never qualify acquisition."""

    return _materialize(
        root=root,
        session_authority=session_authority,
        source_rows=source_rows,
        rank_rows=rank_rows,
        data_freeze_at_ms=data_freeze_at_ms,
        captured_listing_inventory_sha256=semantic_sha256("test-listing-inventory"),
        authenticated_download_inventory_sha256=semantic_sha256(
            "test-download-inventory"
        ),
        monthly_rank_authority_inventory_sha256=semantic_sha256(
            tuple((row.calendar_month, row.receipt_sha256) for row in rank_rows)
        ),
        monthly_rank_bar_session_inventory_sha256=semantic_sha256(
            tuple((row.calendar_month, row.receipt_sha256) for row in rank_rows)
        ),
        acquisition_semantic_receipt_sha256=semantic_sha256("test-acquisition"),
        monthly_rank_semantic_receipt_sha256=semantic_sha256(
            "test-monthly-rank-authority"
        ),
        monthly_rank_bar_semantic_receipt_sha256=semantic_sha256(
            "test-monthly-rank-bar-authority"
        ),
        source_acquisition_audit_receipt_sha256=semantic_sha256(
            "test-acquisition-audit"
        ),
        monthly_rank_audit_receipt_sha256=semantic_sha256("test-rank-audit"),
        monthly_rank_bar_audit_receipt_sha256=semantic_sha256("test-rank-bar-audit"),
        source_transport_qualified=False,
        rank_bar_data_qualified=False,
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )


def parse_massive_profitability_archive_freeze_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityArchiveFreezeV1:
    """Reload exact freeze bytes; generic reload never qualifies acquisition."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze source is not canonical JSON"
        )
    expected_fields = {
        "schema",
        "data_freeze_at_ms",
        "source_archive_start_session_date",
        "source_archive_end_session_date",
        "earliest_feature_base_session_date",
        "earliest_eligible_decision_session_date",
        "latest_h63_mature_decision_session_date",
        "latest_h63_endpoint_session_date",
        "fixed_candidate_session_dates",
        "fixed_outer_test_session_inventories",
        "fixed_lockbox_session_dates",
        "source_sessions",
        "monthly_ranks",
        "candidate_inventory_sha256",
        "captured_listing_inventory_sha256",
        "authenticated_download_inventory_sha256",
        "monthly_rank_authority_inventory_sha256",
        "monthly_rank_bar_session_inventory_sha256",
        "session_authority_receipt_sha256",
        "acquisition_semantic_receipt_sha256",
        "monthly_rank_semantic_receipt_sha256",
        "monthly_rank_bar_semantic_receipt_sha256",
        "protocol_receipt_sha256",
        "specification_sha256",
        "implementation_source_sha256",
        "calendar_geometry_complete",
        "panel_materialization_authorized",
        "predictive_training_authorized",
        "profitability_reporting_authorized",
        "lockbox_access_authorized",
        "semantic_receipt_sha256",
        "source_acquisition_audit_receipt_sha256",
        "monthly_rank_audit_receipt_sha256",
        "monthly_rank_bar_audit_receipt_sha256",
    }
    if set(payload) != expected_fields:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze field inventory differs"
        )
    try:
        source_rows = tuple(
            MassiveProfitabilityArchiveSourceSessionV1(**row)
            for row in payload["source_sessions"]
        )
        rank_rows = tuple(
            MassiveProfitabilityArchiveMonthlyRankV1(**row)
            for row in payload["monthly_ranks"]
        )
        semantic_values = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "semantic_receipt_sha256",
                "source_acquisition_audit_receipt_sha256",
                "monthly_rank_audit_receipt_sha256",
                "monthly_rank_bar_audit_receipt_sha256",
            }
        }
        semantic_values["source_sessions"] = source_rows
        semantic_values["monthly_ranks"] = rank_rows
        semantic_values["fixed_candidate_session_dates"] = tuple(
            payload["fixed_candidate_session_dates"]
        )
        semantic_values["fixed_outer_test_session_inventories"] = tuple(
            tuple(row) for row in payload["fixed_outer_test_session_inventories"]
        )
        semantic_values["fixed_lockbox_session_dates"] = tuple(
            payload["fixed_lockbox_session_dates"]
        )
        audit_receipt = semantic_sha256(
            {
                "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                "source_acquisition_audit_receipt_sha256": payload[
                    "source_acquisition_audit_receipt_sha256"
                ],
                "monthly_rank_audit_receipt_sha256": payload[
                    "monthly_rank_audit_receipt_sha256"
                ],
                "monthly_rank_bar_audit_receipt_sha256": payload[
                    "monthly_rank_bar_audit_receipt_sha256"
                ],
                "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            }
        )
        result = MassiveProfitabilityArchiveFreezeV1(
            **semantic_values,  # type: ignore[arg-type]
            semantic_receipt_sha256=payload["semantic_receipt_sha256"],
            source_acquisition_audit_receipt_sha256=payload[
                "source_acquisition_audit_receipt_sha256"
            ],
            monthly_rank_audit_receipt_sha256=payload[
                "monthly_rank_audit_receipt_sha256"
            ],
            monthly_rank_bar_audit_receipt_sha256=payload[
                "monthly_rank_bar_audit_receipt_sha256"
            ],
            audit_receipt_sha256=audit_receipt,
            loaded_source=loaded_source,
            source_transport_qualified=False,
            rank_bar_data_qualified=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze values are malformed"
        ) from exc
    if raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityArchiveFreezeV1Error(
            "archive freeze regenerated bytes differ"
        )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_DATASET",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SPEC_SHA256",
    "MassiveProfitabilityArchiveFreezeV1",
    "MassiveProfitabilityArchiveFreezeV1Error",
    "MassiveProfitabilityArchiveMonthlyRankV1",
    "MassiveProfitabilityArchiveSourceSessionV1",
    "materialize_massive_profitability_archive_freeze_for_test_v1",
    "materialize_massive_profitability_archive_freeze_v1",
    "parse_massive_profitability_archive_freeze_v1",
]
