"""Prospective runtime capability and historical finalized-source chronology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_listing import (
    MassiveVendorListingEntryV0,
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MassiveAuthenticatedFlatFileDownloadV1,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)
from rl_quant.workflows.massive_production_typed_run_v2 import (
    MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256,
    MassiveProductionTypedRunV2,
)

MASSIVE_TYPED_READINESS_MINIMUM_RUNS_V1 = 20
MASSIVE_TYPED_READINESS_CAPABILITY_V1_AUTHORIZING = False
MASSIVE_TYPED_READINESS_PUBLICATION_SAFETY_MS_V1 = 5 * 60 * 1_000
MASSIVE_TYPED_READINESS_CAPABILITY_V1_SCHEMA = (
    "rl-quant.massive-typed-readiness-capability-v1"
)
MASSIVE_TYPED_READINESS_PANEL_SELECTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "required_runs": MASSIVE_TYPED_READINESS_MINIMUM_RUNS_V1,
        "extremes": (
            "runtime_ms",
            "compressed_bytes",
            "source_rows",
            "ticker_count",
            "correction_event_count",
            "active_event_key_count",
            "security_partition_count",
        ),
        "per_extreme": 5,
        "fill": "production-run-receipt-order",
        "distinct": (
            "session",
            "source-object",
            "scan",
            "persisted-partition",
        ),
    }
)
MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "run_spec": MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256,
        "panel_spec": MASSIVE_TYPED_READINESS_PANEL_SELECTION_V1_SPEC_SHA256,
        "maximum_runtime_ms": 55 * 60 * 1_000,
        "one_execution_environment": True,
        "production-clock-only": True,
    }
)
MASSIVE_HISTORICAL_READINESS_V1_SCHEMA = (
    "rl-quant.massive-historical-finalized-readiness-v1"
)
MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "vendor_available": "committed-listing-last-modified",
        "listing": "independently-reparsed-authenticated-capture",
        "publication_safety_ms": MASSIVE_TYPED_READINESS_PUBLICATION_SAFETY_MS_V1,
        "strategy_ready_upper": "vendor-available+safety+qualified-max-runtime",
        "research_download": "typed-authenticated-get-retained-but-never-used-as-historical-availability",
        "source_selection": "immediately-prior-exchange-session",
        "decision": "12:30:00-America/New_York",
        "panel_materialization_authorized": False,
    }
)
_EASTERN = ZoneInfo("America/New_York")


class MassiveHistoricalReadinessV1Error(ValueError):
    """Runtime or historical readiness evidence differs from the frozen rule."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveHistoricalReadinessV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _workload(run: MassiveProductionTypedRunV2, field: str) -> int:
    engine = run.development_engine_run
    values = {
        "runtime_ms": run.runtime_ms,
        "compressed_bytes": engine.authenticated_download.content_length,
        "source_rows": engine.scan_evidence.source_row_count,
        "ticker_count": engine.scan_evidence.ticker_count,
        "correction_event_count": (
            engine.persisted_partition_manifest.correction_event_count
        ),
        "active_event_key_count": (
            engine.persisted_partition_manifest.active_event_key_count
        ),
        "security_partition_count": len(engine.persisted_partition_manifest.partitions),
    }
    return values[field]


def _select_runs(
    archive_runs: Sequence[MassiveProductionTypedRunV2],
) -> tuple[MassiveProductionTypedRunV2, ...]:
    runs = tuple(archive_runs)
    if len(runs) < MASSIVE_TYPED_READINESS_MINIMUM_RUNS_V1:
        raise MassiveHistoricalReadinessV1Error(
            "typed readiness requires at least 20 production runs"
        )
    if any(not isinstance(run, MassiveProductionTypedRunV2) for run in runs):
        raise MassiveHistoricalReadinessV1Error(
            "typed readiness accepts production-clock runs only"
        )
    selected: dict[str, MassiveProductionTypedRunV2] = {}
    fields = (
        "runtime_ms",
        "compressed_bytes",
        "source_rows",
        "ticker_count",
        "correction_event_count",
        "active_event_key_count",
        "security_partition_count",
    )
    for field in fields:
        ranked = sorted(
            runs,
            key=lambda row: (-_workload(row, field), row.receipt_sha256),
        )
        for run in ranked[:5]:
            selected[run.receipt_sha256] = run
    for run in sorted(runs, key=lambda row: row.receipt_sha256):
        if len(selected) >= MASSIVE_TYPED_READINESS_MINIMUM_RUNS_V1:
            break
        selected[run.receipt_sha256] = run
    return tuple(selected[key] for key in sorted(selected))


@dataclass(frozen=True, slots=True)
class MassiveTypedReadinessCapabilityV1:
    archive_runs: tuple[MassiveProductionTypedRunV2, ...]
    selected_run_receipts: tuple[str, ...]
    execution_environment_receipt_sha256: str
    production_run_spec_receipt_sha256: str
    maximum_runtime_ms: int
    maximum_clock_error_ns: int
    panel_selection_spec_receipt_sha256: str
    capability_spec_receipt_sha256: str
    capability_passed: bool
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_READINESS_CAPABILITY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not MASSIVE_TYPED_READINESS_CAPABILITY_V1_AUTHORIZING
            or
            self.schema != MASSIVE_TYPED_READINESS_CAPABILITY_V1_SCHEMA
            or len(self.archive_runs) < MASSIVE_TYPED_READINESS_MINIMUM_RUNS_V1
            or tuple(run.receipt_sha256 for run in self.archive_runs)
            != tuple(sorted(run.receipt_sha256 for run in self.archive_runs))
            or not self.capability_passed
            or self.maximum_runtime_ms > 55 * 60 * 1_000
            or self.maximum_clock_error_ns < 0
        ):
            raise MassiveHistoricalReadinessV1Error(
                "typed readiness capability identity differs"
            )
        for run in self.archive_runs:
            run.validate()
            if not run.production_timing_qualified:
                raise MassiveHistoricalReadinessV1Error(
                    "synthetic run entered typed readiness capability"
                )
        sessions = tuple(
            run.development_engine_run.source_session_date for run in self.archive_runs
        )
        source_receipts = tuple(
            run.development_engine_run.authenticated_download.loaded_source.receipt.receipt_sha256
            for run in self.archive_runs
        )
        scan_receipts = tuple(
            run.development_engine_run.scan_evidence.receipt_sha256
            for run in self.archive_runs
        )
        partition_receipts = tuple(
            run.development_engine_run.persisted_partition_manifest.receipt_sha256
            for run in self.archive_runs
        )
        if any(
            len(values) != len(set(values))
            for values in (sessions, source_receipts, scan_receipts, partition_receipts)
        ):
            raise MassiveHistoricalReadinessV1Error(
                "typed readiness runs are not distinct"
            )
        environments = {
            run.execution_environment.receipt_sha256 for run in self.archive_runs
        }
        run_specs = {
            run.production_run_spec_receipt_sha256 for run in self.archive_runs
        }
        expected_selected = tuple(
            run.receipt_sha256 for run in _select_runs(self.archive_runs)
        )
        selected = tuple(
            run
            for run in self.archive_runs
            if run.receipt_sha256 in set(self.selected_run_receipts)
        )
        if (
            environments != {self.execution_environment_receipt_sha256}
            or run_specs != {MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256}
            or self.production_run_spec_receipt_sha256
            != MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256
            or self.selected_run_receipts != expected_selected
            or self.maximum_runtime_ms != max(run.runtime_ms for run in selected)
            or self.maximum_clock_error_ns
            != max(run.clock_authority.maximum_clock_error_ns for run in selected)
            or self.panel_selection_spec_receipt_sha256
            != MASSIVE_TYPED_READINESS_PANEL_SELECTION_V1_SPEC_SHA256
            or self.capability_spec_receipt_sha256
            != MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256
        ):
            raise MassiveHistoricalReadinessV1Error(
                "typed readiness capability was not deterministically derived"
            )
        for name in (
            "execution_environment_receipt_sha256",
            "production_run_spec_receipt_sha256",
            "panel_selection_spec_receipt_sha256",
            "capability_spec_receipt_sha256",
            "receipt_sha256",
            *self.selected_run_receipts,
        ):
            _digest("typed readiness receipt", name)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveHistoricalReadinessV1Error(
                "typed readiness capability receipt differs"
            )


def build_massive_typed_readiness_capability_v1(
    archive_runs: Sequence[MassiveProductionTypedRunV2],
) -> MassiveTypedReadinessCapabilityV1:
    if not MASSIVE_TYPED_READINESS_CAPABILITY_V1_AUTHORIZING:
        raise MassiveHistoricalReadinessV1Error(
            "typed readiness v1 is superseded by source-derived runtime authorities"
        )
    if any(not isinstance(run, MassiveProductionTypedRunV2) for run in archive_runs):
        raise MassiveHistoricalReadinessV1Error(
            "typed readiness accepts production-clock runs only"
        )
    runs = tuple(sorted(archive_runs, key=lambda row: row.receipt_sha256))
    selected = _select_runs(runs)
    body = {
        "schema": MASSIVE_TYPED_READINESS_CAPABILITY_V1_SCHEMA,
        "archive_runs": runs,
        "selected_run_receipts": tuple(run.receipt_sha256 for run in selected),
        "execution_environment_receipt_sha256": (
            runs[0].execution_environment.receipt_sha256
        ),
        "production_run_spec_receipt_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256,
        "maximum_runtime_ms": max(run.runtime_ms for run in selected),
        "maximum_clock_error_ns": max(
            run.clock_authority.maximum_clock_error_ns for run in selected
        ),
        "panel_selection_spec_receipt_sha256": (
            MASSIVE_TYPED_READINESS_PANEL_SELECTION_V1_SPEC_SHA256
        ),
        "capability_spec_receipt_sha256": (
            MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256
        ),
        "capability_passed": True,
    }
    result = MassiveTypedReadinessCapabilityV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveHistoricalFinalizedReadinessV1:
    captured_listing: MassiveCapturedFlatFileListingV0
    listing_entry: MassiveVendorListingEntryV0
    research_download: MassiveAuthenticatedFlatFileDownloadV1
    source_session: MassiveExchangeSession
    decision_session: MassiveExchangeSession
    session_authority: MassiveSessionAuthority
    source_session_date: str
    decision_session_date: str
    source_object_key: str
    vendor_available_at_ms: int
    historical_strategy_ready_upper_bound_at_ms: int
    research_downloaded_at_ms: int
    decision_at_ms: int
    qualified_maximum_runtime_ms: int
    listing_entry_receipt_sha256: str
    research_download_receipt_sha256: str
    readiness_capability: MassiveTypedReadinessCapabilityV1
    readiness_capability_receipt_sha256: str
    session_authority_receipt_sha256: str
    readiness_spec_receipt_sha256: str
    historical_availability_qualified: bool
    panel_materialization_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_HISTORICAL_READINESS_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.captured_listing.validate()
        self.listing_entry.validate()
        self.research_download.validate()
        self.source_session.validate()
        self.decision_session.validate()
        self.session_authority.validate()
        self.readiness_capability.validate()
        expected_ready = (
            self.vendor_available_at_ms
            + MASSIVE_TYPED_READINESS_PUBLICATION_SAFETY_MS_V1
            + self.qualified_maximum_runtime_ms
        )
        expected_decision_at_ms = int(
            datetime.combine(
                date.fromisoformat(self.decision_session_date),
                time(12, 30),
                tzinfo=_EASTERN,
            ).timestamp()
            * 1_000
        )
        sessions = tuple(
            row for row in self.session_authority.sessions if row.exchange == "XNYS"
        )
        if (
            self.schema != MASSIVE_HISTORICAL_READINESS_V1_SCHEMA
            or self.source_session_date != self.source_session.session_date
            or self.decision_session_date != self.decision_session.session_date
            or self.source_object_key
            != canonical_massive_trade_object_key(self.source_session_date)
            or self.source_object_key != self.listing_entry.source_object_key
            or self.vendor_available_at_ms
            != self.listing_entry.vendor_last_modified_at_ms
            or self.research_downloaded_at_ms != self.research_download.completed_at_ms
            or self.research_download_receipt_sha256
            != self.research_download.receipt_sha256
            or self.research_download.source_object_key != self.source_object_key
            or self.research_download.listing_acquisition_receipt_sha256
            != self.captured_listing.acquisition_evidence.receipt_sha256
            or self.decision_at_ms != expected_decision_at_ms
            or isinstance(self.research_downloaded_at_ms, bool)
            or not isinstance(self.research_downloaded_at_ms, int)
            or self.research_downloaded_at_ms
            < self.listing_entry.listing_observed_at_ms
            or self.session_authority.resolve(
                exchange="XNYS", session_date=self.source_session_date
            )
            != self.source_session
            or self.session_authority.resolve(
                exchange="XNYS", session_date=self.decision_session_date
            )
            != self.decision_session
            or sessions.index(self.decision_session)
            != sessions.index(self.source_session) + 1
            or self.historical_strategy_ready_upper_bound_at_ms != expected_ready
            or self.qualified_maximum_runtime_ms < 0
            or self.qualified_maximum_runtime_ms > 55 * 60 * 1_000
            or self.qualified_maximum_runtime_ms
            != self.readiness_capability.maximum_runtime_ms
            or self.readiness_capability_receipt_sha256
            != self.readiness_capability.receipt_sha256
            or self.historical_availability_qualified
            is not (expected_ready <= self.decision_at_ms)
            or self.panel_materialization_authorized
        ):
            raise MassiveHistoricalReadinessV1Error(
                "historical finalized readiness differs"
            )
        for name in (
            "listing_entry_receipt_sha256",
            "research_download_receipt_sha256",
            "readiness_capability_receipt_sha256",
            "session_authority_receipt_sha256",
            "readiness_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.listing_entry_receipt_sha256 != self.listing_entry.receipt_sha256
            or self.session_authority_receipt_sha256
            != self.session_authority.receipt_sha256
            or self.readiness_spec_receipt_sha256
            != MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveHistoricalReadinessV1Error(
                "historical finalized readiness receipt differs"
            )


def build_massive_historical_finalized_readiness_v1(
    *,
    captured_listing: MassiveCapturedFlatFileListingV0,
    listing_root: str | Path,
    research_download: MassiveAuthenticatedFlatFileDownloadV1,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    session_authority: MassiveSessionAuthority,
    readiness_capability: MassiveTypedReadinessCapabilityV1,
) -> MassiveHistoricalFinalizedReadinessV1:
    validate_massive_captured_flat_file_listing_v0(
        root=listing_root, captured_listing=captured_listing
    )
    research_download.validate()
    listing_entry = captured_listing.committed_listing.resolve(
        source_object_key=research_download.source_object_key
    )
    listing_entry.validate()
    source_session.validate()
    decision_session.validate()
    session_authority.validate()
    readiness_capability.validate()
    source = session_authority.resolve(
        exchange="XNYS", session_date=source_session.session_date
    )
    decision = session_authority.resolve(
        exchange="XNYS", session_date=decision_session.session_date
    )
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    if (
        source != source_session
        or decision != decision_session
        or sessions.index(decision_session) != sessions.index(source_session) + 1
        or listing_entry.source_object_key
        != canonical_massive_trade_object_key(source_session.session_date)
        or research_download.completed_at_ms < listing_entry.listing_observed_at_ms
        or research_download.listing_acquisition_receipt_sha256
        != captured_listing.acquisition_evidence.receipt_sha256
    ):
        raise MassiveHistoricalReadinessV1Error(
            "historical readiness source chronology differs"
        )
    decision_at_ms = int(
        datetime.combine(
            date.fromisoformat(decision_session.session_date),
            time(12, 30),
            tzinfo=_EASTERN,
        ).timestamp()
        * 1_000
    )
    ready_at_ms = (
        listing_entry.vendor_last_modified_at_ms
        + MASSIVE_TYPED_READINESS_PUBLICATION_SAFETY_MS_V1
        + readiness_capability.maximum_runtime_ms
    )
    body = {
        "schema": MASSIVE_HISTORICAL_READINESS_V1_SCHEMA,
        "captured_listing": captured_listing,
        "listing_entry": listing_entry,
        "research_download": research_download,
        "source_session": source_session,
        "decision_session": decision_session,
        "session_authority": session_authority,
        "source_session_date": source_session.session_date,
        "decision_session_date": decision_session.session_date,
        "source_object_key": listing_entry.source_object_key,
        "vendor_available_at_ms": listing_entry.vendor_last_modified_at_ms,
        "historical_strategy_ready_upper_bound_at_ms": ready_at_ms,
        "research_downloaded_at_ms": research_download.completed_at_ms,
        "decision_at_ms": decision_at_ms,
        "qualified_maximum_runtime_ms": readiness_capability.maximum_runtime_ms,
        "listing_entry_receipt_sha256": listing_entry.receipt_sha256,
        "research_download_receipt_sha256": research_download.receipt_sha256,
        "readiness_capability": readiness_capability,
        "readiness_capability_receipt_sha256": readiness_capability.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "readiness_spec_receipt_sha256": MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256,
        "historical_availability_qualified": ready_at_ms <= decision_at_ms,
        "panel_materialization_authorized": False,
    }
    result = MassiveHistoricalFinalizedReadinessV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256",
    "MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256",
    "MASSIVE_TYPED_READINESS_PANEL_SELECTION_V1_SPEC_SHA256",
    "MassiveHistoricalFinalizedReadinessV1",
    "MassiveHistoricalReadinessV1Error",
    "MassiveTypedReadinessCapabilityV1",
    "build_massive_historical_finalized_readiness_v1",
    "build_massive_typed_readiness_capability_v1",
]
