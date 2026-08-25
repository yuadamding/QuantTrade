"""Authenticated trade-to-bar rederivation for monthly Massive PIT ranks.

The legacy monthly-rank authority proves that a canonical daily-bar artifact is
internally self-consistent.  It does not prove that the values in that artifact
were derived from the authenticated trade object.  This additive generation
closes that gap by rescanning the exact object GET, independently replaying the
persisted correction timeline, and independently reconstructing every bar row.

``source_data_qualified`` on the legacy V2 rank authority is intentionally not
used here.  The profitability gate must consume ``rank_bar_data_qualified``
from this authority instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import TypeVar, cast

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_listing import (
    coverage_session_from_massive_trade_key,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
    validate_massive_persisted_partitions_semantically_v2,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.features.massive_daily_bars_v0 import (
    MassiveDailyBarsArtifactV0,
    MassiveDailyBarsRowV0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveMonthlyRankInputAuthorityV2,
    MassiveProfitabilityProductionAcquisitionV2,
    validate_massive_profitability_production_acquisition_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-monthly-rank-bar-authority-v1"
)
MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source": "exact-authenticated-massive-flat-file-object-get",
        "scan": "whole-file-participant-time-rescan-must-match",
        "partitions": "event-timeline-reparse-and-independent-correction-replay",
        "bars": "independent-condition-authority-rederivation-must-match-v0",
        "rank_sessions": "exact-union-of-v2-monthly-rank-window-sessions",
        "availability": "session-close<=vendor-last-modified<=monthly-activation",
        "common_authorities": (
            "session",
            "identity-routing",
            "condition",
            "correction",
            "event-domain",
        ),
        "legacy-source-data-qualified": "nonauthorizing",
        "profitability-gate-field": "rank_bar_data_qualified",
    }
)

MASSIVE_MONTHLY_RANK_BAR_V1_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_MONTHLY_RANK_BAR_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_MONTHLY_RANK_BAR_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_MONTHLY_RANK_BAR_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveMonthlyRankBarAuthorityV1Error(ValueError):
    """Authenticated rank-session bars do not match their trade partitions."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveMonthlyRankBarAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveMonthlyRankBarAuthorityV1Error(f"{name} must be nonnegative")
    return value


def _session_ms(value_ns: int) -> int:
    value = _nonnegative_int("session timestamp", value_ns)
    if value % 1_000_000:
        raise MassiveMonthlyRankBarAuthorityV1Error(
            "session timestamp is not millisecond aligned"
        )
    return value // 1_000_000


def _independent_bar_row(
    *,
    security_id: str,
    active_rows: Sequence[MassiveExtractedTradeRow],
    condition_authority: MassiveConditionAuthority,
) -> MassiveDailyBarsRowV0:
    """Reconstruct V0 bar semantics without calling its private row builder."""

    rows = tuple(active_rows)
    if not rows:
        raise MassiveMonthlyRankBarAuthorityV1Error(
            "rank bar reconstruction received an empty active partition"
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.canonical_record.participant_timestamp_ns,
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.source_row_number,
            ),
        )
    )
    eligibility = tuple(
        condition_authority.resolve(row.canonical_record.conditions) for row in ordered
    )
    open_close = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[0]
    )
    high_low = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[1]
    )
    volume = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[2]
    )
    open_close_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in open_close
    )
    high_low_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in high_low
    )
    volume_prices = tuple(Decimal(row.canonical_record.price_decimal) for row in volume)
    volume_sizes = tuple(Decimal(row.canonical_record.size_decimal) for row in volume)
    opening = open_close_prices[0] if open_close_prices else Decimal(0)
    closing = open_close_prices[-1] if open_close_prices else Decimal(0)
    high = max(high_low_prices) if high_low_prices else Decimal(0)
    low = min(high_low_prices) if high_low_prices else Decimal(0)
    shares = sum(volume_sizes, Decimal(0))
    dollars = sum(
        (price * size for price, size in zip(volume_prices, volume_sizes, strict=True)),
        Decimal(0),
    )
    combined_price_valid = bool(open_close_prices and high_low_prices)
    high_low_range = (
        Decimal(0)
        if not combined_price_valid or closing == 0
        else (high - low) / closing
    )
    close_location = (
        Decimal(0)
        if not combined_price_valid
        else Decimal("0.5")
        if high == low
        else (closing - low) / (high - low)
    )
    values = tuple(
        float(value)
        for value in (
            opening,
            high,
            low,
            closing,
            shares,
            dollars,
            high_low_range,
            close_location,
        )
    )
    valid = (
        bool(open_close_prices),
        bool(high_low_prices),
        bool(high_low_prices),
        bool(open_close_prices),
        bool(volume),
        bool(volume),
        combined_price_valid and closing != 0,
        combined_price_valid,
    )
    active_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in ordered))
    body: dict[str, object] = {
        "security_id": security_id,
        "values": values,
        "valid": valid,
        "source_active_inventory_sha256": active_inventory,
    }
    result = MassiveDailyBarsRowV0(
        security_id=security_id,
        values=values,
        valid=valid,
        source_active_inventory_sha256=active_inventory,
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveMonthlyRankBarSessionV1:
    source_session_date: str
    session_close_at_ms: int
    earliest_monthly_activation_at_ms: int
    vendor_last_modified_at_ms: int
    authenticated_download_receipt_sha256: str
    whole_file_scan_receipt_sha256: str
    semantic_partition_manifest_receipt_sha256: str
    persisted_partition_manifest_receipt_sha256: str
    daily_bars_artifact_receipt_sha256: str
    committed_bar_row_inventory_sha256: str
    rederived_bar_row_inventory_sha256: str
    rederived_security_count: int
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.source_session_date:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar session date is absent"
            )
        close = _nonnegative_int("rank-bar session close", self.session_close_at_ms)
        activation = _nonnegative_int(
            "rank-bar monthly activation", self.earliest_monthly_activation_at_ms
        )
        vendor = _nonnegative_int(
            "rank-bar vendor availability", self.vendor_last_modified_at_ms
        )
        if not close <= vendor <= activation:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar vendor availability lies outside close and activation"
            )
        if self.rederived_security_count <= 0:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar session contains no reconstructed securities"
            )
        for name in (
            "authenticated_download_receipt_sha256",
            "whole_file_scan_receipt_sha256",
            "semantic_partition_manifest_receipt_sha256",
            "persisted_partition_manifest_receipt_sha256",
            "daily_bars_artifact_receipt_sha256",
            "committed_bar_row_inventory_sha256",
            "rederived_bar_row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.committed_bar_row_inventory_sha256
            != self.rederived_bar_row_inventory_sha256
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "committed and reconstructed rank-bar inventories differ"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar session receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveMonthlyRankBarAuthorityV1:
    rank_input_authority_semantic_receipt_sha256: str
    production_acquisition_receipt_sha256: str
    session_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    event_domain_spec_receipt_sha256: str
    rank_group_inventory_sha256: str
    sessions: tuple[MassiveMonthlyRankBarSessionV1, ...]
    session_inventory_sha256: str
    source_transport_qualified: bool
    rank_bar_data_qualified: bool
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rank_input_authority_semantic_receipt_sha256": (
                self.rank_input_authority_semantic_receipt_sha256
            ),
            "session_authority_receipt_sha256": self.session_authority_receipt_sha256,
            "identity_authority_receipt_sha256": self.identity_authority_receipt_sha256,
            "condition_authority_receipt_sha256": (
                self.condition_authority_receipt_sha256
            ),
            "correction_authority_receipt_sha256": (
                self.correction_authority_receipt_sha256
            ),
            "event_domain_spec_receipt_sha256": (self.event_domain_spec_receipt_sha256),
            "rank_group_inventory_sha256": self.rank_group_inventory_sha256,
            "sessions": tuple(asdict(row) for row in self.sessions),
            "session_inventory_sha256": self.session_inventory_sha256,
            "source_transport_qualified": self.source_transport_qualified,
            "rank_bar_data_qualified": self.rank_bar_data_qualified,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "panel_materialization_authorized": self.panel_materialization_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SCHEMA
            or self.specification_sha256
            != MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SOURCE_SHA256
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar authority identity differs"
            )
        dates = tuple(row.source_session_date for row in self.sessions)
        if not dates or dates != tuple(sorted(set(dates))):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar sessions are not canonical"
            )
        for row in self.sessions:
            row.validate()
        if self.session_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.sessions)
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar session inventory differs"
            )
        if (
            not isinstance(self.source_transport_qualified, bool)
            or not isinstance(self.rank_bar_data_qualified, bool)
            or self.rank_bar_data_qualified
            and not self.source_transport_qualified
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar qualification state differs"
            )
        if any(
            (
                self.panel_materialization_authorized,
                self.predictive_training_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
            )
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar authority cannot authorize profitability work"
            )
        for name in (
            "rank_input_authority_semantic_receipt_sha256",
            "production_acquisition_receipt_sha256",
            "session_authority_receipt_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "event_domain_spec_receipt_sha256",
            "rank_group_inventory_sha256",
            "session_inventory_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "production_acquisition_receipt_sha256": (
                    self.production_acquisition_receipt_sha256
                ),
            }
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank-bar audit receipt differs"
            )


_T = TypeVar("_T")


def _unique_by_date(*, values: Sequence[_T], field: str, name: str) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for value in values:
        session_date = cast(str, getattr(value, field))
        if session_date in result:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                f"{name} duplicates source session {session_date}"
            )
        result[session_date] = value
    return result


def _build_massive_monthly_rank_bar_authority_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    rank_input_authority: MassiveMonthlyRankInputAuthorityV2,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    require_fixed_runtime: bool,
) -> MassiveMonthlyRankBarAuthorityV1:
    session_authority.validate()
    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    rank_input_authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=source_root,
        acquisition=acquisition,
        require_fixed_runtime=require_fixed_runtime,
    )
    if (
        rank_input_authority.acquisition_receipt_sha256 != acquisition.receipt_sha256
        or rank_input_authority.session_authority_receipt_sha256
        != session_authority.receipt_sha256
    ):
        raise MassiveMonthlyRankBarAuthorityV1Error(
            "monthly rank authority and authenticated source authorities differ"
        )

    activations: defaultdict[str, list[int]] = defaultdict(list)
    for group in rank_input_authority.groups:
        for session_date in group.observation_session_dates:
            activations[session_date].append(group.scheduled_effective_at_ms)
    required_dates = tuple(sorted(activations))
    if not required_dates:
        raise MassiveMonthlyRankBarAuthorityV1Error(
            "monthly rank authority has no source sessions"
        )

    # Re-key downloads by the source date encoded in the fixed object key.
    downloads_by_date = {}
    for download in acquisition.authenticated_downloads:
        source_date = coverage_session_from_massive_trade_key(
            download.source_object_key
        )
        if source_date in downloads_by_date:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "authenticated downloads duplicate one source date"
            )
        downloads_by_date[source_date] = download
    scans = _unique_by_date(
        values=scan_evidence,
        field="source_session_date",
        name="whole-file scan",
    )
    semantic_manifests = _unique_by_date(
        values=semantic_partition_manifests,
        field="source_session_date",
        name="semantic partition manifest",
    )
    persisted_manifests = _unique_by_date(
        values=persisted_partition_manifests,
        field="source_session_date",
        name="persisted partition manifest",
    )
    bars_by_date = _unique_by_date(
        values=daily_bars,
        field="source_session_date",
        name="daily bars",
    )
    for name, values in (
        ("authenticated downloads", downloads_by_date),
        ("whole-file scans", scans),
        ("semantic partition manifests", semantic_manifests),
        ("persisted partition manifests", persisted_manifests),
        ("daily bars", bars_by_date),
    ):
        if tuple(sorted(values)) != required_dates:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                f"{name} do not exactly cover monthly rank sessions"
            )

    listing_by_acquisition = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    source_inventory_by_date: dict[str, tuple[int, str, str]] = {}
    for source_date, download in downloads_by_date.items():
        captured = listing_by_acquisition.get(
            download.listing_acquisition_receipt_sha256
        )
        if captured is None:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "authenticated rank source lacks its captured listing"
            )
        entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        source_inventory_by_date[source_date] = (
            entry.vendor_last_modified_at_ms,
            entry.receipt_sha256,
            download.receipt_sha256,
        )
    for group in rank_input_authority.groups:
        if group.daily_bar_inventory_sha256 != semantic_sha256(
            tuple(
                (source_date, bars_by_date[source_date].receipt_sha256)
                for source_date in group.observation_session_dates
            )
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "monthly rank group is not bound to the supplied daily bars"
            )
        if group.authenticated_source_inventory_sha256 != semantic_sha256(
            tuple(
                (source_date, *source_inventory_by_date[source_date])
                for source_date in group.observation_session_dates
            )
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "monthly rank group is not bound to the authenticated sources"
            )
    sessions: list[MassiveMonthlyRankBarSessionV1] = []
    common_event_domain: str | None = None
    for source_date in required_dates:
        download = downloads_by_date[source_date]
        scan = scans[source_date]
        semantic_manifest = semantic_manifests[source_date]
        persisted_manifest = persisted_manifests[source_date]
        bars = bars_by_date[source_date]
        session = session_authority.resolve(exchange="XNYS", session_date=source_date)
        captured = listing_by_acquisition[download.listing_acquisition_receipt_sha256]
        listing_entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        session_close = _session_ms(session.regular_close_ns)
        activation = min(activations[source_date])
        if not (
            session_close <= listing_entry.vendor_last_modified_at_ms <= activation
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank source vendor availability lies outside close and activation"
            )

        rescanned_rows, rescanned = scan_massive_daily_trade_file_v0(
            root=source_root,
            loaded_source=download.loaded_source,
            session_authority=session_authority,
            session=session,
            correction_authority=correction_authority,
        )
        if rescanned != scan:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "whole-file scan was not derived from the authenticated download"
            )
        if (
            scan.loaded_source_receipt_sha256 != download.loaded_source.receipt_sha256
            or scan.source_object_receipt_sha256
            != download.loaded_source.receipt.receipt_sha256
            or scan.source_commit_receipt_sha256
            != download.loaded_source.commit.receipt_sha256
            or len(rescanned_rows) != scan.source_row_count
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "whole-file scan source binding differs"
            )
        if (
            semantic_manifest.source_file_scan_receipt_sha256 != scan.receipt_sha256
            or semantic_manifest.identity_authority_receipt_sha256
            != identity_authority.receipt_sha256
            or semantic_manifest.condition_authority_receipt_sha256
            != condition_authority.receipt_sha256
            or semantic_manifest.correction_authority_receipt_sha256
            != correction_authority.receipt_sha256
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank semantic partition authorities differ"
            )
        event_domain = semantic_manifest.feature_domain_spec_receipt_sha256
        if common_event_domain is None:
            common_event_domain = event_domain
        elif common_event_domain != event_domain:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "rank sessions use different event-domain authorities"
            )
        validate_massive_persisted_partitions_semantically_v2(
            root=persisted_root,
            manifest=persisted_manifest,
            scan_evidence=scan,
            semantic_partition_manifest=semantic_manifest,
            identity_authority=identity_authority,
            correction_authority=correction_authority,
        )
        validate_massive_daily_bars_v0(root=daily_bars_root, artifact=bars)
        if (
            bars.persisted_partition_manifest_receipt_sha256
            != persisted_manifest.receipt_sha256
            or bars.condition_authority_receipt_sha256
            != condition_authority.receipt_sha256
        ):
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "daily bars are not bound to the qualified rank partitions"
            )
        rederived_rows = []
        for partition in persisted_manifest.partitions:
            _, active_rows, _ = load_massive_persisted_security_rows_v2(
                root=persisted_root, partition=partition
            )
            if active_rows:
                rederived_rows.append(
                    _independent_bar_row(
                        security_id=partition.security_id,
                        active_rows=active_rows,
                        condition_authority=condition_authority,
                    )
                )
        ordered_rows = tuple(sorted(rederived_rows, key=lambda row: row.security_id))
        if ordered_rows != bars.rows:
            raise MassiveMonthlyRankBarAuthorityV1Error(
                "daily bars differ from authenticated partition rederivation"
            )
        rederived_inventory = semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered_rows)
        )
        body: dict[str, object] = {
            "source_session_date": source_date,
            "session_close_at_ms": session_close,
            "earliest_monthly_activation_at_ms": activation,
            "vendor_last_modified_at_ms": listing_entry.vendor_last_modified_at_ms,
            "authenticated_download_receipt_sha256": download.receipt_sha256,
            "whole_file_scan_receipt_sha256": scan.receipt_sha256,
            "semantic_partition_manifest_receipt_sha256": (
                semantic_manifest.receipt_sha256
            ),
            "persisted_partition_manifest_receipt_sha256": (
                persisted_manifest.receipt_sha256
            ),
            "daily_bars_artifact_receipt_sha256": bars.receipt_sha256,
            "committed_bar_row_inventory_sha256": bars.row_inventory_sha256,
            "rederived_bar_row_inventory_sha256": rederived_inventory,
            "rederived_security_count": len(ordered_rows),
        }
        row = MassiveMonthlyRankBarSessionV1(
            source_session_date=source_date,
            session_close_at_ms=session_close,
            earliest_monthly_activation_at_ms=activation,
            vendor_last_modified_at_ms=listing_entry.vendor_last_modified_at_ms,
            authenticated_download_receipt_sha256=download.receipt_sha256,
            whole_file_scan_receipt_sha256=scan.receipt_sha256,
            semantic_partition_manifest_receipt_sha256=(
                semantic_manifest.receipt_sha256
            ),
            persisted_partition_manifest_receipt_sha256=(
                persisted_manifest.receipt_sha256
            ),
            daily_bars_artifact_receipt_sha256=bars.receipt_sha256,
            committed_bar_row_inventory_sha256=bars.row_inventory_sha256,
            rederived_bar_row_inventory_sha256=rederived_inventory,
            rederived_security_count=len(ordered_rows),
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        sessions.append(row)
    assert common_event_domain is not None
    ordered_sessions = tuple(sessions)
    source_transport_qualified = (
        require_fixed_runtime and acquisition.fixed_runtime_captured
    )
    rank_bar_data_qualified = source_transport_qualified
    semantic_body: dict[str, object] = {
        "schema": MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SCHEMA,
        "rank_input_authority_semantic_receipt_sha256": (
            rank_input_authority.semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "event_domain_spec_receipt_sha256": common_event_domain,
        "rank_group_inventory_sha256": semantic_sha256(
            tuple(group.receipt_sha256 for group in rank_input_authority.groups)
        ),
        "sessions": tuple(asdict(row) for row in ordered_sessions),
        "session_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered_sessions)
        ),
        "source_transport_qualified": source_transport_qualified,
        "rank_bar_data_qualified": rank_bar_data_qualified,
        "specification_sha256": MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SOURCE_SHA256
        ),
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic_body)
    result = MassiveMonthlyRankBarAuthorityV1(
        rank_input_authority_semantic_receipt_sha256=(
            rank_input_authority.semantic_receipt_sha256
        ),
        production_acquisition_receipt_sha256=acquisition.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        correction_authority_receipt_sha256=correction_authority.receipt_sha256,
        event_domain_spec_receipt_sha256=common_event_domain,
        rank_group_inventory_sha256=semantic_body["rank_group_inventory_sha256"],  # type: ignore[arg-type]
        sessions=ordered_sessions,
        session_inventory_sha256=semantic_body["session_inventory_sha256"],  # type: ignore[arg-type]
        source_transport_qualified=source_transport_qualified,
        rank_bar_data_qualified=rank_bar_data_qualified,
        specification_sha256=MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "production_acquisition_receipt_sha256": acquisition.receipt_sha256,
            }
        ),
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


def build_massive_monthly_rank_bar_authority_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    rank_input_authority: MassiveMonthlyRankInputAuthorityV2,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
) -> MassiveMonthlyRankBarAuthorityV1:
    """Build the authorizing rank-bar gate from fixed-runtime acquisition."""

    return _build_massive_monthly_rank_bar_authority_v1(
        source_root=source_root,
        persisted_root=persisted_root,
        daily_bars_root=daily_bars_root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        acquisition=acquisition,
        rank_input_authority=rank_input_authority,
        scan_evidence=scan_evidence,
        semantic_partition_manifests=semantic_partition_manifests,
        persisted_partition_manifests=persisted_partition_manifests,
        daily_bars=daily_bars,
        require_fixed_runtime=True,
    )


def build_massive_monthly_rank_bar_authority_for_test_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    rank_input_authority: MassiveMonthlyRankInputAuthorityV2,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
) -> MassiveMonthlyRankBarAuthorityV1:
    """Exercise exact rederivation without creating production qualification."""

    return _build_massive_monthly_rank_bar_authority_v1(
        source_root=source_root,
        persisted_root=persisted_root,
        daily_bars_root=daily_bars_root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        acquisition=acquisition,
        rank_input_authority=rank_input_authority,
        scan_evidence=scan_evidence,
        semantic_partition_manifests=semantic_partition_manifests,
        persisted_partition_manifests=persisted_partition_manifests,
        daily_bars=daily_bars,
        require_fixed_runtime=False,
    )


__all__ = [
    "MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SCHEMA",
    "MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_MONTHLY_RANK_BAR_AUTHORITY_V1_SPEC_SHA256",
    "MASSIVE_MONTHLY_RANK_BAR_V1_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_MONTHLY_RANK_BAR_V1_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_MONTHLY_RANK_BAR_V1_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_MONTHLY_RANK_BAR_V1_PROFITABILITY_REPORTING_AUTHORIZED",
    "MassiveMonthlyRankBarAuthorityV1",
    "MassiveMonthlyRankBarAuthorityV1Error",
    "MassiveMonthlyRankBarSessionV1",
    "build_massive_monthly_rank_bar_authority_for_test_v1",
    "build_massive_monthly_rank_bar_authority_v1",
]
