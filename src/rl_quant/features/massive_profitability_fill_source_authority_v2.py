"""Authenticated price-and-volume-forming fill VWAPs for Massive P0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MassiveProfitabilitySecuritySupportV2,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-profitability-fill-source-authority-v2"
)
MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "window": "[15:50,16:00)-America/New_York",
        "population": (
            "terminal-active-and-price-forming-and-volume-forming-participant-time"
        ),
        "source": "daily-input-authority-v1-qualified-persisted-partition",
        "missing": "zero-plus-false-mask-never-shift-execution",
        "support": "all-experiment-supported-securities",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityFillSourceAuthorityV2Error(ValueError):
    """A fill row is detached from the authenticated daily input chain."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFillSourceRowV2:
    session_date: str
    security_id: str
    fill_start_at_ms: int
    fill_end_at_ms: int
    fill_vwap: float
    qualifying_share_volume: float
    qualifying_dollar_volume: float
    qualifying_trade_count: int
    valid: bool
    qualifying_trade_inventory_sha256: str
    persisted_partition_receipt_sha256: str | None
    daily_input_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.session_date
            or not self.security_id
            or self.fill_start_at_ms < 0
            or self.fill_end_at_ms <= self.fill_start_at_ms
            or not isinstance(self.valid, bool)
            or self.qualifying_trade_count < 0
        ):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill row identity or chronology differs"
            )
        values = (
            self.fill_vwap,
            self.qualifying_share_volume,
            self.qualifying_dollar_volume,
        )
        if self.valid:
            if (
                self.qualifying_trade_count <= 0
                or any(value <= 0.0 for value in values)
                or abs(
                    self.fill_vwap
                    - self.qualifying_dollar_volume / self.qualifying_share_volume
                )
                > 1e-12
            ):
                raise MassiveProfitabilityFillSourceAuthorityV2Error(
                    "valid fill values do not reconcile"
                )
        elif self.qualifying_trade_count != 0 or any(value != 0.0 for value in values):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "invalid fill is not a zero placeholder"
            )
        _digest("fill trade inventory", self.qualifying_trade_inventory_sha256)
        _digest("fill daily input row", self.daily_input_row_receipt_sha256)
        if self.persisted_partition_receipt_sha256 is not None:
            _digest("fill persisted partition", self.persisted_partition_receipt_sha256)
        _digest("fill row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFillSourceAuthorityV2:
    session_dates: tuple[str, ...]
    supported_security_ids: tuple[str, ...]
    rows: tuple[MassiveProfitabilityFillSourceRowV2, ...]
    daily_input_authority_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str | None
    security_support_semantic_receipt_sha256: str | None
    condition_authority_receipt_sha256: str
    persisted_manifest_inventory_sha256: str
    row_inventory_sha256: str
    source_data_qualified: bool
    fill_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256
            or not self.session_dates
            or self.session_dates != tuple(sorted(set(self.session_dates)))
            or not self.supported_security_ids
            or self.supported_security_ids
            != tuple(sorted(set(self.supported_security_ids)))
        ):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill authority identity or support differs"
            )
        expected = tuple(
            (session_date, security_id)
            for session_date in self.session_dates
            for security_id in self.supported_security_ids
        )
        if tuple((row.session_date, row.security_id) for row in self.rows) != expected:
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill authority does not contain the exact support rectangle"
            )
        for row in self.rows:
            row.validate()
        if (
            self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not isinstance(self.source_data_qualified, bool)
            or self.fill_source_data_qualified != self.source_data_qualified
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill authority inventory or qualification differs"
            )
        for value in (
            self.origin_plan_semantic_receipt_sha256,
            self.security_support_semantic_receipt_sha256,
        ):
            if value is not None:
                _digest("fill frozen component", value)
        for name in (
            "daily_input_authority_semantic_receipt_sha256",
            "condition_authority_receipt_sha256",
            "persisted_manifest_inventory_sha256",
            "row_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill authority semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "daily_input_audit_receipt_sha256": (
                    self.daily_input_authority_semantic_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill authority audit receipt differs"
            )

    def row(self, *, session_date: str, security_id: str) -> MassiveProfitabilityFillSourceRowV2:
        for value in self.rows:
            if value.session_date == session_date and value.security_id == security_id:
                return value
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            "requested fill row is outside the frozen authority"
        )


def _fill_clock(session_date: str) -> tuple[int, int]:
    day = date.fromisoformat(session_date)
    eastern = ZoneInfo("America/New_York")
    start = datetime.combine(day, time(15, 50), tzinfo=eastern)
    end = datetime.combine(day, time(16, 0), tzinfo=eastern)
    return int(start.timestamp() * 1_000), int(end.timestamp() * 1_000)


def _build_massive_profitability_fill_source_authority_v2(
    *,
    persisted_root: str | Path,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    required_session_dates: Sequence[str],
    supported_security_ids: Sequence[str],
    origin_plan_semantic_receipt_sha256: str | None,
    security_support_semantic_receipt_sha256: str | None,
) -> MassiveProfitabilityFillSourceAuthorityV2:
    session_authority.validate()
    condition_authority.validate()
    daily_input_authority.validate()
    if (
        session_authority.receipt_sha256
        != daily_input_authority.session_authority_receipt_sha256
        or condition_authority.receipt_sha256
        != daily_input_authority.condition_authority_receipt_sha256
    ):
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            "fill session or condition authority differs from daily inputs"
        )
    dates = tuple(sorted(set(required_session_dates)))
    support = tuple(sorted(set(supported_security_ids)))
    if not dates or not support or not set(dates) <= {
        row.source_session_date for row in daily_input_authority.sessions
    } or not set(support) <= set(daily_input_authority.supported_security_ids):
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            "fill requested scope exceeds qualified daily inputs"
        )
    manifests = {row.source_session_date: row for row in persisted_partition_manifests}
    if len(manifests) != len(tuple(persisted_partition_manifests)) or not set(dates) <= set(
        manifests
    ):
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            "fill persisted manifests do not cover requested sessions"
        )
    daily_sessions = {
        row.source_session_date: row for row in daily_input_authority.sessions
    }
    rows: list[MassiveProfitabilityFillSourceRowV2] = []
    for session_date in dates:
        manifest = manifests[session_date]
        manifest.validate()
        if (
            manifest.receipt_sha256
            != daily_sessions[session_date].persisted_partition_manifest_receipt_sha256
        ):
            raise MassiveProfitabilityFillSourceAuthorityV2Error(
                "fill manifest is detached from daily input authority"
            )
        start, end = _fill_clock(session_date)
        partition_by_security = {row.security_id: row for row in manifest.partitions}
        for security_id in support:
            partition = partition_by_security.get(security_id)
            qualifying = []
            if partition is not None:
                _, active, _ = load_massive_persisted_security_rows_v2(
                    root=persisted_root, partition=partition
                )
                for trade in active:
                    at_ms = trade.canonical_record.participant_timestamp_ns // 1_000_000
                    flags = condition_authority.resolve(
                        trade.canonical_record.conditions
                    )
                    if start <= at_ms < end and flags[0] and flags[2]:
                        qualifying.append(trade)
            shares = sum(
                (Decimal(row.canonical_record.size_decimal) for row in qualifying),
                Decimal(0),
            )
            dollars = sum(
                (
                    Decimal(row.canonical_record.price_decimal)
                    * Decimal(row.canonical_record.size_decimal)
                    for row in qualifying
                ),
                Decimal(0),
            )
            valid = bool(qualifying) and shares > 0 and dollars > 0
            daily_row = daily_input_authority.row(
                session_date=session_date, security_id=security_id
            )
            body: dict[str, object] = {
                "session_date": session_date,
                "security_id": security_id,
                "fill_start_at_ms": start,
                "fill_end_at_ms": end,
                "fill_vwap": 0.0 if not valid else float(dollars / shares),
                "qualifying_share_volume": 0.0 if not valid else float(shares),
                "qualifying_dollar_volume": 0.0 if not valid else float(dollars),
                "qualifying_trade_count": 0 if not valid else len(qualifying),
                "valid": valid,
                "qualifying_trade_inventory_sha256": semantic_sha256(
                    tuple(row.receipt_sha256 for row in qualifying)
                ),
                "persisted_partition_receipt_sha256": (
                    None if partition is None else partition.receipt_sha256
                ),
                "daily_input_row_receipt_sha256": daily_row.receipt_sha256,
            }
            result_row = MassiveProfitabilityFillSourceRowV2(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
            result_row.validate()
            rows.append(result_row)
    manifest_inventory = semantic_sha256(
        tuple(manifests[session_date].receipt_sha256 for session_date in dates)
    )
    source_qualified = daily_input_authority.daily_input_data_qualified
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
        "session_dates": dates,
        "supported_security_ids": support,
        "rows": tuple(asdict(row) for row in rows),
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "origin_plan_semantic_receipt_sha256": origin_plan_semantic_receipt_sha256,
        "security_support_semantic_receipt_sha256": (
            security_support_semantic_receipt_sha256
        ),
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "persisted_manifest_inventory_sha256": manifest_inventory,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_data_qualified": source_qualified,
        "fill_source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveProfitabilityFillSourceAuthorityV2(
        session_dates=dates,
        supported_security_ids=support,
        rows=tuple(rows),
        daily_input_authority_semantic_receipt_sha256=(
            daily_input_authority.semantic_receipt_sha256
        ),
        origin_plan_semantic_receipt_sha256=origin_plan_semantic_receipt_sha256,
        security_support_semantic_receipt_sha256=(
            security_support_semantic_receipt_sha256
        ),
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        persisted_manifest_inventory_sha256=manifest_inventory,
        row_inventory_sha256=semantic["row_inventory_sha256"],  # type: ignore[arg-type]
        source_data_qualified=source_qualified,
        fill_source_data_qualified=source_qualified,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "daily_input_audit_receipt_sha256": (
                    daily_input_authority.semantic_receipt_sha256
                ),
            }
        ),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


def build_massive_profitability_fill_source_authority_v2(
    *,
    persisted_root: str | Path,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    security_support: MassiveProfitabilitySecuritySupportV2,
) -> MassiveProfitabilityFillSourceAuthorityV2:
    """Derive every P0 entry and horizon-exit fill from qualified partitions."""

    origin_plan.validate()
    security_support.validate()
    if (
        daily_input_authority.archive_freeze_semantic_receipt_sha256 is None
        or daily_input_authority.security_support_semantic_receipt_sha256
        != security_support.semantic_receipt_sha256
        or security_support.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or daily_input_authority.supported_security_ids
        != security_support.all_supported_security_ids
        or daily_input_authority.normalized_identity_semantic_receipt_sha256
        != security_support.normalized_identity_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityFillSourceAuthorityV2Error(
            "fill origin, support, identity, or frozen daily input binding differs"
        )
    by_date = {
        row.session_date: index for index, row in enumerate(session_authority.sessions)
    }
    dates: set[str] = set()
    for origin in origin_plan.origin_plan_v1.origins:
        index = by_date[origin.decision_session_date]
        dates.add(origin.decision_session_date)
        for offset in (1, 5, 21, 63):
            if index + offset >= len(session_authority.sessions):
                raise MassiveProfitabilityFillSourceAuthorityV2Error(
                    "origin lacks one complete H63 exit-fill path"
                )
            dates.add(session_authority.sessions[index + offset].session_date)
    return _build_massive_profitability_fill_source_authority_v2(
        persisted_root=persisted_root,
        session_authority=session_authority,
        condition_authority=condition_authority,
        daily_input_authority=daily_input_authority,
        persisted_partition_manifests=persisted_partition_manifests,
        required_session_dates=tuple(sorted(dates)),
        supported_security_ids=security_support.all_supported_security_ids,
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        security_support_semantic_receipt_sha256=(
            security_support.semantic_receipt_sha256
        ),
    )


def build_massive_profitability_fill_source_authority_for_test_v2(
    *,
    persisted_root: str | Path,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    required_session_dates: Sequence[str],
    supported_security_ids: Sequence[str],
) -> MassiveProfitabilityFillSourceAuthorityV2:
    return _build_massive_profitability_fill_source_authority_v2(
        persisted_root=persisted_root,
        session_authority=session_authority,
        condition_authority=condition_authority,
        daily_input_authority=daily_input_authority,
        persisted_partition_manifests=persisted_partition_manifests,
        required_session_dates=required_session_dates,
        supported_security_ids=supported_security_ids,
        origin_plan_semantic_receipt_sha256=None,
        security_support_semantic_receipt_sha256=None,
    )


__all__ = [
    "MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA",
    "MassiveProfitabilityFillSourceAuthorityV2",
    "MassiveProfitabilityFillSourceAuthorityV2Error",
    "MassiveProfitabilityFillSourceRowV2",
    "build_massive_profitability_fill_source_authority_for_test_v2",
    "build_massive_profitability_fill_source_authority_v2",
]
