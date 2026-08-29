"""Source-derived morning VWAPs for the adaptive Massive alpha protocol."""

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
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA = "rl-quant.massive-adaptive-fill-source-v1"
MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "window": "[09:35,09:45)-America/New_York",
        "population": "terminal-active-price-and-volume-forming-participant-time",
        "source": "qualified-persisted-partitions-bound-to-daily-input-v1",
        "missing": "zero-plus-false-mask",
        "duration_prior": False,
        "downstream_authorization": False,
    }
)


class MassiveAdaptiveFillSourceV1Error(ValueError):
    """An adaptive fill is detached from its authenticated source roots."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveFillSourceV1Error(f"{name} must be a lowercase SHA-256")
    return value


def adaptive_fill_clock_v1(session_date: str) -> tuple[int, int]:
    day = date.fromisoformat(session_date)
    eastern = ZoneInfo("America/New_York")
    start = datetime.combine(day, time(9, 35), tzinfo=eastern)
    end = datetime.combine(day, time(9, 45), tzinfo=eastern)
    return int(start.timestamp() * 1_000), int(end.timestamp() * 1_000)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFillRowV1:
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
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        expected_start, expected_end = adaptive_fill_clock_v1(self.session_date)
        if (
            not self.security_id
            or (self.fill_start_at_ms, self.fill_end_at_ms)
            != (expected_start, expected_end)
            or not isinstance(self.valid, bool)
            or self.qualifying_trade_count < 0
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill identity or clock differs")
        values = (
            self.fill_vwap,
            self.qualifying_share_volume,
            self.qualifying_dollar_volume,
        )
        if self.valid:
            if (
                self.qualifying_trade_count <= 0
                or any(value <= 0.0 for value in values)
                or abs(self.fill_vwap - self.qualifying_dollar_volume / self.qualifying_share_volume)
                > 1e-12
            ):
                raise MassiveAdaptiveFillSourceV1Error("adaptive fill values do not reconcile")
        elif self.qualifying_trade_count != 0 or any(value != 0.0 for value in values):
            raise MassiveAdaptiveFillSourceV1Error("invalid adaptive fill is not zero-masked")
        for name, value in (
            ("trade inventory", self.qualifying_trade_inventory_sha256),
            ("daily input row", self.daily_input_row_receipt_sha256),
            ("fill row", self.receipt_sha256),
        ):
            _digest(name, value)
        if self.persisted_partition_receipt_sha256 is not None:
            _digest("persisted partition", self.persisted_partition_receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFillSourceV1:
    session_dates: tuple[str, ...]
    supported_security_ids: tuple[str, ...]
    rows: tuple[MassiveAdaptiveFillRowV1, ...]
    daily_input_authority_semantic_receipt_sha256: str
    session_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    persisted_manifest_inventory_sha256: str
    row_inventory_sha256: str
    source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    source_paths_replayed: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        expected = tuple(
            (session_date, security_id)
            for session_date in self.session_dates
            for security_id in self.supported_security_ids
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.fill_rule
            != "next-session-09:35-09:45-et-qualifying-trade-vwap"
            or self.specification_sha256 != MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256
            or self.implementation_source_sha256 != MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256
            or not self.session_dates
            or self.session_dates != tuple(sorted(set(self.session_dates)))
            or not self.supported_security_ids
            or self.supported_security_ids != tuple(sorted(set(self.supported_security_ids)))
            or tuple((row.session_date, row.security_id) for row in self.rows) != expected
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill authority identity differs")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill inventory differs")
        if not isinstance(self.source_data_qualified, bool) or not isinstance(
            self.source_paths_replayed, bool
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill qualification differs")
        if any(
            (
                self.predictive_training_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill authorizes downstream use")
        assert_no_adaptive_hold_semantics(asdict(self))
        for name in (
            "daily_input_authority_semantic_receipt_sha256",
            "session_authority_receipt_sha256",
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
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill semantic receipt differs")
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "daily_input_audit_receipt_sha256": (
                    self.daily_input_authority_semantic_receipt_sha256
                ),
            }
        ):
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill audit receipt differs")

    def row(self, *, session_date: str, security_id: str) -> MassiveAdaptiveFillRowV1:
        for row in self.rows:
            if row.session_date == session_date and row.security_id == security_id:
                return row
        raise MassiveAdaptiveFillSourceV1Error("fill row is outside adaptive support")


def build_massive_adaptive_fill_source_v1(
    *,
    persisted_root: str | Path,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    required_session_dates: Sequence[str],
    supported_security_ids: Sequence[str],
) -> MassiveAdaptiveFillSourceV1:
    """Replay morning fills from the same qualified archive used by daily inputs."""

    session_authority.validate()
    condition_authority.validate()
    daily_input_authority.validate()
    if (
        session_authority.receipt_sha256
        != daily_input_authority.session_authority_receipt_sha256
        or condition_authority.receipt_sha256
        != daily_input_authority.condition_authority_receipt_sha256
    ):
        raise MassiveAdaptiveFillSourceV1Error("adaptive fill roots differ from daily inputs")
    dates = tuple(sorted(set(required_session_dates)))
    support = tuple(sorted(set(supported_security_ids)))
    daily_sessions = {row.source_session_date: row for row in daily_input_authority.sessions}
    if (
        not dates
        or not support
        or not set(dates) <= set(daily_sessions)
        or not set(support) <= set(daily_input_authority.supported_security_ids)
    ):
        raise MassiveAdaptiveFillSourceV1Error("adaptive fill scope exceeds daily inputs")
    manifest_rows = tuple(persisted_partition_manifests)
    manifests = {row.source_session_date: row for row in manifest_rows}
    if len(manifests) != len(manifest_rows) or not set(dates) <= set(manifests):
        raise MassiveAdaptiveFillSourceV1Error("adaptive fill manifests are incomplete")
    rows: list[MassiveAdaptiveFillRowV1] = []
    for session_date in dates:
        manifest = manifests[session_date]
        manifest.validate()
        if manifest.receipt_sha256 != daily_sessions[
            session_date
        ].persisted_partition_manifest_receipt_sha256:
            raise MassiveAdaptiveFillSourceV1Error("adaptive fill manifest is detached")
        start, end = adaptive_fill_clock_v1(session_date)
        partitions = {row.security_id: row for row in manifest.partitions}
        for security_id in support:
            partition = partitions.get(security_id)
            qualifying = []
            if partition is not None:
                _, active, _ = load_massive_persisted_security_rows_v2(
                    root=persisted_root, partition=partition
                )
                for trade in active:
                    at_ms = trade.canonical_record.participant_timestamp_ns // 1_000_000
                    price_forming, _, volume_forming, _ = condition_authority.resolve(
                        trade.canonical_record.conditions
                    )
                    if start <= at_ms < end and price_forming and volume_forming:
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
            body = {
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
            row = MassiveAdaptiveFillRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
            row.validate()
            rows.append(row)
    manifest_inventory = semantic_sha256(
        tuple(manifests[session_date].receipt_sha256 for session_date in dates)
    )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic = {
        "schema": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA,
        "session_dates": dates,
        "supported_security_ids": support,
        "rows": tuple(asdict(row) for row in rows),
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "persisted_manifest_inventory_sha256": manifest_inventory,
        "row_inventory_sha256": row_inventory,
        "source_data_qualified": daily_input_authority.daily_input_data_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveAdaptiveFillSourceV1(
        session_dates=dates,
        supported_security_ids=support,
        rows=tuple(rows),
        daily_input_authority_semantic_receipt_sha256=(
            daily_input_authority.semantic_receipt_sha256
        ),
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        persisted_manifest_inventory_sha256=manifest_inventory,
        row_inventory_sha256=row_inventory,
        source_data_qualified=daily_input_authority.daily_input_data_qualified,
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        specification_sha256=MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "daily_input_audit_receipt_sha256": (
                    daily_input_authority.semantic_receipt_sha256
                ),
            }
        ),
        source_paths_replayed=True,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA",
    "MassiveAdaptiveFillRowV1",
    "MassiveAdaptiveFillSourceV1",
    "MassiveAdaptiveFillSourceV1Error",
    "adaptive_fill_clock_v1",
    "build_massive_adaptive_fill_source_v1",
]
