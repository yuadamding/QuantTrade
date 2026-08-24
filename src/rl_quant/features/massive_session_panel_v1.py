"""Session-aligned permanent-security panel for Massive profitability P0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
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
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_FIELDS,
    MassiveDailyBarsArtifactV0,
    MassiveDailyBarsRowV0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_daily_tape_v0 import (
    MASSIVE_DAILY_TAPE_V0_FIELDS,
    MassiveDailyTapeArtifactV0,
    MassiveDailyTapeRowV0,
    validate_massive_daily_tape_v0,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)


MASSIVE_SESSION_PANEL_V1_SCHEMA = "rl-quant.massive-session-panel-v1"
MASSIVE_SESSION_PANEL_V1_DATASET = "massive-finalized-session-panel-v1"
MASSIVE_SESSION_PANEL_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_SESSION_PANEL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "grid": "one-row-per-permanent-security-per-XNYS-session",
        "membership": "latest-effective-and-available-PIT-membership-at-session-close",
        "listed": "listing<=close<delisting",
        "observed": "common-daily-bars-and-tape-support",
        "halt_no_print": "PIT-member-and-listed-without-regular-session-trade",
        "tradable": "member+listed+observed+valid-close-and-dollar-volume",
        "corrections": "complete-event-timeline-denominator-by-kind",
        "bars_fields": MASSIVE_DAILY_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_DAILY_TAPE_V0_FIELDS,
        "missing": "zero-value-plus-false-mask",
    }
)
MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_SESSION_PANEL_V1_SCHEMA,
        "row_key": ("source_session_index", "security_id"),
        "bars_fields": MASSIVE_DAILY_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_DAILY_TAPE_V0_FIELDS,
    }
)


class MassiveSessionPanelV1Error(ValueError):
    """Session-panel inputs, chronology, or committed bytes differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveSessionPanelV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveSessionPanelV1Error(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class MassiveSessionInputReceiptV1:
    source_session_date: str
    daily_bars_artifact_receipt_sha256: str
    daily_tape_artifact_receipt_sha256: str
    persisted_partition_manifest_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        try:
            date.fromisoformat(self.source_session_date)
        except ValueError as exc:
            raise MassiveSessionPanelV1Error("input session date is invalid") from exc
        for name in (
            "daily_bars_artifact_receipt_sha256",
            "daily_tape_artifact_receipt_sha256",
            "persisted_partition_manifest_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSessionPanelV1Error("session input receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveSessionEventCountsV1:
    source_session_date: str
    security_id: str
    event_timeline_count: int
    replacement_event_count: int
    cancellation_event_count: int
    late_report_event_count: int
    partition_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id:
            raise MassiveSessionPanelV1Error("event-count security ID is absent")
        try:
            date.fromisoformat(self.source_session_date)
        except ValueError as exc:
            raise MassiveSessionPanelV1Error(
                "event-count session date is invalid"
            ) from exc
        counts = (
            self.event_timeline_count,
            self.replacement_event_count,
            self.cancellation_event_count,
            self.late_report_event_count,
        )
        for name, value in zip(
            ("event", "replacement", "cancellation", "late-report"),
            counts,
            strict=True,
        ):
            _count(f"{name} count", value)
        if sum(counts[1:]) > counts[0]:
            raise MassiveSessionPanelV1Error(
                "correction counts exceed the complete event timeline"
            )
        _digest("partition receipt", self.partition_receipt_sha256)
        _digest("event-count receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSessionPanelV1Error("event-count receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveSessionPanelRowV1:
    source_session_index: int
    source_session_date: str
    regular_open_ns: int
    regular_close_ns: int
    security_id: str
    pit_member: bool
    listed: bool
    tradable: bool
    observed_regular_trade: bool
    halt_or_no_print: bool
    bars_values: tuple[float, ...]
    bars_valid: tuple[bool, ...]
    tape_values: tuple[float, ...]
    tape_valid: tuple[bool, ...]
    event_timeline_count: int
    replacement_event_count: int
    cancellation_event_count: int
    late_report_event_count: int
    daily_bars_row_receipt_sha256: str | None
    daily_tape_row_receipt_sha256: str | None
    event_counts_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _count("source session index", self.source_session_index)
        _count("regular open", self.regular_open_ns)
        _count("regular close", self.regular_close_ns)
        if self.regular_close_ns <= self.regular_open_ns:
            raise MassiveSessionPanelV1Error("panel session bounds differ")
        if not self.security_id:
            raise MassiveSessionPanelV1Error("session-panel security ID is absent")
        try:
            date.fromisoformat(self.source_session_date)
        except ValueError as exc:
            raise MassiveSessionPanelV1Error("panel session date is invalid") from exc
        for state_flag in (
            self.pit_member,
            self.listed,
            self.tradable,
            self.observed_regular_trade,
            self.halt_or_no_print,
        ):
            if not isinstance(state_flag, bool):
                raise MassiveSessionPanelV1Error("panel state flags must be Boolean")
        for values, valid, fields in (
            (self.bars_values, self.bars_valid, MASSIVE_DAILY_BARS_V0_FIELDS),
            (self.tape_values, self.tape_valid, MASSIVE_DAILY_TAPE_V0_FIELDS),
        ):
            if (
                len(values) != len(fields)
                or len(valid) != len(fields)
                or any(not isinstance(flag, bool) for flag in valid)
                or any(not math.isfinite(float(value)) for value in values)
            ):
                raise MassiveSessionPanelV1Error("panel values or masks differ")
        if self.observed_regular_trade != (
            self.daily_bars_row_receipt_sha256 is not None
            and self.daily_tape_row_receipt_sha256 is not None
        ):
            raise MassiveSessionPanelV1Error("observed-trade evidence differs")
        close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
        dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
        expected_tradable = (
            self.pit_member
            and self.listed
            and self.observed_regular_trade
            and self.bars_valid[close_index]
            and self.bars_valid[dollar_index]
        )
        if self.tradable != expected_tradable or self.halt_or_no_print != (
            self.pit_member and self.listed and not self.observed_regular_trade
        ):
            raise MassiveSessionPanelV1Error("panel tradability state differs")
        counts = (
            self.event_timeline_count,
            self.replacement_event_count,
            self.cancellation_event_count,
            self.late_report_event_count,
        )
        for name, count_value in zip(
            ("event", "replacement", "cancellation", "late-report"),
            counts,
            strict=True,
        ):
            _count(f"row {name} count", count_value)
        if sum(counts[1:]) > counts[0]:
            raise MassiveSessionPanelV1Error("row correction counts differ")
        optional_receipts = (
            self.daily_bars_row_receipt_sha256,
            self.daily_tape_row_receipt_sha256,
            self.event_counts_receipt_sha256,
        )
        for optional_receipt in optional_receipts:
            if optional_receipt is not None:
                _digest("optional row receipt", optional_receipt)
        if (self.event_timeline_count > 0) != (
            self.event_counts_receipt_sha256 is not None
        ):
            raise MassiveSessionPanelV1Error("event-count evidence presence differs")
        _digest("panel row receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSessionPanelV1Error("panel row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveSessionPanelArtifactV1:
    exchange: str
    start_session_date: str
    end_session_date: str
    session_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    input_receipts: tuple[MassiveSessionInputReceiptV1, ...]
    rows: tuple[MassiveSessionPanelRowV1, ...]
    session_count: int
    security_count: int
    row_count: int
    member_row_count: int
    row_inventory_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_SESSION_PANEL_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_SESSION_PANEL_V1_SCHEMA or self.exchange != "XNYS":
            raise MassiveSessionPanelV1Error("session-panel identity drifted")
        for name in (
            "session_authority_receipt_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "row_inventory_sha256",
            "feature_spec_receipt_sha256",
            "feature_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.feature_spec_receipt_sha256 != MASSIVE_SESSION_PANEL_V1_SPEC_SHA256
            or self.feature_source_sha256 != MASSIVE_SESSION_PANEL_V1_SOURCE_SHA256
        ):
            raise MassiveSessionPanelV1Error("session-panel implementation drifted")
        input_dates = tuple(
            input_row.source_session_date for input_row in self.input_receipts
        )
        if (
            not input_dates
            or input_dates != tuple(sorted(set(input_dates)))
            or input_dates[0] != self.start_session_date
            or input_dates[-1] != self.end_session_date
        ):
            raise MassiveSessionPanelV1Error("session input inventory differs")
        for input_row in self.input_receipts:
            input_row.validate()
        keys = tuple(
            (panel_row.source_session_index, panel_row.security_id)
            for panel_row in self.rows
        )
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveSessionPanelV1Error("session-panel rows are not canonical")
        for panel_row in self.rows:
            panel_row.validate()
        expected_session_count = len(input_dates)
        expected_security_count = len({row.security_id for row in self.rows})
        rows_by_index: dict[int, list[MassiveSessionPanelRowV1]] = {}
        for panel_row in self.rows:
            rows_by_index.setdefault(panel_row.source_session_index, []).append(
                panel_row
            )
        if tuple(rows_by_index) != tuple(range(expected_session_count)):
            raise MassiveSessionPanelV1Error("session-panel indices differ")
        for session_index, session_rows in rows_by_index.items():
            session_coordinates = {
                (
                    panel_row.source_session_date,
                    panel_row.regular_open_ns,
                    panel_row.regular_close_ns,
                )
                for panel_row in session_rows
            }
            if (
                len(session_rows) != expected_security_count
                or len(session_coordinates) != 1
                or session_rows[0].source_session_date != input_dates[session_index]
            ):
                raise MassiveSessionPanelV1Error(
                    "session-panel rectangle coordinates differ"
                )
        if (
            self.session_count != expected_session_count
            or self.security_count != expected_security_count
            or self.row_count != len(self.rows)
            or self.row_count != self.session_count * self.security_count
            or self.member_row_count != sum(row.pit_member for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
        ):
            raise MassiveSessionPanelV1Error("session-panel counts differ")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_SESSION_PANEL_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveSessionPanelV1Error("session-panel source contract differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSessionPanelV1Error("session-panel artifact receipt differs")


def _membership_at(
    identity_authority: PITSecurityUniverseAuthority,
    *,
    security_id: str,
    session_close_ms: int,
) -> bool:
    eligible = tuple(
        row
        for row in identity_authority.membership_events
        if row.security_id == security_id
        and row.effective_at_ms <= session_close_ms
        and row.available_at_ms <= session_close_ms
    )
    if not eligible:
        return False
    latest = max(eligible, key=lambda row: (row.effective_at_ms, row.available_at_ms))
    return latest.is_member


def build_massive_session_panel_rows_v1(
    *,
    sessions: Sequence[MassiveExchangeSession],
    identity_authority: PITSecurityUniverseAuthority,
    bars_by_session: Mapping[str, Mapping[str, MassiveDailyBarsRowV0]],
    tape_by_session: Mapping[str, Mapping[str, MassiveDailyTapeRowV0]],
    event_counts_by_session: Mapping[str, Mapping[str, MassiveSessionEventCountsV1]],
) -> tuple[MassiveSessionPanelRowV1, ...]:
    """Build the exact session grid without compressing absent observations."""

    identity_authority.validate()
    ordered_sessions = tuple(sessions)
    dates = tuple(row.session_date for row in ordered_sessions)
    if not dates or dates != tuple(sorted(set(dates))):
        raise MassiveSessionPanelV1Error("session grid is not chronological")
    if set(bars_by_session) != set(dates) or set(tape_by_session) != set(dates):
        raise MassiveSessionPanelV1Error(
            "daily feature support differs from session grid"
        )
    masters = tuple(
        sorted(identity_authority.security_master, key=lambda row: row.security_id)
    )
    rows: list[MassiveSessionPanelRowV1] = []
    zero_bars = (0.0,) * len(MASSIVE_DAILY_BARS_V0_FIELDS)
    zero_tape = (0.0,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS)
    false_bars = (False,) * len(zero_bars)
    false_tape = (False,) * len(zero_tape)
    for session_index, session in enumerate(ordered_sessions):
        session.validate()
        if session.exchange != "XNYS":
            raise MassiveSessionPanelV1Error("profitability P0 requires XNYS sessions")
        bars = bars_by_session[session.session_date]
        tape = tape_by_session[session.session_date]
        if set(bars) != set(tape):
            raise MassiveSessionPanelV1Error(
                "daily bars and tape security support differs"
            )
        counts = event_counts_by_session.get(session.session_date, {})
        unknown = (set(bars) | set(counts)) - {row.security_id for row in masters}
        if unknown:
            raise MassiveSessionPanelV1Error("daily input references unknown security")
        close_ms = session.regular_close_ns // 1_000_000
        for master in masters:
            bars_row = bars.get(master.security_id)
            tape_row = tape.get(master.security_id)
            count_row = counts.get(master.security_id)
            if bars_row is not None:
                bars_row.validate()
            if tape_row is not None:
                tape_row.validate()
            if count_row is not None:
                count_row.validate()
                if count_row.source_session_date != session.session_date:
                    raise MassiveSessionPanelV1Error("event-count date differs")
            listed = master.listing_at_ms <= close_ms and (
                master.delisting_at_ms is None or close_ms < master.delisting_at_ms
            )
            member = _membership_at(
                identity_authority,
                security_id=master.security_id,
                session_close_ms=close_ms,
            )
            observed = bars_row is not None and tape_row is not None
            bars_values = zero_bars if bars_row is None else bars_row.values
            bars_valid = false_bars if bars_row is None else bars_row.valid
            tape_values = zero_tape if tape_row is None else tape_row.values
            tape_valid = false_tape if tape_row is None else tape_row.valid
            event_counts = (
                (0, 0, 0, 0)
                if count_row is None
                else (
                    count_row.event_timeline_count,
                    count_row.replacement_event_count,
                    count_row.cancellation_event_count,
                    count_row.late_report_event_count,
                )
            )
            close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
            dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
            provisional = MassiveSessionPanelRowV1(
                source_session_index=session_index,
                source_session_date=session.session_date,
                regular_open_ns=session.regular_open_ns,
                regular_close_ns=session.regular_close_ns,
                security_id=master.security_id,
                pit_member=member,
                listed=listed,
                tradable=(
                    member
                    and listed
                    and observed
                    and bars_valid[close_index]
                    and bars_valid[dollar_index]
                ),
                observed_regular_trade=observed,
                halt_or_no_print=member and listed and not observed,
                bars_values=bars_values,
                bars_valid=bars_valid,
                tape_values=tape_values,
                tape_valid=tape_valid,
                event_timeline_count=event_counts[0],
                replacement_event_count=event_counts[1],
                cancellation_event_count=event_counts[2],
                late_report_event_count=event_counts[3],
                daily_bars_row_receipt_sha256=(
                    None if bars_row is None else bars_row.receipt_sha256
                ),
                daily_tape_row_receipt_sha256=(
                    None if tape_row is None else tape_row.receipt_sha256
                ),
                event_counts_receipt_sha256=(
                    None if count_row is None else count_row.receipt_sha256
                ),
                receipt_sha256="0" * 64,
            )
            row = replace(
                provisional,
                receipt_sha256=semantic_sha256(provisional.unsigned()),
            )
            row.validate()
            rows.append(row)
    return tuple(rows)


def _event_counts(
    *,
    persisted_root: str | Path,
    manifests: Sequence[MassivePersistedPartitionManifestV1],
) -> dict[str, dict[str, MassiveSessionEventCountsV1]]:
    result: dict[str, dict[str, MassiveSessionEventCountsV1]] = {}
    for manifest in manifests:
        manifest.validate()
        by_security: dict[str, MassiveSessionEventCountsV1] = {}
        for partition in manifest.partitions:
            events, _, corrections = load_massive_persisted_security_rows_v2(
                root=persisted_root,
                partition=partition,
            )
            kinds = tuple(row["correction_kind"] for row in corrections)
            provisional = MassiveSessionEventCountsV1(
                source_session_date=manifest.source_session_date,
                security_id=partition.security_id,
                event_timeline_count=len(events),
                replacement_event_count=kinds.count("replacement"),
                cancellation_event_count=kinds.count("cancellation"),
                late_report_event_count=kinds.count("late-report"),
                partition_receipt_sha256=partition.receipt_sha256,
                receipt_sha256="0" * 64,
            )
            row = replace(
                provisional,
                receipt_sha256=semantic_sha256(provisional.unsigned()),
            )
            row.validate()
            by_security[row.security_id] = row
        result[manifest.source_session_date] = by_security
    return result


def _payload(artifact: MassiveSessionPanelArtifactV1) -> dict[str, object]:
    return {
        "schema": artifact.schema,
        "exchange": artifact.exchange,
        "start_session_date": artifact.start_session_date,
        "end_session_date": artifact.end_session_date,
        "session_authority_receipt_sha256": artifact.session_authority_receipt_sha256,
        "identity_authority_receipt_sha256": artifact.identity_authority_receipt_sha256,
        "condition_authority_receipt_sha256": artifact.condition_authority_receipt_sha256,
        "input_receipts": tuple(asdict(row) for row in artifact.input_receipts),
        "rows": tuple(asdict(row) for row in artifact.rows),
        "session_count": artifact.session_count,
        "security_count": artifact.security_count,
        "row_count": artifact.row_count,
        "member_row_count": artifact.member_row_count,
        "row_inventory_sha256": artifact.row_inventory_sha256,
        "feature_spec_receipt_sha256": artifact.feature_spec_receipt_sha256,
        "feature_source_sha256": artifact.feature_source_sha256,
    }


def materialize_massive_session_panel_v1(
    *,
    daily_feature_root: str | Path,
    persisted_root: str | Path,
    output_root: str | Path,
    bars_artifacts: Sequence[MassiveDailyBarsArtifactV0],
    tape_artifacts: Sequence[MassiveDailyTapeArtifactV0],
    persisted_manifests: Sequence[MassivePersistedPartitionManifestV1],
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    start_session_date: str,
    end_session_date: str,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveSessionPanelArtifactV1:
    """Publish the exact PIT identity × exchange-session grid."""

    session_authority.validate()
    identity_authority.validate()
    bars = tuple(sorted(bars_artifacts, key=lambda row: row.source_session_date))
    tape = tuple(sorted(tape_artifacts, key=lambda row: row.source_session_date))
    manifests = tuple(
        sorted(persisted_manifests, key=lambda row: row.source_session_date)
    )
    sessions = tuple(
        row
        for row in session_authority.sessions
        if row.exchange == "XNYS"
        and start_session_date <= row.session_date <= end_session_date
    )
    expected_dates = tuple(row.session_date for row in sessions)
    if (
        not expected_dates
        or tuple(row.source_session_date for row in bars) != expected_dates
        or tuple(row.source_session_date for row in tape) != expected_dates
        or tuple(row.source_session_date for row in manifests) != expected_dates
    ):
        raise MassiveSessionPanelV1Error("daily inputs do not exhaust the session grid")
    for bars_artifact in bars:
        validate_massive_daily_bars_v0(root=daily_feature_root, artifact=bars_artifact)
    for tape_artifact in tape:
        validate_massive_daily_tape_v0(root=daily_feature_root, artifact=tape_artifact)
    conditions = {
        bars_artifact.condition_authority_receipt_sha256 for bars_artifact in bars
    } | {tape_artifact.condition_authority_receipt_sha256 for tape_artifact in tape}
    if len(conditions) != 1:
        raise MassiveSessionPanelV1Error("daily condition authorities differ")
    inputs: list[MassiveSessionInputReceiptV1] = []
    for bars_artifact, tape_artifact, manifest in zip(
        bars, tape, manifests, strict=True
    ):
        manifest.validate()
        if (
            manifest.identity_authority_receipt_sha256
            != identity_authority.receipt_sha256
            or bars_artifact.persisted_partition_manifest_receipt_sha256
            != manifest.receipt_sha256
            or tape_artifact.persisted_partition_manifest_receipt_sha256
            != manifest.receipt_sha256
        ):
            raise MassiveSessionPanelV1Error("daily source authority chain differs")
        body = {
            "source_session_date": manifest.source_session_date,
            "daily_bars_artifact_receipt_sha256": bars_artifact.receipt_sha256,
            "daily_tape_artifact_receipt_sha256": tape_artifact.receipt_sha256,
            "persisted_partition_manifest_receipt_sha256": manifest.receipt_sha256,
        }
        inputs.append(
            MassiveSessionInputReceiptV1(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
        )
    bars_by_session = {
        artifact.source_session_date: {row.security_id: row for row in artifact.rows}
        for artifact in bars
    }
    tape_by_session = {
        artifact.source_session_date: {row.security_id: row for row in artifact.rows}
        for artifact in tape
    }
    rows = build_massive_session_panel_rows_v1(
        sessions=sessions,
        identity_authority=identity_authority,
        bars_by_session=bars_by_session,
        tape_by_session=tape_by_session,
        event_counts_by_session=_event_counts(
            persisted_root=persisted_root,
            manifests=manifests,
        ),
    )
    relative = (
        f"massive-profitability-p0/session-panel-v1/"
        f"{start_session_date}-{end_session_date}.json"
    )
    placeholder = MassiveSessionPanelArtifactV1(
        exchange="XNYS",
        start_session_date=start_session_date,
        end_session_date=end_session_date,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        condition_authority_receipt_sha256=next(iter(conditions)),
        input_receipts=tuple(inputs),
        rows=rows,
        session_count=len(sessions),
        security_count=len(identity_authority.security_master),
        row_count=len(rows),
        member_row_count=sum(row.pit_member for row in rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        feature_spec_receipt_sha256=MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        feature_source_sha256=MASSIVE_SESSION_PANEL_V1_SOURCE_SHA256,
        loaded_source=bars[-1].loaded_source,
        receipt_sha256="0" * 64,
    )
    payload = _payload(placeholder)
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_SESSION_PANEL_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root,
        relative_payload_path=relative,
        verified_at_ms=published_at_ms,
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    validate_massive_session_panel_v1(root=output_root, artifact=result)
    return result


def validate_massive_session_panel_v1(
    *,
    root: str | Path,
    artifact: MassiveSessionPanelArtifactV1,
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveSessionPanelV1Error("session-panel source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveSessionPanelV1Error("session-panel bytes differ")


__all__ = [
    "MASSIVE_SESSION_PANEL_V1_DATASET",
    "MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_SESSION_PANEL_V1_SPEC_SHA256",
    "MassiveSessionEventCountsV1",
    "MassiveSessionInputReceiptV1",
    "MassiveSessionPanelArtifactV1",
    "MassiveSessionPanelRowV1",
    "MassiveSessionPanelV1Error",
    "build_massive_session_panel_rows_v1",
    "materialize_massive_session_panel_v1",
    "validate_massive_session_panel_v1",
]
