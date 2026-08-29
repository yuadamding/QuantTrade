"""Replay adaptive economic paths and targets from frozen Massive roots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.accounting import EconomicPosition, apply_corporate_action, mark_position
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA,
    MassiveAdaptiveAlphaTargetsV1,
    MassiveAdaptiveEconomicPathV1,
    build_massive_adaptive_alpha_targets_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import MassiveAdaptiveFillSourceV1
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
    build_massive_adaptive_origin_authority_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
    parse_massive_economic_origin_coverage_v8,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    _listed,
    _ordered_events,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SCHEMA = "rl-quant.massive-adaptive-source-targets-v1"
MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "origin": "close-plus-60-minute-decision-clock",
        "entry": "next-session-09:35-09:45-vwap",
        "boundary_marks": "same-window-vwap-at-offsets-0-1-5-10-21-42-63-126",
        "intermediate_marks": "source-qualified-regular-close",
        "events": "corporate-and-terminal-before-each-economic-mark",
        "path": "complete-offset-0-through-126-or-explicit-missing",
        "duration_prior": False,
        "downstream_authorization": False,
    }
)
_BOUNDARIES = frozenset(
    {0}
    | {bucket.start_offset_sessions for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS}
    | {bucket.end_offset_sessions for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS}
)


class MassiveAdaptiveSourceTargetsV1Error(ValueError):
    """Adaptive target paths do not replay from their source authorities."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveSourceTargetsV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveSourceTargetsV1:
    decision_session_date: str
    fill_session_date: str
    security_ids: tuple[str, ...]
    paths: tuple[MassiveAdaptiveEconomicPathV1, ...]
    targets: MassiveAdaptiveAlphaTargetsV1
    origin_authority_receipt_sha256: str
    decision_clock_receipt_sha256: str
    session_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    terminal_authority_receipt_sha256: str
    economic_coverage_receipt_sha256: str
    economic_path_inventory_sha256: str
    target_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_paths_replayed: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        self.targets.validate()
        for path in self.paths:
            path.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SCHEMA
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256
            or not self.source_paths_replayed
            or self.security_ids != tuple(path.security_id for path in self.paths)
            or self.security_ids != self.targets.security_ids
            or self.decision_session_date != self.targets.decision_session_date
            or self.economic_path_inventory_sha256
            != semantic_sha256(tuple(path.receipt_sha256 for path in self.paths))
            or self.target_receipt_sha256 != self.targets.semantic_receipt_sha256
            or self.origin_authority_receipt_sha256
            != self.targets.origin_receipt_sha256
            or self.fill_source_receipt_sha256 != self.targets.fill_source_receipt_sha256
            or self.terminal_authority_receipt_sha256
            != self.targets.terminal_authority_receipt_sha256
            or self.economic_coverage_receipt_sha256
            != self.targets.economic_coverage_receipt_sha256
        ):
            raise MassiveAdaptiveSourceTargetsV1Error("adaptive source-target binding differs")
        if any(
            (
                self.predictive_training_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptiveSourceTargetsV1Error("source targets authorize downstream use")
        assert_no_adaptive_hold_semantics(asdict(self))
        for name in (
            "decision_clock_receipt_sha256",
            "origin_authority_receipt_sha256",
            "session_authority_receipt_sha256",
            "identity_authority_receipt_sha256",
            "daily_input_authority_receipt_sha256",
            "fill_source_receipt_sha256",
            "terminal_authority_receipt_sha256",
            "economic_coverage_receipt_sha256",
            "economic_path_inventory_sha256",
            "target_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveSourceTargetsV1Error("source-target receipt differs")


def _daily_session(
    authority: MassiveProfitabilityDailyInputAuthorityV1, session_date: str
):
    return next(row for row in authority.sessions if row.source_session_date == session_date)


def _build_path(
    *,
    security_id: str,
    decision_at_ms: int,
    sessions: Sequence,
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input: MassiveProfitabilityDailyInputAuthorityV1,
    identity: PITSecurityUniverseAuthority,
    events: Sequence,
    source_root_receipt: str,
) -> MassiveAdaptiveEconomicPathV1:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    first_fill = fill_source.row(
        session_date=sessions[0].session_date, security_id=security_id
    )
    position = (
        EconomicPosition.from_mapping({security_id: 1.0}) if first_fill.valid else None
    )
    event_index = 0
    applied: list[str] = []
    excluded: list[str] = []
    fallback_offset: int | None = None
    latest_event_available = 0
    economic_at: list[int] = []
    available_at: list[int] = []
    values: list[float] = []
    valid: list[bool] = []
    terminal: list[bool] = []
    kinds: list[str] = []
    receipts: list[str] = []
    for offset, session in enumerate(sessions):
        boundary = offset in _BOUNDARIES
        at_ms = (
            fill_source.row(session_date=session.session_date, security_id=security_id).fill_end_at_ms
            if boundary
            else session.regular_close_ns // 1_000_000
        )
        if offset > 0 and position is not None:
            while event_index < len(events) and events[event_index].effective_at_ms <= at_ms:
                event = events[event_index]
                latest_event_available = max(latest_event_available, event.available_at_ms)
                if event.security_id in position.as_mapping():
                    position = apply_corporate_action(position, event)
                    applied.append(event.event_id)
                    if event.event_id.startswith("FALLBACK:"):
                        fallback_offset = offset
                else:
                    excluded.append(event.event_id)
                event_index += 1
        mark_inputs: list[str] = []
        mark_value = 0.0
        mark_valid = position is not None
        is_terminal = False
        kind = "missing"
        if position is not None:
            if not position.holdings:
                mark_value = position.cash
                mark_valid = True
                is_terminal = True
                kind = "terminal-disposition"
            else:
                marks: dict[str, float] = {}
                for holding_id in position.as_mapping():
                    if not _listed(
                        security_id=holding_id,
                        at_ms=at_ms,
                        identity_authority=identity,
                    ):
                        mark_valid = False
                        break
                    if boundary:
                        row = fill_source.row(
                            session_date=session.session_date, security_id=holding_id
                        )
                        if not row.valid:
                            mark_valid = False
                            break
                        marks[holding_id] = row.fill_vwap
                        mark_inputs.append(row.receipt_sha256)
                    else:
                        row = daily_input.row(
                            session_date=session.session_date, security_id=holding_id
                        )
                        if not row.bars_valid[close_index] or row.bars_values[close_index] <= 0.0:
                            mark_valid = False
                            break
                        marks[holding_id] = row.bars_values[close_index]
                        mark_inputs.append(row.daily_bar_row_receipt_sha256)
                if mark_valid:
                    mark_value = mark_position(position, marks)
                    kind = "market"
        daily_session = _daily_session(daily_input, session.session_date)
        available = max(
            at_ms,
            daily_session.authenticated_get_completed_at_ms,
            latest_event_available,
        )
        receipt_body = {
            "security_id": security_id,
            "offset": offset,
            "economic_at_ms": at_ms,
            "available_at_ms": available,
            "position": None if position is None else asdict(position),
            "mark_inputs": tuple(sorted(mark_inputs)),
            "applied_events": tuple(applied),
            "excluded_events": tuple(excluded),
            "value": mark_value if mark_valid else 0.0,
            "valid": mark_valid,
            "terminal": is_terminal,
            "kind": kind if mark_valid else "missing",
            "source_root_receipt": source_root_receipt,
        }
        economic_at.append(at_ms)
        available_at.append(available)
        values.append(mark_value if mark_valid else 0.0)
        valid.append(mark_valid)
        terminal.append(is_terminal)
        kinds.append(kind if mark_valid else "missing")
        receipts.append(semantic_sha256(receipt_body))
    source_path = semantic_sha256(
        {
            "source_root_receipt": source_root_receipt,
            "security_id": security_id,
            "mark_receipts": tuple(receipts),
        }
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_PATH_V1_SCHEMA,
        "security_id": security_id,
        "decision_at_ms": decision_at_ms,
        "fill_at_ms": economic_at[0],
        "economic_at_ms": tuple(economic_at),
        "available_at_ms": tuple(available_at),
        "values": tuple(values),
        "valid": tuple(valid),
        "terminal": tuple(terminal),
        "mark_kinds": tuple(kinds),
        "mark_receipts": tuple(receipts),
        "unresolved_terminal_fallback_session_offset": fallback_offset,
        "conservative_total_loss_fallback": fallback_offset is not None,
        "source_economic_path_receipt_sha256": source_path,
    }
    result = MassiveAdaptiveEconomicPathV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def build_massive_adaptive_source_targets_v1(
    *,
    economic_coverage_root: str | Path,
    decision_clock: MassiveDecisionClockAuthority,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_source: MassiveAdaptiveFillSourceV1,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    economic_coverage: MassiveEconomicOriginCoverageV8,
    origin_authority: MassiveAdaptiveOriginAuthorityV1,
    built_at_ms: int,
) -> MassiveAdaptiveSourceTargetsV1:
    """Replay 127 economic marks and build exact adaptive targets."""

    decision_clock.validate()
    session_authority.validate()
    identity_authority.validate()
    daily_input_authority.validate()
    fill_source.validate()
    terminal_authority.validate()
    economic_coverage.validate()
    origin_authority.validate()
    replayed_coverage = parse_massive_economic_origin_coverage_v8(
        root=economic_coverage_root,
        loaded_source=economic_coverage.loaded_source,
    )
    if replayed_coverage != economic_coverage:
        raise MassiveAdaptiveSourceTargetsV1Error("economic coverage replay differs")
    if (
        decision_clock.session_authority_receipt_sha256 != session_authority.receipt_sha256
        or daily_input_authority.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or fill_source.session_authority_receipt_sha256 != session_authority.receipt_sha256
        or fill_source.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or fill_source.condition_authority_receipt_sha256
        != daily_input_authority.condition_authority_receipt_sha256
        or economic_coverage.decision_at_ms != decision_clock.decision_at_ns // 1_000_000
        or economic_coverage.terminal_source_receipt_sha256
        != terminal_authority.terminal_source_semantic_receipt_sha256
    ):
        raise MassiveAdaptiveSourceTargetsV1Error("adaptive source roots differ")
    replayed_origin = build_massive_adaptive_origin_authority_v1(
        economic_coverage_root=economic_coverage_root,
        decision_clock=decision_clock,
        session_authority=session_authority,
        identity_authority=identity_authority,
        daily_input_authority=daily_input_authority,
        terminal_authority=terminal_authority,
        economic_coverage=economic_coverage,
    )
    if replayed_origin != origin_authority:
        raise MassiveAdaptiveSourceTargetsV1Error("adaptive origin replay differs")
    support = origin_authority.security_ids
    decision_index = next(
        index
        for index, session in enumerate(session_authority.sessions)
        if session.session_date == decision_clock.session_date
    )
    if decision_index + 127 >= len(session_authority.sessions):
        raise MassiveAdaptiveSourceTargetsV1Error("decision lacks offset-126 fill maturity")
    path_sessions = tuple(session_authority.sessions[decision_index + 1 : decision_index + 128])
    boundary_dates = tuple(path_sessions[offset].session_date for offset in sorted(_BOUNDARIES))
    if not set(boundary_dates) <= set(fill_source.session_dates) or not set(support) <= set(
        fill_source.supported_security_ids
    ):
        raise MassiveAdaptiveSourceTargetsV1Error("adaptive fill source lacks boundary marks")
    end_ms = path_sessions[-1].regular_close_ns // 1_000_000
    events = _ordered_events(
        native_events=economic_coverage.selected_events,
        terminal_authority=terminal_authority,
        identity_authority=identity_authority,
        start_exclusive_at_ms=fill_source.row(
            session_date=path_sessions[0].session_date, security_id=support[0]
        ).fill_end_at_ms,
        end_inclusive_at_ms=end_ms,
    )
    root_receipt = semantic_sha256(
        {
            "decision_clock": decision_clock.receipt_sha256,
            "session_authority": session_authority.receipt_sha256,
            "identity_authority": identity_authority.receipt_sha256,
            "daily_input": daily_input_authority.semantic_receipt_sha256,
            "fill_source": fill_source.semantic_receipt_sha256,
            "terminal_authority": terminal_authority.semantic_receipt_sha256,
            "economic_coverage": economic_coverage.semantic_receipt_sha256,
            "security_ids": support,
            "path_dates": tuple(row.session_date for row in path_sessions),
            "event_inventory": tuple(event.event_id for event in events),
        }
    )
    paths = tuple(
        _build_path(
            security_id=security_id,
            decision_at_ms=decision_clock.decision_at_ns // 1_000_000,
            sessions=path_sessions,
            fill_source=fill_source,
            daily_input=daily_input_authority,
            identity=identity_authority,
            events=events,
            source_root_receipt=root_receipt,
        )
        for security_id in support
    )
    accounting_receipt = semantic_sha256(
        {
            "source_root_receipt": root_receipt,
            "path_inventory": tuple(path.receipt_sha256 for path in paths),
        }
    )
    targets = build_massive_adaptive_alpha_targets_v1(
        decision_session_date=decision_clock.session_date,
        built_at_ms=built_at_ms,
        paths=paths,
        exposure_panel=origin_authority.exposure_panel,
        origin_receipt_sha256=origin_authority.semantic_receipt_sha256,
        economic_accounting_receipt_sha256=accounting_receipt,
        fill_source_receipt_sha256=fill_source.semantic_receipt_sha256,
        terminal_authority_receipt_sha256=terminal_authority.semantic_receipt_sha256,
        economic_coverage_receipt_sha256=economic_coverage.semantic_receipt_sha256,
    )
    path_inventory = semantic_sha256(tuple(path.receipt_sha256 for path in paths))
    semantic = {
        "schema": MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SCHEMA,
        "decision_session_date": decision_clock.session_date,
        "fill_session_date": path_sessions[0].session_date,
        "security_ids": support,
        "paths": tuple(asdict(path) for path in paths),
        "targets": asdict(targets),
        "origin_authority_receipt_sha256": origin_authority.semantic_receipt_sha256,
        "decision_clock_receipt_sha256": decision_clock.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "daily_input_authority_receipt_sha256": daily_input_authority.semantic_receipt_sha256,
        "fill_source_receipt_sha256": fill_source.semantic_receipt_sha256,
        "terminal_authority_receipt_sha256": terminal_authority.semantic_receipt_sha256,
        "economic_coverage_receipt_sha256": economic_coverage.semantic_receipt_sha256,
        "economic_path_inventory_sha256": path_inventory,
        "target_receipt_sha256": targets.semantic_receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveSourceTargetsV1(
        decision_session_date=decision_clock.session_date,
        fill_session_date=path_sessions[0].session_date,
        security_ids=support,
        paths=paths,
        targets=targets,
        origin_authority_receipt_sha256=origin_authority.semantic_receipt_sha256,
        decision_clock_receipt_sha256=decision_clock.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        daily_input_authority_receipt_sha256=daily_input_authority.semantic_receipt_sha256,
        fill_source_receipt_sha256=fill_source.semantic_receipt_sha256,
        terminal_authority_receipt_sha256=terminal_authority.semantic_receipt_sha256,
        economic_coverage_receipt_sha256=economic_coverage.semantic_receipt_sha256,
        economic_path_inventory_sha256=path_inventory,
        target_receipt_sha256=targets.semantic_receipt_sha256,
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        specification_sha256=MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_sha256(semantic),
        source_paths_replayed=True,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SCHEMA",
    "MassiveAdaptiveSourceTargetsV1",
    "MassiveAdaptiveSourceTargetsV1Error",
    "build_massive_adaptive_source_targets_v1",
]
