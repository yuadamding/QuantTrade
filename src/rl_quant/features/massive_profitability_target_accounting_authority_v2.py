"""Source-owned fill-to-fill economic target paths for Massive P0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.accounting import (
    EconomicPosition,
    EconomicValuePoint,
    apply_corporate_action,
    compute_post_fill_total_return,
    mark_position,
)
from rl_quant.alpha.contracts import CorporateActionRecord, TerminalEventRecord
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
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
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MassiveProfitabilityFillSourceAuthorityV2,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-profitability-target-accounting-authority-v2"
)
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "entry": "qualified-[15:50,16:00)-fill",
        "path": "entry-through-H63-complete-session-ledger",
        "horizon_exit": "qualified-[15:50,16:00)-fill-for-every-held-security",
        "intermediate_mark": "qualified-daily-close",
        "events": "post-entry-dividend-split-successor-and-terminal-replay",
        "terminal": "exact-carry-or-conservative-total-loss",
        "cash": "zero-return",
        "caller_economic_paths": "prohibited",
        "performance_authorization": False,
    }
)
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS = (1, 5, 21, 63)


class MassiveProfitabilityTargetAccountingAuthorityV2Error(ValueError):
    """A target path cannot be reconstructed from qualified source inputs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _path_for_security(
    *,
    origin_security_id: str,
    origin: MassiveProfitabilityDecisionOriginV1,
    selected_sessions: Sequence[MassiveExchangeSession],
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_authority: MassiveProfitabilityFillSourceAuthorityV2,
    events: Sequence[CorporateActionRecord | TerminalEventRecord],
) -> MassiveProfitabilityTargetEconomicPathRowV1:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    entry_fill = fill_authority.row(
        session_date=origin.decision_session_date,
        security_id=origin_security_id,
    )
    position = (
        EconomicPosition.from_mapping({origin_security_id: 1.0})
        if entry_fill.valid
        else None
    )
    event_index = 0
    applied: list[str] = []
    excluded: list[str] = []
    values: list[float] = []
    valid: list[bool] = []
    terminal: list[bool] = []
    kinds: list[str] = []
    receipts: list[str] = []
    terminal_offset: int | None = None
    conservative_fallback = False
    for offset, session in enumerate(selected_sessions):
        session_date = session.session_date
        at_ms = session.regular_close_ns // 1_000_000
        if position is not None and offset > 0:
            while event_index < len(events) and events[event_index].effective_at_ms <= at_ms:
                event = events[event_index]
                if event.security_id in position.as_mapping():
                    position = apply_corporate_action(position, event)
                    applied.append(event.event_id)
                    if event.event_id.startswith("FALLBACK:"):
                        conservative_fallback = True
                else:
                    excluded.append(event.event_id)
                event_index += 1
        value: float | None = None
        mark_receipts: list[str] = []
        mark_kind = "market"
        is_terminal = False
        if position is not None:
            if offset == 0:
                value = entry_fill.fill_vwap
                mark_receipts.append(entry_fill.receipt_sha256)
            else:
                marks: dict[str, float] = {}
                missing = False
                for holding_id in position.as_mapping():
                    if offset in MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS:
                        fill = fill_authority.row(
                            session_date=session_date, security_id=holding_id
                        )
                        if not fill.valid:
                            missing = True
                            break
                        marks[holding_id] = fill.fill_vwap
                        mark_receipts.append(fill.receipt_sha256)
                    else:
                        daily = daily_input_authority.row(
                            session_date=session_date, security_id=holding_id
                        )
                        if (
                            not daily.bars_valid[close_index]
                            or daily.bars_values[close_index] <= 0.0
                            or not _listed(
                                security_id=holding_id,
                                at_ms=at_ms,
                                identity_authority=identity_authority,
                            )
                        ):
                            missing = True
                            break
                        marks[holding_id] = daily.bars_values[close_index]
                        assert daily.daily_bar_row_receipt_sha256 is not None
                        mark_receipts.append(daily.daily_bar_row_receipt_sha256)
                if not missing:
                    value = mark_position(position, marks)
            if not position.holdings:
                is_terminal = True
                mark_kind = "terminal-disposition"
                if terminal_offset is None:
                    terminal_offset = offset
        values.append(0.0 if value is None else float(value))
        valid.append(value is not None)
        terminal.append(is_terminal and value is not None)
        kinds.append(mark_kind if value is not None else "market")
        receipts.append(
            semantic_sha256(
                {
                    "origin_security_id": origin_security_id,
                    "session_offset": offset,
                    "position": None if position is None else asdict(position),
                    "mark_receipts": tuple(sorted(mark_receipts)),
                    "applied_event_ids": tuple(applied),
                    "excluded_event_ids": tuple(excluded),
                    "fill_source_authority": fill_authority.semantic_receipt_sha256,
                    "daily_input_authority": (
                        daily_input_authority.semantic_receipt_sha256
                    ),
                }
            )
        )
    fallback_offset = terminal_offset if conservative_fallback else None
    body: dict[str, object] = {
        "security_id": origin_security_id,
        "economic_at_ms": tuple(
            session.regular_close_ns // 1_000_000 for session in selected_sessions
        ),
        "available_at_ms": tuple(
            session.regular_close_ns // 1_000_000 for session in selected_sessions
        ),
        "values": tuple(values),
        "valid": tuple(valid),
        "terminal": tuple(terminal),
        "mark_kinds": tuple(kinds),
        "mark_receipts": tuple(receipts),
        "unresolved_terminal_fallback_session_offset": fallback_offset,
        "conservative_total_loss_fallback": conservative_fallback,
    }
    row = MassiveProfitabilityTargetEconomicPathRowV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    row.validate()
    return row


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetAccountingAuthorityV2:
    origin_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    decision_session_date: str
    session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityTargetEconomicPathRowV1, ...]
    horizons: tuple[int, ...]
    daily_input_authority_semantic_receipt_sha256: str
    fill_source_authority_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    scoped_economic_event_inventory_sha256: str
    row_inventory_sha256: str
    fill_sources_qualified: bool
    economic_values_data_qualified: bool
    terminal_accounting_complete: bool
    conservative_total_loss_target_count: int
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    economic_archive_audit_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "economic_archive_audit_receipt_sha256",
                "audit_receipt_sha256",
            }
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256
            or len(self.session_dates) != 64
            or self.session_dates != tuple(sorted(set(self.session_dates)))
            or self.session_dates[0] != self.decision_session_date
            or self.horizons
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS
        ):
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target accounting V2 identity or interval differs"
            )
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target accounting V2 support differs"
            )
        for row in self.rows:
            row.validate()
        fallback_count = sum(row.conservative_total_loss_fallback for row in self.rows)
        if (
            self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not isinstance(self.fill_sources_qualified, bool)
            or not isinstance(self.economic_values_data_qualified, bool)
            or self.terminal_accounting_complete is not True
            or self.conservative_total_loss_target_count != fallback_count
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target accounting V2 qualification or inventory differs"
            )
        for name in (
            "origin_receipt_sha256",
            "origin_plan_semantic_receipt_sha256",
            "daily_input_authority_semantic_receipt_sha256",
            "fill_source_authority_semantic_receipt_sha256",
            "terminal_authority_semantic_receipt_sha256",
            "scoped_economic_event_inventory_sha256",
            "row_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "economic_archive_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target accounting V2 semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "economic_archive_audit_receipt_sha256": (
                    self.economic_archive_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target accounting V2 audit receipt differs"
            )

    def target(self, *, security_id: str, horizon_sessions: int) -> float:
        if horizon_sessions not in self.horizons:
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target horizon is outside the frozen P0 inventory"
            )
        row = next((value for value in self.rows if value.security_id == security_id), None)
        if row is None:
            raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
                "target security is absent"
            )
        points = tuple(
            EconomicValuePoint(
                session_index=offset,
                economic_at_ms=row.economic_at_ms[offset],
                available_at_ms=row.available_at_ms[offset],
                value=row.values[offset],
                mark_kind=row.mark_kinds[offset],
                terminal=row.terminal[offset],
            )
            for offset in range(64)
            if row.valid[offset]
        )
        return compute_post_fill_total_return(
            points, fill_session_index=0, horizon_sessions=horizon_sessions
        ).simple_return


def build_massive_profitability_target_accounting_authority_v2(
    *,
    root: str | Path,
    origin: MassiveProfitabilityDecisionOriginV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_source_authority: MassiveProfitabilityFillSourceAuthorityV2,
    economic_coverage: MassiveEconomicOriginCoverageV8,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveProfitabilityTargetAccountingAuthorityV2:
    """Own the entry-fill to H63 holding-vintage accounting replay."""

    origin.validate()
    origin_plan.validate()
    session_authority.validate()
    identity_authority.validate()
    daily_input_authority.validate()
    fill_source_authority.validate()
    terminal_authority.validate()
    if origin.receipt_sha256 not in {
        row.receipt_sha256 for row in origin_plan.origin_plan_v1.origins
    }:
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "target origin is absent from the frozen V2 plan"
        )
    reloaded_coverage = parse_massive_economic_origin_coverage_v8(
        root=root, loaded_source=economic_coverage.loaded_source
    )
    if reloaded_coverage.semantic_receipt_sha256 != economic_coverage.semantic_receipt_sha256:
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "target economic coverage differs after committed-byte reparse"
        )
    if (
        reloaded_coverage.terminal_source_receipt_sha256
        != terminal_authority.terminal_source_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "target economic coverage and terminal authority differ"
        )
    sessions = tuple(session_authority.sessions)
    by_date = {row.session_date: index for index, row in enumerate(sessions)}
    decision_index = by_date.get(origin.decision_session_date)
    if decision_index is None or decision_index + 63 >= len(sessions):
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "target origin lacks a complete H63 session path"
        )
    selected_sessions = sessions[decision_index : decision_index + 64]
    session_dates = tuple(row.session_date for row in selected_sessions)
    if not set(session_dates) <= {
        row.source_session_date for row in daily_input_authority.sessions
    }:
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "daily input authority lacks the complete target interval"
        )
    required_fill_dates = {
        session_dates[offset]
        for offset in (
            0,
            *MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        )
    }
    if not required_fill_dates <= set(fill_source_authority.session_dates):
        raise MassiveProfitabilityTargetAccountingAuthorityV2Error(
            "fill source authority lacks an entry or horizon exit session"
        )
    events = _ordered_events(
        native_events=reloaded_coverage.selected_events,
        terminal_authority=terminal_authority,
        identity_authority=identity_authority,
        start_exclusive_at_ms=origin.fill_end_at_ms,
        end_inclusive_at_ms=(selected_sessions[-1].regular_close_ns // 1_000_000),
    )
    rows = tuple(
        _path_for_security(
            origin_security_id=security_id,
            origin=origin,
            selected_sessions=selected_sessions,
            session_authority=session_authority,
            identity_authority=identity_authority,
            daily_input_authority=daily_input_authority,
            fill_authority=fill_source_authority,
            events=events,
        )
        for security_id in sorted(origin.decision_member_security_ids)
    )
    scoped_event_receipts = tuple(
        sorted(
            semantic_sha256(asdict(event))
            for event in events
        )
    )
    fill_qualified = fill_source_authority.fill_source_data_qualified
    economic_qualified = (
        daily_input_authority.daily_input_data_qualified
        and reloaded_coverage.coverage_qualified
        and terminal_authority.structural_terminal_coverage_complete
    )
    fallback_count = sum(row.conservative_total_loss_fallback for row in rows)
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
        "origin_receipt_sha256": origin.receipt_sha256,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "decision_session_date": origin.decision_session_date,
        "session_dates": session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_authority_semantic_receipt_sha256": (
            fill_source_authority.semantic_receipt_sha256
        ),
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority.semantic_receipt_sha256
        ),
        "scoped_economic_event_inventory_sha256": semantic_sha256(
            scoped_event_receipts
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "fill_sources_qualified": fill_qualified,
        "economic_values_data_qualified": economic_qualified,
        "terminal_accounting_complete": True,
        "conservative_total_loss_target_count": fallback_count,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveProfitabilityTargetAccountingAuthorityV2(
        origin_receipt_sha256=origin.receipt_sha256,
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        decision_session_date=origin.decision_session_date,
        session_dates=session_dates,
        rows=rows,
        horizons=MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        daily_input_authority_semantic_receipt_sha256=(
            daily_input_authority.semantic_receipt_sha256
        ),
        fill_source_authority_semantic_receipt_sha256=(
            fill_source_authority.semantic_receipt_sha256
        ),
        terminal_authority_semantic_receipt_sha256=(
            terminal_authority.semantic_receipt_sha256
        ),
        scoped_economic_event_inventory_sha256=semantic[
            "scoped_economic_event_inventory_sha256"
        ],  # type: ignore[arg-type]
        row_inventory_sha256=semantic["row_inventory_sha256"],  # type: ignore[arg-type]
        fill_sources_qualified=fill_qualified,
        economic_values_data_qualified=economic_qualified,
        terminal_accounting_complete=True,
        conservative_total_loss_target_count=fallback_count,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        economic_archive_audit_receipt_sha256=reloaded_coverage.audit_receipt_sha256,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_archive_audit_receipt_sha256": (
                    reloaded_coverage.audit_receipt_sha256
                ),
            }
        ),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS",
    "MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA",
    "MassiveProfitabilityTargetAccountingAuthorityV2",
    "MassiveProfitabilityTargetAccountingAuthorityV2Error",
    "build_massive_profitability_target_accounting_authority_v2",
]
