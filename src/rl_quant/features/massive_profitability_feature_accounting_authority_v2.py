"""Source-derived 64-session economic feature histories for Massive P0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.accounting import EconomicPosition, apply_corporate_action, mark_position
from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
    MassiveNativeEconomicObservationV8,
    parse_massive_economic_origin_coverage_v8,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_feature_accounting_v1 import (
    MassiveProfitabilityFeatureAccountingV1,
    build_massive_profitability_feature_accounting_v1,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
    MassiveProfitabilityTerminalSupportRowV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-profitability-feature-accounting-authority-v2"
)
MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "history": "exact-source-minus-63-through-source",
        "base": "one-share-at-first-valid-listed-economic-mark",
        "events": "effective-no-later-than-source-close",
        "holding_vintage": "pre-base-and-pre-acquisition-events-excluded",
        "marks": "qualified-daily-input-close",
        "cash": "zero-return",
        "future_economic_sources": "audit-only",
        "corporate_action_predictors": "prohibited",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityFeatureAccountingAuthorityV2Error(ValueError):
    """Feature economic values cannot be reconstructed from qualified inputs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _native_event(row: MassiveNativeEconomicObservationV8) -> CorporateActionRecord:
    row.validate()
    result = CorporateActionRecord(
        event_id=row.logical_event_key,
        security_id=row.security_id,
        kind=CorporateActionKind(row.kind),
        effective_at_ms=row.effective_at_ms,
        available_at_ms=max(row.effective_at_ms, row.research_captured_at_ms),
        cash_per_share=row.cash_per_share,
        share_ratio=row.share_ratio,
    )
    result.validate()
    return result


def _terminal_event(
    row: MassiveProfitabilityTerminalSupportRowV1,
) -> TerminalEventRecord | None:
    row.validate()
    if row.resolution_kind == "live-through-coverage":
        return None
    assert row.effective_at_ms is not None
    assert row.provider_available_at_ms is not None
    kind = (
        TerminalEventKind.WORTHLESS
        if row.resolution_kind == "conservative-total-loss"
        else TerminalEventKind(row.resolution_kind)
    )
    event = TerminalEventRecord(
        event_id=(
            f"FALLBACK:{row.listing_delisting_event_id}"
            if row.conservative_total_loss
            else str(row.listing_delisting_event_id)
        ),
        security_id=row.security_id,
        kind=kind,
        effective_at_ms=row.effective_at_ms,
        available_at_ms=row.provider_available_at_ms,
        cash_per_share=row.cash_per_share,
        successor_security_id=row.successor_security_id,
        successor_ratio=row.successor_ratio,
    )
    event.validate()
    return event


def _listed(
    *, security_id: str, at_ms: int, identity_authority: PITSecurityUniverseAuthority
) -> bool:
    master = next(
        row for row in identity_authority.security_master if row.security_id == security_id
    )
    return master.listing_at_ms <= at_ms and (
        master.delisting_at_ms is None or at_ms < master.delisting_at_ms
    )


def _event_domain(
    *, security_id: str, identity_authority: PITSecurityUniverseAuthority
) -> str:
    master = next(
        row for row in identity_authority.security_master if row.security_id == security_id
    )
    return master.corporate_action_chain_id or security_id


def _ordered_events(
    *,
    native_events: Sequence[MassiveNativeEconomicObservationV8],
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    start_exclusive_at_ms: int,
    end_inclusive_at_ms: int,
) -> tuple[CorporateActionRecord | TerminalEventRecord, ...]:
    values: list[CorporateActionRecord | TerminalEventRecord] = [
        _native_event(row)
        for row in native_events
        if start_exclusive_at_ms < row.effective_at_ms <= end_inclusive_at_ms
    ]
    values.extend(
        event
        for row in terminal_authority.rows
        if (event := _terminal_event(row)) is not None
        and start_exclusive_at_ms < event.effective_at_ms <= end_inclusive_at_ms
    )
    groups: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    for event in values:
        groups[
            (
                _event_domain(
                    security_id=event.security_id,
                    identity_authority=identity_authority,
                ),
                event.effective_at_ms,
            )
        ].append(event.event_id)
    ambiguous = tuple(sorted(key for key, ids in groups.items() if len(ids) > 1))
    if ambiguous:
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "same-time noncommuting feature events lack qualified order evidence"
        )
    return tuple(
        sorted(values, key=lambda row: (row.effective_at_ms, row.event_id))
    )


def _economic_history(
    *,
    security_id: str,
    session_dates: Sequence[str],
    session_authority: MassiveSessionAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    events: Sequence[CorporateActionRecord | TerminalEventRecord],
) -> tuple[
    dict[tuple[str, str], float | None],
    dict[tuple[str, str], str],
    tuple[tuple[str, str], ...],
]:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    values: dict[tuple[str, str], float | None] = {}
    receipts: dict[tuple[str, str], str] = {}
    terminal_keys: list[tuple[str, str]] = []
    position: EconomicPosition | None = None
    event_index = 0
    excluded: list[str] = []
    applied: list[str] = []
    for session_date in session_dates:
        session = session_authority.resolve(exchange="XNYS", session_date=session_date)
        at_ms = session.regular_close_ns // 1_000_000
        origin_daily = daily_input_authority.row(
            session_date=session_date, security_id=security_id
        )
        if position is None:
            if (
                origin_daily.bars_valid[close_index]
                and origin_daily.bars_values[close_index] > 0.0
                and _listed(
                    security_id=security_id,
                    at_ms=at_ms,
                    identity_authority=identity_authority,
                )
            ):
                position = EconomicPosition.from_mapping({security_id: 1.0})
                while event_index < len(events) and (
                    events[event_index].effective_at_ms <= at_ms
                ):
                    excluded.append(events[event_index].event_id)
                    event_index += 1
        else:
            while event_index < len(events) and events[event_index].effective_at_ms <= at_ms:
                event = events[event_index]
                if event.security_id in position.as_mapping():
                    position = apply_corporate_action(position, event)
                    applied.append(event.event_id)
                else:
                    excluded.append(event.event_id)
                event_index += 1
        key = (session_date, security_id)
        mark_receipts: list[str] = []
        value: float | None = None
        terminal = False
        if position is not None:
            marks: dict[str, float] = {}
            missing = False
            for holding_id in position.as_mapping():
                row = daily_input_authority.row(
                    session_date=session_date, security_id=holding_id
                )
                if (
                    not row.bars_valid[close_index]
                    or row.bars_values[close_index] <= 0.0
                    or not _listed(
                        security_id=holding_id,
                        at_ms=at_ms,
                        identity_authority=identity_authority,
                    )
                ):
                    missing = True
                    break
                marks[holding_id] = row.bars_values[close_index]
                assert row.daily_bar_row_receipt_sha256 is not None
                mark_receipts.append(row.daily_bar_row_receipt_sha256)
            if not missing:
                value = mark_position(position, marks)
                terminal = not position.holdings
        values[key] = value
        if terminal:
            terminal_keys.append(key)
        receipts[key] = semantic_sha256(
            {
                "session_date": session_date,
                "origin_security_id": security_id,
                "position": None if position is None else asdict(position),
                "mark_receipts": tuple(sorted(mark_receipts)),
                "applied_event_ids": tuple(applied),
                "excluded_event_ids": tuple(excluded),
                "daily_input_authority": (
                    daily_input_authority.semantic_receipt_sha256
                ),
            }
        )
    return values, receipts, tuple(terminal_keys)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFeatureAccountingAuthorityV2:
    origin_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    accounting: MassiveProfitabilityFeatureAccountingV1
    daily_input_authority_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    scoped_economic_event_inventory_sha256: str
    maximum_actual_input_at_ms: int
    economic_values_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    economic_archive_audit_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        result = {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "economic_archive_audit_receipt_sha256",
                "audit_receipt_sha256",
            }
        }
        result["accounting"] = self.accounting.semantic_unsigned() | {
            "semantic_receipt_sha256": self.accounting.semantic_receipt_sha256
        }
        return result

    def validate(self) -> None:
        self.accounting.validate()
        if (
            self.schema
            != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA
            or self.origin_receipt_sha256 != self.accounting.origin_receipt_sha256
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256
            or self.maximum_actual_input_at_ms != self.accounting.feature_cutoff_at_ms
            or not isinstance(self.economic_values_data_qualified, bool)
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
                "feature accounting authority identity or cutoff differs"
            )
        for name in (
            "origin_receipt_sha256",
            "origin_plan_semantic_receipt_sha256",
            "daily_input_authority_semantic_receipt_sha256",
            "terminal_authority_semantic_receipt_sha256",
            "scoped_economic_event_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "economic_archive_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
                "feature accounting V2 semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "economic_archive_audit_receipt_sha256": (
                    self.economic_archive_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
                "feature accounting V2 audit receipt differs"
            )


def build_massive_profitability_feature_accounting_authority_v2(
    *,
    root: str | Path,
    origin: MassiveProfitabilityDecisionOriginV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    economic_coverage: MassiveEconomicOriginCoverageV8,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveProfitabilityFeatureAccountingAuthorityV2:
    """Reconstruct one bounded feature history from committed source roots."""

    origin.validate()
    origin_plan.validate()
    session_authority.validate()
    identity_authority.validate()
    daily_input_authority.validate()
    terminal_authority.validate()
    if origin.receipt_sha256 not in {
        row.receipt_sha256 for row in origin_plan.origin_plan_v1.origins
    }:
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "feature origin is absent from the frozen V2 plan"
        )
    reloaded_coverage = parse_massive_economic_origin_coverage_v8(
        root=root, loaded_source=economic_coverage.loaded_source
    )
    if reloaded_coverage.semantic_receipt_sha256 != economic_coverage.semantic_receipt_sha256:
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "economic coverage differs after committed-byte reparse"
        )
    if (
        reloaded_coverage.terminal_source_receipt_sha256
        != terminal_authority.terminal_source_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "feature economic coverage and terminal authority differ"
        )
    sessions = tuple(session_authority.sessions)
    by_date = {row.session_date: index for index, row in enumerate(sessions)}
    source_index = by_date.get(origin.source_session_date)
    if source_index is None or source_index < 63:
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "feature origin lacks exact 63-session prehistory"
        )
    selected_sessions = sessions[source_index - 63 : source_index + 1]
    session_dates = tuple(row.session_date for row in selected_sessions)
    if not set(session_dates) <= {
        row.source_session_date for row in daily_input_authority.sessions
    }:
        raise MassiveProfitabilityFeatureAccountingAuthorityV2Error(
            "daily input authority lacks the bounded feature rectangle"
        )
    start_time = selected_sessions[0].regular_close_ns // 1_000_000
    events = _ordered_events(
        native_events=reloaded_coverage.selected_events,
        terminal_authority=terminal_authority,
        identity_authority=identity_authority,
        start_exclusive_at_ms=start_time,
        end_inclusive_at_ms=origin.feature_cutoff_at_ms,
    )
    all_values: dict[tuple[str, str], float | None] = {}
    all_receipts: dict[tuple[str, str], str] = {}
    terminals: list[tuple[str, str]] = []
    for security_id in sorted(origin.decision_member_security_ids):
        values, receipts, terminal_keys = _economic_history(
            security_id=security_id,
            session_dates=session_dates,
            session_authority=session_authority,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            events=events,
        )
        all_values.update(values)
        all_receipts.update(receipts)
        terminals.extend(terminal_keys)
    native_events = tuple(
        row
        for row in reloaded_coverage.selected_events
        if start_time < row.effective_at_ms <= origin.feature_cutoff_at_ms
    )
    scoped_event_receipts = tuple(
        sorted(
            {
                *(row.receipt_sha256 for row in native_events),
                *(
                    semantic_sha256(asdict(row))
                    for row in events
                    if isinstance(row, TerminalEventRecord)
                ),
            }
        )
    )
    scoped_coverage_receipt = semantic_sha256(
        {
            "origin_receipt_sha256": origin.receipt_sha256,
            "accounting_lane": reloaded_coverage.accounting_lane,
            "event_receipts": scoped_event_receipts,
            "terminal_authority_semantic_receipt_sha256": (
                terminal_authority.semantic_receipt_sha256
            ),
            "cash_policy_receipt_sha256": (
                reloaded_coverage.cash_policy.receipt_sha256
            ),
        }
    )
    accounting = build_massive_profitability_feature_accounting_v1(
        origin=origin,
        session_authority=session_authority,
        economic_values=all_values,
        mark_receipts=all_receipts,
        terminal_keys=terminals,
        selected_events=native_events,
        economic_coverage_semantic_receipt_sha256=scoped_coverage_receipt,
        economic_coverage_audit_receipt_sha256=reloaded_coverage.audit_receipt_sha256,
    )
    data_qualified = (
        daily_input_authority.daily_input_data_qualified
        and reloaded_coverage.coverage_qualified
        and terminal_authority.structural_terminal_coverage_complete
    )
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
        "origin_receipt_sha256": origin.receipt_sha256,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "accounting": accounting.semantic_unsigned()
        | {"semantic_receipt_sha256": accounting.semantic_receipt_sha256},
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority.semantic_receipt_sha256
        ),
        "scoped_economic_event_inventory_sha256": semantic_sha256(
            scoped_event_receipts
        ),
        "maximum_actual_input_at_ms": origin.feature_cutoff_at_ms,
        "economic_values_data_qualified": data_qualified,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveProfitabilityFeatureAccountingAuthorityV2(
        origin_receipt_sha256=origin.receipt_sha256,
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        accounting=accounting,
        daily_input_authority_semantic_receipt_sha256=(
            daily_input_authority.semantic_receipt_sha256
        ),
        terminal_authority_semantic_receipt_sha256=(
            terminal_authority.semantic_receipt_sha256
        ),
        scoped_economic_event_inventory_sha256=semantic[
            "scoped_economic_event_inventory_sha256"
        ],  # type: ignore[arg-type]
        maximum_actual_input_at_ms=origin.feature_cutoff_at_ms,
        economic_values_data_qualified=data_qualified,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
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
    "MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA",
    "MassiveProfitabilityFeatureAccountingAuthorityV2",
    "MassiveProfitabilityFeatureAccountingAuthorityV2Error",
    "build_massive_profitability_feature_accounting_authority_v2",
]
