"""Package-owned action support and factor exposures for adaptive alpha."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from statistics import fmean, pstdev

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.alpha.targets import OriginExposurePanel
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
    parse_massive_economic_origin_coverage_v8,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    _economic_history,
    _ordered_events,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-origin-authority-v1"
)
MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "support": "latest-complete-effective-and-available-action-membership-group",
        "history": "decision-minus-63-through-decision-regular-close",
        "economic_values": "corporate-and-terminal-accounted-daily-close-path",
        "exposures": (
            "intercept",
            "log_economic_value",
            "log_trailing_63_adv",
            "reversal_5",
            "momentum_21_minus_5",
            "economic_volatility_63",
        ),
        "regression_weight": "trailing-63-session-adv",
        "future_sources": "excluded-from-origin-semantic-identity",
        "duration_prior": False,
        "downstream_authorization": False,
    }
)
MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1 = (
    "intercept",
    "log_economic_value",
    "log_trailing_63_adv",
    "reversal_5",
    "momentum_21_minus_5",
    "economic_volatility_63",
)


class MassiveAdaptiveOriginAuthorityV1Error(ValueError):
    """Adaptive membership or exposure inputs are not causal and source-owned."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOriginAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOriginExposureRowV1:
    security_id: str
    universe_rank: int
    exposures: tuple[float, ...]
    regression_weight: float
    qualified: bool
    membership_row_receipt_sha256: str
    identity_row_receipt_sha256: str
    daily_row_inventory_sha256: str
    economic_history_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.security_id
            or isinstance(self.universe_rank, bool)
            or not isinstance(self.universe_rank, int)
            or self.universe_rank <= 0
            or len(self.exposures) != len(MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1)
            or self.exposures[0] != 1.0
            or self.regression_weight <= 0.0
            or not isinstance(self.qualified, bool)
        ):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "adaptive origin exposure row differs"
            )
        if not self.qualified and (
            self.exposures != (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            or self.regression_weight != 1.0
        ):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "unqualified adaptive exposure is not zero-masked"
            )
        for name in (
            "membership_row_receipt_sha256",
            "identity_row_receipt_sha256",
            "daily_row_inventory_sha256",
            "economic_history_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "adaptive exposure row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOriginAuthorityV1:
    decision_session_date: str
    decision_at_ms: int
    membership_effective_at_ms: int
    membership_available_at_ms: int
    history_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    universe_ranks: tuple[int, ...]
    rows: tuple[MassiveAdaptiveOriginExposureRowV1, ...]
    exposure_panel: OriginExposurePanel
    decision_clock_receipt_sha256: str
    session_authority_receipt_sha256: str
    action_universe_rule_receipt_sha256: str
    membership_group_inventory_sha256: str
    selected_identity_inventory_sha256: str
    selected_daily_session_inventory_sha256: str
    selected_daily_row_inventory_sha256: str
    scoped_economic_event_inventory_sha256: str
    row_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_paths_replayed: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        self.exposure_panel.validate()
        for row in self.rows:
            row.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SCHEMA
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256
            or len(self.history_session_dates) != 64
            or self.history_session_dates != tuple(sorted(set(self.history_session_dates)))
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or self.security_ids != tuple(row.security_id for row in self.rows)
            or self.universe_ranks != tuple(row.universe_rank for row in self.rows)
            or self.exposure_panel.origin_at_ms != self.decision_at_ms
            or self.exposure_panel.available_at_ms > self.decision_at_ms
            or self.exposure_panel.asset_ids != self.security_ids
            or self.exposure_panel.exposure_names
            != MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1
            or self.exposure_panel.exposures != tuple(row.exposures for row in self.rows)
            or self.exposure_panel.regression_weights
            != tuple(row.regression_weight for row in self.rows)
            or self.exposure_panel.qualified_asset_mask
            != tuple(row.qualified for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.membership_available_at_ms > self.decision_at_ms
            or self.membership_effective_at_ms > self.decision_at_ms
            or not self.source_paths_replayed
        ):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "adaptive origin identity or replay differs"
            )
        if any(
            (
                self.predictive_training_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "adaptive origin authorizes downstream use"
            )
        assert_no_adaptive_hold_semantics(asdict(self))
        for name in (
            "decision_clock_receipt_sha256",
            "session_authority_receipt_sha256",
            "action_universe_rule_receipt_sha256",
            "membership_group_inventory_sha256",
            "selected_identity_inventory_sha256",
            "selected_daily_session_inventory_sha256",
            "selected_daily_row_inventory_sha256",
            "scoped_economic_event_inventory_sha256",
            "row_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveOriginAuthorityV1Error(
                "adaptive origin receipt differs"
            )


def build_massive_adaptive_origin_authority_v1(
    *,
    economic_coverage_root: str | Path,
    decision_clock: MassiveDecisionClockAuthority,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    economic_coverage: MassiveEconomicOriginCoverageV8,
) -> MassiveAdaptiveOriginAuthorityV1:
    """Derive action support and residualization exposures without caller rows."""

    decision_clock.validate()
    session_authority.validate()
    identity_authority.validate()
    daily_input_authority.validate()
    terminal_authority.validate()
    economic_coverage.validate()
    if parse_massive_economic_origin_coverage_v8(
        root=economic_coverage_root, loaded_source=economic_coverage.loaded_source
    ) != economic_coverage:
        raise MassiveAdaptiveOriginAuthorityV1Error("economic coverage replay differs")
    action_rule = checked_pit_universe_rule(
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule
    )
    decision_at = decision_clock.decision_at_ns // 1_000_000
    if (
        identity_authority.rule.receipt_sha256 != action_rule.receipt_sha256
        or decision_clock.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or daily_input_authority.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or economic_coverage.decision_at_ms != decision_at
        or economic_coverage.terminal_source_receipt_sha256
        != terminal_authority.terminal_source_semantic_receipt_sha256
    ):
        raise MassiveAdaptiveOriginAuthorityV1Error("adaptive origin roots differ")
    effective_times = tuple(
        sorted(
            {
                row.effective_at_ms
                for row in identity_authority.membership_events
                if row.effective_at_ms <= decision_at and row.available_at_ms <= decision_at
            }
        )
    )
    if not effective_times:
        raise MassiveAdaptiveOriginAuthorityV1Error("no action membership exists at decision")
    effective = effective_times[-1]
    group = tuple(
        row for row in identity_authority.membership_events if row.effective_at_ms == effective
    )
    if not group or any(
        row.available_at_ms > decision_at or row.observation_end_ms >= decision_at
        for row in group
    ):
        raise MassiveAdaptiveOriginAuthorityV1Error("action membership group is not causal")
    members = tuple(row for row in group if row.is_member)
    if not members:
        raise MassiveAdaptiveOriginAuthorityV1Error("action membership is empty")
    member_by_id = {row.security_id: row for row in members}
    support = tuple(sorted(member_by_id))
    decision_index = next(
        index
        for index, row in enumerate(session_authority.sessions)
        if row.session_date == decision_clock.session_date
    )
    if decision_index < 63:
        raise MassiveAdaptiveOriginAuthorityV1Error("origin lacks 64 source sessions")
    history_sessions = tuple(session_authority.sessions[decision_index - 63 : decision_index + 1])
    history_dates = tuple(row.session_date for row in history_sessions)
    daily_sessions = {row.source_session_date: row for row in daily_input_authority.sessions}
    if not set(history_dates) <= set(daily_sessions) or any(
        daily_sessions[day].vendor_last_modified_at_ms > decision_at for day in history_dates
    ):
        raise MassiveAdaptiveOriginAuthorityV1Error("daily history was unavailable at decision")
    events = _ordered_events(
        native_events=economic_coverage.selected_events,
        terminal_authority=terminal_authority,
        identity_authority=identity_authority,
        start_exclusive_at_ms=history_sessions[0].regular_open_ns // 1_000_000 - 1,
        end_inclusive_at_ms=decision_clock.regular_close_ns // 1_000_000,
    )
    event_inventory = semantic_sha256(tuple(event.event_id for event in events))
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    master_by_id = {row.security_id: row for row in identity_authority.security_master}
    rows: list[MassiveAdaptiveOriginExposureRowV1] = []
    for security_id in support:
        values, _, _ = _economic_history(
            security_id=security_id,
            session_dates=history_dates,
            session_authority=session_authority,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            events=events,
        )
        economic = tuple(values[(day, security_id)] for day in history_dates)
        daily_rows = tuple(
            daily_input_authority.row(session_date=day, security_id=security_id)
            for day in history_dates
        )
        daily_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in daily_rows))
        dollar_rows = daily_rows[-63:]
        qualified = (
            all(value is not None and value > 0.0 for value in economic)
            and all(
                row.bars_valid[dollar_index] and row.bars_values[dollar_index] > 0.0
                for row in dollar_rows
            )
        )
        if qualified:
            economic_values = tuple(float(value) for value in economic if value is not None)
            adv = fmean(row.bars_values[dollar_index] for row in dollar_rows)
            log_returns = tuple(
                log(right / left)
                for left, right in zip(economic_values[:-1], economic_values[1:], strict=True)
            )
            exposures = (
                1.0,
                log(economic_values[-1]),
                log(adv),
                -(economic_values[-1] / economic_values[-6] - 1.0),
                economic_values[-6] / economic_values[-22] - 1.0,
                pstdev(log_returns),
            )
            weight = adv
        else:
            exposures = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            weight = 1.0
        membership = member_by_id[security_id]
        history_receipt = semantic_sha256(
            {
                "security_id": security_id,
                "economic_values": economic,
                "daily_row_inventory": daily_inventory,
                "event_inventory": event_inventory,
            }
        )
        body = {
            "security_id": security_id,
            "universe_rank": int(membership.universe_rank or 0),
            "exposures": exposures,
            "regression_weight": weight,
            "qualified": qualified,
            "membership_row_receipt_sha256": semantic_sha256(asdict(membership)),
            "identity_row_receipt_sha256": master_by_id[
                security_id
            ].identity_source_receipt_sha256,
            "daily_row_inventory_sha256": daily_inventory,
            "economic_history_receipt_sha256": history_receipt,
        }
        row = MassiveAdaptiveOriginExposureRowV1(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        row.validate()
        rows.append(row)
    available_at = max(
        max(row.available_at_ms for row in group),
        max(daily_sessions[day].vendor_last_modified_at_ms for day in history_dates),
    )
    panel = OriginExposurePanel(
        origin_at_ms=decision_at,
        available_at_ms=available_at,
        asset_ids=support,
        exposure_names=MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1,
        exposures=tuple(row.exposures for row in rows),
        regression_weights=tuple(row.regression_weight for row in rows),
        qualified_asset_mask=tuple(row.qualified for row in rows),
        source_receipt_sha256=semantic_sha256(
            {
                "rows": tuple(row.receipt_sha256 for row in rows),
                "decision_clock": decision_clock.receipt_sha256,
            }
        ),
    )
    panel.validate()
    membership_inventory = semantic_sha256(tuple(asdict(row) for row in group))
    identity_inventory = semantic_sha256(
        tuple(master_by_id[security_id].identity_source_receipt_sha256 for security_id in support)
    )
    session_inventory = semantic_sha256(
        tuple(daily_sessions[day].receipt_sha256 for day in history_dates)
    )
    daily_inventory = semantic_sha256(tuple(row.daily_row_inventory_sha256 for row in rows))
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic = {
        "schema": MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SCHEMA,
        "decision_session_date": decision_clock.session_date,
        "decision_at_ms": decision_at,
        "membership_effective_at_ms": effective,
        "membership_available_at_ms": max(row.available_at_ms for row in group),
        "history_session_dates": history_dates,
        "security_ids": support,
        "universe_ranks": tuple(row.universe_rank for row in rows),
        "rows": tuple(asdict(row) for row in rows),
        "exposure_panel": asdict(panel),
        "decision_clock_receipt_sha256": decision_clock.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "action_universe_rule_receipt_sha256": action_rule.receipt_sha256,
        "membership_group_inventory_sha256": membership_inventory,
        "selected_identity_inventory_sha256": identity_inventory,
        "selected_daily_session_inventory_sha256": session_inventory,
        "selected_daily_row_inventory_sha256": daily_inventory,
        "scoped_economic_event_inventory_sha256": event_inventory,
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveOriginAuthorityV1(
        decision_session_date=decision_clock.session_date,
        decision_at_ms=decision_at,
        membership_effective_at_ms=effective,
        membership_available_at_ms=max(row.available_at_ms for row in group),
        history_session_dates=history_dates,
        security_ids=support,
        universe_ranks=tuple(row.universe_rank for row in rows),
        rows=tuple(rows),
        exposure_panel=panel,
        decision_clock_receipt_sha256=decision_clock.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        action_universe_rule_receipt_sha256=action_rule.receipt_sha256,
        membership_group_inventory_sha256=membership_inventory,
        selected_identity_inventory_sha256=identity_inventory,
        selected_daily_session_inventory_sha256=session_inventory,
        selected_daily_row_inventory_sha256=daily_inventory,
        scoped_economic_event_inventory_sha256=event_inventory,
        row_inventory_sha256=row_inventory,
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        specification_sha256=MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
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
    "MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1",
    "MassiveAdaptiveOriginAuthorityV1",
    "MassiveAdaptiveOriginAuthorityV1Error",
    "MassiveAdaptiveOriginExposureRowV1",
    "build_massive_adaptive_origin_authority_v1",
]
