"""Source-derived inner-validation checkpoint selection for adaptive profit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    MassiveAdaptiveProfitTraceV1,
)
from rl_quant.evaluation.massive_adaptive_profitability_authority_v1 import (
    MassiveAdaptiveProfitabilityAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)

MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V2_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-candidate-v2"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-selection-v2"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_RULE = (
    "eligible-primary-dollar-profit-active-log-wealth-drawdown-earliest-epoch-v2"
)


class MassiveAdaptiveProfitCheckpointSelectionV2Error(ValueError):
    """Source-derived cost ladders or checkpoint inventories differ."""


def _maximum_drawdown(trace: MassiveAdaptiveProfitTraceV1) -> float:
    peak = trace.initial_capital
    maximum = 0.0
    for row in trace.rows:
        peak = max(peak, row.posttrade_equity)
        maximum = max(maximum, 1.0 - row.posttrade_equity / peak)
    return maximum


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointCandidateV2:
    epoch_index: int
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    inference_plan_receipt_sha256: str
    economic_source_inventory_sha256: str
    primary_cost_basis_points: float
    low_cost_basis_points: float
    high_cost_basis_points: float
    primary_dollar_net_profit: float
    primary_terminal_net_return: float
    low_cost_terminal_net_return: float
    high_cost_terminal_net_return: float
    primary_active_log_wealth: float
    maximum_drawdown: float
    eligibility_failures: tuple[str, ...]
    economically_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        numbers = (
            self.primary_cost_basis_points,
            self.low_cost_basis_points,
            self.high_cost_basis_points,
            self.primary_dollar_net_profit,
            self.primary_terminal_net_return,
            self.low_cost_terminal_net_return,
            self.high_cost_terminal_net_return,
            self.primary_active_log_wealth,
            self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V2_SCHEMA
            or self.epoch_index < 0
            or any(not math.isfinite(value) for value in numbers)
            or not self.low_cost_basis_points
            < self.primary_cost_basis_points
            < self.high_cost_basis_points
            or self.maximum_drawdown < 0.0
            or self.eligibility_failures
            != tuple(sorted(set(self.eligibility_failures)))
            or self.economically_eligible != (not self.eligibility_failures)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
                "adaptive checkpoint candidate differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    @property
    def selection_key(self) -> tuple[float, ...]:
        self.validate()
        return (
            self.primary_dollar_net_profit,
            self.primary_active_log_wealth,
            -self.maximum_drawdown,
            -float(self.epoch_index),
        )


def build_massive_adaptive_profit_checkpoint_candidate_v2(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    primary_trace: MassiveAdaptiveProfitTraceV1,
    low_cost_trace: MassiveAdaptiveProfitTraceV1,
    high_cost_trace: MassiveAdaptiveProfitTraceV1,
    primary_authority: MassiveAdaptiveProfitabilityAuthorityV1,
    low_cost_authority: MassiveAdaptiveProfitabilityAuthorityV1,
    high_cost_authority: MassiveAdaptiveProfitabilityAuthorityV1,
) -> MassiveAdaptiveProfitCheckpointCandidateV2:
    """Derive one candidate using replayed 10/20/40-bp economic traces."""

    for value in (
        checkpoint,
        forecast_archive,
        primary_trace,
        low_cost_trace,
        high_cost_trace,
        primary_authority,
        low_cost_authority,
        high_cost_authority,
    ):
        value.validate()
    traces = (low_cost_trace, primary_trace, high_cost_trace)
    authorities = (low_cost_authority, primary_authority, high_cost_authority)
    if (
        forecast_archive.checkpoint_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or forecast_archive.checkpoint_source_receipt_sha256
        != checkpoint.loaded_source.receipt_sha256
        or forecast_archive.model_state_receipt_sha256
        != checkpoint.model_state_receipt_sha256
        or any(
            authority.trace_receipt_sha256 != trace.semantic_receipt_sha256
            for authority, trace in zip(authorities, traces, strict=True)
        )
        or any(
            trace.forecast_archive_receipt_sha256
            != forecast_archive.semantic_receipt_sha256
            for trace in traces
        )
        or any(trace.evaluation_role != "inner_validation" for trace in traces)
        or len({trace.inference_plan_receipt_sha256 for trace in traces}) != 1
        or len({trace.fill_source_receipt_sha256 for trace in traces}) != 1
        or len({trace.daily_input_receipt_sha256 for trace in traces}) != 1
        or len({trace.identity_authority_receipt_sha256 for trace in traces}) != 1
        or len(
            {trace.economic_event_authority_inventory_sha256 for trace in traces}
        )
        != 1
        or len({trace.initial_capital for trace in traces}) != 1
        or tuple(row.decision_session_date for row in low_cost_trace.rows)
        != tuple(row.decision_session_date for row in primary_trace.rows)
        or tuple(row.decision_session_date for row in high_cost_trace.rows)
        != tuple(row.decision_session_date for row in primary_trace.rows)
        or any(
            not trace.frozen_actions_replayed
            or trace.frozen_decision_trace_receipt_sha256
            != primary_trace.semantic_receipt_sha256
            for trace in (low_cost_trace, high_cost_trace)
        )
        or primary_trace.frozen_actions_replayed
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
            "checkpoint and frozen-action economic evidence differ"
        )
    target_inventory = tuple(
        row.decision_target_receipt_sha256 for row in primary_trace.rows
    )
    if any(
        tuple(row.decision_target_receipt_sha256 for row in trace.rows)
        != target_inventory
        for trace in (low_cost_trace, high_cost_trace)
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
            "cost ladder did not reuse the primary target inventory"
        )
    low_return = low_cost_trace.final_equity / low_cost_trace.initial_capital - 1.0
    primary_return = primary_trace.final_equity / primary_trace.initial_capital - 1.0
    high_return = high_cost_trace.final_equity / high_cost_trace.initial_capital - 1.0
    failures: list[str] = []
    if primary_return <= 0.0:
        failures.append("nonpositive-primary-net-return")
    if high_return < 0.0:
        failures.append("negative-high-cost-net-return")
    if primary_trace.cumulative_active_log_return <= 0.0:
        failures.append("nonpositive-active-log-wealth")
    if not low_return >= primary_return >= high_return:
        failures.append("nonmonotone-frozen-action-cost-ladder")
    source_qualified = bool(
        isinstance(checkpoint, MassiveAdaptiveCheckpointV1)
        and isinstance(forecast_archive, MassiveAdaptiveForecastArchiveV2)
        and all(isinstance(trace, MassiveAdaptiveProfitTraceV1) for trace in traces)
        and all(
            isinstance(authority, MassiveAdaptiveProfitabilityAuthorityV1)
            for authority in authorities
        )
        and checkpoint.development_training_authorized
        and forecast_archive.development_forecast_authorized
        and all(trace.source_data_qualified for trace in traces)
        and all(authority.development_profitability_authorized for authority in authorities)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V2_SCHEMA,
        "epoch_index": checkpoint.epoch_index,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": checkpoint.loaded_source.receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": low_cost_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": high_cost_trace.semantic_receipt_sha256,
        "inference_plan_receipt_sha256": primary_trace.inference_plan_receipt_sha256,
        "economic_source_inventory_sha256": semantic_sha256(
            (
                primary_trace.fill_source_receipt_sha256,
                primary_trace.daily_input_receipt_sha256,
                primary_trace.identity_authority_receipt_sha256,
                primary_trace.economic_event_authority_inventory_sha256,
            )
        ),
        "primary_cost_basis_points": primary_trace.transaction_cost_basis_points,
        "low_cost_basis_points": low_cost_trace.transaction_cost_basis_points,
        "high_cost_basis_points": high_cost_trace.transaction_cost_basis_points,
        "primary_dollar_net_profit": (
            primary_trace.final_equity - primary_trace.initial_capital
        ),
        "primary_terminal_net_return": primary_return,
        "low_cost_terminal_net_return": low_return,
        "high_cost_terminal_net_return": high_return,
        "primary_active_log_wealth": primary_trace.cumulative_active_log_return,
        "maximum_drawdown": _maximum_drawdown(primary_trace),
        "eligibility_failures": tuple(sorted(failures)),
        "economically_eligible": not failures,
        "source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveProfitCheckpointCandidateV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointSelectionV2:
    selected_epoch_index: int
    selected_checkpoint_receipt_sha256: str
    selected_candidate_receipt_sha256: str
    candidate_epoch_indices: tuple[int, ...]
    candidate_receipts: tuple[str, ...]
    eligible_epoch_indices: tuple[int, ...]
    inference_plan_receipt_sha256: str
    economic_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    selection_rule: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_RULE
    development_checkpoint_selection_authorized: bool = False
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "development_checkpoint_selection_authorized",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_SCHEMA
            or self.selection_rule
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_RULE
            or self.candidate_epoch_indices
            != tuple(sorted(set(self.candidate_epoch_indices)))
            or len(self.candidate_epoch_indices) != len(self.candidate_receipts)
            or self.eligible_epoch_indices
            != tuple(sorted(set(self.eligible_epoch_indices)))
            or self.selected_epoch_index not in self.eligible_epoch_indices
            or not isinstance(self.source_data_qualified, bool)
            or self.development_checkpoint_selection_authorized
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
                "adaptive checkpoint selection differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def select_massive_adaptive_profit_checkpoint_v2(
    candidates: Sequence[MassiveAdaptiveProfitCheckpointCandidateV2],
) -> MassiveAdaptiveProfitCheckpointSelectionV2:
    """Select only after deriving every metric from common frozen-action traces."""

    ordered = tuple(sorted(candidates, key=lambda row: row.epoch_index))
    for candidate in ordered:
        candidate.validate()
    if (
        not ordered
        or len({row.epoch_index for row in ordered}) != len(ordered)
        or len({row.inference_plan_receipt_sha256 for row in ordered}) != 1
        or len({row.economic_source_inventory_sha256 for row in ordered}) != 1
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
            "checkpoint candidates do not share one validation experiment"
        )
    eligible = tuple(row for row in ordered if row.economically_eligible)
    if not eligible:
        raise MassiveAdaptiveProfitCheckpointSelectionV2Error(
            "no source-derived checkpoint passed the economic ladder"
        )
    selected = max(eligible, key=lambda row: row.selection_key)
    source_qualified = all(row.source_data_qualified for row in ordered)
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_SCHEMA,
        "selected_epoch_index": selected.epoch_index,
        "selected_checkpoint_receipt_sha256": selected.checkpoint_receipt_sha256,
        "selected_candidate_receipt_sha256": selected.semantic_receipt_sha256,
        "candidate_epoch_indices": tuple(row.epoch_index for row in ordered),
        "candidate_receipts": tuple(row.semantic_receipt_sha256 for row in ordered),
        "eligible_epoch_indices": tuple(row.epoch_index for row in eligible),
        "inference_plan_receipt_sha256": selected.inference_plan_receipt_sha256,
        "economic_source_inventory_sha256": selected.economic_source_inventory_sha256,
        "source_data_qualified": source_qualified,
        "selection_rule": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_RULE,
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveProfitCheckpointSelectionV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        development_checkpoint_selection_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V2_RULE",
    "MassiveAdaptiveProfitCheckpointCandidateV2",
    "MassiveAdaptiveProfitCheckpointSelectionV2",
    "MassiveAdaptiveProfitCheckpointSelectionV2Error",
    "build_massive_adaptive_profit_checkpoint_candidate_v2",
    "select_massive_adaptive_profit_checkpoint_v2",
]
