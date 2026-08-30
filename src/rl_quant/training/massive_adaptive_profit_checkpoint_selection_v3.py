"""Fold-bound checkpoint selection from corrected adaptive profit traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v2 import (
    MassiveAdaptiveProfitTraceV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)

MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V3_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-candidate-v3"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V3_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-selection-v3"
)


class MassiveAdaptiveProfitCheckpointSelectionV3Error(ValueError):
    """Checkpoint economics or fold provenance do not reconcile."""


def _maximum_drawdown(trace: MassiveAdaptiveProfitTraceV2) -> float:
    peak = trace.initial_capital
    maximum = 0.0
    for row in trace.rows:
        peak = max(peak, row.posttrade_equity)
        maximum = max(maximum, 1.0 - row.posttrade_equity / peak)
    return maximum


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointCandidateV3:
    fold_index: int
    epoch_index: int
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    training_cutoff_session_date: str
    inner_validation_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
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
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        values = (
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
            self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V3_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or self.epoch_index < 0
            or not self.training_cutoff_session_date
            or any(not math.isfinite(value) for value in values)
            or not self.low_cost_basis_points
            < self.primary_cost_basis_points
            < self.high_cost_basis_points
            or self.maximum_drawdown < 0.0
            or self.eligibility_failures != tuple(sorted(set(self.eligibility_failures)))
            or self.economically_eligible != (not self.eligibility_failures)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
                "adaptive checkpoint candidate v3 differs"
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


def build_massive_adaptive_profit_checkpoint_candidate_v3(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    primary_trace: MassiveAdaptiveProfitTraceV2,
    low_cost_trace: MassiveAdaptiveProfitTraceV2,
    high_cost_trace: MassiveAdaptiveProfitTraceV2,
) -> MassiveAdaptiveProfitCheckpointCandidateV3:
    """Derive one candidate only after exact fold/checkpoint reconciliation."""

    for value in (
        checkpoint,
        training_window_plan,
        forecast_archive,
        calibration,
        primary_trace,
        low_cost_trace,
        high_cost_trace,
    ):
        value.validate()
    traces = (low_cost_trace, primary_trace, high_cost_trace)
    fold = training_window_plan.fold_index
    checkpoint_source = getattr(
        getattr(checkpoint, "loaded_source", None), "receipt_sha256", None
    )
    if (
        training_window_plan.split_role != "training"
        or checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or forecast_archive.fold_index != fold
        or forecast_archive.checkpoint_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or forecast_archive.model_state_receipt_sha256
        != checkpoint.model_state_receipt_sha256
        or forecast_archive.training_window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or calibration.fold_index != fold
        or calibration.checkpoint_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or calibration.model_state_receipt_sha256
        != checkpoint.model_state_receipt_sha256
        or calibration.training_window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or any(trace.fold_index != fold for trace in traces)
        or any(
            trace.checkpoint_receipt_sha256 != checkpoint.semantic_receipt_sha256
            or trace.model_state_receipt_sha256 != checkpoint.model_state_receipt_sha256
            or trace.training_window_plan_receipt_sha256
            != training_window_plan.semantic_receipt_sha256
            or trace.forecast_archive_receipt_sha256
            != forecast_archive.semantic_receipt_sha256
            or trace.calibration_receipt_sha256 != calibration.semantic_receipt_sha256
            or trace.inference_plan_receipt_sha256
            != forecast_archive.inference_plan_receipt_sha256
            for trace in traces
        )
        or len({trace.initial_capital for trace in traces}) != 1
        or len({trace.maximum_fill_participation for trace in traces}) != 1
        or len({trace.compiler_config_receipt_sha256 for trace in traces}) != 1
        or len({trace.identity_authority_receipt_sha256 for trace in traces}) != 1
        or not primary_trace.deterministic_profitability_replayed
        or not low_cost_trace.deterministic_profitability_replayed
        or not high_cost_trace.deterministic_profitability_replayed
        or checkpoint_source is None
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
            "candidate checkpoint, fold, calibration, and traces differ"
        )
    targets = tuple(row.decision_target_receipt_sha256 for row in primary_trace.rows)
    if (
        tuple(row.decision_target_receipt_sha256 for row in low_cost_trace.rows)
        != targets
        or tuple(row.decision_target_receipt_sha256 for row in high_cost_trace.rows)
        != targets
        or not low_cost_trace.frozen_actions_replayed
        or not high_cost_trace.frozen_actions_replayed
        or primary_trace.frozen_actions_replayed
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
            "candidate cost ladder does not replay frozen primary actions"
        )
    low_return = low_cost_trace.final_equity / low_cost_trace.initial_capital - 1.0
    primary_return = primary_trace.final_equity / primary_trace.initial_capital - 1.0
    high_return = high_cost_trace.final_equity / high_cost_trace.initial_capital - 1.0
    failures: list[str] = []
    if primary_return <= 0.0:
        failures.append("primary-net-return")
    if high_return < 0.0:
        failures.append("high-cost-net-return")
    if primary_trace.cumulative_active_log_return <= 0.0:
        failures.append("active-log-wealth")
    if not low_return > primary_return > high_return:
        failures.append("cost-ladder")
    cutoff = max(row.origin_session_date for row in training_window_plan.rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_CANDIDATE_V3_SCHEMA,
        "fold_index": fold,
        "epoch_index": checkpoint.epoch_index,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": checkpoint_source,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "training_window_plan_receipt_sha256": training_window_plan.semantic_receipt_sha256,
        "training_cutoff_session_date": cutoff,
        "inner_validation_plan_receipt_sha256": forecast_archive.inference_plan_receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": low_cost_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": high_cost_trace.semantic_receipt_sha256,
        "economic_source_inventory_sha256": semantic_sha256(
            tuple(trace.source_inventory_sha256 for trace in traces)
        ),
        "primary_cost_basis_points": primary_trace.transaction_cost_basis_points,
        "low_cost_basis_points": low_cost_trace.transaction_cost_basis_points,
        "high_cost_basis_points": high_cost_trace.transaction_cost_basis_points,
        "primary_dollar_net_profit": primary_trace.final_equity
        - primary_trace.initial_capital,
        "primary_terminal_net_return": primary_return,
        "low_cost_terminal_net_return": low_return,
        "high_cost_terminal_net_return": high_return,
        "primary_active_log_wealth": primary_trace.cumulative_active_log_return,
        "maximum_drawdown": _maximum_drawdown(primary_trace),
        "eligibility_failures": tuple(sorted(failures)),
        "economically_eligible": not failures,
        "source_data_qualified": bool(
            checkpoint.development_training_authorized
            and forecast_archive.development_forecast_authorized
            and calibration.development_calibration_authorized
            and all(trace.source_data_qualified for trace in traces)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveProfitCheckpointCandidateV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointSelectionV3:
    fold_index: int
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    training_cutoff_session_date: str
    inner_validation_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    selected_candidate_receipt_sha256: str
    candidate_inventory_sha256: str
    candidate_count: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V3_SCHEMA
            or self.fold_index < 0
            or self.candidate_count <= 0
            or not self.training_cutoff_session_date
            or not isinstance(self.source_data_qualified, bool)
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
                "adaptive checkpoint selection v3 differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def select_massive_adaptive_profit_checkpoint_v3(
    candidates: Sequence[MassiveAdaptiveProfitCheckpointCandidateV3],
) -> MassiveAdaptiveProfitCheckpointSelectionV3:
    ordered = tuple(sorted(candidates, key=lambda row: row.semantic_receipt_sha256))
    if not ordered:
        raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
            "checkpoint selection v3 has no candidates"
        )
    for candidate in ordered:
        candidate.validate()
    lineage = {
        (
            row.fold_index,
            row.training_window_plan_receipt_sha256,
            row.inner_validation_plan_receipt_sha256,
            row.calibration_receipt_sha256,
        )
        for row in ordered
    }
    eligible = tuple(row for row in ordered if row.economically_eligible)
    if len(lineage) != 1 or not eligible:
        raise MassiveAdaptiveProfitCheckpointSelectionV3Error(
            "checkpoint candidates span folds or none is economically eligible"
        )
    selected = max(eligible, key=lambda row: row.selection_key)
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V3_SCHEMA,
        "fold_index": selected.fold_index,
        "selected_checkpoint_receipt_sha256": selected.checkpoint_receipt_sha256,
        "selected_model_state_receipt_sha256": selected.model_state_receipt_sha256,
        "training_window_plan_receipt_sha256": selected.training_window_plan_receipt_sha256,
        "training_cutoff_session_date": selected.training_cutoff_session_date,
        "inner_validation_plan_receipt_sha256": selected.inner_validation_plan_receipt_sha256,
        "calibration_receipt_sha256": selected.calibration_receipt_sha256,
        "selected_candidate_receipt_sha256": selected.semantic_receipt_sha256,
        "candidate_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered)
        ),
        "candidate_count": len(ordered),
        "source_data_qualified": all(row.source_data_qualified for row in ordered),
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveProfitCheckpointSelectionV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveProfitCheckpointCandidateV3",
    "MassiveAdaptiveProfitCheckpointSelectionV3",
    "MassiveAdaptiveProfitCheckpointSelectionV3Error",
    "build_massive_adaptive_profit_checkpoint_candidate_v3",
    "select_massive_adaptive_profit_checkpoint_v3",
]
