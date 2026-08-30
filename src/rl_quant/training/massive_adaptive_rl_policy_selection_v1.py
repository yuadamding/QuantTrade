"""Economic inner-validation selection for adaptive RL policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptiveRLCheckpointV1

if TYPE_CHECKING:
    from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    )


MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-trace-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-candidate-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-policy-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SCHEMA,
            "payload": "canonical-json-policy-selection-and-candidates",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "role": "inner-validation-only",
        "primary_cost_basis_points": 20.0,
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "stress": "same-target-inventory",
        "primary_metric": "incremental-strategy-minus-neutral-log-wealth",
        "terminal": "liquidation-adjusted",
        "maximum_drawdown": 0.25,
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLPolicySelectionV1Error(ValueError):
    """RL validation traces, candidates, or committed selection differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyTraceV1:
    fold_index: int
    evaluation_role: str
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    update_index: int
    training_forecast_authority_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    transaction_cost_basis_points: float
    initial_capital: float
    decision_session_dates: tuple[str, ...]
    transition_receipts: tuple[str, ...]
    decision_target_inventory_sha256: str
    economic_source_inventory_sha256: str
    strategy_active_log_returns: tuple[float, ...]
    incremental_rl_log_returns: tuple[float, ...]
    cumulative_strategy_active_log_return: float
    cumulative_incremental_rl_log_return: float
    terminal_liquidation_adjusted_return: float
    maximum_drawdown: float
    frozen_targets_replayed: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or self.evaluation_role
            not in {"training_control", "inner_validation", "outer_test"}
            or isinstance(self.update_index, bool)
            or self.update_index < 0
            or not math.isfinite(self.transaction_cost_basis_points)
            or self.transaction_cost_basis_points < 0.0
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0.0
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or not self.decision_session_dates
            or len(self.transition_receipts) != len(self.decision_session_dates)
            or len(self.strategy_active_log_returns) != len(self.decision_session_dates)
            or len(self.incremental_rl_log_returns) != len(self.decision_session_dates)
            or any(
                not math.isfinite(value)
                for value in (
                    *self.strategy_active_log_returns,
                    *self.incremental_rl_log_returns,
                    self.cumulative_strategy_active_log_return,
                    self.cumulative_incremental_rl_log_return,
                    self.terminal_liquidation_adjusted_return,
                    self.maximum_drawdown,
                )
            )
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or abs(
                sum(self.strategy_active_log_returns)
                - self.cumulative_strategy_active_log_return
            )
            > 1.0e-12
            or abs(
                sum(self.incremental_rl_log_returns)
                - self.cumulative_incremental_rl_log_return
            )
            > 1.0e-12
            or not isinstance(self.frozen_targets_replayed, bool)
            or not isinstance(self.source_data_qualified, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV1Error(
                "adaptive RL policy trace differs"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.forecast_archive_receipt_sha256,
            self.inference_plan_receipt_sha256,
            self.calibration_receipt_sha256,
            *self.transition_receipts,
            self.decision_target_inventory_sha256,
            self.economic_source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy trace", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_policy_trace_v1(
    *,
    fold_index: int,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    forecast_archive_receipt_sha256: str,
    inference_plan_receipt_sha256: str,
    calibration_receipt_sha256: str,
    transaction_cost_basis_points: float,
    initial_capital: float,
    transitions: Sequence[MassiveAdaptiveRLTransitionV1],
    frozen_targets_replayed: bool,
    evaluation_role: str = "inner_validation",
) -> MassiveAdaptiveRLPolicyTraceV1:
    """Derive policy economics only from complete environment transitions."""

    checkpoint.validate()
    rows = tuple(transitions)
    if (
        checkpoint.training_forecast_authority_receipt_sha256 is None
        or not rows
        or any(row.truncated for row in rows)
        or any(row.terminated for row in rows[:-1])
        or not rows[-1].terminated
    ):
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "adaptive RL policy trace is not one complete authorized episode"
        )
    for row in rows:
        row.validate()
    dates = tuple(row.economic_step.strategy_execution.decision_session_date for row in rows)
    final_equity = rows[-1].strategy_liquidation_adjusted_equity
    running_peak = initial_capital
    maximum_drawdown = 0.0
    for row in rows:
        equity = row.economic_step.strategy_posttrade_book.marked_equity
        running_peak = max(running_peak, equity)
        maximum_drawdown = max(maximum_drawdown, 1.0 - equity / running_peak)
    target_inventory = semantic_sha256(
        tuple(
            (
                row.policy_decision.security_ids,
                row.policy_decision.target_weights,
            )
            for row in rows
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_V1_SCHEMA,
        "fold_index": fold_index,
        "evaluation_role": evaluation_role,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "update_index": checkpoint.update_index,
        "training_forecast_authority_receipt_sha256": (
            checkpoint.training_forecast_authority_receipt_sha256
        ),
        "forecast_archive_receipt_sha256": _digest(
            "forecast archive receipt", forecast_archive_receipt_sha256
        ),
        "inference_plan_receipt_sha256": _digest(
            "inference plan receipt", inference_plan_receipt_sha256
        ),
        "calibration_receipt_sha256": _digest(
            "calibration receipt", calibration_receipt_sha256
        ),
        "transaction_cost_basis_points": float(transaction_cost_basis_points),
        "initial_capital": float(initial_capital),
        "decision_session_dates": dates,
        "transition_receipts": tuple(row.semantic_receipt_sha256 for row in rows),
        "decision_target_inventory_sha256": target_inventory,
        "economic_source_inventory_sha256": semantic_sha256(
            tuple(row.economic_step.source_inventory_sha256 for row in rows)
        ),
        "strategy_active_log_returns": tuple(
            row.strategy_active_log_return for row in rows
        ),
        "incremental_rl_log_returns": tuple(
            row.incremental_rl_log_return for row in rows
        ),
        "cumulative_strategy_active_log_return": sum(
            row.strategy_active_log_return for row in rows
        ),
        "cumulative_incremental_rl_log_return": sum(
            row.incremental_rl_log_return for row in rows
        ),
        "terminal_liquidation_adjusted_return": final_equity / initial_capital - 1.0,
        "maximum_drawdown": maximum_drawdown,
        "frozen_targets_replayed": frozen_targets_replayed,
        "source_data_qualified": bool(
            checkpoint.development_rl_training_authorized
            and all(row.source_data_qualified for row in rows)
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPolicyTraceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyCandidateV1:
    fold_index: int
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    update_index: int
    training_forecast_authority_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    decision_target_inventory_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fixed_action_receipt_sha256: str
    fixed_control_validation_trace_receipt_sha256: str
    best_fixed_control_incremental_log_wealth: float
    ppo_minus_best_fixed_control_log_wealth: float
    primary_incremental_rl_log_wealth: float
    primary_strategy_active_log_wealth: float
    high_cost_terminal_return: float
    maximum_drawdown: float
    eligibility_failures: tuple[str, ...]
    economically_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V1_SCHEMA
            or self.fold_index < 0
            or self.update_index < 0
            or any(
                not math.isfinite(value)
                for value in (
                    self.primary_incremental_rl_log_wealth,
                    self.best_fixed_control_incremental_log_wealth,
                    self.ppo_minus_best_fixed_control_log_wealth,
                    self.primary_strategy_active_log_wealth,
                    self.high_cost_terminal_return,
                    self.maximum_drawdown,
                )
            )
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or self.eligibility_failures
            != tuple(sorted(set(self.eligibility_failures)))
            or self.economically_eligible != (not self.eligibility_failures)
            or not isinstance(self.source_data_qualified, bool)
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV1Error(
                "adaptive RL policy candidate differs"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.primary_trace_receipt_sha256,
            self.low_cost_trace_receipt_sha256,
            self.high_cost_trace_receipt_sha256,
            self.decision_target_inventory_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_fixed_action_receipt_sha256,
            self.fixed_control_validation_trace_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy candidate", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    @property
    def selection_key(self) -> tuple[float, ...]:
        self.validate()
        return (
            self.ppo_minus_best_fixed_control_log_wealth,
            self.primary_incremental_rl_log_wealth,
            self.primary_strategy_active_log_wealth,
            -self.maximum_drawdown,
            -float(self.update_index),
        )


def build_massive_adaptive_rl_policy_candidate_v1(
    *,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    primary_trace: MassiveAdaptiveRLPolicyTraceV1,
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    fixed_control_selection_authority: (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    ),
    fixed_control_validation_trace: MassiveAdaptiveRLPolicyTraceV1,
) -> MassiveAdaptiveRLPolicyCandidateV1:
    checkpoint.validate()
    fixed_control_selection_authority.validate()
    fixed_control_validation_trace.validate()
    fixed_selection = fixed_control_selection_authority.runtime_selection
    traces = (low_cost_trace, primary_trace, high_cost_trace)
    for trace in traces:
        trace.validate()
    if (
        tuple(trace.transaction_cost_basis_points for trace in traces)
        != (10.0, 20.0, 40.0)
        or any(trace.checkpoint_receipt_sha256 != checkpoint.semantic_receipt_sha256 for trace in traces)
        or len({trace.fold_index for trace in traces}) != 1
        or len({trace.forecast_archive_receipt_sha256 for trace in traces}) != 1
        or len({trace.inference_plan_receipt_sha256 for trace in traces}) != 1
        or len({trace.calibration_receipt_sha256 for trace in traces}) != 1
        or len({trace.decision_target_inventory_sha256 for trace in traces}) != 1
        or any(trace.evaluation_role != "inner_validation" for trace in traces)
        or primary_trace.frozen_targets_replayed
        or not low_cost_trace.frozen_targets_replayed
        or not high_cost_trace.frozen_targets_replayed
        or not fixed_control_selection_authority.runtime_selection_replayed
        or fixed_selection is None
        or fixed_selection.fold_index != primary_trace.fold_index
        or fixed_control_validation_trace.fold_index != primary_trace.fold_index
        or fixed_control_validation_trace.evaluation_role != "inner_validation"
        or fixed_control_validation_trace.transaction_cost_basis_points != 20.0
        or fixed_control_validation_trace.frozen_targets_replayed
        or fixed_control_validation_trace.forecast_archive_receipt_sha256
        != primary_trace.forecast_archive_receipt_sha256
        or fixed_control_validation_trace.inference_plan_receipt_sha256
        != primary_trace.inference_plan_receipt_sha256
        or fixed_control_validation_trace.calibration_receipt_sha256
        != primary_trace.calibration_receipt_sha256
        or fixed_control_validation_trace.economic_source_inventory_sha256
        != primary_trace.economic_source_inventory_sha256
    ):
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "adaptive RL policy cost ladder or checkpoint differs"
        )
    failures: list[str] = []
    if primary_trace.cumulative_incremental_rl_log_return <= 0.0:
        failures.append("incremental-log-wealth")
    fixed_incremental = (
        fixed_control_validation_trace.cumulative_incremental_rl_log_return
    )
    ppo_minus_fixed = (
        primary_trace.cumulative_incremental_rl_log_return - fixed_incremental
    )
    if ppo_minus_fixed <= 0.0:
        failures.append("best-fixed-control")
    if primary_trace.cumulative_strategy_active_log_return <= 0.0:
        failures.append("strategy-active-log-wealth")
    if high_cost_trace.terminal_liquidation_adjusted_return < 0.0:
        failures.append("high-cost-terminal-return")
    returns = tuple(trace.terminal_liquidation_adjusted_return for trace in traces)
    if not returns[0] >= returns[1] >= returns[2]:
        failures.append("cost-ladder")
    if primary_trace.maximum_drawdown > 0.25:
        failures.append("maximum-drawdown")
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V1_SCHEMA,
        "fold_index": primary_trace.fold_index,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "update_index": checkpoint.update_index,
        "training_forecast_authority_receipt_sha256": (
            primary_trace.training_forecast_authority_receipt_sha256
        ),
        "primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": low_cost_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": high_cost_trace.semantic_receipt_sha256,
        "decision_target_inventory_sha256": (
            primary_trace.decision_target_inventory_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            fixed_control_selection_authority.semantic_receipt_sha256
        ),
        "selected_fixed_action_receipt_sha256": (
            fixed_selection.selected_action_receipt_sha256
        ),
        "fixed_control_validation_trace_receipt_sha256": (
            fixed_control_validation_trace.semantic_receipt_sha256
        ),
        "best_fixed_control_incremental_log_wealth": fixed_incremental,
        "ppo_minus_best_fixed_control_log_wealth": ppo_minus_fixed,
        "primary_incremental_rl_log_wealth": (
            primary_trace.cumulative_incremental_rl_log_return
        ),
        "primary_strategy_active_log_wealth": (
            primary_trace.cumulative_strategy_active_log_return
        ),
        "high_cost_terminal_return": (
            high_cost_trace.terminal_liquidation_adjusted_return
        ),
        "maximum_drawdown": primary_trace.maximum_drawdown,
        "eligibility_failures": tuple(sorted(failures)),
        "economically_eligible": not failures,
        "source_data_qualified": bool(
            all(trace.source_data_qualified for trace in traces)
            and fixed_control_validation_trace.source_data_qualified
            and fixed_control_selection_authority.source_data_qualified
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPolicyCandidateV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicySelectionV1:
    fold_index: int
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    selected_update_index: int
    training_forecast_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_candidate_receipt_sha256: str
    candidate_inventory_sha256: str
    candidate_count: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SCHEMA
            or self.fold_index < 0
            or self.selected_update_index < 0
            or self.candidate_count <= 0
            or not isinstance(self.source_data_qualified, bool)
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV1Error(
                "adaptive RL policy selection differs"
            )
        for value in (
            self.selected_checkpoint_receipt_sha256,
            self.selected_model_state_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_candidate_receipt_sha256,
            self.candidate_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy selection", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def select_massive_adaptive_rl_policy_v1(
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV1],
) -> MassiveAdaptiveRLPolicySelectionV1:
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    if not ordered:
        raise MassiveAdaptiveRLPolicySelectionV1Error("RL policy candidates are absent")
    for candidate in ordered:
        candidate.validate()
    lineage = {
        (
            candidate.fold_index,
            candidate.training_forecast_authority_receipt_sha256,
            candidate.fixed_control_selection_authority_receipt_sha256,
        )
        for candidate in ordered
    }
    eligible = tuple(candidate for candidate in ordered if candidate.economically_eligible)
    if len(lineage) != 1 or not eligible:
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "RL policy candidates span folds or none is eligible"
        )
    selected = max(eligible, key=lambda value: value.selection_key)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SCHEMA,
        "fold_index": selected.fold_index,
        "selected_checkpoint_receipt_sha256": selected.checkpoint_receipt_sha256,
        "selected_model_state_receipt_sha256": selected.model_state_receipt_sha256,
        "selected_update_index": selected.update_index,
        "training_forecast_authority_receipt_sha256": (
            selected.training_forecast_authority_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            selected.fixed_control_selection_authority_receipt_sha256
        ),
        "selected_candidate_receipt_sha256": selected.semantic_receipt_sha256,
        "candidate_inventory_sha256": semantic_sha256(
            tuple(candidate.semantic_receipt_sha256 for candidate in ordered)
        ),
        "candidate_count": len(ordered),
        "source_data_qualified": all(candidate.source_data_qualified for candidate in ordered),
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveRLPolicySelectionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicySelectionAuthorityV1:
    selection_receipt_sha256: str
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_selection: MassiveAdaptiveRLPolicySelectionV1 | None
    runtime_candidates: tuple[MassiveAdaptiveRLPolicyCandidateV1, ...] | None
    runtime_selection_replayed: bool
    development_policy_selection_authorized: bool
    outer_evaluation_authorized: bool
    reinforcement_learning_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        runtime = self.runtime_selection is not None and self.runtime_candidates is not None
        expected_authorized = runtime and self.source_data_qualified
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_selection_replayed != runtime
            or self.development_policy_selection_authorized != expected_authorized
            or self.outer_evaluation_authorized != expected_authorized
            or self.reinforcement_learning_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV1Error(
                "adaptive RL policy selection authority differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.selection_receipt_sha256
        ):
            raise MassiveAdaptiveRLPolicySelectionV1Error(
                "adaptive RL policy selection source transaction differs"
            )
        if runtime:
            assert self.runtime_selection is not None
            assert self.runtime_candidates is not None
            self.runtime_selection.validate()
            for candidate in self.runtime_candidates:
                candidate.validate()
            if (
                self.runtime_selection.semantic_receipt_sha256
                != self.selection_receipt_sha256
                or self.runtime_selection.candidate_inventory_sha256
                != self.candidate_inventory_sha256
            ):
                raise MassiveAdaptiveRLPolicySelectionV1Error(
                    "adaptive RL runtime policy selection differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _authority_payload(
    *,
    selection: MassiveAdaptiveRLPolicySelectionV1,
    candidates: tuple[MassiveAdaptiveRLPolicyCandidateV1, ...],
) -> dict[str, object]:
    return {
        "selection": asdict(selection),
        "candidates": tuple(asdict(candidate) for candidate in candidates),
    }


def _load_authority_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[MassiveAdaptiveRLPolicySelectionV1, tuple[MassiveAdaptiveRLPolicyCandidateV1, ...]]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, Mapping) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "RL policy selection payload is not canonical JSON"
        )
    selection_payload = dict(payload["selection"])
    candidates_payload = tuple(dict(value) for value in payload["candidates"])
    candidates = tuple(
        MassiveAdaptiveRLPolicyCandidateV1(
            **{
                **value,
                "eligibility_failures": tuple(value["eligibility_failures"]),
            }
        )
        for value in candidates_payload
    )
    selection = MassiveAdaptiveRLPolicySelectionV1(**selection_payload)
    selection.validate()
    for candidate in candidates:
        candidate.validate()
    return selection, candidates


def parse_massive_adaptive_rl_policy_selection_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicySelectionAuthorityV1:
    selection, _candidates = _load_authority_payload(
        root=root, loaded_source=loaded_source
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SCHEMA,
        "selection_receipt_sha256": selection.semantic_receipt_sha256,
        "candidate_inventory_sha256": selection.candidate_inventory_sha256,
        "source_data_qualified": selection.source_data_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPolicySelectionAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded_source,
        runtime_selection=None,
        runtime_candidates=None,
        runtime_selection_replayed=False,
        development_policy_selection_authorized=False,
        outer_evaluation_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_policy_selection_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV1],
) -> MassiveAdaptiveRLPolicySelectionAuthorityV1:
    parsed = parse_massive_adaptive_rl_policy_selection_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed_selection, committed_candidates = _load_authority_payload(
        root=root, loaded_source=authority.loaded_source
    )
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    rebuilt = select_massive_adaptive_rl_policy_v1(ordered)
    if committed_candidates != ordered or committed_selection != rebuilt:
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "RL policy selection authority does not replay"
        )
    result = MassiveAdaptiveRLPolicySelectionAuthorityV1(
        **parsed.semantic_unsigned(),  # type: ignore[arg-type]
        semantic_receipt_sha256=parsed.semantic_receipt_sha256,
        loaded_source=parsed.loaded_source,
        runtime_selection=rebuilt,
        runtime_candidates=ordered,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=rebuilt.source_data_qualified,
        outer_evaluation_authorized=rebuilt.source_data_qualified,
        reinforcement_learning_authorized=rebuilt.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_policy_selection_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV1],
    committed_at_ms: int,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLPolicySelectionV1Error(
            "RL policy selection artifact ID is not path safe"
        )
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    selection = select_massive_adaptive_rl_policy_v1(ordered)
    relative = f"massive-adaptive/rl-policy-selection-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                _authority_payload(selection=selection, candidates=ordered)
            )
        ),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=selection.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-POLICY-SELECTION-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_policy_selection_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_policy_selection_authority_v1(
            root=root, loaded_source=loaded
        ),
        candidates=ordered,
    )


__all__ = [
    "MassiveAdaptiveRLPolicyCandidateV1",
    "MassiveAdaptiveRLPolicySelectionAuthorityV1",
    "MassiveAdaptiveRLPolicySelectionV1",
    "MassiveAdaptiveRLPolicySelectionV1Error",
    "MassiveAdaptiveRLPolicyTraceV1",
    "authorize_massive_adaptive_rl_policy_selection_authority_v1",
    "build_massive_adaptive_rl_policy_candidate_v1",
    "build_massive_adaptive_rl_policy_trace_v1",
    "materialize_massive_adaptive_rl_policy_selection_authority_v1",
    "parse_massive_adaptive_rl_policy_selection_authority_v1",
    "select_massive_adaptive_rl_policy_v1",
]
