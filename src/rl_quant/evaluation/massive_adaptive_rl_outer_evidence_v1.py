"""Fold-bound frozen-policy outer evidence for adaptive RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Sequence

from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_outer_evidence_v1 import (
    _nonwrapping_fold_cluster_lcb,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v1 import (
    MassiveAdaptiveFrozenRLPolicyV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV1,
    MassiveAdaptiveRLPolicyTraceV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
    MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
)


MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V1_SCHEMA = "rl-quant.massive-adaptive-rl-outer-plan-v1"
MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-cost-fold-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-evidence-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "folds": 4,
        "sessions_per_fold": 126,
        "primary_capital": 10_000_000.0,
        "cost_ladder": (10.0, 20.0, 40.0),
        "policy": "one-committed-frozen-policy-per-fold",
        "primary_contrasts": (
            "strategy-active-log-return",
            "strategy-minus-neutral-log-return",
            "strategy-minus-training-selected-fixed-control",
        ),
        "bootstrap": "same-fold-cluster-nonwrapping-63-session-v1",
        "minimum_positive_folds": 3,
        "maximum_fold_drawdown": 0.25,
        "cost_ladder_monotonicity": (
            "derived-report-gate-not-structural-validity"
        ),
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLOuterEvidenceV1Error(ValueError):
    """The frozen policy, outer fold, or paired economic traces differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterPlanV1:
    fold_index: int
    supervised_checkpoint_receipt_sha256: str
    supervised_model_state_receipt_sha256: str
    calibration_receipt_sha256: str
    outer_inference_plan_receipt_sha256: str
    outer_forecast_archive_receipt_sha256: str
    rl_policy_selection_authority_receipt_sha256: str
    frozen_rl_policy_receipt_sha256: str
    frozen_rl_policy_model_state_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    compiler_config_receipt_sha256: str
    benchmark_specification: str
    initial_book_specification: str
    primary_capital: float
    primary_cost_basis_points: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "outer_evaluation_authorized"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V1_SCHEMA
            or not 0 <= self.fold_index < MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or self.benchmark_specification != "shared-buy-and-drift-book-v1"
            or self.initial_book_specification != "all-books-cash-v1"
            or self.primary_capital != 10_000_000.0
            or self.primary_cost_basis_points != 20.0
            or not isinstance(self.source_data_qualified, bool)
            or self.outer_evaluation_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceV1Error(
                "adaptive RL outer plan differs"
            )
        for value in (
            self.supervised_checkpoint_receipt_sha256,
            self.supervised_model_state_receipt_sha256,
            self.calibration_receipt_sha256,
            self.outer_inference_plan_receipt_sha256,
            self.outer_forecast_archive_receipt_sha256,
            self.rl_policy_selection_authority_receipt_sha256,
            self.frozen_rl_policy_receipt_sha256,
            self.frozen_rl_policy_model_state_receipt_sha256,
            self.observation_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            self.compiler_config_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL outer plan", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_plan_v1(
    *,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
    outer_forecast_archive: MassiveAdaptiveOuterForecastArchiveV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    policy_selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1,
) -> MassiveAdaptiveRLOuterPlanV1:
    """Freeze every nonmarket input before one outer fold is evaluated."""

    for value in (
        outer_inference_plan,
        outer_forecast_archive,
        calibration,
        policy_selection_authority,
        frozen_policy,
        compiler_config,
    ):
        value.validate()
    selection = policy_selection_authority.runtime_selection
    if (
        selection is None
        or not policy_selection_authority.runtime_selection_replayed
        or len(
            {
                outer_inference_plan.fold_index,
                outer_forecast_archive.fold_index,
                calibration.fold_index,
                selection.fold_index,
                frozen_policy.fold_index,
            }
        )
        != 1
        or outer_forecast_archive.outer_inference_plan_receipt_sha256
        != outer_inference_plan.semantic_receipt_sha256
        or outer_forecast_archive.selected_checkpoint_receipt_sha256
        != outer_inference_plan.selected_checkpoint_receipt_sha256
        or calibration.checkpoint_receipt_sha256
        != outer_forecast_archive.selected_checkpoint_receipt_sha256
        or calibration.model_state_receipt_sha256
        != outer_forecast_archive.model_state_receipt_sha256
        or calibration.training_window_plan_receipt_sha256
        != outer_forecast_archive.training_window_plan_receipt_sha256
        or frozen_policy.policy_selection_authority_receipt_sha256
        != policy_selection_authority.semantic_receipt_sha256
        or frozen_policy.policy_selection_receipt_sha256
        != selection.semantic_receipt_sha256
        or frozen_policy.selected_rl_checkpoint_receipt_sha256
        != selection.selected_checkpoint_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterEvidenceV1Error(
            "adaptive RL outer plan components differ"
        )
    source_qualified = bool(
        outer_inference_plan.outer_inference_authorized
        and outer_forecast_archive.outer_forecast_authorized
        and calibration.development_calibration_authorized
        and policy_selection_authority.outer_evaluation_authorized
        and frozen_policy.development_outer_policy_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V1_SCHEMA,
        "fold_index": outer_inference_plan.fold_index,
        "supervised_checkpoint_receipt_sha256": (
            outer_forecast_archive.selected_checkpoint_receipt_sha256
        ),
        "supervised_model_state_receipt_sha256": (
            outer_forecast_archive.model_state_receipt_sha256
        ),
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "outer_inference_plan_receipt_sha256": (
            outer_inference_plan.semantic_receipt_sha256
        ),
        "outer_forecast_archive_receipt_sha256": (
            outer_forecast_archive.semantic_receipt_sha256
        ),
        "rl_policy_selection_authority_receipt_sha256": (
            policy_selection_authority.semantic_receipt_sha256
        ),
        "frozen_rl_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "frozen_rl_policy_model_state_receipt_sha256": (
            frozen_policy.frozen_model_state_receipt_sha256
        ),
        "observation_specification_sha256": (
            frozen_policy.observation_specification_sha256
        ),
        "action_specification_sha256": frozen_policy.action_specification_sha256,
        "reward_specification_sha256": frozen_policy.reward_specification_sha256,
        "compiler_config_receipt_sha256": compiler_config.receipt_sha256,
        "benchmark_specification": "shared-buy-and-drift-book-v1",
        "initial_book_specification": "all-books-cash-v1",
        "primary_capital": 10_000_000.0,
        "primary_cost_basis_points": 20.0,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLOuterPlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=source_qualified,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterCostFoldV1:
    fold_index: int
    outer_plan_receipt_sha256: str
    frozen_rl_policy_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    primary_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    best_fixed_control_trace_receipt_sha256: str
    decision_target_inventory_sha256: str
    primary_strategy_active_log_returns: tuple[float, ...]
    primary_incremental_rl_log_returns: tuple[float, ...]
    primary_ppo_minus_fixed_control_log_returns: tuple[float, ...]
    low_cost_terminal_return: float
    primary_terminal_return: float
    high_cost_terminal_return: float
    maximum_drawdown: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def terminal_return_ladder_monotone(self) -> bool:
        """Whether this fold's observed cost ladder satisfies the final gate."""

        return bool(
            self.low_cost_terminal_return
            >= self.primary_terminal_return
            >= self.high_cost_terminal_return
        )

    def validate(self) -> None:
        values = (
            *self.primary_strategy_active_log_returns,
            *self.primary_incremental_rl_log_returns,
            *self.primary_ppo_minus_fixed_control_log_returns,
            self.low_cost_terminal_return,
            self.primary_terminal_return,
            self.high_cost_terminal_return,
            self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA
            or not 0 <= self.fold_index < MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or len(self.primary_strategy_active_log_returns)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or len(self.primary_incremental_rl_log_returns)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or len(self.primary_ppo_minus_fixed_control_log_returns)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or any(not math.isfinite(value) for value in values)
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceV1Error(
                "adaptive RL outer cost fold differs"
            )
        for value in (
            self.outer_plan_receipt_sha256,
            self.frozen_rl_policy_receipt_sha256,
            self.low_cost_trace_receipt_sha256,
            self.primary_trace_receipt_sha256,
            self.high_cost_trace_receipt_sha256,
            self.best_fixed_control_trace_receipt_sha256,
            self.decision_target_inventory_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL outer cost fold", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_cost_fold_v1(
    *,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    primary_trace: MassiveAdaptiveRLPolicyTraceV1,
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    best_fixed_control_trace: MassiveAdaptiveRLPolicyTraceV1,
) -> MassiveAdaptiveRLOuterCostFoldV1:
    outer_plan.validate()
    frozen_policy.validate()
    traces = (low_cost_trace, primary_trace, high_cost_trace)
    for trace in traces:
        trace.validate()
    best_fixed_control_trace.validate()
    if (
        not outer_plan.outer_evaluation_authorized
        or not frozen_policy.development_outer_policy_authorized
        or outer_plan.frozen_rl_policy_receipt_sha256
        != frozen_policy.semantic_receipt_sha256
        or tuple(trace.transaction_cost_basis_points for trace in traces)
        != (10.0, 20.0, 40.0)
        or any(trace.evaluation_role != "outer_test" for trace in traces)
        or any(trace.fold_index != outer_plan.fold_index for trace in traces)
        or any(
            trace.checkpoint_receipt_sha256
            != frozen_policy.selected_rl_checkpoint_receipt_sha256
            for trace in traces
        )
        or len({trace.decision_target_inventory_sha256 for trace in traces}) != 1
        or primary_trace.frozen_targets_replayed
        or not low_cost_trace.frozen_targets_replayed
        or not high_cost_trace.frozen_targets_replayed
        or best_fixed_control_trace.evaluation_role != "outer_test"
        or best_fixed_control_trace.fold_index != outer_plan.fold_index
        or best_fixed_control_trace.transaction_cost_basis_points != 20.0
        or best_fixed_control_trace.frozen_targets_replayed
        or best_fixed_control_trace.forecast_archive_receipt_sha256
        != primary_trace.forecast_archive_receipt_sha256
        or best_fixed_control_trace.inference_plan_receipt_sha256
        != primary_trace.inference_plan_receipt_sha256
        or best_fixed_control_trace.calibration_receipt_sha256
        != primary_trace.calibration_receipt_sha256
        or best_fixed_control_trace.decision_session_dates
        != primary_trace.decision_session_dates
    ):
        raise MassiveAdaptiveRLOuterEvidenceV1Error(
            "adaptive RL outer trace or frozen cost ladder differs"
        )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_COST_FOLD_V1_SCHEMA,
        "fold_index": outer_plan.fold_index,
        "outer_plan_receipt_sha256": outer_plan.semantic_receipt_sha256,
        "frozen_rl_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": low_cost_trace.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": high_cost_trace.semantic_receipt_sha256,
        "best_fixed_control_trace_receipt_sha256": (
            best_fixed_control_trace.semantic_receipt_sha256
        ),
        "decision_target_inventory_sha256": (
            primary_trace.decision_target_inventory_sha256
        ),
        "primary_strategy_active_log_returns": (
            primary_trace.strategy_active_log_returns
        ),
        "primary_incremental_rl_log_returns": (
            primary_trace.incremental_rl_log_returns
        ),
        "primary_ppo_minus_fixed_control_log_returns": tuple(
            policy - fixed
            for policy, fixed in zip(
                primary_trace.incremental_rl_log_returns,
                best_fixed_control_trace.incremental_rl_log_returns,
                strict=True,
            )
        ),
        "low_cost_terminal_return": low_cost_trace.terminal_liquidation_adjusted_return,
        "primary_terminal_return": (primary_trace.terminal_liquidation_adjusted_return),
        "high_cost_terminal_return": (
            high_cost_trace.terminal_liquidation_adjusted_return
        ),
        "maximum_drawdown": primary_trace.maximum_drawdown,
        "source_data_qualified": bool(
            outer_plan.source_data_qualified
            and all(trace.source_data_qualified for trace in traces)
            and best_fixed_control_trace.source_data_qualified
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLOuterCostFoldV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceV1:
    fold_indices: tuple[int, ...]
    fold_receipts: tuple[str, ...]
    fold_inventory_sha256: str
    mean_strategy_active_log_return: float
    mean_incremental_rl_log_return: float
    mean_ppo_minus_fixed_control_log_return: float
    mean_high_cost_terminal_return: float
    strategy_active_log_return_lcb95: float
    incremental_rl_log_return_lcb95: float
    ppo_minus_fixed_control_log_return_lcb95: float
    positive_strategy_fold_count: int
    positive_incremental_fold_count: int
    positive_ppo_minus_fixed_control_fold_count: int
    cost_ladder_monotone: bool
    maximum_fold_drawdown: float
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_development_conclusion_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "outer_development_conclusion_authorized",
            }
        }

    def validate(self) -> None:
        gates = {
            "strategy-active-lcb-positive": self.strategy_active_log_return_lcb95 > 0.0,
            "incremental-rl-lcb-positive": self.incremental_rl_log_return_lcb95 > 0.0,
            "ppo-minus-fixed-control-lcb-positive": (
                self.ppo_minus_fixed_control_log_return_lcb95 > 0.0
            ),
            "high-cost-mean-return-nonnegative": self.mean_high_cost_terminal_return
            >= 0.0,
            "positive-strategy-folds-at-least-three": self.positive_strategy_fold_count
            >= 3,
            "positive-incremental-folds-at-least-three": self.positive_incremental_fold_count
            >= 3,
            "positive-ppo-minus-fixed-folds-at-least-three": (
                self.positive_ppo_minus_fixed_control_fold_count >= 3
            ),
            "cost-ladder-monotone": self.cost_ladder_monotone,
            "maximum-fold-drawdown": self.maximum_fold_drawdown <= 0.25,
        }
        numbers = (
            self.mean_strategy_active_log_return,
            self.mean_incremental_rl_log_return,
            self.mean_ppo_minus_fixed_control_log_return,
            self.mean_high_cost_terminal_return,
            self.strategy_active_log_return_lcb95,
            self.incremental_rl_log_return_lcb95,
            self.ppo_minus_fixed_control_log_return_lcb95,
            self.maximum_fold_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SCHEMA
            or self.fold_indices != tuple(range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1))
            or len(self.fold_receipts) != MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or self.fold_inventory_sha256 != semantic_sha256(self.fold_receipts)
            or any(not math.isfinite(value) for value in numbers)
            or self.passed_gate_names
            != tuple(sorted(name for name, passed in gates.items() if passed))
            or self.failed_gate_names
            != tuple(sorted(name for name, passed in gates.items() if not passed))
            or not isinstance(self.source_data_qualified, bool)
            or self.outer_development_conclusion_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceV1Error(
                "adaptive RL outer evidence differs"
            )
        for value in (
            *self.fold_receipts,
            self.fold_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL outer evidence", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_evidence_v1(
    folds: Sequence[MassiveAdaptiveRLOuterCostFoldV1],
) -> MassiveAdaptiveRLOuterEvidenceV1:
    ordered = tuple(sorted(folds, key=lambda value: value.fold_index))
    for fold in ordered:
        fold.validate()
    if tuple(fold.fold_index for fold in ordered) != tuple(
        range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1)
    ):
        raise MassiveAdaptiveRLOuterEvidenceV1Error(
            "adaptive RL outer evidence requires exactly folds zero through three"
        )
    active = tuple(fold.primary_strategy_active_log_returns for fold in ordered)
    incremental = tuple(fold.primary_incremental_rl_log_returns for fold in ordered)
    ppo_minus_fixed = tuple(
        fold.primary_ppo_minus_fixed_control_log_returns for fold in ordered
    )
    active_lcb = _nonwrapping_fold_cluster_lcb(active)
    incremental_lcb = _nonwrapping_fold_cluster_lcb(incremental)
    ppo_minus_fixed_lcb = _nonwrapping_fold_cluster_lcb(ppo_minus_fixed)
    mean_active = mean(value for fold in active for value in fold)
    mean_incremental = mean(value for fold in incremental for value in fold)
    mean_ppo_minus_fixed = mean(value for fold in ppo_minus_fixed for value in fold)
    mean_high = mean(fold.high_cost_terminal_return for fold in ordered)
    positive_strategy = sum(fold.primary_terminal_return > 0.0 for fold in ordered)
    positive_incremental = sum(
        sum(fold.primary_incremental_rl_log_returns) > 0.0 for fold in ordered
    )
    positive_ppo_minus_fixed = sum(
        sum(fold.primary_ppo_minus_fixed_control_log_returns) > 0.0 for fold in ordered
    )
    ladder = all(fold.terminal_return_ladder_monotone for fold in ordered)
    maximum_drawdown = max(fold.maximum_drawdown for fold in ordered)
    gates = {
        "strategy-active-lcb-positive": active_lcb > 0.0,
        "incremental-rl-lcb-positive": incremental_lcb > 0.0,
        "ppo-minus-fixed-control-lcb-positive": ppo_minus_fixed_lcb > 0.0,
        "high-cost-mean-return-nonnegative": mean_high >= 0.0,
        "positive-strategy-folds-at-least-three": positive_strategy >= 3,
        "positive-incremental-folds-at-least-three": positive_incremental >= 3,
        "positive-ppo-minus-fixed-folds-at-least-three": (
            positive_ppo_minus_fixed >= 3
        ),
        "cost-ladder-monotone": ladder,
        "maximum-fold-drawdown": maximum_drawdown <= 0.25,
    }
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SCHEMA,
        "fold_indices": tuple(fold.fold_index for fold in ordered),
        "fold_receipts": tuple(fold.semantic_receipt_sha256 for fold in ordered),
        "fold_inventory_sha256": semantic_sha256(
            tuple(fold.semantic_receipt_sha256 for fold in ordered)
        ),
        "mean_strategy_active_log_return": mean_active,
        "mean_incremental_rl_log_return": mean_incremental,
        "mean_ppo_minus_fixed_control_log_return": mean_ppo_minus_fixed,
        "mean_high_cost_terminal_return": mean_high,
        "strategy_active_log_return_lcb95": active_lcb,
        "incremental_rl_log_return_lcb95": incremental_lcb,
        "ppo_minus_fixed_control_log_return_lcb95": ppo_minus_fixed_lcb,
        "positive_strategy_fold_count": positive_strategy,
        "positive_incremental_fold_count": positive_incremental,
        "positive_ppo_minus_fixed_control_fold_count": positive_ppo_minus_fixed,
        "cost_ladder_monotone": ladder,
        "maximum_fold_drawdown": maximum_drawdown,
        "passed_gate_names": tuple(
            sorted(name for name, passed in gates.items() if passed)
        ),
        "failed_gate_names": tuple(
            sorted(name for name, passed in gates.items() if not passed)
        ),
        "source_data_qualified": all(fold.source_data_qualified for fold in ordered),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveRLOuterEvidenceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_development_conclusion_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLOuterCostFoldV1",
    "MassiveAdaptiveRLOuterEvidenceV1",
    "MassiveAdaptiveRLOuterEvidenceV1Error",
    "MassiveAdaptiveRLOuterPlanV1",
    "build_massive_adaptive_rl_outer_cost_fold_v1",
    "build_massive_adaptive_rl_outer_evidence_v1",
    "build_massive_adaptive_rl_outer_plan_v1",
]
