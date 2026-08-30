"""Checkpoint-owned primary rollout and frozen-target cost stresses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    MassiveAdaptiveRLCheckpointPolicyTraceV1,
    evaluate_massive_adaptive_rl_checkpoint_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_v1,
)


MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-cost-ladder-v1"
)
MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SPEC_SHA256 = semantic_sha256(
    {
        "costs_basis_points": (10.0, 20.0, 40.0),
        "primary": "checkpoint-deterministic-actions-and-compiler",
        "stress": "same-primary-target-weights-no-policy-or-compiler-rerun",
        "economics": "same-fill-event-and-three-book-kernel",
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLCostLadderV1Error(ValueError):
    """Primary actions, frozen targets, or cost-rung roots differ."""


def _environment_identity(
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> tuple[object, ...]:
    return (
        environment.forecast_archive.semantic_receipt_sha256,
        environment.calibration.semantic_receipt_sha256,
        environment.inference_plan.semantic_receipt_sha256,
        environment.fill_source.semantic_receipt_sha256,
        environment.daily_input_authority.semantic_receipt_sha256,
        environment.identity_authority.receipt_sha256,
        None
        if environment.economic_event_archive is None
        else environment.economic_event_archive.receipt_sha256,
        environment.compiler_config.receipt_sha256,
        environment.initial_capital,
        environment.maximum_fill_participation,
        tuple(
            (date, row.semantic_receipt_sha256)
            for date, row in environment.roots.items()
        ),
        tuple(
            (date, row.semantic_receipt_sha256)
            for date, row in environment.contexts.items()
        ),
    )


def replay_massive_adaptive_rl_frozen_target_transitions_v1(
    *,
    primary_action_evidence: Sequence[object],
    primary_transitions: Sequence[MassiveAdaptiveRLTransitionV1],
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> tuple[MassiveAdaptiveRLTransitionV1, ...]:
    """Replay an attached primary target inventory through one stress book."""

    evidence_rows = tuple(primary_action_evidence)
    primary_rows = tuple(primary_transitions)
    if not evidence_rows or len(evidence_rows) != len(primary_rows):
        raise MassiveAdaptiveRLCostLadderV1Error(
            "frozen-target replay evidence inventory differs"
        )
    environment.reset()
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    for index, (evidence, primary_transition) in enumerate(
        zip(evidence_rows, primary_rows, strict=True)
    ):
        row = environment.inference_plan.rows[index]
        if (
            getattr(evidence, "decision_session_date", None)
            != row.decision_session_date
            or getattr(evidence, "observation_receipt_sha256", None)
            != primary_transition.observation_receipt_sha256
        ):
            raise MassiveAdaptiveRLCostLadderV1Error(
                "primary action evidence and frozen chronology differ"
            )
        values = tuple(getattr(evidence, "action_values", ()))
        if len(values) != 10:
            raise MassiveAdaptiveRLCostLadderV1Error(
                "frozen-target action evidence differs"
            )
        action = build_massive_adaptive_rl_action_v1(
            bucket_controls=values[:7],
            uncertainty_control=values[7],
            risk_control=values[8],
            trade_cost_control=values[9],
        )
        next_observation, _reward, terminated, truncated, info = environment.step(
            action,
            frozen_control=primary_transition.compiler_control,
            frozen_decision=primary_transition.policy_decision,
        )
        if truncated:
            raise MassiveAdaptiveRLCostLadderV1Error(
                "frozen-target cost replay cannot truncate"
            )
        transition = info.get("transition")
        if not isinstance(transition, MassiveAdaptiveRLTransitionV1):
            raise MassiveAdaptiveRLCostLadderV1Error(
                "frozen-target transition is absent"
            )
        if (
            transition.policy_decision.security_ids
            != primary_transition.policy_decision.security_ids
            or transition.policy_decision.target_weights
            != primary_transition.policy_decision.target_weights
            or not transition.economic_step.frozen_targets_replayed
        ):
            raise MassiveAdaptiveRLCostLadderV1Error(
                "cost stress changed a primary target"
            )
        transitions.append(transition)
        if terminated != (index == len(primary_rows) - 1):
            raise MassiveAdaptiveRLCostLadderV1Error(
                "frozen-target termination differs"
            )
        if not terminated:
            assert next_observation is not None
    return tuple(transitions)


def _replay_frozen_targets(
    *,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    primary: MassiveAdaptiveRLCheckpointPolicyTraceV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> tuple[MassiveAdaptiveRLPolicyTraceV1, tuple[MassiveAdaptiveRLTransitionV1, ...]]:
    checkpoint = checkpoint_authority.runtime_checkpoint
    if checkpoint is None:
        raise MassiveAdaptiveRLCostLadderV1Error(
            "frozen-target replay has no runtime checkpoint"
        )
    transitions = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=primary.action_evidence,
        primary_transitions=primary.transitions,
        environment=environment,
    )
    trace = build_massive_adaptive_rl_policy_trace_v1(
        fold_index=primary.fold_index,
        checkpoint=checkpoint,
        forecast_archive_receipt_sha256=(
            environment.forecast_archive.semantic_receipt_sha256
        ),
        inference_plan_receipt_sha256=(
            environment.inference_plan.semantic_receipt_sha256
        ),
        calibration_receipt_sha256=environment.calibration.semantic_receipt_sha256,
        transaction_cost_basis_points=environment.transaction_cost_basis_points,
        initial_capital=environment.initial_capital,
        transitions=transitions,
        frozen_targets_replayed=True,
        evaluation_role=primary.evaluation_role,
    )
    return trace, transitions


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCostLadderV1:
    fold_index: int
    evaluation_role: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    primary: MassiveAdaptiveRLCheckpointPolicyTraceV1
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1
    low_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    high_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    decision_target_inventory_sha256: str
    low_cost_transition_inventory_sha256: str
    high_cost_transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_policy_selection_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "evaluation_role": self.evaluation_role,
            "checkpoint_authority_receipt_sha256": (
                self.checkpoint_authority_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "primary_receipt_sha256": self.primary.semantic_receipt_sha256,
            "low_cost_trace_receipt_sha256": (
                self.low_cost_trace.semantic_receipt_sha256
            ),
            "high_cost_trace_receipt_sha256": (
                self.high_cost_trace.semantic_receipt_sha256
            ),
            "decision_target_inventory_sha256": (
                self.decision_target_inventory_sha256
            ),
            "low_cost_transition_inventory_sha256": (
                self.low_cost_transition_inventory_sha256
            ),
            "high_cost_transition_inventory_sha256": (
                self.high_cost_transition_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.primary.validate()
        self.low_cost_trace.validate()
        self.high_cost_trace.validate()
        for transition in (*self.low_cost_transitions, *self.high_cost_transitions):
            transition.validate()
        expected = self.source_data_qualified
        traces = (
            self.low_cost_trace,
            self.primary.policy_trace,
            self.high_cost_trace,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SCHEMA
            or self.fold_index != self.primary.fold_index
            or self.evaluation_role != self.primary.evaluation_role
            or self.checkpoint_receipt_sha256 != self.primary.checkpoint_receipt_sha256
            or tuple(row.transaction_cost_basis_points for row in traces)
            != (10.0, 20.0, 40.0)
            or self.primary.policy_trace.frozen_targets_replayed
            or not self.low_cost_trace.frozen_targets_replayed
            or not self.high_cost_trace.frozen_targets_replayed
            or len({row.decision_target_inventory_sha256 for row in traces}) != 1
            or self.decision_target_inventory_sha256
            != self.primary.policy_trace.decision_target_inventory_sha256
            or self.low_cost_transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.low_cost_transitions)
            )
            or self.high_cost_transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.high_cost_transitions)
            )
            or tuple(row.terminal_liquidation_adjusted_return for row in traces)
            != tuple(
                sorted(
                    (row.terminal_liquidation_adjusted_return for row in traces),
                    reverse=True,
                )
            )
            or self.development_policy_selection_authorized
            != (expected and self.evaluation_role == "inner_validation")
            or self.outer_evaluation_authorized
            != (expected and self.evaluation_role == "outer_test")
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLCostLadderV1Error(
                "adaptive RL frozen-target cost ladder differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1(
    *,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    fold_index: int,
    evaluation_role: str,
) -> MassiveAdaptiveRLCostLadderV1:
    """Evaluate the actor once, then replay its target sequence at 10/40 bp."""

    environments = (
        low_cost_environment,
        primary_environment,
        high_cost_environment,
    )
    if (
        tuple(row.transaction_cost_basis_points for row in environments)
        != (10.0, 20.0, 40.0)
        or len({_environment_identity(row) for row in environments}) != 1
    ):
        raise MassiveAdaptiveRLCostLadderV1Error(
            "adaptive RL cost environments differ beyond transaction cost"
        )
    primary = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=primary_environment,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
    )
    low_trace, low_transitions = _replay_frozen_targets(
        checkpoint_authority=checkpoint_authority,
        primary=primary,
        environment=low_cost_environment,
    )
    high_trace, high_transitions = _replay_frozen_targets(
        checkpoint_authority=checkpoint_authority,
        primary=primary,
        environment=high_cost_environment,
    )
    source_qualified = bool(
        primary.source_data_qualified
        and low_trace.source_data_qualified
        and high_trace.source_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SCHEMA,
        "fold_index": fold_index,
        "evaluation_role": evaluation_role,
        "checkpoint_authority_receipt_sha256": (
            checkpoint_authority.semantic_receipt_sha256
        ),
        "checkpoint_receipt_sha256": primary.checkpoint_receipt_sha256,
        "primary": primary,
        "low_cost_trace": low_trace,
        "high_cost_trace": high_trace,
        "low_cost_transitions": low_transitions,
        "high_cost_transitions": high_transitions,
        "decision_target_inventory_sha256": (
            primary.policy_trace.decision_target_inventory_sha256
        ),
        "low_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in low_transitions)
        ),
        "high_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in high_transitions)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_COST_LADDER_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLCostLadderV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_policy_selection_authorized=(
            source_qualified and evaluation_role == "inner_validation"
        ),
        outer_evaluation_authorized=(
            source_qualified and evaluation_role == "outer_test"
        ),
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLCostLadderV1",
    "MassiveAdaptiveRLCostLadderV1Error",
    "evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1",
    "replay_massive_adaptive_rl_frozen_target_transitions_v1",
]
