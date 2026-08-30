"""Checkpoint-owned deterministic policy evaluation for adaptive RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path

import torch

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    MassiveAdaptiveRLActionV1,
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256,
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


MASSIVE_ADAPTIVE_RL_POLICY_ACTION_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-action-evidence-v1"
)
MASSIVE_ADAPTIVE_RL_CHECKPOINT_POLICY_TRACE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-checkpoint-policy-trace-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SPEC_SHA256 = semantic_sha256(
    {
        "policy": "replayed-checkpoint-actor-eval-mode",
        "action": "deterministic-distribution-control",
        "economics": "shared-three-book-environment",
        "targets": "primary-policy-run-only",
        "updates": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLPolicyEvaluatorV1Error(ValueError):
    """The checkpoint, observation, deterministic action, or trace differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _policy_from_state(
    state: dict[str, torch.Tensor], *, device: torch.device
) -> MassiveAdaptivePPOActorCriticV1:
    try:
        actor_input = state["actor.0.weight"]
        actor_hidden = state["actor.2.weight"]
    except KeyError as error:
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive PPO checkpoint omits the registered actor architecture"
        ) from error
    if (
        actor_input.ndim != 2
        or actor_hidden.ndim != 2
        or actor_input.shape[0] != actor_hidden.shape[0]
        or actor_hidden.shape[0] != actor_hidden.shape[1]
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive PPO checkpoint architecture differs"
        )
    model = MassiveAdaptivePPOActorCriticV1(
        observation_dim=int(actor_input.shape[1]),
        hidden_dim=int(actor_input.shape[0]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyActionEvidenceV1:
    decision_session_date: str
    observation_receipt_sha256: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    actor_state_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    deterministic_action_implementation_sha256: str
    distribution_parameter_receipt_sha256: str
    action_values: tuple[float, ...]
    action_receipt_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_ACTION_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_ACTION_EVIDENCE_V1_SCHEMA
            or not self.decision_session_date
            or len(self.action_values) != 10
            or any(not -1.0 <= value <= 1.0 for value in self.action_values)
            or self.action_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
            or self.reward_specification_sha256
            != MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
            or self.deterministic_action_implementation_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SOURCE_SHA256
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
                "adaptive deterministic policy action evidence differs"
            )
        for value in (
            self.observation_receipt_sha256,
            self.checkpoint_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.actor_state_receipt_sha256,
            self.observation_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            self.deterministic_action_implementation_sha256,
            self.distribution_parameter_receipt_sha256,
            self.action_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive deterministic action evidence", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCheckpointPolicyTraceV1:
    fold_index: int
    evaluation_role: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    policy_trace: MassiveAdaptiveRLPolicyTraceV1
    action_evidence: tuple[MassiveAdaptiveRLPolicyActionEvidenceV1, ...]
    transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    action_evidence_inventory_sha256: str
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    checkpoint_policy_replayed: bool
    development_policy_evaluation_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_CHECKPOINT_POLICY_TRACE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "evaluation_role": self.evaluation_role,
            "checkpoint_authority_receipt_sha256": (
                self.checkpoint_authority_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "policy_trace_receipt_sha256": self.policy_trace.semantic_receipt_sha256,
            "action_evidence_inventory_sha256": self.action_evidence_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.policy_trace.validate()
        for evidence_row in self.action_evidence:
            evidence_row.validate()
        for transition_row in self.transitions:
            transition_row.validate()
        runtime = bool(self.action_evidence and self.transitions)
        expected = runtime and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_CHECKPOINT_POLICY_TRACE_V1_SCHEMA
            or self.evaluation_role not in {"inner_validation", "outer_test"}
            or self.fold_index != self.policy_trace.fold_index
            or self.evaluation_role != self.policy_trace.evaluation_role
            or self.checkpoint_receipt_sha256
            != self.policy_trace.checkpoint_receipt_sha256
            or self.model_state_receipt_sha256
            != self.policy_trace.model_state_receipt_sha256
            or len(self.action_evidence) != len(self.transitions)
            or len(self.transitions) != len(self.policy_trace.transition_receipts)
            or tuple(row.observation_receipt_sha256 for row in self.action_evidence)
            != tuple(row.observation_receipt_sha256 for row in self.transitions)
            or tuple(row.action_receipt_sha256 for row in self.action_evidence)
            != tuple(row.action_receipt_sha256 for row in self.transitions)
            or self.action_evidence_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.action_evidence)
            )
            or self.transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.transitions)
            )
            or self.checkpoint_policy_replayed != runtime
            or self.development_policy_evaluation_authorized
            != (expected and self.evaluation_role == "inner_validation")
            or self.outer_evaluation_authorized
            != (expected and self.evaluation_role == "outer_test")
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
                "adaptive checkpoint policy trace differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _action_evidence(
    *,
    decision_session_date: str,
    observation_receipt_sha256: str,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    distribution: MassiveAdaptiveBoundedControlDistributionV1,
    action: MassiveAdaptiveRLActionV1,
    action_values: tuple[float, ...],
) -> MassiveAdaptiveRLPolicyActionEvidenceV1:
    checkpoint = checkpoint_authority.runtime_checkpoint
    assert checkpoint is not None
    distribution_receipt = semantic_sha256(
        (
            _tensor_receipt(distribution.mean),
            _tensor_receipt(distribution.log_std),
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_ACTION_EVIDENCE_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "observation_receipt_sha256": observation_receipt_sha256,
        "checkpoint_authority_receipt_sha256": (
            checkpoint_authority.semantic_receipt_sha256
        ),
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "actor_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "observation_specification_sha256": (
            checkpoint.observation_specification_sha256
        ),
        "action_specification_sha256": checkpoint.action_specification_sha256,
        "reward_specification_sha256": checkpoint.reward_specification_sha256,
        "deterministic_action_implementation_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SOURCE_SHA256
        ),
        "distribution_parameter_receipt_sha256": distribution_receipt,
        "action_values": action_values,
        "action_receipt_sha256": action.semantic_receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPolicyActionEvidenceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def evaluate_massive_adaptive_rl_checkpoint_v1(
    *,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    fold_index: int,
    evaluation_role: str,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLCheckpointPolicyTraceV1:
    """Load the committed actor and derive every validation/outer action."""

    checkpoint_authority.validate()
    chronology_authority.validate()
    checkpoint = checkpoint_authority.runtime_checkpoint
    if (
        checkpoint is None
        or not checkpoint_authority.runtime_checkpoint_replayed
        or evaluation_role not in {"inner_validation", "outer_test"}
        or environment.inference_plan.fold_index != fold_index
        or tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        == ()
        or chronology_authority.fold_index != fold_index
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive policy evaluation roots are not replayed"
        )
    if evaluation_role == "inner_validation" and getattr(
        environment.inference_plan, "inference_role", None
    ) != "inner_validation":
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive policy selection requires inner-validation chronology"
        )
    if evaluation_role == "inner_validation" and (
        not chronology_authority.development_policy_selection_authorized
        or chronology_authority.validation_inference_plan_receipt_sha256
        != environment.inference_plan.semantic_receipt_sha256
        or chronology_authority.rl_validation_origin_dates
        != tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive policy evaluation is outside its selection chronology"
        )
    if evaluation_role == "outer_test" and not bool(
        getattr(environment.inference_plan, "outer_inference_authorized", False)
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive outer policy evaluation requires an outer plan"
        )
    if evaluation_role == "outer_test" and (
        not chronology_authority.outer_evaluation_authorized
        or chronology_authority.outer_inference_plan_receipt_sha256
        != environment.inference_plan.semantic_receipt_sha256
        or chronology_authority.outer_origin_dates
        != tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
    ):
        raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
            "adaptive policy evaluation is outside its outer chronology"
        )
    policy = _policy_from_state(checkpoint.model_state, device=torch.device(device))
    observation, _ = environment.reset()
    evidence: list[MassiveAdaptiveRLPolicyActionEvidenceV1] = []
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    with torch.inference_mode():
        while True:
            tensor = torch.tensor(
                observation.values,
                dtype=torch.float32,
                device=torch.device(device),
            ).unsqueeze(0)
            output = policy({"adaptive_state": tensor})
            distribution = output.distribution
            if not isinstance(distribution, MassiveAdaptiveBoundedControlDistributionV1):
                raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
                    "adaptive PPO policy emitted an unregistered distribution"
                )
            values = tuple(
                float(value)
                for value in distribution.deterministic_action()[0].cpu().tolist()
            )
            action = build_massive_adaptive_rl_action_v1(
                bucket_controls=values[:7],
                uncertainty_control=values[7],
                risk_control=values[8],
                trade_cost_control=values[9],
            )
            row = environment.inference_plan.rows[environment.state.chronology_cursor]
            evidence.append(
                _action_evidence(
                    decision_session_date=row.decision_session_date,
                    observation_receipt_sha256=observation.semantic_receipt_sha256,
                    checkpoint_authority=checkpoint_authority,
                    distribution=distribution,
                    action=action,
                    action_values=values,
                )
            )
            next_observation, _reward, terminated, truncated, info = environment.step(
                action
            )
            if truncated:
                raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
                    "deterministic policy evaluation cannot truncate"
                )
            transition = info.get("transition")
            if not isinstance(transition, MassiveAdaptiveRLTransitionV1):
                raise MassiveAdaptiveRLPolicyEvaluatorV1Error(
                    "adaptive policy evaluation transition is absent"
                )
            transitions.append(transition)
            if terminated:
                break
            assert next_observation is not None
            observation = next_observation
    trace = build_massive_adaptive_rl_policy_trace_v1(
        fold_index=fold_index,
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
        transitions=tuple(transitions),
        frozen_targets_replayed=False,
        evaluation_role=evaluation_role,
    )
    source_qualified = bool(
        checkpoint_authority.source_data_qualified
        and all(row.source_data_qualified for row in transitions)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_CHECKPOINT_POLICY_TRACE_V1_SCHEMA,
        "fold_index": fold_index,
        "evaluation_role": evaluation_role,
        "checkpoint_authority_receipt_sha256": (
            checkpoint_authority.semantic_receipt_sha256
        ),
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "policy_trace": trace,
        "action_evidence": tuple(evidence),
        "transitions": tuple(transitions),
        "action_evidence_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in evidence)
        ),
        "transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in transitions)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_EVALUATOR_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLCheckpointPolicyTraceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        checkpoint_policy_replayed=True,
        development_policy_evaluation_authorized=(
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
    "MassiveAdaptiveRLCheckpointPolicyTraceV1",
    "MassiveAdaptiveRLPolicyActionEvidenceV1",
    "MassiveAdaptiveRLPolicyEvaluatorV1Error",
    "evaluate_massive_adaptive_rl_checkpoint_v1",
]
