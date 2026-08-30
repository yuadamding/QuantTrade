"""Frozen-checkpoint outer rollout and create-only replay authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v1 import (
    MassiveAdaptiveRLOuterPlanV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    _policy_from_state,
    _tensor_receipt,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v1 import (
    MassiveAdaptiveFrozenRLPolicyV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)


MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-frozen-rl-action-evidence-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-rollout-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-rollout-authority-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-outer-rollout-authority-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "policy": "one-fold-bound-frozen-policy",
        "updates": False,
        "action": "deterministic-checkpoint-replay",
        "primary_cost_basis_points": 20.0,
        "economics": "shared-three-book-environment",
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA,
            "payload": "outer-trace-and-frozen-policy-actions",
            "promotion": "reload-policy-rerun-actions-and-economics",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLOuterRolloutV1Error(ValueError):
    """The outer plan, frozen actor, action inventory, or economics differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterRolloutV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFrozenRLActionEvidenceV1:
    decision_session_date: str
    observation_receipt_sha256: str
    frozen_policy_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    frozen_model_state_receipt_sha256: str
    distribution_parameter_receipt_sha256: str
    action_values: tuple[float, ...]
    action_receipt_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA
            or not self.decision_session_date
            or len(self.action_values) != 10
            or any(not -1.0 <= value <= 1.0 for value in self.action_values[:9])
            or not 0.0 <= self.action_values[9] <= 1.0
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterRolloutV1Error(
                "frozen adaptive RL action evidence differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterRolloutV1:
    fold_index: int
    outer_plan_receipt_sha256: str
    frozen_policy_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    frozen_model_state_receipt_sha256: str
    policy_trace: MassiveAdaptiveRLPolicyTraceV1
    action_evidence: tuple[MassiveAdaptiveFrozenRLActionEvidenceV1, ...]
    transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    action_inventory_sha256: str
    transition_inventory_sha256: str
    decision_target_inventory_sha256: str
    environment_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    frozen_policy_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "outer_plan_receipt_sha256": self.outer_plan_receipt_sha256,
            "frozen_policy_receipt_sha256": self.frozen_policy_receipt_sha256,
            "selected_checkpoint_receipt_sha256": (
                self.selected_checkpoint_receipt_sha256
            ),
            "frozen_model_state_receipt_sha256": (
                self.frozen_model_state_receipt_sha256
            ),
            "policy_trace_receipt_sha256": self.policy_trace.semantic_receipt_sha256,
            "action_inventory_sha256": self.action_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "decision_target_inventory_sha256": self.decision_target_inventory_sha256,
            "environment_source_inventory_sha256": (
                self.environment_source_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.policy_trace.validate()
        for row in self.action_evidence:
            row.validate()
        for row in self.transitions:
            row.validate()
        runtime = bool(self.action_evidence and self.transitions)
        expected = runtime and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SCHEMA
            or self.policy_trace.evaluation_role != "outer_test"
            or self.policy_trace.fold_index != self.fold_index
            or self.policy_trace.checkpoint_receipt_sha256
            != self.selected_checkpoint_receipt_sha256
            or len(self.action_evidence) != len(self.transitions)
            or self.action_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.action_evidence)
            )
            or self.transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.transitions)
            )
            or self.decision_target_inventory_sha256
            != self.policy_trace.decision_target_inventory_sha256
            or tuple(row.action_receipt_sha256 for row in self.action_evidence)
            != tuple(row.action_receipt_sha256 for row in self.transitions)
            or self.frozen_policy_replayed != runtime
            or self.outer_evaluation_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterRolloutV1Error(
                "adaptive RL outer rollout differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def run_massive_adaptive_rl_outer_rollout_v1(
    *,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLOuterRolloutV1:
    """Run one immutable actor over one fold-bound outer chronology."""

    outer_plan.validate()
    frozen_policy.validate()
    chronology_authority.validate()
    if (
        frozen_policy.runtime_model_state is None
        or not frozen_policy.runtime_policy_replayed
        or outer_plan.fold_index != frozen_policy.fold_index
        or chronology_authority.fold_index != outer_plan.fold_index
        or not chronology_authority.outer_evaluation_authorized
        or chronology_authority.outer_inference_plan_receipt_sha256
        != outer_plan.outer_inference_plan_receipt_sha256
        or chronology_authority.outer_origin_dates
        != tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        or outer_plan.frozen_rl_policy_receipt_sha256
        != frozen_policy.semantic_receipt_sha256
        or outer_plan.frozen_rl_policy_model_state_receipt_sha256
        != frozen_policy.frozen_model_state_receipt_sha256
        or environment.inference_plan.fold_index != outer_plan.fold_index
        or not bool(
            getattr(environment.inference_plan, "outer_inference_authorized", False)
        )
        or environment.inference_plan.semantic_receipt_sha256
        != outer_plan.outer_inference_plan_receipt_sha256
        or environment.forecast_archive.semantic_receipt_sha256
        != outer_plan.outer_forecast_archive_receipt_sha256
        or environment.calibration.semantic_receipt_sha256
        != outer_plan.calibration_receipt_sha256
        or environment.compiler_config.receipt_sha256
        != outer_plan.compiler_config_receipt_sha256
        or environment.initial_capital != outer_plan.primary_capital
        or environment.transaction_cost_basis_points
        != outer_plan.primary_cost_basis_points
    ):
        raise MassiveAdaptiveRLOuterRolloutV1Error(
            "adaptive outer environment differs from its frozen plan"
        )
    policy = _policy_from_state(
        frozen_policy.runtime_model_state, device=torch.device(device)
    )
    observation, _ = environment.reset()
    evidence: list[MassiveAdaptiveFrozenRLActionEvidenceV1] = []
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
                raise MassiveAdaptiveRLOuterRolloutV1Error(
                    "frozen adaptive actor emitted an unregistered distribution"
                )
            values = tuple(
                float(value)
                for value in distribution.deterministic_action()[0].cpu().tolist()
            )
            action = build_massive_adaptive_rl_action_v1(
                bucket_controls=values[:7],
                uncertainty_control=values[7],
                risk_control=values[8],
                turnover_control=values[9],
            )
            decision_date = environment.inference_plan.rows[
                environment.state.chronology_cursor
            ].decision_session_date
            distribution_receipt = semantic_sha256(
                (
                    _tensor_receipt(distribution.mean),
                    _tensor_receipt(distribution.log_std),
                    _tensor_receipt(distribution.turnover_alpha),
                    _tensor_receipt(distribution.turnover_beta),
                )
            )
            evidence_body = {
                "schema": MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA,
                "decision_session_date": decision_date,
                "observation_receipt_sha256": observation.semantic_receipt_sha256,
                "frozen_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
                "selected_checkpoint_receipt_sha256": (
                    frozen_policy.selected_rl_checkpoint_receipt_sha256
                ),
                "frozen_model_state_receipt_sha256": (
                    frozen_policy.frozen_model_state_receipt_sha256
                ),
                "distribution_parameter_receipt_sha256": distribution_receipt,
                "action_values": values,
                "action_receipt_sha256": action.semantic_receipt_sha256,
                "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
            }
            evidence_row = MassiveAdaptiveFrozenRLActionEvidenceV1(
                **evidence_body,  # type: ignore[arg-type]
                semantic_receipt_sha256=semantic_sha256(evidence_body),
            )
            evidence_row.validate()
            evidence.append(evidence_row)
            next_observation, _reward, terminated, truncated, info = environment.step(
                action
            )
            if truncated:
                raise MassiveAdaptiveRLOuterRolloutV1Error(
                    "outer adaptive rollout cannot truncate"
                )
            transition = info.get("transition")
            if not isinstance(transition, MassiveAdaptiveRLTransitionV1):
                raise MassiveAdaptiveRLOuterRolloutV1Error(
                    "outer adaptive transition is absent"
                )
            transitions.append(transition)
            if terminated:
                break
            assert next_observation is not None
            observation = next_observation
    trace = build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=outer_plan.fold_index,
        checkpoint_receipt_sha256=(
            frozen_policy.selected_rl_checkpoint_receipt_sha256
        ),
        model_state_receipt_sha256=(
            frozen_policy.selected_rl_checkpoint_model_state_receipt_sha256
        ),
        update_index=frozen_policy.selected_update_index,
        training_forecast_authority_receipt_sha256=(
            frozen_policy.training_forecast_authority_receipt_sha256
        ),
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
        evaluation_role="outer_test",
        checkpoint_source_data_qualified=frozen_policy.source_data_qualified,
    )
    source_qualified = bool(
        outer_plan.outer_evaluation_authorized
        and frozen_policy.development_outer_policy_authorized
        and all(row.source_data_qualified for row in transitions)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SCHEMA,
        "fold_index": outer_plan.fold_index,
        "outer_plan_receipt_sha256": outer_plan.semantic_receipt_sha256,
        "frozen_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "selected_checkpoint_receipt_sha256": (
            frozen_policy.selected_rl_checkpoint_receipt_sha256
        ),
        "frozen_model_state_receipt_sha256": (
            frozen_policy.frozen_model_state_receipt_sha256
        ),
        "policy_trace": trace,
        "action_evidence": tuple(evidence),
        "transitions": tuple(transitions),
        "action_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in evidence)
        ),
        "transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in transitions)
        ),
        "decision_target_inventory_sha256": trace.decision_target_inventory_sha256,
        "environment_source_inventory_sha256": environment.source_inventory_sha256,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterRolloutV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        frozen_policy_replayed=True,
        outer_evaluation_authorized=source_qualified,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterRolloutAuthorityV1:
    fold_index: int
    outer_plan_receipt_sha256: str
    frozen_policy_receipt_sha256: str
    outer_rollout_receipt_sha256: str
    policy_trace_receipt_sha256: str
    action_inventory_sha256: str
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rollout: MassiveAdaptiveRLOuterRolloutV1 | None
    runtime_rollout_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "outer_plan_receipt_sha256": self.outer_plan_receipt_sha256,
            "frozen_policy_receipt_sha256": self.frozen_policy_receipt_sha256,
            "outer_rollout_receipt_sha256": self.outer_rollout_receipt_sha256,
            "policy_trace_receipt_sha256": self.policy_trace_receipt_sha256,
            "action_inventory_sha256": self.action_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_rollout is not None
        expected = runtime and self.source_data_qualified
        if self.runtime_rollout is not None:
            self.runtime_rollout.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.outer_rollout_receipt_sha256
            or self.runtime_rollout_replayed != runtime
            or self.outer_evaluation_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterRolloutV1Error(
                "adaptive RL outer rollout authority differs"
            )
        if runtime and self.runtime_rollout is not None and (
            self.runtime_rollout.fold_index != self.fold_index
            or self.runtime_rollout.outer_plan_receipt_sha256
            != self.outer_plan_receipt_sha256
            or self.runtime_rollout.frozen_policy_receipt_sha256
            != self.frozen_policy_receipt_sha256
            or self.runtime_rollout.semantic_receipt_sha256
            != self.outer_rollout_receipt_sha256
            or self.runtime_rollout.policy_trace.semantic_receipt_sha256
            != self.policy_trace_receipt_sha256
            or self.runtime_rollout.action_inventory_sha256
            != self.action_inventory_sha256
            or self.runtime_rollout.transition_inventory_sha256
            != self.transition_inventory_sha256
        ):
            raise MassiveAdaptiveRLOuterRolloutV1Error(
                "adaptive runtime outer rollout differs from its authority"
            )
        for value in (
            self.outer_plan_receipt_sha256,
            self.frozen_policy_receipt_sha256,
            self.outer_rollout_receipt_sha256,
            self.policy_trace_receipt_sha256,
            self.action_inventory_sha256,
            self.transition_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL outer rollout authority", value)


def _payload(rollout: MassiveAdaptiveRLOuterRolloutV1) -> dict[str, object]:
    return {
        "fold_index": rollout.fold_index,
        "outer_plan_receipt_sha256": rollout.outer_plan_receipt_sha256,
        "frozen_policy_receipt_sha256": rollout.frozen_policy_receipt_sha256,
        "selected_checkpoint_receipt_sha256": (
            rollout.selected_checkpoint_receipt_sha256
        ),
        "frozen_model_state_receipt_sha256": (
            rollout.frozen_model_state_receipt_sha256
        ),
        "policy_trace": asdict(rollout.policy_trace),
        "action_evidence": tuple(asdict(row) for row in rollout.action_evidence),
        "action_inventory_sha256": rollout.action_inventory_sha256,
        "transition_inventory_sha256": rollout.transition_inventory_sha256,
        "decision_target_inventory_sha256": rollout.decision_target_inventory_sha256,
        "environment_source_inventory_sha256": (
            rollout.environment_source_inventory_sha256
        ),
        "source_data_qualified": rollout.source_data_qualified,
        "outer_rollout_receipt_sha256": rollout.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterRolloutV1Error(
            "adaptive RL outer rollout payload is not canonical JSON"
        )
    return dict(value)


def parse_massive_adaptive_rl_outer_rollout_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterRolloutAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA,
        "fold_index": int(payload["fold_index"]),
        "outer_plan_receipt_sha256": str(payload["outer_plan_receipt_sha256"]),
        "frozen_policy_receipt_sha256": str(payload["frozen_policy_receipt_sha256"]),
        "outer_rollout_receipt_sha256": str(payload["outer_rollout_receipt_sha256"]),
        "policy_trace_receipt_sha256": str(
            dict(payload["policy_trace"])["semantic_receipt_sha256"]  # type: ignore[arg-type]
        ),
        "action_inventory_sha256": str(payload["action_inventory_sha256"]),
        "transition_inventory_sha256": str(payload["transition_inventory_sha256"]),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterRolloutAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_rollout=None,
        runtime_rollout_replayed=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_outer_rollout_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLOuterRolloutAuthorityV1,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLOuterRolloutAuthorityV1:
    parsed = parse_massive_adaptive_rl_outer_rollout_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    replayed = run_massive_adaptive_rl_outer_rollout_v1(
        outer_plan=outer_plan,
        frozen_policy=frozen_policy,
        chronology_authority=chronology_authority,
        environment=environment,
        device=device,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(replayed)
    ):
        raise MassiveAdaptiveRLOuterRolloutV1Error(
            "adaptive RL outer rollout does not replay from the frozen policy"
        )
    result = replace(
        parsed,
        runtime_rollout=replayed,
        runtime_rollout_replayed=True,
        outer_evaluation_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_outer_rollout_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    committed_at_ms: int,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLOuterRolloutAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLOuterRolloutV1Error(
            "adaptive RL outer rollout artifact ID is not path safe"
        )
    rollout = run_massive_adaptive_rl_outer_rollout_v1(
        outer_plan=outer_plan,
        frozen_policy=frozen_policy,
        chronology_authority=chronology_authority,
        environment=environment,
        device=device,
    )
    relative = f"massive-adaptive/rl-outer-rollout-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(rollout))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=rollout.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-OUTER-ROLLOUT-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_outer_rollout_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_outer_rollout_authority_v1(
            root=root, loaded_source=loaded
        ),
        outer_plan=outer_plan,
        frozen_policy=frozen_policy,
        chronology_authority=chronology_authority,
        environment=environment,
        device=device,
    )


__all__ = [
    "MassiveAdaptiveFrozenRLActionEvidenceV1",
    "MassiveAdaptiveRLOuterRolloutAuthorityV1",
    "MassiveAdaptiveRLOuterRolloutV1",
    "MassiveAdaptiveRLOuterRolloutV1Error",
    "authorize_massive_adaptive_rl_outer_rollout_authority_v1",
    "materialize_massive_adaptive_rl_outer_rollout_authority_v1",
    "parse_massive_adaptive_rl_outer_rollout_authority_v1",
    "run_massive_adaptive_rl_outer_rollout_v1",
]
