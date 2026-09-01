"""Canonical manifest and training lifecycle for Massive adaptive RL.

This workflow is deliberately narrower than a historical launcher.  It owns
the complete causal multi-block PPO fit and publishes every registered policy
candidate in both forms required downstream:

* a prequential checkpoint that can resume the block runner exactly; and
* a policy checkpoint that the deterministic validation evaluator can reopen.

No action, transition, return, or P&L array is accepted by this workflow.
Outer access and profitability reporting remain downstream, separately
authorized boundaries.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_evaluator_v1 import (
    MassiveAdaptiveRLFixedControlEvaluationV1,
    evaluate_massive_adaptive_rl_fixed_control_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    MassiveAdaptiveRLCostLadderAuthorityV1,
    materialize_massive_adaptive_rl_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MassiveAdaptiveRLPolicyTraceAuthorityV1,
    materialize_massive_adaptive_rl_policy_trace_authority_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256,
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256,
    MassiveAdaptivePPOConfigV1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_checkpoint_authority_v1 import (
    MassiveAdaptivePrequentialPPOCheckpointAuthorityV1,
    materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MassiveAdaptivePPOTrainingRunV1,
    MassiveAdaptivePrequentialPPORunnerV1,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
    materialize_massive_adaptive_rl_checkpoint_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
    materialize_massive_adaptive_rl_fixed_control_fit_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_protocol_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_authority_v1 import (
    MassiveAdaptiveRLFitEnvironmentAuthorityV1,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v1"
)
MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-training-workflow-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_WORKFLOW_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-workflow-v1"
)
MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256 = semantic_sha256(
    {
        "training": "all-authorized-prequential-blocks-in-order",
        "candidate_publication": (
            "exact-runner-resume-authority",
            "checkpoint-owned-policy-evaluation-authority",
        ),
        "caller_actions": False,
        "caller_transitions": False,
        "caller_returns": False,
        "validation_context": "one-shared-receipt-for-all-candidates-and-controls",
        "seed_policy": "one-canonical-predeclared-seed-no-selection",
        "validation_access_during_fit": False,
        "outer_access": False,
        "lockbox_access": False,
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1 = (
    "active-return-lcb95-positive",
    "incremental-return-lcb95-positive",
    "ppo-minus-fixed-control-lcb95-positive",
    "high-cost-liquidation-adjusted-return-nonnegative",
    "positive-active-folds-at-least-three",
    "positive-incremental-folds-at-least-three",
    "positive-ppo-minus-fixed-folds-at-least-three",
    "frozen-target-cost-ladders-monotone",
    "maximum-fold-drawdown-at-most-0.25",
)


class MassiveAdaptiveRLWorkflowV1Error(ValueError):
    """The experiment manifest or package-owned training lifecycle differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLWorkflowV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL experiment ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV1:
    experiment_id: str
    fold_indices: tuple[int, ...]
    prequential_block_sessions: int
    candidate_update_indices: tuple[int, ...]
    seeds: tuple[int, ...]
    seed_policy: str
    ppo_config: MassiveAdaptivePPOConfigV1
    primary_capital: float
    cost_ladder_basis_points: tuple[float, ...]
    primary_cost_basis_points: float
    maximum_fill_participation: float
    fixed_control_registry_receipt_sha256: str
    policy_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    benchmark_specification: str
    initial_book_specification: str
    maximum_fold_drawdown: float
    outer_gate_names: tuple[str, ...]
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    workflow_specification_sha256: str = MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256
    workflow_implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "fold_indices": self.fold_indices,
            "prequential_block_sessions": self.prequential_block_sessions,
            "candidate_update_indices": self.candidate_update_indices,
            "seeds": self.seeds,
            "seed_policy": self.seed_policy,
            "ppo_config": asdict(self.ppo_config),
            "primary_capital": self.primary_capital,
            "cost_ladder_basis_points": self.cost_ladder_basis_points,
            "primary_cost_basis_points": self.primary_cost_basis_points,
            "maximum_fill_participation": self.maximum_fill_participation,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "policy_specification_sha256": self.policy_specification_sha256,
            "action_specification_sha256": self.action_specification_sha256,
            "reward_specification_sha256": self.reward_specification_sha256,
            "benchmark_specification": self.benchmark_specification,
            "initial_book_specification": self.initial_book_specification,
            "maximum_fold_drawdown": self.maximum_fold_drawdown,
            "outer_gate_names": self.outer_gate_names,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "workflow_specification_sha256": self.workflow_specification_sha256,
            "workflow_implementation_source_sha256": (
                self.workflow_implementation_source_sha256
            ),
        }

    def validate(self) -> None:
        self.ppo_config.validate()
        registry = build_massive_adaptive_rl_fixed_control_registry_v1()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V1_SCHEMA
            or _artifact_id(self.experiment_id) != self.experiment_id
            or self.fold_indices != (0, 1, 2, 3)
            or self.prequential_block_sessions not in {21, 63}
            or not self.candidate_update_indices
            or self.candidate_update_indices
            != tuple(sorted(set(self.candidate_update_indices)))
            or any(value <= 0 for value in self.candidate_update_indices)
            or len(self.seeds) != 1
            or self.seeds != tuple(sorted(set(self.seeds)))
            or any(isinstance(value, bool) or value < 0 for value in self.seeds)
            or self.seed_policy != "canonical-fixed-seed-v1"
            or self.ppo_config.seed != self.seeds[0]
            or self.primary_capital != 10_000_000.0
            or self.cost_ladder_basis_points != (10.0, 20.0, 40.0)
            or self.primary_cost_basis_points != 20.0
            or self.maximum_fill_participation != 0.02
            or self.fixed_control_registry_receipt_sha256
            != registry.semantic_receipt_sha256
            or self.policy_specification_sha256
            != MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256
            or self.action_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
            or self.reward_specification_sha256
            != MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
            or self.benchmark_specification != "shared-buy-and-drift-book-v1"
            or self.initial_book_specification != "all-books-cash-v1"
            or self.maximum_fold_drawdown != 0.25
            or self.outer_gate_names != MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.workflow_specification_sha256
            != MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256
            or self.workflow_implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL experiment manifest differs"
            )
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.policy_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            self.protocol_receipt_sha256,
            self.workflow_specification_sha256,
            self.workflow_implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL experiment manifest", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_experiment_manifest_v1(
    *,
    experiment_id: str,
    prequential_block_sessions: int = 21,
    candidate_update_indices: tuple[int, ...] = (1,),
    seeds: tuple[int, ...] = (17,),
    ppo_config: MassiveAdaptivePPOConfigV1 | None = None,
) -> MassiveAdaptiveRLExperimentManifestV1:
    config = ppo_config or MassiveAdaptivePPOConfigV1(seed=seeds[0] if seeds else 0)
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V1_SCHEMA,
        "experiment_id": _artifact_id(experiment_id),
        "fold_indices": (0, 1, 2, 3),
        "prequential_block_sessions": prequential_block_sessions,
        "candidate_update_indices": candidate_update_indices,
        "seeds": seeds,
        "seed_policy": "canonical-fixed-seed-v1",
        "ppo_config": config,
        "primary_capital": 10_000_000.0,
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "primary_cost_basis_points": 20.0,
        "maximum_fill_participation": 0.02,
        "fixed_control_registry_receipt_sha256": registry.semantic_receipt_sha256,
        "policy_specification_sha256": MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256,
        "action_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
        ),
        "reward_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
        ),
        "benchmark_specification": "shared-buy-and-drift-book-v1",
        "initial_book_specification": "all-books-cash-v1",
        "maximum_fold_drawdown": 0.25,
        "outer_gate_names": MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "workflow_specification_sha256": MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256,
        "workflow_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLExperimentManifestV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLExperimentManifestV1(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_manifest_v1(
    path: str | Path,
) -> MassiveAdaptiveRLExperimentManifestV1:
    source = Path(path)
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL experiment manifest is not canonical JSON"
        )
    payload = dict(value)
    for name in (
        "fold_indices",
        "candidate_update_indices",
        "seeds",
        "cost_ladder_basis_points",
        "outer_gate_names",
    ):
        payload[name] = tuple(cast(list[object], payload[name]))
    payload["ppo_config"] = MassiveAdaptivePPOConfigV1(
        **cast(dict[str, object], payload["ppo_config"])  # type: ignore[arg-type]
    )
    result = MassiveAdaptiveRLExperimentManifestV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def write_massive_adaptive_rl_experiment_manifest_v1(
    *, path: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV1
) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(manifest)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL experiment manifest is create-only"
        ) from error


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrainingWorkflowV1:
    experiment_manifest_receipt_sha256: str
    fold_index: int
    seed: int
    training_forecast_authority_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    training_run: MassiveAdaptivePPOTrainingRunV1
    runner_checkpoint_authorities: tuple[
        MassiveAdaptivePrequentialPPOCheckpointAuthorityV1, ...
    ]
    policy_checkpoint_authorities: tuple[MassiveAdaptiveRLCheckpointAuthorityV1, ...]
    candidate_update_indices: tuple[int, ...]
    runner_checkpoint_inventory_sha256: str
    policy_checkpoint_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_manifest_receipt_sha256": (
                self.experiment_manifest_receipt_sha256
            ),
            "fold_index": self.fold_index,
            "seed": self.seed,
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "chronology_authority_receipt_sha256": (
                self.chronology_authority_receipt_sha256
            ),
            "fixed_control_fit_authority_receipt_sha256": (
                self.fixed_control_fit_authority.semantic_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority.semantic_receipt_sha256
            ),
            "training_run_receipt_sha256": self.training_run.semantic_receipt_sha256,
            "candidate_update_indices": self.candidate_update_indices,
            "runner_checkpoint_inventory_sha256": (
                self.runner_checkpoint_inventory_sha256
            ),
            "policy_checkpoint_inventory_sha256": (
                self.policy_checkpoint_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.fixed_control_fit_authority.validate()
        self.fixed_control_selection_authority.validate()
        self.training_run.validate()
        for runner_authority in self.runner_checkpoint_authorities:
            runner_authority.validate()
        for policy_authority in self.policy_checkpoint_authorities:
            policy_authority.validate()
        runner_updates = tuple(
            authority.runtime_checkpoint.ppo_checkpoint.update_index
            for authority in self.runner_checkpoint_authorities
            if authority.runtime_checkpoint is not None
        )
        policy_updates = tuple(
            authority.runtime_checkpoint.update_index
            for authority in self.policy_checkpoint_authorities
            if authority.runtime_checkpoint is not None
        )
        runtime = bool(
            self.runner_checkpoint_authorities
            and self.policy_checkpoint_authorities
            and len(runner_updates) == len(self.runner_checkpoint_authorities)
            and len(policy_updates) == len(self.policy_checkpoint_authorities)
        )
        expected = runtime and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V1_SCHEMA
            or self.fold_index < 0
            or self.seed < 0
            or self.candidate_update_indices != runner_updates
            or self.candidate_update_indices != policy_updates
            or self.runner_checkpoint_inventory_sha256
            != semantic_sha256(
                tuple(
                    authority.semantic_receipt_sha256
                    for authority in self.runner_checkpoint_authorities
                )
            )
            or self.policy_checkpoint_inventory_sha256
            != semantic_sha256(
                tuple(
                    authority.semantic_receipt_sha256
                    for authority in self.policy_checkpoint_authorities
                )
            )
            or self.training_run.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority_receipt_sha256
            or self.training_run.rl_chronology_authority_receipt_sha256
            != self.chronology_authority_receipt_sha256
            or self.fixed_control_fit_authority.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority_receipt_sha256
            or self.fixed_control_fit_authority.chronology_authority_receipt_sha256
            != self.chronology_authority_receipt_sha256
            or not self.fixed_control_fit_authority.runtime_fit_replayed
            or not self.fixed_control_selection_authority.runtime_selection_replayed
            or self.fixed_control_selection_authority.runtime_selection is None
            or self.fixed_control_selection_authority.runtime_selection.candidate_inventory_sha256
            != self.fixed_control_fit_authority.candidate_inventory_sha256
            or self.training_run.update_count != self.candidate_update_indices[-1]
            or self.development_rl_training_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL training workflow differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationWorkflowV1:
    experiment_manifest_receipt_sha256: str
    training_workflow_receipt_sha256: str
    fold_index: int
    checkpoint_authority_receipts: tuple[str, ...]
    validation_context_receipt_sha256: str
    fixed_control_evaluation: MassiveAdaptiveRLFixedControlEvaluationV1
    policy_trace_authorities: tuple[MassiveAdaptiveRLPolicyTraceAuthorityV1, ...]
    cost_ladder_authorities: tuple[MassiveAdaptiveRLCostLadderAuthorityV1, ...]
    policy_trace_authority_inventory_sha256: str
    cost_ladder_authority_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_policy_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_WORKFLOW_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_manifest_receipt_sha256": (
                self.experiment_manifest_receipt_sha256
            ),
            "training_workflow_receipt_sha256": (self.training_workflow_receipt_sha256),
            "fold_index": self.fold_index,
            "checkpoint_authority_receipts": self.checkpoint_authority_receipts,
            "validation_context_receipt_sha256": (
                self.validation_context_receipt_sha256
            ),
            "fixed_control_evaluation_receipt_sha256": (
                self.fixed_control_evaluation.semantic_receipt_sha256
            ),
            "policy_trace_authority_inventory_sha256": (
                self.policy_trace_authority_inventory_sha256
            ),
            "cost_ladder_authority_inventory_sha256": (
                self.cost_ladder_authority_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.fixed_control_evaluation.validate()
        for trace_authority in self.policy_trace_authorities:
            trace_authority.validate()
        for ladder_authority in self.cost_ladder_authorities:
            ladder_authority.validate()
        _digest(
            "adaptive RL validation context",
            self.validation_context_receipt_sha256,
        )
        expected = self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_WORKFLOW_V1_SCHEMA
            or self.fold_index < 0
            or not self.checkpoint_authority_receipts
            or len(self.checkpoint_authority_receipts)
            != len(self.policy_trace_authorities)
            or len(self.checkpoint_authority_receipts)
            != len(self.cost_ladder_authorities)
            or tuple(
                row.checkpoint_authority_receipt_sha256
                for row in self.policy_trace_authorities
            )
            != self.checkpoint_authority_receipts
            or tuple(
                row.checkpoint_authority_receipt_sha256
                for row in self.cost_ladder_authorities
            )
            != self.checkpoint_authority_receipts
            or any(
                row.evaluation_role != "inner_validation"
                or row.fold_index != self.fold_index
                for row in self.policy_trace_authorities
            )
            or any(
                row.evaluation_role != "inner_validation"
                or row.fold_index != self.fold_index
                for row in self.cost_ladder_authorities
            )
            or self.fixed_control_evaluation.fold_index != self.fold_index
            or self.fixed_control_evaluation.validation_context_receipt_sha256
            != self.validation_context_receipt_sha256
            or tuple(
                row.policy_trace_receipt_sha256 for row in self.policy_trace_authorities
            )
            != tuple(
                row.primary_trace_receipt_sha256 for row in self.cost_ladder_authorities
            )
            or self.policy_trace_authority_inventory_sha256
            != semantic_sha256(
                tuple(
                    row.semantic_receipt_sha256 for row in self.policy_trace_authorities
                )
            )
            or self.cost_ladder_authority_inventory_sha256
            != semantic_sha256(
                tuple(
                    row.semantic_receipt_sha256 for row in self.cost_ladder_authorities
                )
            )
            or self.development_policy_evaluation_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL validation workflow differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def run_massive_adaptive_rl_training_workflow_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV1,
    fold_index: int,
    seed: int,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    fit_environment_authorities: (
        Mapping[str, MassiveAdaptiveRLFitEnvironmentAuthorityV1] | None
    ) = None,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLTrainingWorkflowV1:
    """Fit all blocks and publish every registered candidate checkpoint."""

    manifest.validate()
    training_authority.validate()
    chronology_authority.validate()
    if (
        fold_index not in manifest.fold_indices
        or seed not in manifest.seeds
        or training_authority.outer_fold_index != fold_index
        or chronology_authority.fold_index != fold_index
        or training_authority.block_sessions != manifest.prequential_block_sessions
        or not training_authority.reinforcement_learning_authorized
        or not chronology_authority.development_rl_training_authorized
    ):
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL training inputs differ from the experiment manifest"
        )
    config = replace(manifest.ppo_config, seed=seed)
    config.validate()
    model = MassiveAdaptivePPOActorCriticV1(observation_dim=90)
    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        fit_environment_authorities=fit_environment_authorities,
        model=model,
        config=config,
        device=device,
    )
    runner_authorities: list[MassiveAdaptivePrequentialPPOCheckpointAuthorityV1] = []
    policy_authorities: list[MassiveAdaptiveRLCheckpointAuthorityV1] = []
    schedule = set(manifest.candidate_update_indices)
    while not runner.training_complete:
        runner.run_next_update()
        snapshot = runner.checkpoint()
        update_index = snapshot.ppo_checkpoint.update_index
        if update_index not in schedule:
            continue
        identifier = (
            f"{manifest.experiment_id}-fold{fold_index}-seed{seed}-update{update_index}"
        )
        runner_authorities.append(
            materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
                root=artifact_root,
                artifact_id=f"{identifier}-runner",
                runner=runner,
                committed_at_ms=committed_at_ms + update_index * 2,
            )
        )
        policy_authorities.append(
            materialize_massive_adaptive_rl_checkpoint_authority_v1(
                root=artifact_root,
                artifact_id=f"{identifier}-policy",
                checkpoint=snapshot.ppo_checkpoint,
                training_forecast_authority=training_authority,
                committed_at_ms=committed_at_ms + update_index * 2 + 1,
            )
        )
    training_run = runner.run_to_completion()
    observed_updates = tuple(
        authority.runtime_checkpoint.ppo_checkpoint.update_index
        for authority in runner_authorities
        if authority.runtime_checkpoint is not None
    )
    if observed_updates != manifest.candidate_update_indices:
        raise MassiveAdaptiveRLWorkflowV1Error(
            "registered candidate update was not reached exactly once"
        )
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    fixed_control_fit_authority = (
        materialize_massive_adaptive_rl_fixed_control_fit_authority_v1(
            root=artifact_root,
            artifact_id=(
                f"{manifest.experiment_id}-fold{fold_index}-seed{seed}-fixed-fit"
            ),
            training_authority=training_authority,
            chronology_authority=chronology_authority,
            environments=environments,
            fit_environment_authorities=fit_environment_authorities,
            committed_at_ms=committed_at_ms + observed_updates[-1] * 2 + 2,
            registry=registry,
        )
    )
    fixed_control_selection_authority = (
        materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1(
            root=artifact_root,
            artifact_id=(
                f"{manifest.experiment_id}-fold{fold_index}-seed{seed}-fixed-selection"
            ),
            fit_authority=fixed_control_fit_authority,
            committed_at_ms=committed_at_ms + observed_updates[-1] * 2 + 3,
        )
    )
    source_qualified = bool(
        training_run.development_rl_training_authorized
        and all(
            authority.development_rl_training_authorized
            for authority in runner_authorities
        )
        and all(
            authority.development_rl_training_authorized
            for authority in policy_authorities
        )
        and fixed_control_fit_authority.development_control_fit_authorized
        and fixed_control_selection_authority.development_control_selection_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V1_SCHEMA,
        "experiment_manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "fold_index": fold_index,
        "seed": seed,
        "training_forecast_authority_receipt_sha256": (
            training_authority.semantic_receipt_sha256
        ),
        "chronology_authority_receipt_sha256": (
            chronology_authority.semantic_receipt_sha256
        ),
        "fixed_control_fit_authority": fixed_control_fit_authority,
        "fixed_control_selection_authority": fixed_control_selection_authority,
        "training_run": training_run,
        "runner_checkpoint_authorities": tuple(runner_authorities),
        "policy_checkpoint_authorities": tuple(policy_authorities),
        "candidate_update_indices": observed_updates,
        "runner_checkpoint_inventory_sha256": semantic_sha256(
            tuple(authority.semantic_receipt_sha256 for authority in runner_authorities)
        ),
        "policy_checkpoint_inventory_sha256": semantic_sha256(
            tuple(authority.semantic_receipt_sha256 for authority in policy_authorities)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLTrainingWorkflowV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_rl_training_authorized=source_qualified,
    )
    result = MassiveAdaptiveRLTrainingWorkflowV1(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
            "development_rl_training_authorized": source_qualified,
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def run_massive_adaptive_rl_validation_workflow_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV1,
    training_workflow: MassiveAdaptiveRLTrainingWorkflowV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[
        str,
        tuple[
            MassiveAdaptiveProfitabilityEnvV1,
            MassiveAdaptiveProfitabilityEnvV1,
            MassiveAdaptiveProfitabilityEnvV1,
        ],
    ],
    fixed_control_environment: MassiveAdaptiveProfitabilityEnvV1,
    artifact_root: str | Path,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationWorkflowV1:
    """Rerun every candidate actor and publish its frozen-target cost ladder."""

    manifest.validate()
    training_workflow.validate()
    chronology_authority.validate()
    if (
        training_workflow.experiment_manifest_receipt_sha256
        != manifest.semantic_receipt_sha256
        or chronology_authority.semantic_receipt_sha256
        != training_workflow.chronology_authority_receipt_sha256
        or not chronology_authority.development_policy_selection_authorized
        or set(environments)
        != {
            authority.semantic_receipt_sha256
            for authority in training_workflow.policy_checkpoint_authorities
        }
    ):
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL validation registry differs from training"
        )
    traces: list[MassiveAdaptiveRLPolicyTraceAuthorityV1] = []
    ladders: list[MassiveAdaptiveRLCostLadderAuthorityV1] = []
    shared_validation_context: str | None = None
    for index, checkpoint_authority in enumerate(
        training_workflow.policy_checkpoint_authorities
    ):
        low, primary, high = environments[checkpoint_authority.semantic_receipt_sha256]
        context_receipts = {
            row.validation_context_receipt_sha256 for row in (low, primary, high)
        }
        if len(context_receipts) != 1:
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL cost rungs do not share one validation context"
            )
        candidate_context = next(iter(context_receipts))
        if shared_validation_context is None:
            shared_validation_context = candidate_context
        elif candidate_context != shared_validation_context:
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL candidates do not share one validation context"
            )
        if (
            tuple(row.transaction_cost_basis_points for row in (low, primary, high))
            != manifest.cost_ladder_basis_points
            or primary.initial_capital != manifest.primary_capital
            or any(
                row.maximum_fill_participation != manifest.maximum_fill_participation
                for row in (low, primary, high)
            )
        ):
            raise MassiveAdaptiveRLWorkflowV1Error(
                "adaptive RL validation economics differ from the manifest"
            )
        artifact = (
            f"{manifest.experiment_id}-fold{training_workflow.fold_index}"
            f"-seed{training_workflow.seed}-candidate{index}"
        )
        trace = materialize_massive_adaptive_rl_policy_trace_authority_v1(
            root=artifact_root,
            artifact_id=f"{artifact}-primary-trace",
            checkpoint_authority=checkpoint_authority,
            chronology_authority=chronology_authority,
            environment=primary,
            fold_index=training_workflow.fold_index,
            evaluation_role="inner_validation",
            committed_at_ms=committed_at_ms + index * 2,
        )
        ladder = materialize_massive_adaptive_rl_cost_ladder_authority_v1(
            root=artifact_root,
            artifact_id=f"{artifact}-cost-ladder",
            checkpoint_authority=checkpoint_authority,
            chronology_authority=chronology_authority,
            primary_environment=primary,
            low_cost_environment=low,
            high_cost_environment=high,
            fold_index=training_workflow.fold_index,
            evaluation_role="inner_validation",
            committed_at_ms=committed_at_ms + index * 2 + 1,
        )
        if trace.policy_trace_receipt_sha256 != ladder.primary_trace_receipt_sha256:
            raise MassiveAdaptiveRLWorkflowV1Error(
                "checkpoint-owned primary validation replay is not exact"
            )
        traces.append(trace)
        ladders.append(ladder)
    if shared_validation_context is None:
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL validation context is absent"
        )
    fixed_control_evaluation = evaluate_massive_adaptive_rl_fixed_control_v1(
        registry=build_massive_adaptive_rl_fixed_control_registry_v1(),
        fit_authority=training_workflow.fixed_control_fit_authority,
        selection_authority=training_workflow.fixed_control_selection_authority,
        chronology_authority=chronology_authority,
        environment=fixed_control_environment,
    )
    if (
        fixed_control_evaluation.validation_context_receipt_sha256
        != shared_validation_context
    ):
        raise MassiveAdaptiveRLWorkflowV1Error(
            "adaptive RL fixed control does not share the validation context"
        )
    source_qualified = bool(
        training_workflow.development_rl_training_authorized
        and all(row.development_policy_evaluation_authorized for row in traces)
        and all(row.development_policy_selection_authorized for row in ladders)
        and fixed_control_evaluation.development_policy_selection_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_WORKFLOW_V1_SCHEMA,
        "experiment_manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_workflow_receipt_sha256": training_workflow.semantic_receipt_sha256,
        "fold_index": training_workflow.fold_index,
        "checkpoint_authority_receipts": tuple(
            row.semantic_receipt_sha256
            for row in training_workflow.policy_checkpoint_authorities
        ),
        "validation_context_receipt_sha256": shared_validation_context,
        "fixed_control_evaluation": fixed_control_evaluation,
        "policy_trace_authorities": tuple(traces),
        "cost_ladder_authorities": tuple(ladders),
        "policy_trace_authority_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in traces)
        ),
        "cost_ladder_authority_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ladders)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_WORKFLOW_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLValidationWorkflowV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_policy_evaluation_authorized=source_qualified,
    )
    result = MassiveAdaptiveRLValidationWorkflowV1(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
            "development_policy_evaluation_authorized": source_qualified,
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _manifest_command(args: argparse.Namespace) -> int:
    config = MassiveAdaptivePPOConfigV1(seed=args.seed)
    manifest = build_massive_adaptive_rl_experiment_manifest_v1(
        experiment_id=args.experiment_id,
        prequential_block_sessions=args.block_sessions,
        candidate_update_indices=tuple(args.candidate_update),
        seeds=(args.seed,),
        ppo_config=config,
    )
    write_massive_adaptive_rl_experiment_manifest_v1(
        path=args.output,
        manifest=manifest,
    )
    print(manifest.semantic_receipt_sha256)
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    manifest = load_massive_adaptive_rl_experiment_manifest_v1(args.manifest)
    print(manifest.semantic_receipt_sha256)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quanttrade-adaptive-rl",
        description="Massive adaptive RL immutable experiment workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser(
        "manifest",
        help="Create one immutable, nonauthorizing experiment manifest.",
    )
    manifest.add_argument("--experiment-id", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--block-sessions", type=int, choices=(21, 63), default=21)
    manifest.add_argument("--candidate-update", type=int, action="append")
    manifest.add_argument("--seed", type=int, default=17)
    manifest.set_defaults(handler=_manifest_command)
    validate = commands.add_parser(
        "validate",
        help="Validate an immutable manifest without opening data or outcomes.",
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(handler=_validate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest" and args.candidate_update is None:
        args.candidate_update = [1]
    return int(args.handler(args))


__all__ = [
    "MassiveAdaptiveRLExperimentManifestV1",
    "MassiveAdaptiveRLTrainingWorkflowV1",
    "MassiveAdaptiveRLValidationWorkflowV1",
    "MassiveAdaptiveRLWorkflowV1Error",
    "build_massive_adaptive_rl_experiment_manifest_v1",
    "load_massive_adaptive_rl_experiment_manifest_v1",
    "main",
    "run_massive_adaptive_rl_training_workflow_v1",
    "run_massive_adaptive_rl_validation_workflow_v1",
    "write_massive_adaptive_rl_experiment_manifest_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
