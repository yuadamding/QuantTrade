"""Manifest-V3-owned execution of one adaptive-RL fit fold.

This is the first authorizing adapter from reconstructed runtime sources into
the reusable PPO and fixed-control components.  Callers cannot supply
environment mappings, chronology dates, actions, transitions, checkpoint
winners, or fixed-control results.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256,
    massive_adaptive_ppo_initial_model_state_receipt_v1,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_registry_v1 import (
    MassiveAdaptiveRLFitEnvironmentRegistryV1,
)
from rl_quant.training.massive_adaptive_rl_fold_fit_chronology_authority_v1 import (
    MassiveAdaptiveRLFoldFitChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    capture_massive_adaptive_rl_execution_environment_v1,
    execution_environment_relative_path_v1,
    load_massive_adaptive_rl_execution_environment_authority_v1,
    massive_adaptive_rl_deterministic_execution_v1,
    materialize_massive_adaptive_rl_execution_environment_authority_v1,
    verify_massive_adaptive_rl_execution_environment_replay_v1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_inputs_v1 import (
    MassiveAdaptiveRLFoldFitInputsAuthorityV1,
    fold_fit_inputs_relative_path_v1,
    load_massive_adaptive_rl_fold_fit_inputs_authority_v1,
    materialize_massive_adaptive_rl_fold_fit_inputs_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)
from rl_quant.workflows.massive_adaptive_rl_process_state_v1 import (
    preserve_massive_adaptive_rl_process_rng_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLTrainingWorkflowV2,
    run_massive_adaptive_rl_training_workflow_v2,
    verify_massive_adaptive_rl_training_workflow_v2,
)


MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fold-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA,
        "encoding": "canonical-json-receipt-envelope",
        "runtime_replay": "nested-checkpoint-and-fixed-control-reauthorization",
    }
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "inputs": "manifest-v3-and-witnessed-runtime-sources-v1-only",
        "forecast": "package-built-training-forecast-authority-v2",
        "chronology": "package-built-fit-only-date-role-commitment",
        "environments": "one-manifest-v3-runtime-registry",
        "ppo": "complete-session-derived-candidate-schedule",
        "fixed_controls": "complete-registered-fit-grid-and-fc06-selection",
        "candidate_provenance": "exact-traversed-environment-prefix",
        "transition_coverage": "exact-fit-origin-dates",
        "model_initialization": "scoped-canonical-seed-and-initial-state-receipt",
        "execution_environment": "persisted-clean-deterministic-runtime-authority-v1",
        "completed_verification": (
            "strictly-read-only-existing-evidence-graph-and-rng-sandbox"
        ),
        "repair": "explicit-and-pre-aggregate-only",
        "caller_environments": False,
        "caller_actions": False,
        "caller_transitions": False,
        "caller_results": False,
        "profitability_reporting": False,
        "outer_access": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFoldFitV1Error(ValueError):
    """One fold fit did not replay from its registered source graph."""


class MassiveAdaptiveRLFoldFitExecutionLeaseUnavailable(
    MassiveAdaptiveRLFoldFitV1Error
):
    """Another process owns this fold's expensive execution stage."""


class MassiveAdaptiveRLFoldFitCompletedEvidenceError(
    MassiveAdaptiveRLFoldFitV1Error
):
    """A completed fold cannot enter the mutable repair path."""


def _execution_environment_artifact_id(*, experiment_id: str, fold_index: int) -> str:
    fold_fit_authority_relative_path_v1(
        experiment_id=experiment_id,
        fold_index=fold_index,
    )
    return f"{experiment_id}-fold{fold_index}"


def _source_transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    receipt = payload.with_name(payload.name + ".receipt.json")
    commit = payload.with_name(payload.name + ".commit.json")
    paths = (payload, receipt, commit)
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit source transaction is incomplete"
        )
    return all(present)


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldFitV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldFitAuthorityV1:
    experiment_id: str
    outer_fold_index: int
    manifest_v3_receipt_sha256: str
    base_manifest_v2_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    execution_device_specification: str
    training_seed: int
    model_initialization_specification_sha256: str
    initial_model_state_receipt_sha256: str
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1
    fit_inputs_authority: MassiveAdaptiveRLFoldFitInputsAuthorityV1
    training_forecast_authority: MassiveAdaptiveRLTrainingForecastAuthorityV2
    fit_chronology_authority: MassiveAdaptiveRLFoldFitChronologyAuthorityV1
    fit_environment_registry: MassiveAdaptiveRLFitEnvironmentRegistryV1
    training_workflow: MassiveAdaptiveRLTrainingWorkflowV2
    fit_environment_registry_receipt_sha256: str
    fit_environment_mapping_receipt_sha256: str
    ppo_training_workflow_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    candidate_checkpoint_authority_receipts: tuple[str, ...]
    candidate_checkpoint_inventory_sha256: str
    candidate_traversed_environment_receipts: tuple[tuple[str, ...], ...]
    candidate_traversed_environment_inventory_sha256: str
    final_training_checkpoint_authority_receipt_sha256: str
    transition_decision_session_dates: tuple[str, ...]
    transition_inventory_sha256: str
    fit_origin_inventory_sha256: str
    source_data_qualified: bool
    runtime_fit_replayed: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA
    _manifest: MassiveAdaptiveRLExperimentManifestV3 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "outer_fold_index": self.outer_fold_index,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "base_manifest_v2_receipt_sha256": (
                self.base_manifest_v2_receipt_sha256
            ),
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "execution_device_specification": self.execution_device_specification,
            "training_seed": self.training_seed,
            "model_initialization_specification_sha256": (
                self.model_initialization_specification_sha256
            ),
            "initial_model_state_receipt_sha256": (
                self.initial_model_state_receipt_sha256
            ),
            "execution_environment_authority_receipt_sha256": (
                self.execution_environment_authority.semantic_receipt_sha256
            ),
            "fit_inputs_authority_receipt_sha256": (
                self.fit_inputs_authority.semantic_receipt_sha256
            ),
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority.semantic_receipt_sha256
            ),
            "fit_chronology_authority_receipt_sha256": (
                self.fit_chronology_authority.semantic_receipt_sha256
            ),
            "fit_environment_registry_receipt_sha256": (
                self.fit_environment_registry_receipt_sha256
            ),
            "fit_environment_mapping_receipt_sha256": (
                self.fit_environment_mapping_receipt_sha256
            ),
            "ppo_training_workflow_receipt_sha256": (
                self.ppo_training_workflow_receipt_sha256
            ),
            "fixed_control_fit_authority_receipt_sha256": (
                self.fixed_control_fit_authority_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "candidate_checkpoint_authority_receipts": (
                self.candidate_checkpoint_authority_receipts
            ),
            "candidate_checkpoint_inventory_sha256": (
                self.candidate_checkpoint_inventory_sha256
            ),
            "candidate_traversed_environment_receipts": (
                self.candidate_traversed_environment_receipts
            ),
            "candidate_traversed_environment_inventory_sha256": (
                self.candidate_traversed_environment_inventory_sha256
            ),
            "final_training_checkpoint_authority_receipt_sha256": (
                self.final_training_checkpoint_authority_receipt_sha256
            ),
            "transition_decision_session_dates": (
                self.transition_decision_session_dates
            ),
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "fit_origin_inventory_sha256": self.fit_origin_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "runtime_fit_replayed": self.runtime_fit_replayed,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def source_transaction_verified(self) -> bool:
        """Whether this completed fold was reopened from persisted bytes."""

        return self._loaded_source is not None

    @property
    def development_stage_authorized(self) -> bool:
        """Whether this persisted fold fit may authorize its aggregate stage."""

        return bool(
            self.development_rl_training_authorized
            and self.source_data_qualified
            and self.runtime_fit_replayed
            and self.execution_environment_authority.development_execution_authorized
            and self.fit_inputs_authority.development_stage_authorized
            and self.source_transaction_verified
        )

    def validate(self) -> None:
        if type(self.execution_environment_authority) is not (
            MassiveAdaptiveRLExecutionEnvironmentAuthorityV1
        ) or type(self.fit_inputs_authority) is not (
            MassiveAdaptiveRLFoldFitInputsAuthorityV1
        ):
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold fit requires exact persisted input witnesses"
            )
        self.training_forecast_authority.validate()
        self.fit_chronology_authority.validate()
        self.fit_environment_registry.validate()
        self.training_workflow.validate()
        self.execution_environment_authority.validate()
        self.fit_inputs_authority.validate()
        runtime_present = self._manifest is not None and self._runtime_sources is not None
        partial_runtime = (self._manifest is None) != (self._runtime_sources is None)
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            self._manifest.validate()
            self._runtime_sources.validate()
        if self._loaded_source is not None:
            self._loaded_source.validate()
        workflow = self.training_workflow.runtime_workflow
        workflow.validate()
        schedule = self.training_workflow.candidate_schedule
        expected_environment_receipts = tuple(
            row.semantic_receipt_sha256
            for row in self.fit_environment_registry.environment_authorities
        )
        runner_authorities = workflow.runner_checkpoint_authorities
        policy_authorities = workflow.policy_checkpoint_authorities
        runner_checkpoints = tuple(
            row.runtime_checkpoint for row in runner_authorities
        )
        policy_checkpoints = tuple(
            row.runtime_checkpoint for row in policy_authorities
        )
        checkpoints_present = bool(
            runner_checkpoints
            and policy_checkpoints
            and all(row is not None for row in runner_checkpoints)
            and all(row is not None for row in policy_checkpoints)
        )
        traversed: list[tuple[str, ...]] = []
        prefix_dates_valid = checkpoints_present
        if checkpoints_present:
            for update_index, runner_checkpoint, policy_checkpoint in zip(
                schedule.candidate_update_indices,
                runner_checkpoints,
                policy_checkpoints,
                strict=True,
            ):
                assert runner_checkpoint is not None
                assert policy_checkpoint is not None
                expected_receipts = expected_environment_receipts[:update_index]
                expected_dates = self.training_forecast_authority.origin_session_dates[
                    : update_index
                    * self.training_forecast_authority.block_sessions
                ]
                traversed.append(expected_receipts)
                prefix_dates_valid = bool(
                    prefix_dates_valid
                    and runner_checkpoint.fit_environment_authority_receipts
                    == expected_receipts
                    and policy_checkpoint.fit_environment_authority_receipts
                    == expected_receipts
                    and runner_checkpoint.transition_decision_session_dates
                    == expected_dates
                    and policy_checkpoint.transition_decision_session_dates
                    == expected_dates
                )
        traversed_rows = tuple(traversed)
        fixed_fit = workflow.fixed_control_fit_authority
        fixed_run = fixed_fit.runtime_fit_run
        fixed_dates_valid = bool(
            fixed_run is not None
            and fixed_fit.runtime_fit_replayed
            and all(
                row.decision_session_dates
                == self.training_forecast_authority.origin_session_dates
                for row in fixed_run.traces
            )
        )
        final_training_run = workflow.training_run
        expected_qualified = bool(
            runtime_present
            and checkpoints_present
            and prefix_dates_valid
            and fixed_dates_valid
            and self.training_forecast_authority.source_data_qualified
            and self.fit_chronology_authority.source_data_qualified
            and self.fit_environment_registry.source_data_qualified
            and self.execution_environment_authority.development_execution_authorized
            and self.fit_inputs_authority.source_data_qualified
            and self.training_workflow.development_rl_training_authorized
            and final_training_run.source_data_qualified
            and final_training_run.transition_decision_session_dates
            == self.training_forecast_authority.origin_session_dates
            and final_training_run.fit_environment_authority_receipts
            == expected_environment_receipts
        )
        runtime_receipt = (
            None
            if self._runtime_sources is None
            else self._runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA
            or partial_runtime
            or not runtime_present
            or self._manifest is None
            or self._runtime_sources is None
            or not self.experiment_id
            or self.outer_fold_index not in range(4)
            or self.experiment_id != self._manifest.experiment_id
            or self.experiment_id != self._runtime_sources.experiment_id
            or self.manifest_v3_receipt_sha256
            != self._manifest.semantic_receipt_sha256
            or self.manifest_v3_receipt_sha256
            != self._runtime_sources.manifest_v3_receipt_sha256
            or self.base_manifest_v2_receipt_sha256
            != self._manifest.base_manifest.semantic_receipt_sha256
            or self.runtime_sources_receipt_sha256
            != self._runtime_sources.semantic_receipt_sha256
            or self.runtime_graph_witness_receipt_sha256 != runtime_receipt
            or self.execution_device_specification
            != self._manifest.execution_device_specification
            or isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed != self._manifest.base_manifest.seeds[0]
            or self.training_seed != workflow.seed
            or self.model_initialization_specification_sha256
            != MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256
            or self.model_initialization_specification_sha256
            != workflow.model_initialization_specification_sha256
            or self.initial_model_state_receipt_sha256
            != workflow.initial_model_state_receipt_sha256
            or self.execution_environment_authority.experiment_id
            != self.experiment_id
            or self.execution_environment_authority.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self.execution_environment_authority.execution_device_specification
            != self.execution_device_specification
            or self.execution_environment_authority.training_seed
            != self.training_seed
            or self.execution_environment_authority.initial_model_state_receipt_sha256
            != self.initial_model_state_receipt_sha256
            or not self.execution_environment_authority.source_transaction_verified
            or not self.execution_environment_authority.runtime_environment_replayed
            or self.fit_inputs_authority.experiment_id != self.experiment_id
            or self.fit_inputs_authority.outer_fold_index != self.outer_fold_index
            or self.fit_inputs_authority.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self.fit_inputs_authority.runtime_sources_receipt_sha256
            != self.runtime_sources_receipt_sha256
            or self.fit_inputs_authority.runtime_graph_witness_receipt_sha256
            != self.runtime_graph_witness_receipt_sha256
            or self.fit_inputs_authority.execution_environment_authority.semantic_receipt_sha256
            != self.execution_environment_authority.semantic_receipt_sha256
            or self.fit_inputs_authority.training_forecast_authority.semantic_receipt_sha256
            != self.training_forecast_authority.semantic_receipt_sha256
            or self.fit_inputs_authority.fit_chronology_authority.semantic_receipt_sha256
            != self.fit_chronology_authority.semantic_receipt_sha256
            or self.fit_inputs_authority.fit_environment_registry.semantic_receipt_sha256
            != self.fit_environment_registry.semantic_receipt_sha256
            or not self.fit_inputs_authority.runtime_inputs_replayed
            or self.training_forecast_authority.outer_fold_index
            != self.outer_fold_index
            or self.fit_chronology_authority.fold_index != self.outer_fold_index
            or self.fit_environment_registry.outer_fold_index
            != self.outer_fold_index
            or self.fit_environment_registry.experiment_id != self.experiment_id
            or self.fit_environment_registry.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self.fit_environment_registry.runtime_sources_receipt_sha256
            != self.runtime_sources_receipt_sha256
            or self.fit_environment_registry.runtime_graph_witness_receipt_sha256
            != self.runtime_graph_witness_receipt_sha256
            or self.fit_environment_registry.forecast_archive_receipts
            != self.training_forecast_authority.source_forecast_archive_receipts
            or len(expected_environment_receipts)
            != len(self.training_forecast_authority.blocks)
            or self.training_workflow.experiment_manifest_receipt_sha256
            != self.base_manifest_v2_receipt_sha256
            or workflow.fold_index != self.outer_fold_index
            or workflow.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority.semantic_receipt_sha256
            or workflow.chronology_authority_receipt_sha256
            != self.fit_chronology_authority.semantic_receipt_sha256
            or self.fit_environment_registry_receipt_sha256
            != self.fit_environment_registry.semantic_receipt_sha256
            or self.fit_environment_mapping_receipt_sha256
            != self.fit_environment_registry.environment_registry_receipt_sha256
            or self.ppo_training_workflow_receipt_sha256
            != self.training_workflow.semantic_receipt_sha256
            or self.fixed_control_fit_authority_receipt_sha256
            != workflow.fixed_control_fit_authority.semantic_receipt_sha256
            or self.fixed_control_selection_authority_receipt_sha256
            != workflow.fixed_control_selection_authority.semantic_receipt_sha256
            or self.candidate_checkpoint_authority_receipts
            != tuple(row.semantic_receipt_sha256 for row in policy_authorities)
            or self.candidate_checkpoint_inventory_sha256
            != semantic_sha256(self.candidate_checkpoint_authority_receipts)
            or self.candidate_traversed_environment_receipts != traversed_rows
            or self.candidate_traversed_environment_inventory_sha256
            != semantic_sha256(traversed_rows)
            or not policy_authorities
            or self.final_training_checkpoint_authority_receipt_sha256
            != policy_authorities[-1].semantic_receipt_sha256
            or self.transition_decision_session_dates
            != final_training_run.transition_decision_session_dates
            or self.transition_decision_session_dates
            != self.training_forecast_authority.origin_session_dates
            or self.transition_inventory_sha256
            != final_training_run.transition_inventory_sha256
            or self.fit_origin_inventory_sha256
            != self.training_forecast_authority.rl_fit_prefix_inventory_sha256
            or self.source_data_qualified != expected_qualified
            or self.runtime_fit_replayed != expected_qualified
            or self.development_rl_training_authorized != expected_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold-fit authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold-fit source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.base_manifest_v2_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            self.model_initialization_specification_sha256,
            self.initial_model_state_receipt_sha256,
            self.execution_environment_authority.semantic_receipt_sha256,
            self.fit_inputs_authority.semantic_receipt_sha256,
            self.training_forecast_authority.semantic_receipt_sha256,
            self.fit_chronology_authority.semantic_receipt_sha256,
            self.fit_environment_registry_receipt_sha256,
            self.fit_environment_mapping_receipt_sha256,
            self.ppo_training_workflow_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            *self.candidate_checkpoint_authority_receipts,
            self.candidate_checkpoint_inventory_sha256,
            self.candidate_traversed_environment_inventory_sha256,
            self.final_training_checkpoint_authority_receipt_sha256,
            self.transition_inventory_sha256,
            self.fit_origin_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fold-fit authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _assemble_massive_adaptive_rl_fold_fit_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    selected_device: torch.device,
    execution_environment: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    fit_inputs: MassiveAdaptiveRLFoldFitInputsAuthorityV1,
    workflow: MassiveAdaptiveRLTrainingWorkflowV2,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    runtime_workflow = workflow.runtime_workflow
    policy_authorities = runtime_workflow.policy_checkpoint_authorities
    traversed = tuple(
        tuple(
            checkpoint.runtime_checkpoint.fit_environment_authority_receipts
            if checkpoint.runtime_checkpoint is not None
            else ()
        )
        for checkpoint in policy_authorities
    )
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if runtime_receipt is None or not policy_authorities:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit runtime evidence is incomplete"
        )
    training = fit_inputs.training_forecast_authority
    fit_registry = fit_inputs.fit_environment_registry
    chronology = fit_inputs.fit_chronology_authority
    training_run = runtime_workflow.training_run
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "outer_fold_index": outer_fold_index,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_v2_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "execution_device_specification": str(selected_device),
        "training_seed": runtime_workflow.seed,
        "model_initialization_specification_sha256": (
            runtime_workflow.model_initialization_specification_sha256
        ),
        "initial_model_state_receipt_sha256": (
            runtime_workflow.initial_model_state_receipt_sha256
        ),
        "execution_environment_authority": execution_environment,
        "fit_inputs_authority": fit_inputs,
        "training_forecast_authority": training,
        "fit_chronology_authority": chronology,
        "fit_environment_registry": fit_registry,
        "training_workflow": workflow,
        "fit_environment_registry_receipt_sha256": (
            fit_registry.semantic_receipt_sha256
        ),
        "fit_environment_mapping_receipt_sha256": (
            fit_registry.environment_registry_receipt_sha256
        ),
        "ppo_training_workflow_receipt_sha256": workflow.semantic_receipt_sha256,
        "fixed_control_fit_authority_receipt_sha256": (
            runtime_workflow.fixed_control_fit_authority.semantic_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            runtime_workflow.fixed_control_selection_authority.semantic_receipt_sha256
        ),
        "candidate_checkpoint_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in policy_authorities
        ),
        "candidate_checkpoint_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in policy_authorities)
        ),
        "candidate_traversed_environment_receipts": traversed,
        "candidate_traversed_environment_inventory_sha256": semantic_sha256(
            traversed
        ),
        "final_training_checkpoint_authority_receipt_sha256": (
            policy_authorities[-1].semantic_receipt_sha256
        ),
        "transition_decision_session_dates": (
            training_run.transition_decision_session_dates
        ),
        "transition_inventory_sha256": training_run.transition_inventory_sha256,
        "fit_origin_inventory_sha256": training.rl_fit_prefix_inventory_sha256,
        "source_data_qualified": True,
        "runtime_fit_replayed": True,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFoldFitAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_rl_training_authorized=True,
        _manifest=manifest,
        _runtime_sources=runtime_sources,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def prepare_massive_adaptive_rl_fold_fit_inputs_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    resume: bool = False,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    """Persist or replay one fold's complete inputs before PPO execution."""

    manifest.validate()
    runtime_sources.validate()
    selected_device = torch.device(
        manifest.execution_device_specification if device is None else device
    )
    if (
        outer_fold_index not in manifest.base_manifest.fold_indices
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or str(selected_device) != manifest.execution_device_specification
    ):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit manifest, sources, fold, or device differ"
        )
    resolved_artifact_root = Path(artifact_root)
    try:
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit artifact root is unavailable"
        ) from error
    initial_model_state_receipt = (
        massive_adaptive_ppo_initial_model_state_receipt_v1(
            seed=manifest.base_manifest.seeds[0]
        )
    )
    with massive_adaptive_rl_deterministic_execution_v1(device=selected_device):
        execution_environment_artifact_id = _execution_environment_artifact_id(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        )
        execution_environment_exists = _source_transaction_exists(
            root=resolved_artifact_root,
            relative=execution_environment_relative_path_v1(
                artifact_id=execution_environment_artifact_id
            ),
        )
        if resume and execution_environment_exists:
            execution_environment = (
                verify_massive_adaptive_rl_execution_environment_replay_v1(
                    authority=(
                        load_massive_adaptive_rl_execution_environment_authority_v1(
                            root=resolved_artifact_root,
                            artifact_id=execution_environment_artifact_id,
                            verified_at_ms=committed_at_ms,
                        )
                    ),
                    manifest=manifest,
                    initial_model_state_receipt_sha256=(
                        initial_model_state_receipt
                    ),
                    device=selected_device,
                )
            )
        elif allow_materialize:
            captured_execution_environment = (
                capture_massive_adaptive_rl_execution_environment_v1(
                    manifest=manifest,
                    initial_model_state_receipt_sha256=(
                        initial_model_state_receipt
                    ),
                    device=selected_device,
                )
            )
            if not captured_execution_environment.source_data_qualified:
                raise MassiveAdaptiveRLFoldFitV1Error(
                    "adaptive RL fold fit requires a clean qualified execution source"
                )
            persisted_execution_environment = (
                materialize_massive_adaptive_rl_execution_environment_authority_v1(
                    root=resolved_artifact_root,
                    artifact_id=execution_environment_artifact_id,
                    authority=captured_execution_environment,
                    committed_at_ms=committed_at_ms,
                )
            )
            execution_environment = (
                verify_massive_adaptive_rl_execution_environment_replay_v1(
                    authority=persisted_execution_environment,
                    manifest=manifest,
                    initial_model_state_receipt_sha256=(
                        initial_model_state_receipt
                    ),
                    device=selected_device,
                )
            )
        else:
            raise MassiveAdaptiveRLFoldFitCompletedEvidenceError(
                "completed adaptive RL fold-fit execution environment is absent"
            )
        fit_inputs_relative = fold_fit_inputs_relative_path_v1(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        )
        fit_inputs_exist = _source_transaction_exists(
            root=resolved_artifact_root,
            relative=fit_inputs_relative,
        )
        if resume and fit_inputs_exist:
            fit_inputs = load_massive_adaptive_rl_fold_fit_inputs_authority_v1(
                root=resolved_artifact_root,
                manifest=manifest,
                runtime_sources=runtime_sources,
                execution_environment_authority=execution_environment,
                outer_fold_index=outer_fold_index,
                verified_at_ms=committed_at_ms,
            )
        elif allow_materialize:
            fit_inputs = materialize_massive_adaptive_rl_fold_fit_inputs_authority_v1(
                root=resolved_artifact_root,
                manifest=manifest,
                runtime_sources=runtime_sources,
                execution_environment_authority=execution_environment,
                outer_fold_index=outer_fold_index,
                committed_at_ms=committed_at_ms,
            )
        else:
            raise MassiveAdaptiveRLFoldFitCompletedEvidenceError(
                "completed adaptive RL fold-fit inputs are absent"
            )
    return fit_inputs


def run_massive_adaptive_rl_fold_fit_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    resume: bool = False,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Execute one complete PPO and fixed-control fit from reconstructed sources."""

    fit_inputs = prepare_massive_adaptive_rl_fold_fit_inputs_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        artifact_root=artifact_root,
        committed_at_ms=committed_at_ms,
        device=device,
        resume=resume,
    )
    selected_device = torch.device(
        manifest.execution_device_specification if device is None else device
    )
    resolved_artifact_root = Path(artifact_root)
    execution_environment = fit_inputs.execution_environment_authority
    with massive_adaptive_rl_deterministic_execution_v1(device=selected_device):
        training = fit_inputs.training_forecast_authority
        fit_registry = fit_inputs.fit_environment_registry
        chronology = fit_inputs.fit_chronology_authority
        environments = fit_registry.build_environments()
        environment_authorities = {
            receipt: fit_registry.authority(receipt)
            for receipt in fit_registry.forecast_archive_receipts
        }
        workflow = run_massive_adaptive_rl_training_workflow_v2(
            manifest=manifest.base_manifest,
            fold_index=outer_fold_index,
            seed=manifest.base_manifest.seeds[0],
            training_authority=training,
            chronology_authority=chronology,
            environments=environments,
            fit_environment_authorities=environment_authorities,
            artifact_root=resolved_artifact_root,
            committed_at_ms=committed_at_ms,
            device=selected_device,
            resume=resume,
        )
    return _assemble_massive_adaptive_rl_fold_fit_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        selected_device=selected_device,
        execution_environment=execution_environment,
        fit_inputs=fit_inputs,
        workflow=workflow,
    )


def fold_fit_authority_relative_path_v1(
    *, experiment_id: str, fold_index: int
) -> str:
    if (
        not experiment_id
        or any(
            not (character.isalnum() or character in "-_")
            for character in experiment_id
        )
        or fold_index not in range(4)
    ):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit artifact identity differs"
        )
    return (
        "massive-adaptive/rl-fold-fit-authority-v1/"
        f"{experiment_id}-fold{fold_index}.json"
    )


def _fold_fit_payload(
    authority: MassiveAdaptiveRLFoldFitAuthorityV1,
) -> dict[str, object]:
    authority.validate()
    return {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }


def _load_fold_fit_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit authority is not canonical JSON"
        )
    return dict(cast(Mapping[str, object], value))


def materialize_massive_adaptive_rl_fold_fit_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFoldFitAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Publish a completed fold aggregate after every nested artifact replays."""

    authority.validate()
    resolved_root = Path(root)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit artifact root is unavailable"
        ) from error
    relative = fold_fit_authority_relative_path_v1(
        experiment_id=authority.experiment_id,
        fold_index=authority.outer_fold_index,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_fold_fit_payload(authority))),
        root=resolved_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-FOLD-FIT-V1-"
            f"{authority.experiment_id}-fold{authority.outer_fold_index}"
        ),
    )
    loaded = load_massive_source_bundle(
        root=resolved_root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    if canonical_json_file_bytes(
        _load_fold_fit_payload(root=resolved_root, loaded_source=loaded)
    ) != canonical_json_file_bytes(_fold_fit_payload(authority)):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "published adaptive RL fold-fit authority differs"
        )
    result = replace(authority, _loaded_source=loaded)
    result.validate()
    return result


def _verify_massive_adaptive_rl_fold_fit_authority_v1_unpreserved(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    verified_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Reconstruct a completed fold strictly from existing immutable evidence."""

    manifest.validate()
    runtime_sources.validate()
    selected_device = torch.device(
        manifest.execution_device_specification if device is None else device
    )
    if (
        outer_fold_index not in manifest.base_manifest.fold_indices
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or str(selected_device) != manifest.execution_device_specification
    ):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit verification roots differ"
        )
    initial_model_state_receipt = (
        massive_adaptive_ppo_initial_model_state_receipt_v1(
            seed=manifest.base_manifest.seeds[0]
        )
    )
    with massive_adaptive_rl_deterministic_execution_v1(device=selected_device):
        execution_environment_artifact_id = _execution_environment_artifact_id(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        )
        execution_environment = (
            verify_massive_adaptive_rl_execution_environment_replay_v1(
                authority=(
                    load_massive_adaptive_rl_execution_environment_authority_v1(
                        root=root,
                        artifact_id=execution_environment_artifact_id,
                        verified_at_ms=verified_at_ms,
                    )
                ),
                manifest=manifest,
                initial_model_state_receipt_sha256=initial_model_state_receipt,
                device=selected_device,
            )
        )
        fit_inputs = load_massive_adaptive_rl_fold_fit_inputs_authority_v1(
            root=root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            execution_environment_authority=execution_environment,
            outer_fold_index=outer_fold_index,
            verified_at_ms=verified_at_ms,
        )
        training = fit_inputs.training_forecast_authority
        fit_registry = fit_inputs.fit_environment_registry
        chronology = fit_inputs.fit_chronology_authority
        environments = fit_registry.build_environments()
        environment_authorities = {
            receipt: fit_registry.authority(receipt)
            for receipt in fit_registry.forecast_archive_receipts
        }
        workflow = verify_massive_adaptive_rl_training_workflow_v2(
            manifest=manifest.base_manifest,
            fold_index=outer_fold_index,
            seed=manifest.base_manifest.seeds[0],
            training_authority=training,
            chronology_authority=chronology,
            environments=environments,
            fit_environment_authorities=environment_authorities,
            artifact_root=root,
            verified_at_ms=verified_at_ms,
            device=selected_device,
        )
    replayed = _assemble_massive_adaptive_rl_fold_fit_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        selected_device=selected_device,
        execution_environment=execution_environment,
        fit_inputs=fit_inputs,
        workflow=workflow,
    )
    committed = _load_fold_fit_payload(root=root, loaded_source=loaded_source)
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _fold_fit_payload(replayed)
    ):
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit authority did not replay"
        )
    result = replace(replayed, _loaded_source=loaded_source)
    result.validate()
    return result


def verify_massive_adaptive_rl_fold_fit_authority_v1(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    verified_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Strictly replay a fold while preserving process-global RNG state."""

    selected_device = torch.device(
        manifest.execution_device_specification if device is None else device
    )
    with preserve_massive_adaptive_rl_process_rng_state_v1(
        include_cuda=selected_device.type == "cuda"
    ):
        return _verify_massive_adaptive_rl_fold_fit_authority_v1_unpreserved(
            root=root,
            loaded_source=loaded_source,
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=outer_fold_index,
            verified_at_ms=verified_at_ms,
            device=device,
        )


def authorize_massive_adaptive_rl_fold_fit_authority_v1(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    committed_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Compatibility alias for strict, read-only completed-fold verification."""

    return verify_massive_adaptive_rl_fold_fit_authority_v1(
        root=root,
        loaded_source=loaded_source,
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        verified_at_ms=committed_at_ms,
        device=device,
    )


def load_massive_adaptive_rl_fold_fit_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    committed_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=fold_fit_authority_relative_path_v1(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        ),
        verified_at_ms=committed_at_ms,
    )
    return verify_massive_adaptive_rl_fold_fit_authority_v1(
        root=root,
        loaded_source=loaded,
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        verified_at_ms=committed_at_ms,
        device=device,
    )


@contextmanager
def _fold_fit_execution_lease(
    *, root: str | Path, experiment_id: str, fold_index: int
) -> Iterator[None]:
    fold_fit_authority_relative_path_v1(
        experiment_id=experiment_id,
        fold_index=fold_index,
    )
    resolved_root = Path(root)
    lease_directory = (
        resolved_root / "massive-adaptive" / "rl-fold-fit-leases-v1"
    )
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
        if resolved_root.is_symlink():
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold-fit artifact root is a symlink"
            )
        lease_directory.mkdir(parents=True, exist_ok=True)
        if lease_directory.is_symlink():
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold-fit lease directory is a symlink"
            )
        lease_path = lease_directory / f"{experiment_id}-fold{fold_index}.lock"
        descriptor = os.open(
            lease_path,
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
    except OSError as error:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit execution lease is unavailable"
        ) from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFoldFitV1Error(
                "adaptive RL fold-fit execution lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLFoldFitExecutionLeaseUnavailable(
                "adaptive RL fold-fit execution lease is already held"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _repair_massive_adaptive_rl_fold_fit_generation_v1_unlocked(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    relative = fold_fit_authority_relative_path_v1(
        experiment_id=manifest.experiment_id,
        fold_index=outer_fold_index,
    )
    if _source_transaction_exists(root=artifact_root, relative=relative):
        raise MassiveAdaptiveRLFoldFitCompletedEvidenceError(
            "completed adaptive RL fold-fit evidence cannot be repaired"
        )
    result = run_massive_adaptive_rl_fold_fit_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
        artifact_root=artifact_root,
        committed_at_ms=committed_at_ms,
        device=device,
        resume=True,
    )
    final_update = result.training_workflow.candidate_schedule.candidate_update_indices[
        -1
    ]
    return materialize_massive_adaptive_rl_fold_fit_authority_v1(
        root=artifact_root,
        authority=result,
        committed_at_ms=committed_at_ms + final_update * 3 + 4,
    )


def repair_massive_adaptive_rl_fold_fit_generation_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Resume an incomplete generation; refuse to mutate a completed fold graph."""

    with _fold_fit_execution_lease(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
        fold_index=outer_fold_index,
    ):
        return _repair_massive_adaptive_rl_fold_fit_generation_v1_unlocked(
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=outer_fold_index,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms,
            device=device,
        )


def prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    """Prepare or strictly replay fold inputs under the fold execution lease."""

    with _fold_fit_execution_lease(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
        fold_index=outer_fold_index,
    ):
        return prepare_massive_adaptive_rl_fold_fit_inputs_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=outer_fold_index,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms,
            device=device,
            resume=True,
            allow_materialize=allow_materialize,
        )


def run_or_resume_massive_adaptive_rl_fold_fit_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Replay a completed fold or resume its latest exact PPO boundary."""

    with _fold_fit_execution_lease(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
        fold_index=outer_fold_index,
    ):
        relative = fold_fit_authority_relative_path_v1(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        )
        if _source_transaction_exists(root=artifact_root, relative=relative):
            return load_massive_adaptive_rl_fold_fit_authority_v1(
                root=artifact_root,
                manifest=manifest,
                runtime_sources=runtime_sources,
                outer_fold_index=outer_fold_index,
                committed_at_ms=committed_at_ms,
                device=device,
            )
        return _repair_massive_adaptive_rl_fold_fit_generation_v1_unlocked(
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=outer_fold_index,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms,
            device=device,
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldFitAuthorityV1",
    "MassiveAdaptiveRLFoldFitCompletedEvidenceError",
    "MassiveAdaptiveRLFoldFitExecutionLeaseUnavailable",
    "MassiveAdaptiveRLFoldFitV1Error",
    "authorize_massive_adaptive_rl_fold_fit_authority_v1",
    "fold_fit_authority_relative_path_v1",
    "load_massive_adaptive_rl_fold_fit_authority_v1",
    "materialize_massive_adaptive_rl_fold_fit_authority_v1",
    "prepare_massive_adaptive_rl_fold_fit_inputs_v1",
    "prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1",
    "repair_massive_adaptive_rl_fold_fit_generation_v1",
    "run_or_resume_massive_adaptive_rl_fold_fit_v1",
    "run_massive_adaptive_rl_fold_fit_v1",
    "verify_massive_adaptive_rl_fold_fit_authority_v1",
]
