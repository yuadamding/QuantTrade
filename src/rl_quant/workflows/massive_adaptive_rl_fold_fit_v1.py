"""Manifest-V3-owned execution of one adaptive-RL fit fold.

This is the first authorizing adapter from reconstructed runtime sources into
the reusable PPO and fixed-control components.  Callers cannot supply
environment mappings, chronology dates, actions, transitions, checkpoint
winners, or fixed-control results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_registry_v1 import (
    MassiveAdaptiveRLFitEnvironmentRegistryV1,
    build_massive_adaptive_rl_fit_environment_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fold_fit_chronology_authority_v1 import (
    MassiveAdaptiveRLFoldFitChronologyAuthorityV1,
    build_massive_adaptive_rl_fold_fit_chronology_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
    build_massive_adaptive_rl_training_forecast_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLTrainingWorkflowV2,
    run_massive_adaptive_rl_training_workflow_v2,
)


MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
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

    def validate(self) -> None:
        self.training_forecast_authority.validate()
        self.fit_chronology_authority.validate()
        self.fit_environment_registry.validate()
        self.training_workflow.validate()
        runtime_present = self._manifest is not None and self._runtime_sources is not None
        partial_runtime = (self._manifest is None) != (self._runtime_sources is None)
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            self._manifest.validate()
            self._runtime_sources.validate()
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
        for value in (
            self.manifest_v3_receipt_sha256,
            self.base_manifest_v2_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
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


def run_massive_adaptive_rl_fold_fit_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFoldFitAuthorityV1:
    """Execute one complete PPO and fixed-control fit from reconstructed sources."""

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
    fold = runtime_sources.fold(outer_fold_index)
    training = build_massive_adaptive_rl_training_forecast_authority_v2(
        outer_fold_index=outer_fold_index,
        block_sessions=manifest.base_manifest.prequential_block_sessions,
        split_plan=runtime_sources.split_plan,
        forecast_archives=fold.fit_forecast_archives,
        training_window_plans=fold.training_windows,
        checkpoint_choices=fold.checkpoint_choices,
        calibrations=fold.calibrations,
    )
    fit_registry = build_massive_adaptive_rl_fit_environment_registry_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
    )
    chronology = build_massive_adaptive_rl_fold_fit_chronology_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        training_forecast_authority=training,
    )
    environments = fit_registry.build_environments()
    environment_authorities = {
        receipt: fit_registry.authority(receipt)
        for receipt in fit_registry.forecast_archive_receipts
    }
    resolved_artifact_root = Path(artifact_root)
    try:
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit artifact root is unavailable"
        ) from error
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
    )
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
    runtime_receipt = runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    if runtime_receipt is None or not policy_authorities:
        raise MassiveAdaptiveRLFoldFitV1Error(
            "adaptive RL fold-fit runtime evidence is incomplete"
        )
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


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldFitAuthorityV1",
    "MassiveAdaptiveRLFoldFitV1Error",
    "run_massive_adaptive_rl_fold_fit_v1",
]
