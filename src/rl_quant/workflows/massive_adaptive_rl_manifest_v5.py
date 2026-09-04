"""Immutable prequential development-profitability protocol manifest.

Manifest V4 fixes candidate ranking but can be interpreted by both the legacy
all-four validation root and the causally correct prequential root.  V5 gives
the prequential interpretation its own receipt.  It freezes the release
edges, stage order, diagnostic continuation rule, and exact authority
generations that future stages must implement.  Constructing this manifest
opens no validation, outer, or lockbox outcome.  Physical code, dependency,
and numerical-runtime identities are frozen separately immediately before the
first validation outcome, so completing the preregistered writer does not
silently change the scientific protocol receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    MassiveAdaptiveRLExperimentManifestV4,
    _parse_base_manifest_v3,
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_validation_execution_environment_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v5"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-manifest-v5-registration-v1"
)
MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-execution-implementation-registration-v1"
)
MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__).with_name("massive_adaptive_rl_manifest_v5_registration.py")
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256 = file_sha256(
    Path(__file__).with_name("massive_adaptive_rl_experiment_runner_v5.py")
)
MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256 = file_sha256(
    Path(__file__).with_name("massive_adaptive_rl_experiment_runner_v4.py")
)
MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA,
        "path": "one-fixed-registration-path-per-experiment",
        "publication": "create-only-source-transaction",
        "chronology": "before-any-prequential-validation-input",
        "legacy_all_four_evidence": "rejected",
        "legacy_writer_after_registration": "rejected",
        "generic_reload": "nonauthorizing",
        "validation_outcomes": False,
        "outer_access": False,
        "profitability_reporting": False,
    }
)
MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256 = (
    semantic_sha256(
        {
            "manifest": "exact-scientific-manifest-v5",
            "manifest_registration": "exact-source-transaction-replayed",
            "chronology": "after-initial-inputs-before-first-validation-outcome",
            "source": "clean-git-commit-tree-and-complete-runtime-source-inventory",
            "dependency_lock": "exact-physical-sha256",
            "runtime": "python-pytorch-numpy-cpu-and-thread-attestation",
            "implementation_inventory": "fixed-package-owned-relative-paths",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
            "mutation_after_registration": "new-experiment-required",
            "outer_access": False,
            "profitability_reporting": False,
        }
    )
)

MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1 = (0, 1)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1 = (2, 3)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1: tuple[
    int | None, ...
] = (None, None, 0, 1)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1 = ((0, 2), (1, 3))
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1 = (
    "trained",
    "initial-validation-inputs-committed",
    "policy-0-frozen",
    "policy-1-frozen",
    "outer-0-sealed",
    "validation-2-released",
    "policy-2-frozen",
    "outer-1-sealed",
    "validation-3-released",
    "policy-3-frozen",
    "outer-2-sealed",
    "outer-3-sealed",
    "profitability-report-published",
    "full-cold-replay-verified",
)

MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-release-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-outcome-authority-v3"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-validation-authority-v3"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-authority-v4"
)
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-frozen-rl-policy-v2"
)
MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-frozen-fc06-v2"
)
MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-walk-forward-policy-schedule-v1"
)
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA = (
    "rl-quant.massive-adaptive-outer-access-commitment-v2"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-rollout-authority-v2"
)
MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-fold-seal-authority-v1"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-profitability-report-authority-v2"
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-prequential-experiment-state-v1"
)

MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "initial_release": {
            "folds": (0, 1),
            "predecessor": "completed-four-fold-fit",
        },
        "later_releases": (
            {"fold": 2, "predecessor": "authenticated-outer-fold-0-seal"},
            {"fold": 3, "predecessor": "authenticated-outer-fold-1-seal"},
        ),
        "caller_sealed_fold_indices": False,
        "source_transaction_chronology": "strict",
    }
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "checkpoint_actions": "deterministic-tanh-actor-mean",
        "primary_cost_basis_points": 20.0,
        "stress_cost_basis_points": (10.0, 40.0),
        "stress_replay": "same-primary-target-position-trace",
        "fixed_control": "fit-selected-fc06-on-same-registry",
        "initial_book": "all-cash",
        "execution_environment": "exact-registered-cpu-attestation",
        "caller_environment": False,
        "caller_actions_targets_metrics_candidates": False,
    }
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-validation-release-authority-v1-child",
        "computation": MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256,
        "candidate_membership": "release-authority-exact-inventory",
        "execution_environment": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
        ),
        "legacy_all_four_barrier": False,
    }
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-validation-release-authority-v1-child",
        "outcome_specification": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256
        ),
        "candidate_population": "exact-fold-release-inventory",
        "fold_economics": "package-derived-from-outcome-v3-inventory",
        "legacy_all_four_barrier": False,
    }
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-fold-validation-authority-v3",
        "ranking_specification": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_specification": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "tie_breaking_specification": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "legacy_selection_authority_v3_adapter": False,
        "diagnostic_fallback": "freezeable-but-positive-authorization-prohibited",
    }
)
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "selection": "exact-policy-selection-authority-v4",
        "contents": (
            "selected-checkpoint-authority",
            "actor-state",
            "critic-state",
            "model-state",
            "normalization-state",
            "observation-action-reward-specifications",
            "validation-release-lineage",
            "scientific-execution-fingerprint",
        ),
        "diagnostic_policy_freeze": True,
        "updates_after_freeze": False,
    }
)
MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256 = semantic_sha256(
    {
        "selection": "exact-policy-selection-authority-v4-fold",
        "fit": "exact-fixed-control-fit-and-selection",
        "contents": "selected-fc06-action-and-complete-source-lineage",
        "diagnostic_control_freeze": True,
        "updates_after_freeze": False,
    }
)
MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "folds": (0, 1, 2, 3),
        "freeze_order": "prequential-stage-sequence",
        "outer_zero_prerequisite": "frozen-ppo-and-fc06-folds-zero-one",
        "outer_one_prerequisite": "already-frozen-fold-one",
        "outer_two_three_prerequisite": "corresponding-frozen-fold",
        "all_four_before_outer_zero": False,
        "diagnostic_schedule": "executable-but-positive-authorization-prohibited",
    }
)
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-frozen-rl-policy-v2-and-fold-fixed-control",
        "policy_freeze_precedes_outer_input": True,
        "fold_local": True,
        "one_time_access": True,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "inputs": (
            "exact-frozen-ppo-policy-v2",
            "exact-frozen-fc06-v2",
            "exact-outer-access-commitment-v2",
        ),
        "strategies": ("ppo", "fc06", "buy-and-drift-benchmark"),
        "shared_economics": True,
        "primary_cost_basis_points": 20.0,
        "stress_cost_basis_points": (10.0, 40.0),
        "stress_replay": "frozen-primary-action-and-target-position-trace",
        "policy_updates": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "shared_economics": ("ppo", "fc06", "buy-and-drift-benchmark"),
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "cost_ladder_actions": "frozen-primary-action-and-target-position-trace",
        "terminal_liquidation": True,
        "release_capabilities": ((0, 2), (1, 3)),
    }
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "economic_witness": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256,
        "economic_witness_source": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256
        ),
        "inputs": "four-authenticated-outer-fold-seals",
        "qualified_and_diagnostic_schedules_reported": True,
        "execution_complete_is_not_profitability": True,
    }
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "stages": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "append_only": True,
        "immediate_predecessor_receipt": True,
        "stage_artifact_source_transaction": True,
        "gaps_branches_duplicates": "rejected",
        "verification": "read-only",
    }
)

MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1 = (
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA,
    MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA,
)

MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": "exact-manifest-v5",
        "registration": (
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
        ),
        "execution_implementation_registration": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
        ),
        "experiment_global_lock": MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256,
        "artifact_root_writer_lock": (
            MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256
        ),
        "direct_materialization_lock": (
            MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256
        ),
        "initial_inputs": MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
        "initial_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
        "withheld_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
        "legacy_writers_after_registration": "rejected",
        "progression": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "outcome_implementation": "must-match-execution-registration",
        "verification": "read-only",
    }
)

MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256 = semantic_sha256(
    {
        "base_manifest": "exact-validation-selection-v4",
        "prequential_plan": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256,
        "initial_inputs": MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
        "initial_validation_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
        "withheld_validation_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
        "release_prerequisites": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1,
        "release_edges": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1,
        "stage_sequence": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "authority_generations": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1,
        "no_eligible_candidate_policy": MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
        "legacy_manifest_v4_writer": False,
        "authoritative_writer": "massive-adaptive-rl-experiment-runner-v5",
        "authoritative_writer_specification": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
        ),
        "execution_implementation_registration": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
        ),
        "manifest_registration": (
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
        ),
        "experiment_global_lock": MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256,
        "artifact_root_writer_lock": (
            MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256
        ),
        "direct_materialization_lock": (
            MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256
        ),
        "validation_environment": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
        ),
        "validation_release": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256,
        "validation_economic_core": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256
        ),
        "validation_outcome": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256,
        "fold_validation": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256,
        "policy_selection": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256,
        "frozen_policy": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256,
        "frozen_fc06": MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256,
        "walk_forward_policy_schedule": (
            MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256
        ),
        "outer_access": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256,
        "outer_rollout": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256,
        "outer_fold_seal": MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256,
        "profitability_report": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256,
        "state": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256,
        "profitability_reporting": False,
        "lockbox": False,
        "live_trading": False,
    }
)


class MassiveAdaptiveRLExperimentManifestV5Error(ValueError):
    """The prequential experiment protocol was not preregistered exactly."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV5:
    base_manifest: MassiveAdaptiveRLExperimentManifestV4
    prequential_validation_plan_specification_sha256: str
    initial_validation_inputs_specification_sha256: str
    validation_execution_environment_specification_sha256: str
    experiment_global_lock_specification_sha256: str
    artifact_root_writer_lock_specification_sha256: str
    direct_materialization_lock_specification_sha256: str
    manifest_v5_registration_specification_sha256: str
    initial_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    validation_release_prerequisite_outer_fold_indices: tuple[int | None, ...]
    outer_to_validation_release_edges: tuple[tuple[int, int], ...]
    prequential_stage_sequence: tuple[str, ...]
    authority_generation_names: tuple[str, ...]
    validation_release_specification_sha256: str
    validation_economic_core_specification_sha256: str
    validation_outcome_v3_specification_sha256: str
    fold_validation_v3_specification_sha256: str
    policy_selection_v4_specification_sha256: str
    frozen_policy_v2_specification_sha256: str
    frozen_fc06_v2_specification_sha256: str
    walk_forward_policy_schedule_v1_specification_sha256: str
    outer_access_v2_specification_sha256: str
    outer_rollout_v2_specification_sha256: str
    outer_fold_seal_v1_specification_sha256: str
    profitability_report_v2_specification_sha256: str
    prequential_state_v1_specification_sha256: str
    no_eligible_candidate_policy: str
    diagnostic_only_continuation_required: bool
    legacy_manifest_v4_materialization_authorized: bool
    authoritative_writer_generation: str
    authoritative_writer_specification_sha256: str
    execution_implementation_registration_specification_sha256: str
    semantic_receipt_sha256: str
    validation_outcome_access_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA

    @property
    def experiment_id(self) -> str:
        return self.base_manifest.experiment_id

    @property
    def execution_device_specification(self) -> str:
        return self.base_manifest.execution_device_specification

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_manifest_v4_receipt_sha256": (
                self.base_manifest.semantic_receipt_sha256
            ),
            **{
                key: value
                for key, value in asdict(self).items()
                if key
                not in {
                    "schema",
                    "base_manifest",
                    "semantic_receipt_sha256",
                }
            },
        }

    def validate(self) -> None:
        if type(self.base_manifest) is not MassiveAdaptiveRLExperimentManifestV4:
            raise MassiveAdaptiveRLExperimentManifestV5Error(
                "adaptive RL Manifest V5 requires exact Manifest V4"
            )
        self.base_manifest.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA
            or self.prequential_validation_plan_specification_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256
            or self.initial_validation_inputs_specification_sha256
            != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
            or self.validation_execution_environment_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
            or self.experiment_global_lock_specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256
            or self.artifact_root_writer_lock_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256
            or self.direct_materialization_lock_specification_sha256
            != MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256
            or self.manifest_v5_registration_specification_sha256
            != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
            or self.initial_validation_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
            or self.withheld_validation_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
            or self.validation_release_prerequisite_outer_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1
            or self.outer_to_validation_release_edges
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1
            or self.prequential_stage_sequence
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
            or self.authority_generation_names
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1
            or self.validation_release_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256
            or self.validation_economic_core_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256
            or self.validation_outcome_v3_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256
            or self.fold_validation_v3_specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256
            or self.policy_selection_v4_specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256
            or self.frozen_policy_v2_specification_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256
            or self.frozen_fc06_v2_specification_sha256
            != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256
            or self.walk_forward_policy_schedule_v1_specification_sha256
            != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256
            or self.outer_access_v2_specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256
            or self.outer_rollout_v2_specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256
            or self.outer_fold_seal_v1_specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256
            or self.profitability_report_v2_specification_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256
            or self.prequential_state_v1_specification_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256
            or self.no_eligible_candidate_policy
            != MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
            or not self.diagnostic_only_continuation_required
            or self.legacy_manifest_v4_materialization_authorized
            or self.authoritative_writer_generation
            != "massive-adaptive-rl-experiment-runner-v5"
            or self.authoritative_writer_specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
            or self.execution_implementation_registration_specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
            or self.validation_outcome_access_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentManifestV5Error(
                "adaptive RL experiment Manifest V5 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_experiment_manifest_v5(
    *,
    experiment_id: str,
    prequential_block_sessions: int = 63,
    seeds: tuple[int, ...] = (17,),
    ppo_config: MassiveAdaptivePPOConfigV1 | None = None,
    execution_device_specification: str = "cpu",
) -> MassiveAdaptiveRLExperimentManifestV5:
    base = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=experiment_id,
        prequential_block_sessions=prequential_block_sessions,
        seeds=seeds,
        ppo_config=ppo_config,
        execution_device_specification=execution_device_specification,
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA,
        "base_manifest": base,
        "prequential_validation_plan_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256
        ),
        "initial_validation_inputs_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
        ),
        "validation_execution_environment_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
        ),
        "experiment_global_lock_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256
        ),
        "artifact_root_writer_lock_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256
        ),
        "direct_materialization_lock_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256
        ),
        "manifest_v5_registration_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
        ),
        "initial_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
        ),
        "withheld_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
        ),
        "validation_release_prerequisite_outer_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1
        ),
        "outer_to_validation_release_edges": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1
        ),
        "prequential_stage_sequence": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "authority_generation_names": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1
        ),
        "validation_release_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256
        ),
        "validation_economic_core_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256
        ),
        "validation_outcome_v3_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256
        ),
        "fold_validation_v3_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256
        ),
        "policy_selection_v4_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256
        ),
        "frozen_policy_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256
        ),
        "frozen_fc06_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256
        ),
        "walk_forward_policy_schedule_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256
        ),
        "outer_access_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256
        ),
        "outer_rollout_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256
        ),
        "outer_fold_seal_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256
        ),
        "profitability_report_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256
        ),
        "prequential_state_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256
        ),
        "no_eligible_candidate_policy": (
            MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        ),
        "diagnostic_only_continuation_required": True,
        "legacy_manifest_v4_materialization_authorized": False,
        "authoritative_writer_generation": "massive-adaptive-rl-experiment-runner-v5",
        "authoritative_writer_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
        ),
        "execution_implementation_registration_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256
        ),
        "validation_outcome_access_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLExperimentManifestV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLExperimentManifestV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def write_massive_adaptive_rl_experiment_manifest_v5(
    *, path: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(manifest)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL experiment Manifest V5 is create-only"
        ) from error


def _parse_base_manifest_v4(value: object) -> MassiveAdaptiveRLExperimentManifestV4:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 base Manifest V4 is malformed"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v3(payload["base_manifest"])
    for name in (
        "candidate_ranking_metric_names",
        "candidate_tie_breaking_rule_names",
        "validation_eligibility_criteria",
        "validation_gate_names",
        "final_gate_names",
    ):
        payload[name] = tuple(cast(list[str], payload[name]))
    result = MassiveAdaptiveRLExperimentManifestV4(**payload)
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_manifest_v5(
    path: str | Path,
) -> MassiveAdaptiveRLExperimentManifestV5:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL experiment Manifest V5 is not canonical JSON"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v4(payload["base_manifest"])
    for name in (
        "initial_validation_fold_indices",
        "withheld_validation_fold_indices",
        "validation_release_prerequisite_outer_fold_indices",
        "prequential_stage_sequence",
        "authority_generation_names",
    ):
        payload[name] = tuple(cast(list[object], payload[name]))
    payload["outer_to_validation_release_edges"] = tuple(
        tuple(cast(list[int], edge))
        for edge in cast(list[list[int]], payload["outer_to_validation_release_edges"])
    )
    result = MassiveAdaptiveRLExperimentManifestV5(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1",
    "MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ECONOMIC_CORE_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256",
    "MassiveAdaptiveRLExperimentManifestV5",
    "MassiveAdaptiveRLExperimentManifestV5Error",
    "build_massive_adaptive_rl_experiment_manifest_v5",
    "load_massive_adaptive_rl_experiment_manifest_v5",
    "write_massive_adaptive_rl_experiment_manifest_v5",
]
