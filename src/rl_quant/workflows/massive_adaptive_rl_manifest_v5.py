"""Immutable prequential development-profitability protocol manifest.

Manifest V4 fixes candidate ranking but can be interpreted by both the legacy
all-four validation root and the causally correct prequential root.  V5 gives
the prequential interpretation its own receipt.  It freezes the release
edges, stage order, diagnostic continuation rule, and exact authority
generations that future stages must implement.  Constructing this manifest
opens no validation, outer, or lockbox outcome.  Physical code, dependency,
and numerical-runtime identities are frozen separately before any validation
input is materialized, so completing the preregistered writer does not
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
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    massive_adaptive_rl_fixed_control_scientific_inventory_v1,
)
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
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
)
from rl_quant.workflows.massive_adaptive_rl_validation_execution_environment_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v5"
)
MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-scientific-protocol-v1"
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
MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "scope": "exact-v5-native-tree-dependency-lock-and-runtime",
        "suite": "package-owned-real-synthetic-market-vertical",
        "economic_mocks": False,
        "required_invariants": (
            "one-step-position-return-lag",
            "unchanged-position-zero-turnover-cost",
            "nonmonotone-fixed-target-cost-ladder-remains-reportable",
            "terminal-liquidation-compounding-identity",
            "ppo-fc06-benchmark-shared-economics",
            "outer-zero-seal-before-validation-two-release",
            "outer-one-seal-before-validation-three-release",
            "diagnostic-continuation-without-positive-authorization",
            "interruption-resume-receipt-equivalence",
            "predecessor-tamper-rejection",
            "nonmaterializing-cold-replay",
            "real-v5-vertical-without-economic-mocks",
        ),
        "result": (
            "exact-node-inventory-command-pass-count-nonpass-outcomes-exit-and-"
            "duration-normalized-output-receipt"
        ),
    }
)
MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256 = (
    semantic_sha256(
        {
            "manifest": "exact-scientific-manifest-v5",
            "manifest_registration": "exact-source-transaction-replayed",
            "chronology": "after-training-before-any-validation-input",
            "source": "clean-git-commit-tree-and-complete-runtime-source-inventory",
            "dependency_lock": "exact-physical-sha256",
            "runtime": "python-pytorch-numpy-cpu-and-thread-attestation",
            "implementation_inventory": "fixed-package-owned-relative-paths",
            "vertical_qualification": (
                MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256
            ),
            "vertical_qualification_execution": "once-before-publication",
            "replay": "identity-check-without-pytest-execution",
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
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1: tuple[int | None, ...] = (
    None,
    None,
    0,
    1,
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1 = ((0, 2), (1, 3))
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1 = (
    "trained",
    "execution-implementation-registered",
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
            "predecessors": (
                "completed-four-fold-fit",
                "execution-implementation-registration",
            ),
        },
        "later_releases": (
            {
                "fold": 2,
                "predecessor": "authenticated-outer-fold-0-seal-and-state-head",
            },
            {
                "fold": 3,
                "predecessor": "authenticated-outer-fold-1-seal-and-state-head",
            },
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
        "predecessor": "exact-replay-authorized-prequential-state-head",
        "outer_stage_order": (
            "policy-1-frozen",
            "policy-2-frozen",
            "policy-3-frozen",
            "outer-2-sealed",
        ),
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
        "cost_ladder_monotonicity": "observed-report-gate-not-structural-validity",
        "policy_updates": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "shared_economics": ("ppo", "fc06", "buy-and-drift-benchmark"),
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "cost_ladder_actions": "frozen-primary-action-and-target-position-trace",
        "cost_ladder_monotonicity": "observed-report-gate-not-seal-validity",
        "terminal_liquidation": True,
        "release_capabilities": ((0, 2), (1, 3)),
    }
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "economic_witness": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256,
        "inputs": "four-authenticated-outer-fold-seals",
        "qualified_and_diagnostic_schedules_reported": True,
        "nonmonotone_cost_ladder": "complete-report-with-failed-gate",
        "execution_complete_is_not_profitability": True,
    }
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "stages": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "append_only": True,
        "immediate_predecessor_receipt": True,
        "stage_artifact_source_transaction": True,
        "stage_artifact_generation": "exact-by-stage",
        "stage_artifact_runtime_authorization": "required-before-transition",
        "gaps_branches_duplicates": "rejected",
        "diagnostic_schedule_cannot_become_qualified": True,
        "failed_profitability_gates": "completed-report-state",
        "generic_reload": "integrity-only-nonauthorizing",
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
        "registration": (MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256),
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
        "initial_inputs_prerequisite": (
            "exact-replayed-execution-implementation-registration"
        ),
        "initial_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
        "withheld_folds": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
        "legacy_writers_after_registration": "rejected",
        "progression": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
        "outcome_implementation": "must-match-execution-registration",
        "verification": "package-owned-nonmaterializing-completion-proof",
    }
)

MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256 = semantic_sha256(
    {
        "scientific_protocol": MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SCHEMA,
        "base_v2_v3_v4": "compatibility-witness-not-semantic-identity",
        "physical_implementation": "separate-create-only-registration",
        "profitability_reporting": False,
        "lockbox": False,
        "live_trading": False,
    }
)

MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SCHEMA,
        "projection": "explicit-v2-through-v5-scientific-choices",
        "source_hashes": False,
        "implementation_receipts": False,
        "compatibility_manifest_receipts": False,
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


def massive_adaptive_rl_scientific_protocol_projection_v1(
    base_manifest: MassiveAdaptiveRLExperimentManifestV4,
) -> dict[str, object]:
    """Return the source-hash-free scientific choices witnessed by V2--V5.

    The nested V2--V4 manifests remain persisted compatibility witnesses, but
    their implementation and source identities must not determine the V5
    scientific receipt.  This projection commits the numerical fixed-control
    inventory directly instead of inheriting its implementation-bearing
    registry receipt.
    """

    base_v3 = base_manifest.base_manifest
    base_v2 = base_v3.base_manifest
    fixed_control_inventory = (
        massive_adaptive_rl_fixed_control_scientific_inventory_v1()
    )
    return {
        "schema": MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SCHEMA,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SPEC_SHA256
        ),
        "experiment_id": base_v2.experiment_id,
        "training": {
            "fold_indices": base_v2.fold_indices,
            "prequential_block_sessions": base_v2.prequential_block_sessions,
            "candidate_elapsed_sessions": base_v2.candidate_elapsed_sessions,
            "seeds": base_v2.seeds,
            "seed_policy": base_v2.seed_policy,
            "ppo_config": asdict(base_v2.ppo_config),
            "fold_candidate_schedule_receipts": (
                base_v2.fold_candidate_schedule_receipts
            ),
        },
        "economics": {
            "primary_capital": base_v2.primary_capital,
            "cost_ladder_basis_points": base_v2.cost_ladder_basis_points,
            "primary_cost_basis_points": base_v2.primary_cost_basis_points,
            "maximum_fill_participation": base_v2.maximum_fill_participation,
            "fixed_control_scientific_inventory": fixed_control_inventory,
            "fixed_control_scientific_inventory_sha256": semantic_sha256(
                fixed_control_inventory
            ),
            "policy_specification_sha256": base_v2.policy_specification_sha256,
            "action_specification_sha256": base_v2.action_specification_sha256,
            "reward_specification_sha256": base_v2.reward_specification_sha256,
            "benchmark_specification": base_v2.benchmark_specification,
            "initial_book_specification": base_v2.initial_book_specification,
            "maximum_fold_drawdown": base_v2.maximum_fold_drawdown,
            "outer_gate_names": base_v2.outer_gate_names,
        },
        "reporting": {
            "profitability_report_specification_sha256": (
                base_v3.profitability_report_specification_sha256
            ),
            "final_gate_names_v3": base_v3.final_gate_names,
            "bootstrap_specification": base_v3.bootstrap_specification,
            "bootstrap_replicates": base_v3.bootstrap_replicates,
            "bootstrap_block_sessions": base_v3.bootstrap_block_sessions,
            "bootstrap_seed": base_v3.bootstrap_seed,
            "annualization_sessions": base_v3.annualization_sessions,
            "risk_free_return_specification": (base_v3.risk_free_return_specification),
            "execution_device_specification": (base_v3.execution_device_specification),
        },
        "selection": {
            "validation_selection_specification_sha256": (
                base_manifest.validation_selection_specification_sha256
            ),
            "candidate_ranking_specification_sha256": (
                base_manifest.candidate_ranking_specification_sha256
            ),
            "candidate_ranking_metric_names": (
                base_manifest.candidate_ranking_metric_names
            ),
            "candidate_tie_breaking_specification_sha256": (
                base_manifest.candidate_tie_breaking_specification_sha256
            ),
            "candidate_tie_breaking_rule_names": (
                base_manifest.candidate_tie_breaking_rule_names
            ),
            "validation_eligibility_criteria": (
                base_manifest.validation_eligibility_criteria
            ),
            "no_eligible_candidate_policy": (
                base_manifest.no_eligible_candidate_policy
            ),
            "validation_gate_names": base_manifest.validation_gate_names,
            "final_gate_names_v4": base_manifest.final_gate_names,
        },
        "prequential": {
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
            "stage_sequence": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
            "authority_generations": (
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
            "diagnostic_only_continuation_required": True,
        },
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }


def _validate_base_manifest_v4_compatibility_witness(
    base_manifest: MassiveAdaptiveRLExperimentManifestV4,
) -> None:
    """Verify persisted V2--V4 receipts without consulting current source hashes."""

    if type(base_manifest) is not MassiveAdaptiveRLExperimentManifestV4:
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 requires exact Manifest V4"
        )
    base_v3 = base_manifest.base_manifest
    if type(base_v3) is not MassiveAdaptiveRLExperimentManifestV3:
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 compatibility Manifest V3 differs"
        )
    base_v2 = base_v3.base_manifest
    if type(base_v2) is not MassiveAdaptiveRLExperimentManifestV2:
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 compatibility Manifest V2 differs"
        )
    base_v2.ppo_config.validate()
    if any(
        authority.semantic_receipt_sha256
        != semantic_sha256(authority.semantic_unsigned())
        for authority in (base_v2, base_v3, base_manifest)
    ):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 compatibility receipt chain differs"
        )
    expected = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=base_v2.experiment_id,
        prequential_block_sessions=base_v2.prequential_block_sessions,
        seeds=base_v2.seeds,
        ppo_config=base_v2.ppo_config,
        execution_device_specification=base_v3.execution_device_specification,
    )
    if massive_adaptive_rl_scientific_protocol_projection_v1(
        base_manifest
    ) != massive_adaptive_rl_scientific_protocol_projection_v1(expected):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 compatibility scientific choices differ"
        )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV5:
    base_manifest: MassiveAdaptiveRLExperimentManifestV4
    scientific_protocol_projection_sha256: str
    prequential_validation_plan_specification_sha256: str
    initial_validation_inputs_specification_sha256: str
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
            "experiment_id": self.experiment_id,
            "scientific_protocol_projection_sha256": (
                self.scientific_protocol_projection_sha256
            ),
            "validation_outcome_access_authorized": False,
            "outer_access_authorized": False,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "live_trading_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        _validate_base_manifest_v4_compatibility_witness(self.base_manifest)
        expected_projection = massive_adaptive_rl_scientific_protocol_projection_v1(
            self.base_manifest
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SCHEMA
            or self.scientific_protocol_projection_sha256
            != semantic_sha256(expected_projection)
            or self.prequential_validation_plan_specification_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256
            or self.initial_validation_inputs_specification_sha256
            != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
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
            or self.validation_outcome_access_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
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
        "scientific_protocol_projection_sha256": semantic_sha256(
            massive_adaptive_rl_scientific_protocol_projection_v1(base)
        ),
        "prequential_validation_plan_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256
        ),
        "initial_validation_inputs_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
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


def _parse_base_manifest_v2_compatibility(
    value: object,
) -> MassiveAdaptiveRLExperimentManifestV2:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 base Manifest V2 is malformed"
        )
    payload = dict(value)
    for name in (
        "fold_indices",
        "candidate_elapsed_sessions",
        "seeds",
        "cost_ladder_basis_points",
        "outer_gate_names",
        "fold_candidate_schedule_receipts",
    ):
        payload[name] = tuple(cast(list[object], payload[name]))
    payload["ppo_config"] = MassiveAdaptivePPOConfigV1(
        **cast(dict[str, object], payload["ppo_config"])  # type: ignore[arg-type]
    )
    return MassiveAdaptiveRLExperimentManifestV2(**payload)  # type: ignore[arg-type]


def _parse_base_manifest_v3_compatibility(
    value: object,
) -> MassiveAdaptiveRLExperimentManifestV3:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 base Manifest V3 is malformed"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v2_compatibility(
        payload["base_manifest"]
    )
    payload["final_gate_names"] = tuple(cast(list[str], payload["final_gate_names"]))
    return MassiveAdaptiveRLExperimentManifestV3(**payload)  # type: ignore[arg-type]


def _parse_base_manifest_v4(value: object) -> MassiveAdaptiveRLExperimentManifestV4:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV5Error(
            "adaptive RL Manifest V5 base Manifest V4 is malformed"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v3_compatibility(
        payload["base_manifest"]
    )
    for name in (
        "candidate_ranking_metric_names",
        "candidate_tie_breaking_rule_names",
        "validation_eligibility_criteria",
        "validation_gate_names",
        "final_gate_names",
    ):
        payload[name] = tuple(cast(list[str], payload[name]))
    return MassiveAdaptiveRLExperimentManifestV4(**payload)  # type: ignore[arg-type]


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
    "MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SPEC_SHA256",
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
    "MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256",
    "MassiveAdaptiveRLExperimentManifestV5",
    "MassiveAdaptiveRLExperimentManifestV5Error",
    "build_massive_adaptive_rl_experiment_manifest_v5",
    "load_massive_adaptive_rl_experiment_manifest_v5",
    "massive_adaptive_rl_scientific_protocol_projection_v1",
    "write_massive_adaptive_rl_experiment_manifest_v5",
]
