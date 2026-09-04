from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import torch

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v3 import (
    MassiveAdaptiveRLFoldValidationAuthorityV3,
    run_or_resume_massive_adaptive_rl_fold_validation_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_outcome_authority_v3 import (
    MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3,
    MassiveAdaptiveRLValidationOutcomeAuthorityV3,
    MassiveAdaptiveRLValidationOutcomeAuthorityV3Error,
    run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3,
    run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256,
    MassiveAdaptiveFrozenRLPolicyV2,
    load_massive_adaptive_frozen_rl_policy_v2,
    run_or_resume_massive_adaptive_frozen_rl_policy_v2,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptiveRLCheckpointV1
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
    load_massive_adaptive_rl_frozen_fc06_v2,
    run_or_resume_massive_adaptive_rl_frozen_fc06_v2,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v4 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV4,
    run_or_resume_massive_adaptive_rl_policy_selection_authority_v4,
)
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    run_or_resume_massive_adaptive_rl_initial_validation_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    _issue_massive_adaptive_rl_manifest_v5_capability_v1,
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type, /, **values):
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _generic_outcome() -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    body = {
        "experiment_id": "v5-initial-components",
        "manifest_v5_receipt_sha256": _digest("manifest"),
        "scientific_protocol_projection_sha256": _digest("protocol"),
        "release_authority_receipt_sha256": _digest("release"),
        "release_source_receipt_sha256": _digest("release-source"),
        "release_commit_receipt_sha256": _digest("release-commit"),
        "release_committed_at_ms": 10,
        "execution_implementation_registration_receipt_sha256": _digest(
            "implementation"
        ),
        "scientific_execution_fingerprint_sha256": _digest("environment"),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "fold_index": 0,
        "outcome_kind": MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3,
        "subject_receipt_sha256": _digest("checkpoint-authority"),
        "validation_sources_v2_receipt_sha256": _digest("validation-sources"),
        "validation_registry_v2_receipt_sha256": _digest("validation-registry"),
        "validation_context_receipt_sha256": _digest("validation-context"),
        "checkpoint_authority_receipt_sha256": _digest("checkpoint-authority"),
        "checkpoint_receipt_sha256": _digest("checkpoint"),
        "model_state_receipt_sha256": _digest("model-state"),
        "update_index": 126,
        "fixed_control_fit_authority_receipt_sha256": _digest("fixed-fit"),
        "fixed_control_selection_authority_receipt_sha256": _digest("fixed-selection"),
        "selected_fc06_action_receipt_sha256": _digest("fc06-action"),
        "economic_witness_receipt_sha256": _digest("ladder"),
        "primary_trace_receipt_sha256": _digest("primary"),
        "low_cost_trace_receipt_sha256": _digest("low"),
        "high_cost_trace_receipt_sha256": _digest("high"),
        "decision_target_inventory_sha256": _digest("targets"),
        "primary_transition_inventory_sha256": _digest("primary-transitions"),
        "low_cost_transition_inventory_sha256": _digest("low-transitions"),
        "high_cost_transition_inventory_sha256": _digest("high-transitions"),
        "primary_incremental_log_wealth": 0.01,
        "primary_active_log_wealth": 0.02,
        "low_cost_terminal_liquidation_adjusted_return": 0.03,
        "primary_cost_terminal_liquidation_adjusted_return": 0.02,
        "high_cost_terminal_liquidation_adjusted_return": 0.01,
        "maximum_drawdown": 0.10,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLValidationOutcomeAuthorityV3(
        **body,
        semantic_receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _generic_selection(
    *, eligible: bool
) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
    checkpoint = _digest("checkpoint-authority")
    candidate = _digest("candidate")
    failures = () if eligible else ("primary-incremental-rl-positive",)
    body = {
        "experiment_id": "v5-initial-components",
        "manifest_v5_receipt_sha256": _digest("manifest"),
        "scientific_protocol_projection_sha256": _digest("protocol"),
        "execution_implementation_registration_receipt_sha256": _digest(
            "implementation"
        ),
        "scientific_execution_fingerprint_sha256": _digest("environment"),
        "validation_release_authority_receipt_sha256": _digest("release"),
        "fold_validation_authority_receipt_sha256": _digest("fold-validation"),
        "fold_validation_source_receipt_sha256": _digest("fold-source"),
        "fold_validation_commit_receipt_sha256": _digest("fold-commit"),
        "fold_validation_committed_at_ms": 20,
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "fold_index": 0,
        "selection_computation_receipt_sha256": _digest("selection-computation"),
        "selected_checkpoint_authority_receipt_sha256": checkpoint,
        "selected_checkpoint_receipt_sha256": _digest("checkpoint"),
        "selected_model_state_receipt_sha256": _digest("model-state"),
        "selected_update_index": 126,
        "selected_candidate_receipt_sha256": candidate,
        "selected_candidate_validation_eligible": eligible,
        "validation_eligibility_failures": failures,
        "selection_pool_kind": "eligible" if eligible else "all-no-eligible",
        "expected_candidate_checkpoint_authority_receipts": (checkpoint,),
        "candidate_receipts": (candidate,),
        "candidate_inventory_sha256": semantic_sha256((candidate,)),
        "ranked_candidate_receipts": (candidate,),
        "ranked_candidate_inventory_sha256": semantic_sha256((candidate,)),
        "training_forecast_authority_receipt_sha256": _digest("training-forecast"),
        "fixed_control_selection_authority_receipt_sha256": _digest("fixed-selection"),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLPolicySelectionAuthorityV4(
        **body,
        semantic_receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _generic_fold_validation() -> MassiveAdaptiveRLFoldValidationAuthorityV3:
    checkpoint = _digest("checkpoint-authority")
    outcome = _digest("ppo-outcome")
    candidate = _digest("candidate")
    body = {
        "experiment_id": "v5-initial-components",
        "manifest_v5_receipt_sha256": _digest("manifest"),
        "scientific_protocol_projection_sha256": _digest("protocol"),
        "release_authority_receipt_sha256": _digest("release"),
        "release_source_receipt_sha256": _digest("release-source"),
        "release_commit_receipt_sha256": _digest("release-commit"),
        "release_committed_at_ms": 10,
        "execution_implementation_registration_receipt_sha256": _digest(
            "implementation"
        ),
        "scientific_execution_fingerprint_sha256": _digest("environment"),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "fold_index": 0,
        "expected_candidate_checkpoint_authority_receipts": (checkpoint,),
        "ppo_validation_outcome_receipts": (outcome,),
        "ppo_validation_outcome_source_receipts": (_digest("ppo-source"),),
        "ppo_validation_outcome_commit_receipts": (_digest("ppo-commit"),),
        "ppo_validation_outcome_committed_at_ms": (11,),
        "fc06_validation_outcome_receipt_sha256": _digest("fc06-outcome"),
        "fc06_validation_outcome_source_receipt_sha256": _digest("fc06-source"),
        "fc06_validation_outcome_commit_receipt_sha256": _digest("fc06-commit"),
        "fc06_validation_outcome_committed_at_ms": 12,
        "fixed_control_fit_authority_receipt_sha256": _digest("fixed-fit"),
        "fixed_control_selection_authority_receipt_sha256": _digest("fixed-selection"),
        "selected_fc06_action_receipt_sha256": _digest("fc06-action"),
        "candidate_receipts": (candidate,),
        "candidate_inventory_sha256": semantic_sha256((candidate,)),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLFoldValidationAuthorityV3(
        **body,
        semantic_receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _generic_frozen_ppo() -> MassiveAdaptiveFrozenRLPolicyV2:
    normalization = semantic_sha256(
        (
            MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256,
            _digest("observation-spec"),
        )
    )
    body = {
        "experiment_id": "v5-initial-components",
        "manifest_v5_receipt_sha256": _digest("manifest"),
        "scientific_protocol_projection_sha256": _digest("protocol"),
        "execution_implementation_registration_receipt_sha256": _digest(
            "implementation"
        ),
        "scientific_execution_fingerprint_sha256": _digest("environment"),
        "validation_release_authority_receipt_sha256": _digest("release"),
        "fold_validation_authority_receipt_sha256": _digest("fold-validation"),
        "policy_selection_authority_receipt_sha256": _digest("selection"),
        "policy_selection_source_receipt_sha256": _digest("selection-source"),
        "policy_selection_commit_receipt_sha256": _digest("selection-commit"),
        "policy_selection_committed_at_ms": 30,
        "fold_index": 0,
        "selected_checkpoint_authority_receipt_sha256": _digest("checkpoint-authority"),
        "selected_checkpoint_authority_source_receipt_sha256": _digest(
            "checkpoint-source"
        ),
        "selected_checkpoint_authority_commit_receipt_sha256": _digest(
            "checkpoint-commit"
        ),
        "selected_checkpoint_authority_committed_at_ms": 5,
        "selected_checkpoint_receipt_sha256": _digest("checkpoint"),
        "selected_model_state_receipt_sha256": _digest("model-state"),
        "selected_update_index": 126,
        "selected_candidate_validation_eligible": True,
        "validation_eligibility_failures": (),
        "training_forecast_authority_receipt_sha256": _digest("training-forecast"),
        "actor_state_receipt_sha256": _digest("actor"),
        "critic_state_receipt_sha256": _digest("critic"),
        "frozen_model_state_receipt_sha256": _digest("frozen-model"),
        "actor_state_keys": ("actor.0.weight",),
        "critic_state_keys": ("critic.0.weight",),
        "actor_optimizer_state_provenance_receipt_sha256": _digest("actor-optimizer"),
        "critic_optimizer_state_provenance_receipt_sha256": _digest("critic-optimizer"),
        "normalization_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256
        ),
        "normalization_state_receipt_sha256": normalization,
        "ppo_config_receipt_sha256": _digest("ppo-config"),
        "observation_specification_sha256": _digest("observation-spec"),
        "action_specification_sha256": _digest("action-spec"),
        "reward_specification_sha256": _digest("reward-spec"),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveFrozenRLPolicyV2(
        **body,
        semantic_receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _generic_frozen_fc06() -> MassiveAdaptiveRLFrozenFC06V2:
    body = {
        "experiment_id": "v5-initial-components",
        "manifest_v5_receipt_sha256": _digest("manifest"),
        "scientific_protocol_projection_sha256": _digest("protocol"),
        "execution_implementation_registration_receipt_sha256": _digest(
            "implementation"
        ),
        "scientific_execution_fingerprint_sha256": _digest("environment"),
        "validation_release_authority_receipt_sha256": _digest("release"),
        "fold_validation_authority_receipt_sha256": _digest("fold-validation"),
        "policy_selection_authority_receipt_sha256": _digest("selection"),
        "policy_selection_source_receipt_sha256": _digest("selection-source"),
        "policy_selection_commit_receipt_sha256": _digest("selection-commit"),
        "policy_selection_committed_at_ms": 30,
        "fold_index": 0,
        "fixed_control_fit_authority_receipt_sha256": _digest("fixed-fit"),
        "fixed_control_selection_authority_receipt_sha256": _digest("fixed-selection"),
        "fixed_control_selection_source_receipt_sha256": _digest(
            "fixed-selection-source"
        ),
        "fixed_control_selection_commit_receipt_sha256": _digest(
            "fixed-selection-commit"
        ),
        "fixed_control_selection_committed_at_ms": 6,
        "selected_control_id": "FC03",
        "selected_action_receipt_sha256": _digest("action"),
        "selected_action_values": (0.0,) * 10,
        "selected_candidate_validation_eligible": False,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLFrozenFC06V2(
        **body,
        semantic_receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def test_v5_initial_component_apis_do_not_accept_economic_results() -> None:
    cases = (
        (
            run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3,
            {
                "root",
                "manifest",
                "release",
                "fold_index",
                "checkpoint_authority",
                "allow_materialize",
            },
        ),
        (
            run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3,
            {"root", "manifest", "release", "fold_index", "allow_materialize"},
        ),
        (
            run_or_resume_massive_adaptive_rl_fold_validation_v3,
            {"root", "manifest", "release", "fold_index", "allow_materialize"},
        ),
        (
            run_or_resume_massive_adaptive_rl_policy_selection_authority_v4,
            {"root", "manifest", "fold_validation", "allow_materialize"},
        ),
        (
            run_or_resume_massive_adaptive_frozen_rl_policy_v2,
            {"root", "manifest", "selection", "allow_materialize"},
        ),
        (
            run_or_resume_massive_adaptive_rl_frozen_fc06_v2,
            {"root", "manifest", "selection", "allow_materialize"},
        ),
        (
            run_or_resume_massive_adaptive_rl_initial_validation_execution_v1,
            {
                "root",
                "manifest",
                "manifest_registration",
                "execution_registration",
                "validation_release",
                "allow_materialize",
            },
        ),
    )
    prohibited = {
        "environment",
        "actions",
        "targets",
        "metrics",
        "candidates",
        "selected_checkpoint",
        "device",
        "artifact_id",
    }
    for function, expected in cases:
        parameters = set(inspect.signature(function).parameters)
        assert parameters == expected
        assert not parameters.intersection(prohibited)

    hints = get_type_hints(
        run_or_resume_massive_adaptive_rl_initial_validation_execution_v1
    )
    assert hints["manifest"] is MassiveAdaptiveRLExperimentManifestV5
    assert hints["validation_release"] is MassiveAdaptiveRLValidationReleaseAuthorityV1
    for function in (
        run_or_resume_massive_adaptive_frozen_rl_policy_v2,
        run_or_resume_massive_adaptive_rl_frozen_fc06_v2,
    ):
        assert get_type_hints(function)["selection"] is (
            MassiveAdaptiveRLPolicySelectionAuthorityV4
        )


def test_generic_v5_evidence_and_freezes_remain_nonauthorizing() -> None:
    outcome = _generic_outcome()
    fold_validation = _generic_fold_validation()
    selection_eligible = _generic_selection(eligible=True)
    selection_diagnostic = _generic_selection(eligible=False)
    frozen_ppo = _generic_frozen_ppo()
    frozen_fc06 = _generic_frozen_fc06()

    assert not outcome.development_stage_authorized
    assert not fold_validation.development_stage_authorized
    assert not selection_eligible.development_stage_authorized
    assert not selection_eligible.positive_profitability_authorization_eligible
    assert not selection_diagnostic.positive_profitability_authorization_eligible
    assert not frozen_ppo.development_stage_authorized
    assert not frozen_ppo.positive_profitability_authorization_eligible
    assert not frozen_fc06.development_stage_authorized
    assert not any(
        row.outer_access_authorized
        for row in (
            outcome,
            fold_validation,
            selection_eligible,
            selection_diagnostic,
            frozen_ppo,
            frozen_fc06,
        )
    )


def test_validation_outcome_rejects_nonfinite_economics() -> None:
    outcome = _generic_outcome()
    changed = replace(
        outcome,
        primary_incremental_log_wealth=float("nan"),
    )
    with pytest.raises(
        MassiveAdaptiveRLValidationOutcomeAuthorityV3Error,
        match="outcome V3 differs",
    ):
        changed.validate()


def test_portable_flags_cannot_promote_v5_components() -> None:
    selection = _generic_selection(eligible=True)
    promoted = replace(
        selection,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=True,
        policy_freezing_authorized=True,
    )
    with pytest.raises(ValueError, match="policy-selection authority V4 differs"):
        promoted.validate()

    frozen = _generic_frozen_ppo()
    promoted_frozen = replace(
        frozen,
        runtime_policy_replayed=True,
        development_outer_policy_authorized=True,
    )
    with pytest.raises(ValueError, match="frozen PPO policy V2 differs"):
        promoted_frozen.validate()


def test_initial_validation_execution_capability_is_fold_and_path_scoped(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="v5-initial-writer-scope"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path, manifest=manifest
    )
    capability = _issue_massive_adaptive_rl_manifest_v5_capability_v1(
        root=tmp_path,
        authority=registration,
        writer_role="initial-validation-execution",
        allowed_fold_indices=(0, 1),
    )
    allowed_paths = (
        "adaptive-rl/v5-initial-writer-scope/validation-outcome-v3/fold-0/fc06.json",
        "adaptive-rl/v5-initial-writer-scope/validation-outcome-v3/"
        f"fold-1/ppo-{_digest('checkpoint-authority')}.json",
        "adaptive-rl/v5-initial-writer-scope/fold-validation-v3/fold-0.json",
        "adaptive-rl/v5-initial-writer-scope/policy-selection-v4/fold-1.json",
        "adaptive-rl/v5-initial-writer-scope/frozen-policy-v2/fold-0.pt",
        "adaptive-rl/v5-initial-writer-scope/frozen-fc06-v2/fold-1.json",
    )
    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=tmp_path, capability=capability
    ):
        for committed_at_ms, allowed in enumerate(allowed_paths, start=1):
            publish_massive_source_object(
                stream=BytesIO(canonical_json_file_bytes({"authorized": True})),
                root=tmp_path,
                relative_payload_path=allowed,
                dataset_id="v5-writer-scope-test",
                source_object_key=allowed,
                requested_at_ms=committed_at_ms,
                downloaded_at_ms=committed_at_ms,
                schema_sha256=_digest("writer-scope-schema"),
                entitlement_receipt_sha256=_digest("writer-scope-entitlement"),
                committed_at_ms=committed_at_ms,
            )
        with pytest.raises(
            MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
            match="does not authorize",
        ):
            disallowed = (
                "adaptive-rl/v5-initial-writer-scope/validation-outcome-v3/"
                "fold-2/test.json"
            )
            publish_massive_source_object(
                stream=BytesIO(canonical_json_file_bytes({"authorized": False})),
                root=tmp_path,
                relative_payload_path=disallowed,
                dataset_id="v5-writer-scope-test",
                source_object_key=disallowed,
                requested_at_ms=2,
                downloaded_at_ms=2,
                schema_sha256=_digest("writer-scope-schema"),
                entitlement_receipt_sha256=_digest("writer-scope-entitlement"),
                committed_at_ms=2,
            )
        with pytest.raises(
            MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
            match="does not authorize",
        ):
            noncanonical = (
                "adaptive-rl/v5-initial-writer-scope/validation-outcome-v3/"
                "fold-0/caller-chosen.json"
            )
            publish_massive_source_object(
                stream=BytesIO(canonical_json_file_bytes({"authorized": False})),
                root=tmp_path,
                relative_payload_path=noncanonical,
                dataset_id="v5-writer-scope-test",
                source_object_key=noncanonical,
                requested_at_ms=3,
                downloaded_at_ms=3,
                schema_sha256=_digest("writer-scope-schema"),
                entitlement_receipt_sha256=_digest("writer-scope-entitlement"),
                committed_at_ms=3,
            )
    assert all((tmp_path / allowed).is_file() for allowed in allowed_paths)


def test_frozen_ppo_persists_inference_state_without_optimizer_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="v5-frozen-ppo-payload"
    )
    model_state = {
        name: value.detach().clone()
        for name, value in MassiveAdaptivePPOActorCriticV1(observation_dim=90)
        .state_dict()
        .items()
    }
    checkpoint = _typed_shell(
        MassiveAdaptiveRLCheckpointV1,
        model_state=model_state,
        semantic_receipt_sha256=_digest("checkpoint"),
        model_state_receipt_sha256=_digest("model-state"),
        update_index=126,
        actor_optimizer_state_receipt_sha256=_digest("actor-optimizer"),
        critic_optimizer_state_receipt_sha256=_digest("critic-optimizer"),
        ppo_config_receipt_sha256=_digest("ppo-config"),
        observation_specification_sha256=_digest("observation-spec"),
        action_specification_sha256=_digest("action-spec"),
        reward_specification_sha256=_digest("reward-spec"),
        source_data_qualified=True,
    )
    checkpoint_transaction = SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256=_digest("checkpoint-source")),
        commit=SimpleNamespace(
            receipt_sha256=_digest("checkpoint-commit"), committed_at_ms=5
        ),
    )
    checkpoint_authority = _typed_shell(
        MassiveAdaptiveRLCheckpointAuthorityV1,
        semantic_receipt_sha256=_digest("checkpoint-authority"),
        checkpoint_source_receipt_sha256=_digest("checkpoint-source"),
        source_data_qualified=True,
        loaded_source=checkpoint_transaction,
        runtime_checkpoint=checkpoint,
    )
    selection_transaction = SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256=_digest("selection-source")),
        commit=SimpleNamespace(
            receipt_sha256=_digest("selection-commit"), committed_at_ms=30
        ),
    )
    selection = _typed_shell(
        MassiveAdaptiveRLPolicySelectionAuthorityV4,
        semantic_receipt_sha256=_digest("selection"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        scientific_protocol_projection_sha256=(
            manifest.scientific_protocol_projection_sha256
        ),
        execution_implementation_registration_receipt_sha256=_digest("implementation"),
        scientific_execution_fingerprint_sha256=_digest("environment"),
        validation_release_authority_receipt_sha256=_digest("release"),
        fold_validation_authority_receipt_sha256=_digest("fold-validation"),
        fold_index=0,
        selected_checkpoint_authority_receipt_sha256=(
            checkpoint_authority.semantic_receipt_sha256
        ),
        selected_checkpoint_receipt_sha256=checkpoint.semantic_receipt_sha256,
        selected_model_state_receipt_sha256=(checkpoint.model_state_receipt_sha256),
        selected_update_index=checkpoint.update_index,
        selected_candidate_validation_eligible=True,
        validation_eligibility_failures=(),
        training_forecast_authority_receipt_sha256=_digest("training-forecast"),
        source_data_qualified=True,
        _runtime_selected_checkpoint=checkpoint_authority,
        _loaded_source=selection_transaction,
    )
    monkeypatch.setattr(MassiveAdaptiveRLCheckpointV1, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLCheckpointAuthorityV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV4, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV4,
        "development_stage_authorized",
        property(lambda _: True),
    )

    frozen = run_or_resume_massive_adaptive_frozen_rl_policy_v2(
        root=tmp_path,
        manifest=manifest,
        selection=selection,
    )
    assert frozen.development_stage_authorized
    assert frozen.positive_profitability_authorization_eligible
    assert frozen.selected_checkpoint_authority_commit_receipt_sha256 == _digest(
        "checkpoint-commit"
    )
    assert frozen.actor_state_keys
    assert frozen.critic_state_keys

    payload_path = (
        tmp_path
        / "adaptive-rl"
        / manifest.experiment_id
        / "frozen-policy-v2"
        / "fold-0.pt"
    )
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    assert set(payload) == {"metadata", "model_state"}
    assert "actor_optimizer_state" not in payload
    assert "critic_optimizer_state" not in payload

    object.__setattr__(
        selection,
        "training_forecast_authority_receipt_sha256",
        _digest("changed-training-forecast"),
    )
    with pytest.raises(ValueError, match="frozen PPO runtime state differs"):
        frozen.validate()
    object.__setattr__(
        selection,
        "training_forecast_authority_receipt_sha256",
        _digest("training-forecast"),
    )

    generic = load_massive_adaptive_frozen_rl_policy_v2(
        root=tmp_path, manifest=manifest, fold_index=0
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_policy_replayed
    assert not generic.development_stage_authorized

    replayed = run_or_resume_massive_adaptive_frozen_rl_policy_v2(
        root=tmp_path,
        manifest=manifest,
        selection=selection,
        allow_materialize=False,
    )
    assert replayed.semantic_receipt_sha256 == frozen.semantic_receipt_sha256
    assert all(
        torch.equal(replayed.runtime_model_state[name], value)
        for name, value in model_state.items()
    )


def test_frozen_fc06_persists_the_exact_fit_selected_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="v5-frozen-fc06-payload"
    )
    selected_control_id, action = registered_massive_adaptive_rl_constant_actions_v1()[
        2
    ]
    fixed_fit = _typed_shell(
        MassiveAdaptiveRLFixedControlFitAuthorityV1,
        semantic_receipt_sha256=_digest("fixed-fit"),
        source_data_qualified=True,
    )
    fixed_transaction = SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256=_digest("fixed-selection-source")),
        commit=SimpleNamespace(
            receipt_sha256=_digest("fixed-selection-commit"), committed_at_ms=6
        ),
    )
    fixed_selection = _typed_shell(
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
        semantic_receipt_sha256=_digest("fixed-selection"),
        loaded_source=fixed_transaction,
        runtime_selection=SimpleNamespace(
            selected_control_id=selected_control_id,
            selected_action_receipt_sha256=action.semantic_receipt_sha256,
        ),
        source_data_qualified=True,
    )
    workflow = SimpleNamespace(
        fixed_control_fit_authority=fixed_fit,
        fixed_control_selection_authority=fixed_selection,
    )
    fold_validation = SimpleNamespace(
        fixed_control_fit_authority_receipt_sha256=fixed_fit.semantic_receipt_sha256,
        release_authority=SimpleNamespace(
            four_fold_fit_authority=SimpleNamespace(
                fold_fit=lambda _fold: SimpleNamespace(
                    training_workflow=SimpleNamespace(runtime_workflow=workflow)
                )
            )
        ),
    )
    selection_transaction = SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256=_digest("selection-source")),
        commit=SimpleNamespace(
            receipt_sha256=_digest("selection-commit"), committed_at_ms=30
        ),
    )
    selection = _typed_shell(
        MassiveAdaptiveRLPolicySelectionAuthorityV4,
        semantic_receipt_sha256=_digest("selection"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        scientific_protocol_projection_sha256=(
            manifest.scientific_protocol_projection_sha256
        ),
        execution_implementation_registration_receipt_sha256=_digest("implementation"),
        scientific_execution_fingerprint_sha256=_digest("environment"),
        validation_release_authority_receipt_sha256=_digest("release"),
        fold_validation_authority_receipt_sha256=_digest("fold-validation"),
        fold_index=0,
        fixed_control_selection_authority_receipt_sha256=(
            fixed_selection.semantic_receipt_sha256
        ),
        selected_candidate_validation_eligible=False,
        source_data_qualified=True,
        _runtime_fold_validation=fold_validation,
        _loaded_source=selection_transaction,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFixedControlFitAuthorityV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV4, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV4,
        "development_stage_authorized",
        property(lambda _: True),
    )

    frozen = run_or_resume_massive_adaptive_rl_frozen_fc06_v2(
        root=tmp_path,
        manifest=manifest,
        selection=selection,
    )
    assert frozen.development_stage_authorized
    assert frozen.selected_control_id == selected_control_id
    assert frozen.selected_action_receipt_sha256 == action.semantic_receipt_sha256
    assert frozen.runtime_action == action
    assert not frozen.selected_candidate_validation_eligible

    object.__setattr__(selection, "selected_candidate_validation_eligible", True)
    with pytest.raises(ValueError, match="frozen FC06 runtime replay differs"):
        frozen.validate()
    object.__setattr__(selection, "selected_candidate_validation_eligible", False)

    generic = load_massive_adaptive_rl_frozen_fc06_v2(
        root=tmp_path, manifest=manifest, fold_index=0
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_control_replayed
    assert not generic.development_stage_authorized

    replayed = run_or_resume_massive_adaptive_rl_frozen_fc06_v2(
        root=tmp_path,
        manifest=manifest,
        selection=selection,
        allow_materialize=False,
    )
    assert replayed.semantic_receipt_sha256 == frozen.semantic_receipt_sha256
    assert replayed.runtime_action == action
