from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MassiveAdaptiveRLFoldValidationAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_evidence_v2 import (
    MassiveAdaptiveRLFoldValidationAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256,
    MassiveAdaptiveRLPolicySelectionAuthorityV3,
    MassiveAdaptiveRLPolicySelectionV3Error,
    authorize_massive_adaptive_rl_policy_selection_authority_v3,
    build_massive_adaptive_rl_policy_selection_authority_v3,
    materialize_massive_adaptive_rl_policy_selection_authority_v3,
    parse_massive_adaptive_rl_policy_selection_authority_v3,
    policy_selection_authority_relative_path_v3,
    policy_selection_v2_witness_artifact_id_v3,
    policy_selection_v2_witness_relative_path_v3,
    run_or_resume_massive_adaptive_rl_policy_selection_authority_v3,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    build_massive_adaptive_rl_experiment_manifest_v4,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _generic_authority(
    *, manifest, fold_index: int = 0
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    candidate_count = fold_index + 1
    checkpoints = tuple(
        _digest(("checkpoint-authority", fold_index, ordinal))
        for ordinal in range(candidate_count)
    )
    candidates = tuple(
        _digest(("candidate", fold_index, ordinal))
        for ordinal in range(candidate_count)
    )
    ranked = tuple(reversed(candidates))
    selected = ranked[0]
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": fold_index,
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "fold_validation_v2_receipt_sha256": _digest("fold-validation-v2"),
        "fold_validation_v2_source_receipt_sha256": _digest(
            "fold-validation-v2-source"
        ),
        "fold_validation_v2_commit_receipt_sha256": _digest(
            "fold-validation-v2-commit"
        ),
        "fold_validation_v2_committed_at_ms": 12,
        "four_fold_validation_inputs_v2_receipt_sha256": _digest("barrier-v2"),
        "four_fold_validation_inputs_v2_source_receipt_sha256": _digest(
            "barrier-v2-source"
        ),
        "four_fold_validation_inputs_v2_commit_receipt_sha256": _digest(
            "barrier-v2-commit"
        ),
        "four_fold_validation_inputs_v2_committed_at_ms": 8,
        "source_bundle_v2_receipt_sha256": _digest("source-bundle-v2"),
        "source_bundle_v2_source_receipt_sha256": _digest("source-bundle-v2-source"),
        "source_bundle_v2_commit_receipt_sha256": _digest("source-bundle-v2-commit"),
        "runtime_source_graph_v2_receipt_sha256": _digest("runtime-graph-v2"),
        "runtime_source_graph_v2_witness_receipt_sha256": _digest(
            "runtime-graph-v2-witness"
        ),
        "replay_dependency_index_v2_receipt_sha256": _digest("dependency-index-v2"),
        "runtime_sources_v2_receipt_sha256": _digest("runtime-sources-v2"),
        "training_source_projection_sha256": _digest("training-projection"),
        "validation_source_projection_sha256": _digest("validation-projection"),
        "validation_sources_v2_receipt_sha256": _digest("validation-sources-v2"),
        "validation_sources_v2_source_receipt_sha256": _digest(
            "validation-sources-v2-source"
        ),
        "validation_sources_v2_commit_receipt_sha256": _digest(
            "validation-sources-v2-commit"
        ),
        "validation_registry_v2_receipt_sha256": _digest("registry-v2"),
        "validation_registry_v2_source_receipt_sha256": _digest("registry-v2-source"),
        "validation_registry_v2_commit_receipt_sha256": _digest("registry-v2-commit"),
        "primary_outcome_v2_receipts": tuple(
            _digest(("primary-v2", ordinal)) for ordinal in range(candidate_count)
        ),
        "primary_outcome_v2_source_receipts": tuple(
            _digest(("primary-v2-source", ordinal))
            for ordinal in range(candidate_count)
        ),
        "primary_outcome_v2_commit_receipts": tuple(
            _digest(("primary-v2-commit", ordinal))
            for ordinal in range(candidate_count)
        ),
        "ladder_outcome_v2_receipts": tuple(
            _digest(("ladder-v2", ordinal)) for ordinal in range(candidate_count)
        ),
        "ladder_outcome_v2_source_receipts": tuple(
            _digest(("ladder-v2-source", ordinal)) for ordinal in range(candidate_count)
        ),
        "ladder_outcome_v2_commit_receipts": tuple(
            _digest(("ladder-v2-commit", ordinal)) for ordinal in range(candidate_count)
        ),
        "fixed_control_outcome_v2_receipt_sha256": _digest("fc06-v2"),
        "fixed_control_outcome_v2_source_receipt_sha256": _digest("fc06-v2-source"),
        "fixed_control_outcome_v2_commit_receipt_sha256": _digest("fc06-v2-commit"),
        "base_fold_validation_v1_receipt_sha256": _digest("fold-validation-v1"),
        "base_policy_selection_authority_v2_receipt_sha256": _digest(
            "selection-authority-v2"
        ),
        "base_policy_selection_authority_v2_source_receipt_sha256": _digest(
            "selection-authority-v2-source"
        ),
        "base_policy_selection_authority_v2_commit_receipt_sha256": _digest(
            "selection-authority-v2-commit"
        ),
        "base_policy_selection_authority_v2_committed_at_ms": 13,
        "selection_v2_receipt_sha256": _digest("selection-v2"),
        "expected_candidate_checkpoint_authority_receipts": checkpoints,
        "candidate_receipts": candidates,
        "candidate_inventory_sha256": semantic_sha256(candidates),
        "ranked_candidate_receipts": ranked,
        "ranked_candidate_inventory_sha256": semantic_sha256(ranked),
        "candidate_checkpoint_inventory_sha256": semantic_sha256(checkpoints),
        "selected_checkpoint_authority_receipt_sha256": checkpoints[-1],
        "selected_checkpoint_receipt_sha256": _digest("selected-checkpoint"),
        "selected_model_state_receipt_sha256": _digest("selected-model-state"),
        "selected_update_index": 1_000,
        "selected_candidate_receipt_sha256": selected,
        "selected_candidate_validation_eligible": True,
        "validation_eligibility_failures": (),
        "selection_pool_kind": "eligible",
        "source_data_qualified": True,
        "positive_profitability_authorization_eligible": True,
        "validation_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLPolicySelectionAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _publish_generic(
    *, tmp_path: Path, manifest, authority: MassiveAdaptiveRLPolicySelectionAuthorityV3
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    relative = policy_selection_authority_relative_path_v3(
        manifest=manifest,
        fold_index=authority.fold_index,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=14,
        downloaded_at_ms=14,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=14,
        request_id="ADAPTIVE-RL-POLICY-SELECTION-V3-GENERIC",
    )
    return parse_massive_adaptive_rl_policy_selection_authority_v3(
        root=tmp_path,
        loaded_source=load_massive_source_bundle(
            root=tmp_path,
            relative_payload_path=relative,
            verified_at_ms=15,
        ),
    )


def test_persisted_policy_selection_v3_binds_v2_lineage_but_reload_is_nonauthorizing(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v3-generic"
    )
    authority = _generic_authority(manifest=manifest)

    generic = _publish_generic(
        tmp_path=tmp_path,
        manifest=manifest,
        authority=authority,
    )

    assert generic.source_transaction_verified
    assert generic.fold_validation_v2_receipt_sha256 == _digest("fold-validation-v2")
    assert generic.four_fold_validation_inputs_v2_receipt_sha256 == _digest(
        "barrier-v2"
    )
    assert generic.runtime_sources_v2_receipt_sha256 == _digest("runtime-sources-v2")
    assert generic.primary_outcome_v2_receipts == (_digest(("primary-v2", 0)),)
    assert generic.base_policy_selection_authority_v2_receipt_sha256 == _digest(
        "selection-authority-v2"
    )
    assert generic.positive_profitability_authorization_eligible
    assert not generic.runtime_selection_replayed
    assert not generic.development_policy_selection_authorized
    assert not generic.policy_freezing_authorized
    assert not generic.outer_diagnostic_preparation_authorized
    assert not generic.development_stage_authorized
    assert not generic.profitability_reporting_authorized
    assert not generic.outer_evaluation_authorized
    assert not generic.lockbox_access_authorized


def test_policy_selection_v3_requires_complete_candidate_and_outcome_inventories() -> (
    None
):
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v3-inventory"
    )
    authority = _generic_authority(manifest=manifest, fold_index=3)
    malformed = replace(
        authority,
        primary_outcome_v2_receipts=authority.primary_outcome_v2_receipts[:-1],
        semantic_receipt_sha256="0" * 64,
    )
    malformed = replace(
        malformed,
        semantic_receipt_sha256=semantic_sha256(malformed.semantic_unsigned()),
    )

    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV3Error,
        match="authority V3 differs",
    ):
        malformed.validate()


def test_policy_selection_v3_requires_v2_fold_and_strict_chronology() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v3-boundary"
    )
    authority = _generic_authority(manifest=manifest)
    malformed = replace(
        authority,
        base_policy_selection_authority_v2_committed_at_ms=(
            authority.fold_validation_v2_committed_at_ms
        ),
        semantic_receipt_sha256="0" * 64,
    )
    malformed = replace(
        malformed,
        semantic_receipt_sha256=semantic_sha256(malformed.semantic_unsigned()),
    )

    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV3Error,
        match="chronology differs",
    ):
        malformed.validate()
    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV3Error,
        match="exact V2 authority types",
    ):
        build_massive_adaptive_rl_policy_selection_authority_v3(
            manifest=manifest,
            validation_authority=object(),  # type: ignore[arg-type]
            base_selection_authority_v2=object(),  # type: ignore[arg-type]
        )


def test_policy_selection_v3_authorizing_api_accepts_only_fold_validation_v2() -> None:
    functions = (
        build_massive_adaptive_rl_policy_selection_authority_v3,
        authorize_massive_adaptive_rl_policy_selection_authority_v3,
        materialize_massive_adaptive_rl_policy_selection_authority_v3,
        run_or_resume_massive_adaptive_rl_policy_selection_authority_v3,
    )
    for function in functions:
        parameters = inspect.signature(function).parameters
        annotation = get_type_hints(function)["validation_authority"]
        assert annotation is MassiveAdaptiveRLFoldValidationAuthorityV2
        assert annotation is not MassiveAdaptiveRLFoldValidationAuthorityV1
        assert "artifact_id" not in parameters
        assert "candidates" not in parameters
        assert "metrics" not in parameters


def test_policy_selection_v3_paths_are_manifest_and_fold_canonical() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v3-paths"
    )
    witness_id = policy_selection_v2_witness_artifact_id_v3(
        manifest=manifest,
        fold_index=2,
    )

    assert witness_id == (f"v3-witness-v4-{manifest.semantic_receipt_sha256}-fold-2")
    assert policy_selection_v2_witness_relative_path_v3(
        manifest=manifest,
        fold_index=2,
    ).endswith(f"/{witness_id}.json")
    assert policy_selection_authority_relative_path_v3(
        manifest=manifest,
        fold_index=2,
    ).endswith(f"/v4-{manifest.semantic_receipt_sha256}-fold-2.json")
    assert len(MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256) == 64
    assert len(MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256) == 64
