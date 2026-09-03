from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import inspect
from typing import get_type_hints

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    MassiveAdaptiveRLFourFoldValidationInputsV2Error,
    four_fold_validation_inputs_authority_relative_path_v2,
    parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v2,
    validate_massive_adaptive_rl_validation_outcome_barrier_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MassiveAdaptiveRLPolicyTraceAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_evidence_v2 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
    MassiveAdaptiveRLFoldValidationAuthorityV2,
    build_massive_adaptive_rl_fold_validation_authority_v2,
    load_massive_adaptive_rl_validation_outcome_authority_v2,
    materialize_massive_adaptive_rl_validation_outcome_authority_v2,
    parse_massive_adaptive_rl_fold_validation_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    validation_primary_trace_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
    massive_adaptive_rl_validation_downstream_evidence_exists_v2,
    parse_massive_adaptive_rl_validation_environment_registry_v2,
    parse_massive_adaptive_rl_validation_sources_authority_v2,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v2,
    validation_environment_registry_relative_path_v2,
    validation_sources_authority_relative_path_v2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _publish(
    *,
    root,
    relative: str,
    dataset: str,
    schema: str,
    body: dict[str, object],
    entitlement: str,
    committed_at_ms: int,
):
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=root,
        relative_payload_path=relative,
        dataset_id=dataset,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=schema,
        entitlement_receipt_sha256=entitlement,
        committed_at_ms=committed_at_ms,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )


def _generic_source_v2(
    tmp_path, manifest
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "fold_index": 0,
        "source_bundle_v2_receipt_sha256": _digest("bundle-v2"),
        "source_bundle_v2_source_receipt_sha256": _digest("bundle-v2-source"),
        "source_bundle_v2_commit_receipt_sha256": _digest("bundle-v2-commit"),
        "source_bundle_v2_committed_at_ms": 1,
        "runtime_source_graph_v2_receipt_sha256": _digest("graph-v2"),
        "runtime_source_graph_v2_source_receipt_sha256": _digest("graph-v2-source"),
        "runtime_source_graph_v2_commit_receipt_sha256": _digest("graph-v2-commit"),
        "runtime_source_graph_v2_committed_at_ms": 2,
        "runtime_source_graph_v2_witness_receipt_sha256": _digest("graph-v2-witness"),
        "replay_dependency_index_v2_receipt_sha256": _digest("index-v2"),
        "replay_dependency_index_v2_source_receipt_sha256": _digest("index-v2-source"),
        "replay_dependency_index_v2_commit_receipt_sha256": _digest("index-v2-commit"),
        "replay_dependency_index_v2_committed_at_ms": 3,
        "runtime_sources_v2_receipt_sha256": _digest("runtime-v2"),
        "base_runtime_sources_v1_receipt_sha256": _digest("runtime-v1"),
        "base_runtime_source_graph_v1_witness_receipt_sha256": _digest(
            "graph-v1-witness"
        ),
        "training_source_projection_sha256": _digest("training-projection"),
        "validation_source_projection_sha256": _digest("validation-projection"),
        "base_validation_sources_v1_receipt_sha256": _digest("validation-source-v1"),
        "base_validation_sources_v1_source_receipt_sha256": _digest(
            "validation-source-v1-source"
        ),
        "base_validation_sources_v1_commit_receipt_sha256": _digest(
            "validation-source-v1-commit"
        ),
        "base_validation_sources_v1_committed_at_ms": 3,
        "validation_origin_inputs_receipt_sha256": _digest("origin-inputs"),
        "validation_tensor_session_dates": ("2024-01-01", "2024-01-02"),
        "validation_decision_session_dates": ("2024-01-02",),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLValidationSourcesAuthorityV2(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    authority = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    relative = validation_sources_authority_relative_path_v2(
        manifest=manifest, fold_index=0
    )
    loaded = _publish(
        root=tmp_path,
        relative=relative,
        dataset=MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET,
        schema=MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
        body=authority.semantic_unsigned(),
        entitlement=authority.semantic_receipt_sha256,
        committed_at_ms=4,
    )
    return parse_massive_adaptive_rl_validation_sources_authority_v2(
        root=tmp_path, loaded_source=loaded
    )


def _generic_registry_v2(
    tmp_path, manifest, source: MassiveAdaptiveRLValidationSourcesAuthorityV2
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": 0,
        "runtime_sources_v2_receipt_sha256": source.runtime_sources_v2_receipt_sha256,
        "source_bundle_v2_receipt_sha256": source.source_bundle_v2_receipt_sha256,
        "runtime_source_graph_v2_receipt_sha256": (
            source.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            source.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            source.replay_dependency_index_v2_receipt_sha256
        ),
        "training_source_projection_sha256": source.training_source_projection_sha256,
        "validation_source_projection_sha256": (
            source.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipt_sha256": source.semantic_receipt_sha256,
        "validation_sources_v2_source_receipt_sha256": source.source_receipt_sha256,
        "validation_sources_v2_commit_receipt_sha256": (
            source.source_transaction_receipt_sha256
        ),
        "validation_sources_v2_committed_at_ms": 4,
        "base_validation_registry_v1_receipt_sha256": _digest("registry-v1"),
        "base_validation_registry_v1_source_receipt_sha256": _digest(
            "registry-v1-source"
        ),
        "base_validation_registry_v1_commit_receipt_sha256": _digest(
            "registry-v1-commit"
        ),
        "base_validation_registry_v1_committed_at_ms": 5,
        "cost_basis_points": (10.0, 20.0, 40.0),
        "environment_authority_receipts": tuple(
            _digest(("environment", cost)) for cost in (10.0, 20.0, 40.0)
        ),
        "validation_context_receipt_sha256": _digest("validation-context"),
        "initial_capital": 10_000_000.0,
        "maximum_fill_participation": 0.02,
        "source_data_qualified": True,
    }
    values["environment_authority_inventory_sha256"] = semantic_sha256(
        values["environment_authority_receipts"]
    )
    provisional = MassiveAdaptiveRLValidationEnvironmentRegistryV2(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    authority = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    relative = validation_environment_registry_relative_path_v2(
        manifest=manifest, fold_index=0
    )
    loaded = _publish(
        root=tmp_path,
        relative=relative,
        dataset=MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET,
        schema=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256
        ),
        body=authority.semantic_unsigned(),
        entitlement=authority.semantic_receipt_sha256,
        committed_at_ms=6,
    )
    return parse_massive_adaptive_rl_validation_environment_registry_v2(
        root=tmp_path, loaded_source=loaded
    )


def _generic_barrier_v2(
    tmp_path,
    manifest,
    source: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    candidate_inventories = tuple(
        tuple(
            _digest(("candidate", fold_index, candidate_index))
            for candidate_index in range(fold_index + 1)
        )
        for fold_index in range(4)
    )
    source_receipts = (
        source.semantic_receipt_sha256,
        *tuple(_digest(("source-v2", index)) for index in range(1, 4)),
    )
    source_object_receipts = (
        source.source_receipt_sha256,
        *tuple(_digest(("source-v2-object", index)) for index in range(1, 4)),
    )
    source_commit_receipts = (
        source.source_transaction_receipt_sha256,
        *tuple(_digest(("source-v2-commit", index)) for index in range(1, 4)),
    )
    registry_receipts = (
        registry.semantic_receipt_sha256,
        *tuple(_digest(("registry-v2", index)) for index in range(1, 4)),
    )
    registry_object_receipts = (
        registry.source_receipt_sha256,
        *tuple(_digest(("registry-v2-object", index)) for index in range(1, 4)),
    )
    registry_commit_receipts = (
        registry.source_transaction_receipt_sha256,
        *tuple(_digest(("registry-v2-commit", index)) for index in range(1, 4)),
    )
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "source_bundle_v2_receipt_sha256": source.source_bundle_v2_receipt_sha256,
        "source_bundle_v2_source_receipt_sha256": (
            source.source_bundle_v2_source_receipt_sha256
        ),
        "source_bundle_v2_commit_receipt_sha256": (
            source.source_bundle_v2_commit_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            source.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            source.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            source.replay_dependency_index_v2_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": source.runtime_sources_v2_receipt_sha256,
        "base_runtime_sources_v1_receipt_sha256": (
            source.base_runtime_sources_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_witness_receipt_sha256": (
            source.base_runtime_source_graph_v1_witness_receipt_sha256
        ),
        "training_source_projection_sha256": source.training_source_projection_sha256,
        "validation_source_projection_sha256": (
            source.validation_source_projection_sha256
        ),
        "fold_indices": (0, 1, 2, 3),
        "validation_sources_v2_receipts": source_receipts,
        "validation_sources_v2_source_receipts": source_object_receipts,
        "validation_sources_v2_commit_receipts": source_commit_receipts,
        "validation_sources_v2_committed_at_ms": (4, 4, 4, 4),
        "validation_environment_registry_v2_receipts": registry_receipts,
        "validation_registry_v2_source_receipts": registry_object_receipts,
        "validation_registry_v2_commit_receipts": registry_commit_receipts,
        "validation_registry_v2_committed_at_ms": (6, 6, 6, 6),
        "base_validation_sources_v1_receipts": tuple(
            _digest(("base-source-v1", index)) for index in range(4)
        ),
        "base_validation_registry_v1_receipts": (
            registry.base_validation_registry_v1_receipt_sha256,
            *tuple(_digest(("base-registry-v1", index)) for index in range(1, 4)),
        ),
        "validation_context_receipts": (
            registry.validation_context_receipt_sha256,
            *tuple(_digest(("context", index)) for index in range(1, 4)),
        ),
        "validation_decision_session_date_inventories": tuple(
            ((f"2024-01-0{fold_index + 2}",)) for fold_index in range(4)
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": (
            candidate_inventories
        ),
        "base_four_fold_validation_inputs_v1_receipt_sha256": _digest("barrier-v1"),
        "base_four_fold_validation_inputs_v1_source_receipt_sha256": _digest(
            "barrier-v1-source"
        ),
        "base_four_fold_validation_inputs_v1_commit_receipt_sha256": _digest(
            "barrier-v1-commit"
        ),
        "base_four_fold_validation_inputs_v1_committed_at_ms": 7,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    authority = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    relative = four_fold_validation_inputs_authority_relative_path_v2(manifest=manifest)
    loaded = _publish(
        root=tmp_path,
        relative=relative,
        dataset=MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET,
        schema=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        body=authority.semantic_unsigned(),
        entitlement=authority.semantic_receipt_sha256,
        committed_at_ms=8,
    )
    return parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
        root=tmp_path, loaded_source=loaded
    )


def _synthetic_primary_v1(
    *, tmp_path, manifest, source, registry, barrier
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    checkpoint = barrier.expected_candidate_checkpoint_authority_receipt_inventories[0][
        0
    ]
    policy_trace_receipt = _digest("policy-trace")
    relative = validation_primary_trace_relative_path_v1(
        manifest=manifest,
        fold_index=0,
        checkpoint_authority_receipt_sha256=checkpoint,
    )
    loaded = _publish(
        root=tmp_path,
        relative=relative,
        dataset=MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET,
        schema=MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        body={"synthetic": "V1 transition witness"},
        entitlement=policy_trace_receipt,
        committed_at_ms=9,
    )
    return MassiveAdaptiveRLPolicyTraceAuthorityV1(
        fold_index=0,
        evaluation_role="inner_validation",
        checkpoint_authority_receipt_sha256=checkpoint,
        checkpoint_receipt_sha256=_digest("checkpoint"),
        model_state_receipt_sha256=_digest("model-state"),
        validation_context_receipt_sha256=(registry.validation_context_receipt_sha256),
        validation_sources_authority_receipt_sha256=(
            source.base_validation_sources_v1_receipt_sha256
        ),
        validation_environment_registry_receipt_sha256=(
            registry.base_validation_registry_v1_receipt_sha256
        ),
        validation_environment_registry_source_receipt_sha256=(
            registry.base_validation_registry_v1_source_receipt_sha256
        ),
        validation_environment_registry_commit_receipt_sha256=(
            registry.base_validation_registry_v1_commit_receipt_sha256
        ),
        validation_environment_registry_committed_at_ms=(
            registry.base_validation_registry_v1_committed_at_ms
        ),
        four_fold_validation_inputs_authority_receipt_sha256=(
            barrier.base_four_fold_validation_inputs_v1_receipt_sha256
        ),
        four_fold_validation_inputs_source_receipt_sha256=(
            barrier.base_four_fold_validation_inputs_v1_source_receipt_sha256
        ),
        four_fold_validation_inputs_commit_receipt_sha256=(
            barrier.base_four_fold_validation_inputs_v1_commit_receipt_sha256
        ),
        four_fold_validation_inputs_committed_at_ms=(
            barrier.base_four_fold_validation_inputs_v1_committed_at_ms
        ),
        validation_environment_authority_receipt_sha256=_digest("environment-20"),
        environment_source_inventory_sha256=_digest("environment-inventory"),
        economic_compatibility_receipt_sha256=_digest("economic-compatibility"),
        policy_trace_receipt_sha256=policy_trace_receipt,
        action_evidence_inventory_sha256=_digest("actions"),
        transition_inventory_sha256=_digest("transitions"),
        source_data_qualified=True,
        semantic_receipt_sha256=_digest("primary-authority-v1"),
        loaded_source=loaded,
        runtime_trace=None,
        runtime_trace_replayed=False,
        runtime_validation_environment_registry=None,
        runtime_validation_environment_registry_replayed=False,
        runtime_four_fold_validation_inputs_authority=None,
        runtime_four_fold_validation_inputs_replayed=False,
        runtime_validation_environment_authority=None,
        runtime_validation_environment_replayed=False,
        development_policy_evaluation_authorized=True,
        outer_evaluation_authorized=False,
    )


def test_v2_validation_inputs_are_persisted_nonauthorizing_generations(
    tmp_path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-generation-v2-generic"
    )
    source = _generic_source_v2(tmp_path, manifest)
    registry = _generic_registry_v2(tmp_path, manifest, source)
    barrier = _generic_barrier_v2(tmp_path, manifest, source, registry)

    assert source.source_transaction_verified
    assert registry.source_transaction_verified
    assert barrier.source_transaction_verified
    assert not source.development_stage_authorized
    assert not registry.development_stage_authorized
    assert not barrier.development_stage_authorized
    assert source.training_source_projection_sha256 != (
        source.validation_source_projection_sha256
    )
    assert barrier.runtime_sources_v2_receipt_sha256 == (
        source.runtime_sources_v2_receipt_sha256
    )
    assert barrier.source_transaction_committed_at_ms == 8


def test_v2_outcome_envelope_requires_barrier_membership_and_is_nonauthorizing_on_reload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-outcome-generation-v2"
    )
    source = _generic_source_v2(tmp_path, manifest)
    registry = _generic_registry_v2(tmp_path, manifest, source)
    barrier = _generic_barrier_v2(tmp_path, manifest, source, registry)
    base = _synthetic_primary_v1(
        tmp_path=tmp_path,
        manifest=manifest,
        source=source,
        registry=registry,
        barrier=barrier,
    )

    monkeypatch.setattr(
        MassiveAdaptiveRLPolicyTraceAuthorityV1,
        "validate",
        lambda _self: None,
    )
    for authority_type in (
        MassiveAdaptiveRLValidationSourcesAuthorityV2,
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    ):
        monkeypatch.setattr(authority_type, "validate", lambda _self: None)
        monkeypatch.setattr(
            authority_type,
            "development_stage_authorized",
            property(lambda _self: True),
        )

    committed = materialize_massive_adaptive_rl_validation_outcome_authority_v2(
        root=tmp_path,
        manifest=manifest,
        base_outcome=base,
        validation_sources_v2=source,
        validation_registry_v2=registry,
        four_fold_validation_inputs_v2=barrier,
        committed_at_ms=10,
    )
    assert committed.development_stage_authorized
    assert committed.base_outcome_receipt_sha256 == base.semantic_receipt_sha256
    assert committed.validation_registry_v2_receipt_sha256 == (
        registry.semantic_receipt_sha256
    )
    assert committed.four_fold_validation_inputs_v2_receipt_sha256 == (
        barrier.semantic_receipt_sha256
    )
    assert committed.runtime_sources_v2_receipt_sha256 == (
        barrier.runtime_sources_v2_receipt_sha256
    )
    assert massive_adaptive_rl_validation_downstream_evidence_exists_v2(
        root=tmp_path,
        manifest=manifest,
        fold_index=0,
    )

    generic = load_massive_adaptive_rl_validation_outcome_authority_v2(
        root=tmp_path,
        manifest=manifest,
        fold_index=0,
        outcome_kind="ppo-primary",
        subject_receipt_sha256=base.checkpoint_authority_receipt_sha256,
        verified_at_ms=11,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_evidence_replayed
    assert not generic.development_stage_authorized

    with pytest.raises(
        MassiveAdaptiveRLFourFoldValidationInputsV2Error,
        match="preregistered population",
    ):
        validate_massive_adaptive_rl_validation_outcome_barrier_v2(
            authority=barrier,
            validation_environment_registry=registry,
            fold_index=0,
            outcome_committed_at_ms=9,
            checkpoint_authority_receipt_sha256=_digest("not-a-candidate"),
        )
    with pytest.raises(
        MassiveAdaptiveRLFourFoldValidationInputsV2Error,
        match="exact V2",
    ):
        validate_massive_adaptive_rl_validation_outcome_barrier_v2(
            authority=barrier,
            validation_environment_registry=object(),  # type: ignore[arg-type]
            fold_index=0,
            outcome_committed_at_ms=9,
        )


def test_v2_fold_schema_binds_v2_inputs_and_outcome_envelopes(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-validation-generation-v2"
    )
    source = _generic_source_v2(tmp_path, manifest)
    registry = _generic_registry_v2(tmp_path, manifest, source)
    barrier = _generic_barrier_v2(tmp_path, manifest, source, registry)
    candidate = barrier.expected_candidate_checkpoint_authority_receipt_inventories[0][
        0
    ]
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": 0,
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "runtime_sources_v2_receipt_sha256": barrier.runtime_sources_v2_receipt_sha256,
        "training_source_projection_sha256": (
            barrier.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            barrier.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipt_sha256": source.semantic_receipt_sha256,
        "validation_sources_v2_source_receipt_sha256": source.source_receipt_sha256,
        "validation_sources_v2_commit_receipt_sha256": (
            source.source_transaction_receipt_sha256
        ),
        "validation_registry_v2_receipt_sha256": registry.semantic_receipt_sha256,
        "validation_registry_v2_source_receipt_sha256": registry.source_receipt_sha256,
        "validation_registry_v2_commit_receipt_sha256": (
            registry.source_transaction_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_receipt_sha256": (
            barrier.semantic_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_source_receipt_sha256": (
            barrier.source_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_commit_receipt_sha256": (
            barrier.source_transaction_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_committed_at_ms": 8,
        "expected_checkpoint_authority_receipts": (candidate,),
        "primary_outcome_v2_receipts": (_digest("primary-v2"),),
        "primary_outcome_v2_source_receipts": (_digest("primary-v2-source"),),
        "primary_outcome_v2_commit_receipts": (_digest("primary-v2-commit"),),
        "ladder_outcome_v2_receipts": (_digest("ladder-v2"),),
        "ladder_outcome_v2_source_receipts": (_digest("ladder-v2-source"),),
        "ladder_outcome_v2_commit_receipts": (_digest("ladder-v2-commit"),),
        "fixed_control_outcome_v2_receipt_sha256": _digest("fc06-v2"),
        "fixed_control_outcome_v2_source_receipt_sha256": _digest("fc06-v2-source"),
        "fixed_control_outcome_v2_commit_receipt_sha256": _digest("fc06-v2-commit"),
        "base_fold_validation_v1_receipt_sha256": _digest("fold-v1"),
        "base_fold_validation_v1_source_receipt_sha256": _digest("fold-v1-source"),
        "base_fold_validation_v1_commit_receipt_sha256": _digest("fold-v1-commit"),
        "base_fold_validation_v1_committed_at_ms": 11,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLFoldValidationAuthorityV2(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    authority = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    relative = (
        "massive-adaptive/rl-fold-validation-authority-v2/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-0.json"
    )
    loaded = _publish(
        root=tmp_path,
        relative=relative,
        dataset=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET,
        schema=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
        body=authority.semantic_unsigned(),
        entitlement=authority.semantic_receipt_sha256,
        committed_at_ms=12,
    )
    generic = parse_massive_adaptive_rl_fold_validation_authority_v2(
        root=tmp_path, loaded_source=loaded
    )

    assert generic.source_transaction_verified
    assert not generic.runtime_validation_replayed
    assert not generic.development_stage_authorized
    assert generic.validation_sources_v2_receipt_sha256 == (
        source.semantic_receipt_sha256
    )
    assert generic.validation_registry_v2_receipt_sha256 == (
        registry.semantic_receipt_sha256
    )
    assert generic.four_fold_validation_inputs_v2_receipt_sha256 == (
        barrier.semantic_receipt_sha256
    )


def test_v2_authorizing_surfaces_do_not_accept_v1_runtime_sources() -> None:
    source_annotation = get_type_hints(
        prepare_or_resume_massive_adaptive_rl_validation_sources_v2
    )["runtime_sources_v2"]
    fold_parameters = inspect.signature(
        build_massive_adaptive_rl_fold_validation_authority_v2
    ).parameters

    assert source_annotation is MassiveAdaptiveRLRuntimeSourcesV2
    assert source_annotation is not MassiveAdaptiveRLRuntimeSourcesV1
    assert "validation_sources_v2" in fold_parameters
    assert "validation_registry_v2" in fold_parameters
    assert "four_fold_validation_inputs_v2" in fold_parameters
    assert "primary_outcomes_v2" in fold_parameters
    assert "ladder_outcomes_v2" in fold_parameters
