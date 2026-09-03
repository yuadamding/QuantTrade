from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_source_bundle_v2 as bundle_v2_module
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v2 import (
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error,
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v2,
    load_massive_adaptive_rl_runtime_source_graph_authority_v2,
    prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLReplayDependencyIndexV1,
    MassiveAdaptiveRLRuntimeSourcesV1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLMixedRuntimeSourceGenerationError,
    authorize_massive_adaptive_rl_replay_dependency_index_v2,
    build_massive_adaptive_rl_replay_dependency_index_v2,
    build_massive_adaptive_rl_runtime_sources_v2,
    load_massive_adaptive_rl_replay_dependency_index_v2,
    prepare_or_resume_massive_adaptive_rl_replay_dependency_index_v2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    load_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v2 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA,
    MassiveAdaptiveRLLegacySourceBundleV1Error,
    authorize_massive_adaptive_rl_source_bundle_v2,
    build_massive_adaptive_rl_source_bundle_v2,
    load_massive_adaptive_rl_source_bundle_v2,
    prepare_or_resume_massive_adaptive_rl_source_bundle_v2,
)
from test_massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    _allow_synthetic_domain_types,
    _authorized_source_bundle,
)


def _authorized_v1_graph(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_id: str
):
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, experiment_id
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    generic_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
    )
    generic_graph = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=generic_bundle,
    )
    graph = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic_graph,
        source_bundle=generic_bundle,
        runtime_sources=runtimes,
    )
    return manifest, source_bundle, graph


def test_source_bundle_v2_is_create_only_replayable_and_rejects_legacy_shape(
    tmp_path: Path,
) -> None:
    manifest, _runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "validation-complete-source-bundle-v2"
    )
    committed = prepare_or_resume_massive_adaptive_rl_source_bundle_v2(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        source_bundle_v1=source_bundle,
        committed_at_ms=100_000,
    )

    assert committed.schema == MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA
    assert committed.development_stage_authorized
    assert len(committed.validation_feature_artifact_receipts) == 4
    assert len(committed.validation_action_artifact_receipts) == 4
    assert (
        committed.training_source_projection_sha256
        != committed.validation_source_projection_sha256
    )
    changed_validation_inventory = tuple(
        (
            role,
            fold_index,
            semantic_sha256("changed-validation-artifact")
            if role == "validation-origin-feature-inventory" and fold_index == 0
            else receipt,
        )
        for role, fold_index, receipt in committed.artifact_key_receipt_inventory
    )
    assert (
        bundle_v2_module._projection_receipt(
            changed_validation_inventory,
            validation=False,
        )
        == committed.training_source_projection_sha256
    )
    assert (
        bundle_v2_module._projection_receipt(
            changed_validation_inventory,
            validation=True,
        )
        != committed.validation_source_projection_sha256
    )

    generic = load_massive_adaptive_rl_source_bundle_v2(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        verified_at_ms=100_001,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_base_bundle_replayed
    assert not generic.source_data_qualified
    assert not generic.development_stage_authorized
    replayed = authorize_massive_adaptive_rl_source_bundle_v2(
        authority=generic,
        manifest=manifest.base_manifest,
        source_bundle_v1=source_bundle,
    )
    assert replayed.semantic_receipt_sha256 == committed.semantic_receipt_sha256
    assert replayed.development_stage_authorized

    legacy_shape = replace(
        source_bundle,
        artifacts=tuple(
            row
            for row in source_bundle.artifacts
            if not row.role.startswith("validation-origin-")
        ),
    )
    with pytest.raises(
        MassiveAdaptiveRLLegacySourceBundleV1Error,
        match="legacy or invalid",
    ):
        build_massive_adaptive_rl_source_bundle_v2(
            manifest=manifest.base_manifest,
            source_bundle_v1=legacy_shape,
        )


def test_runtime_source_graph_v2_binds_exact_source_bundle_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, source_bundle, graph_v1 = _authorized_v1_graph(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        experiment_id="validation-complete-runtime-graph-v2",
    )
    bundle_v2 = prepare_or_resume_massive_adaptive_rl_source_bundle_v2(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        source_bundle_v1=source_bundle,
        committed_at_ms=110_000,
    )
    graph_v2 = prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle_v2=bundle_v2,
        runtime_source_graph_v1=graph_v1,
        committed_at_ms=110_001,
    )

    assert graph_v2.development_stage_authorized
    assert graph_v2.source_bundle_v2_receipt_sha256 == (
        bundle_v2.semantic_receipt_sha256
    )
    assert graph_v2.base_runtime_source_graph_v1_receipt_sha256 == (
        graph_v1.semantic_receipt_sha256
    )
    assert len(graph_v2.validation_feature_row_receipts) == 4
    assert len(graph_v2.validation_action_row_receipts) == 4

    generic = load_massive_adaptive_rl_runtime_source_graph_authority_v2(
        source_root=tmp_path,
        manifest=manifest,
        verified_at_ms=110_002,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_graph_replayed
    assert generic.runtime_authority_receipt_sha256 is None
    replayed = authorize_massive_adaptive_rl_runtime_source_graph_authority_v2(
        authority=generic,
        manifest=manifest,
        source_bundle_v2=bundle_v2,
        runtime_source_graph_v1=graph_v1,
    )
    assert replayed.semantic_receipt_sha256 == graph_v2.semantic_receipt_sha256
    assert replayed.runtime_authority_receipt_sha256 is not None

    forged = replace(
        generic,
        source_bundle_v2_receipt_sha256=semantic_sha256("different-v2-bundle"),
        semantic_receipt_sha256="0" * 64,
    )
    forged = replace(
        forged,
        semantic_receipt_sha256=semantic_sha256(forged.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error,
        match="source transaction differs",
    ):
        forged.validate()


def test_dependency_index_and_runtime_sources_v2_reject_missing_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, source_bundle, graph_v1 = _authorized_v1_graph(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        experiment_id="validation-complete-runtime-sources-v2",
    )
    bundle_v2 = prepare_or_resume_massive_adaptive_rl_source_bundle_v2(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        source_bundle_v1=source_bundle,
        committed_at_ms=120_000,
    )
    graph_v2 = prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle_v2=bundle_v2,
        runtime_source_graph_v1=graph_v1,
        committed_at_ms=120_001,
    )

    feature_receipts = tuple(
        (semantic_sha256(("validation-feature", fold_index)),)
        for fold_index in range(4)
    )
    action_receipts = tuple(
        (semantic_sha256(("validation-action", fold_index)),) for fold_index in range(4)
    )
    folds = tuple(
        SimpleNamespace(
            outer_fold_index=fold_index,
            validation_features=(
                SimpleNamespace(
                    semantic_receipt_sha256=feature_receipts[fold_index][0]
                ),
            ),
            validation_action_origins=(
                SimpleNamespace(semantic_receipt_sha256=action_receipts[fold_index][0]),
            ),
        )
        for fold_index in range(4)
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLRuntimeSourcesV1,
        "validate",
        lambda _self: None,
    )
    runtime_v1 = MassiveAdaptiveRLRuntimeSourcesV1(
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        source_bundle_receipt_sha256=source_bundle.semantic_receipt_sha256,
        replay_dependency_index_receipt_sha256=semantic_sha256("base-index-v1"),
        runtime_source_graph_authority=graph_v1,
        session_authority=SimpleNamespace(),  # type: ignore[arg-type]
        condition_authority=SimpleNamespace(),  # type: ignore[arg-type]
        persisted_partition_manifests=(),
        identity_authority=SimpleNamespace(),  # type: ignore[arg-type]
        economic_event_archive=SimpleNamespace(),  # type: ignore[arg-type]
        daily_input_authority=SimpleNamespace(),  # type: ignore[arg-type]
        fill_source=SimpleNamespace(),  # type: ignore[arg-type]
        split_plan=SimpleNamespace(),  # type: ignore[arg-type]
        folds=folds,  # type: ignore[arg-type]
        replay_dependency_receipts=tuple(
            receipt
            for inventories in (feature_receipts, action_receipts)
            for rows in inventories
            for receipt in rows
        ),
        source_data_qualified=True,
        semantic_receipt_sha256=semantic_sha256("runtime-sources-v1"),
    )
    index_rows = tuple(
        SimpleNamespace(semantic_receipt_sha256=receipt)
        for inventories in (feature_receipts, action_receipts)
        for rows in inventories
        for receipt in rows
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLReplayDependencyIndexV1,
        "validate",
        lambda _self: None,
    )
    base_index = MassiveAdaptiveRLReplayDependencyIndexV1(
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        base_manifest_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        source_bundle_receipt_sha256=source_bundle.semantic_receipt_sha256,
        persisted_runtime_source_graph_receipt_sha256=graph_v1.semantic_receipt_sha256,
        runtime_source_graph_witness_receipt_sha256=(
            graph_v1.runtime_authority_receipt_sha256 or ""
        ),
        rows=index_rows,  # type: ignore[arg-type]
        row_inventory_sha256=semantic_sha256("base-row-inventory"),
        object_inventory_sha256=semantic_sha256("base-object-inventory"),
        dependency_edge_inventory_sha256=semantic_sha256("base-edge-inventory"),
        committed_source_data_qualified=True,
        semantic_receipt_sha256=semantic_sha256("base-index-v1"),
        specification_sha256=(
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
        ),
        implementation_source_sha256=(
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
        ),
    )

    index_v2 = prepare_or_resume_massive_adaptive_rl_replay_dependency_index_v2(
        source_root=tmp_path,
        manifest=manifest,
        runtime_source_graph_v2=graph_v2,
        replay_dependency_index_v1=base_index,
        runtime_sources_v1=runtime_v1,
        committed_at_ms=120_002,
    )
    assert index_v2.development_stage_authorized
    generic = load_massive_adaptive_rl_replay_dependency_index_v2(
        source_root=tmp_path,
        manifest=manifest,
        verified_at_ms=120_003,
    )
    assert not generic.runtime_index_replayed
    replayed = authorize_massive_adaptive_rl_replay_dependency_index_v2(
        authority=generic,
        manifest=manifest,
        runtime_source_graph_v2=graph_v2,
        replay_dependency_index_v1=base_index,
        runtime_sources_v1=runtime_v1,
    )
    runtime_v2 = build_massive_adaptive_rl_runtime_sources_v2(
        manifest=manifest,
        source_bundle_v2=bundle_v2,
        runtime_source_graph_v2=graph_v2,
        replay_dependency_index_v2=replayed,
        runtime_sources_v1=runtime_v1,
    )
    assert runtime_v2.source_data_qualified
    assert runtime_v2.base_runtime_sources_v1 is runtime_v1
    assert runtime_v2.validation_feature_receipt_inventories == feature_receipts
    assert runtime_v2.validation_action_receipt_inventories == action_receipts

    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        "validate",
        lambda _self: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        "development_stage_authorized",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        "source_transaction_verified",
        property(lambda _self: True),
    )
    fit = MassiveAdaptiveRLFourFoldFitAuthorityV1(
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        runtime_sources_receipt_sha256=runtime_v1.semantic_receipt_sha256,
        runtime_graph_witness_receipt_sha256=(
            graph_v1.runtime_authority_receipt_sha256 or ""
        ),
        four_fold_fit_inputs_authority_receipt_sha256=semantic_sha256("fit-inputs"),
        fold_indices=(0, 1, 2, 3),
        fold_fit_authority_receipts=tuple(
            semantic_sha256(("fit", fold_index)) for fold_index in range(4)
        ),
        fold_fit_input_authority_receipts=tuple(
            semantic_sha256(("fit-input", fold_index)) for fold_index in range(4)
        ),
        execution_environment_authority_receipts=tuple(
            semantic_sha256(("environment", fold_index)) for fold_index in range(4)
        ),
        scientific_execution_fingerprint_sha256=semantic_sha256("science"),
        physical_worker_compatibility_sha256=semantic_sha256("worker"),
        source_data_qualified=True,
        runtime_fit_replayed=True,
        semantic_receipt_sha256=semantic_sha256("four-fold-fit"),
        development_rl_training_authorized=True,
    )
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_v2,
        four_fold_fit_authority=fit,
    )
    with pytest.raises(
        MassiveAdaptiveRLMixedRuntimeSourceGenerationError,
        match="mixed",
    ):
        validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
            runtime_sources_v2=runtime_v2,
            four_fold_fit_authority=replace(
                fit,
                runtime_sources_receipt_sha256=semantic_sha256("other-runtime"),
            ),
        )

    missing = replace(base_index, rows=base_index.rows[:-1])
    with pytest.raises(
        MassiveAdaptiveRLMixedRuntimeSourceGenerationError,
        match="absent",
    ):
        build_massive_adaptive_rl_replay_dependency_index_v2(
            manifest=manifest,
            runtime_source_graph_v2=graph_v2,
            replay_dependency_index_v1=missing,
            runtime_sources_v1=runtime_v1,
        )
