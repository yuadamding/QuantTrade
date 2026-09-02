from __future__ import annotations

import inspect
from dataclasses import replace
from io import BytesIO

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    materialize_massive_adaptive_rl_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_validation_authority_v1 import (
    materialize_massive_adaptive_rl_fixed_control_validation_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    materialize_massive_adaptive_rl_policy_trace_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
    MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    MassiveAdaptiveRLValidationInputsV1Error,
    MassiveAdaptiveRLValidationSourcesAuthorityV1,
    materialize_massive_adaptive_rl_validation_environment_registry_v1,
    materialize_massive_adaptive_rl_validation_sources_authority_v1,
    parse_massive_adaptive_rl_validation_environment_registry_v1,
    parse_massive_adaptive_rl_validation_sources_authority_v1,
    validation_decision_tensor_relative_path_v1,
    validation_cost_ladder_relative_path_v1,
    validation_environment_registry_relative_path_v1,
    validation_fixed_control_relative_path_v1,
    validation_forecast_archive_relative_path_v1,
    validation_primary_trace_relative_path_v1,
    validation_sources_authority_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    build_massive_adaptive_rl_experiment_manifest_v4,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _generic_sources(manifest) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "runtime_sources_receipt_sha256": _digest("runtime-sources"),
        "runtime_graph_witness_receipt_sha256": _digest("runtime-witness"),
        "fold_index": 0,
        "supervised_lineage_receipt_sha256": _digest("lineage"),
        "supervised_training_window_receipt_sha256": _digest("window"),
        "supervised_checkpoint_choice_receipt_sha256": _digest("choice"),
        "supervised_checkpoint_receipt_sha256": _digest("checkpoint"),
        "supervised_checkpoint_source_receipt_sha256": _digest("checkpoint-source"),
        "supervised_model_state_receipt_sha256": _digest("model-state"),
        "supervised_model_spec_receipt_sha256": _digest("model-spec"),
        "calibration_receipt_sha256": _digest("calibration"),
        "validation_decision_tensor_receipt_sha256": _digest("tensor"),
        "validation_decision_tensor_source_receipt_sha256": _digest("tensor-source"),
        "validation_inference_plan_receipt_sha256": _digest("plan"),
        "validation_forecast_archive_receipt_sha256": _digest("forecast"),
        "validation_forecast_source_receipt_sha256": _digest("forecast-source"),
        "validation_chronology_authority_receipt_sha256": _digest("chronology"),
        "validation_tensor_session_dates": ("2023-12-29", "2024-01-02"),
        "validation_decision_session_dates": ("2024-01-02",),
        "validation_full_decision_root_inventory_sha256": _digest("full-roots"),
        "validation_origin_decision_root_inventory_sha256": _digest("origin-roots"),
        "validation_context_origin_inventory_sha256": _digest("contexts"),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLValidationSourcesAuthorityV1(
        **values,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def _environment_authority(
    manifest, sources, cost: float
) -> MassiveAdaptiveRLValidationEnvironmentAuthorityV1:
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "validation_sources_authority_receipt_sha256": (
            sources.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": sources.runtime_sources_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": (
            sources.runtime_graph_witness_receipt_sha256
        ),
        "fold_index": 0,
        "transaction_cost_basis_points": cost,
        "forecast_archive_receipt_sha256": (
            sources.validation_forecast_archive_receipt_sha256
        ),
        "inference_plan_receipt_sha256": (
            sources.validation_inference_plan_receipt_sha256
        ),
        "calibration_receipt_sha256": sources.calibration_receipt_sha256,
        "decision_root_inventory_sha256": _digest("decision-roots"),
        "context_origin_inventory_sha256": _digest("origin-contexts"),
        "daily_input_authority_receipt_sha256": _digest("daily"),
        "fill_source_receipt_sha256": _digest("fills"),
        "identity_authority_receipt_sha256": _digest("identity"),
        "economic_event_archive_receipt_sha256": _digest("events"),
        "compiler_config_receipt_sha256": _digest("compiler"),
        "initial_capital": 10_000_000.0,
        "maximum_fill_participation": 0.02,
        "validation_context_receipt_sha256": _digest("shared-context"),
        "environment_source_inventory_sha256": _digest(("environment", cost)),
        "economic_compatibility_receipt_sha256": _digest(("economics", cost)),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLValidationEnvironmentAuthorityV1(
        **values,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def _generic_registry(manifest, sources):
    authorities = tuple(
        _environment_authority(manifest, sources, cost) for cost in (10.0, 20.0, 40.0)
    )
    values = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "validation_sources_authority_receipt_sha256": (
            sources.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": sources.runtime_sources_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": (
            sources.runtime_graph_witness_receipt_sha256
        ),
        "fold_index": 0,
        "cost_basis_points": (10.0, 20.0, 40.0),
        "environment_authorities": authorities,
        "environment_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in authorities
        ),
        "environment_authority_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in authorities)
        ),
        "validation_context_receipt_sha256": _digest("shared-context"),
        "initial_capital": 10_000_000.0,
        "maximum_fill_participation": 0.02,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLValidationEnvironmentRegistryV1(
        **values,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def test_validation_input_paths_are_manifest_and_fold_canonical() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="canonical-validation-inputs"
    )
    paths = (
        validation_decision_tensor_relative_path_v1(manifest=manifest, fold_index=2),
        validation_forecast_archive_relative_path_v1(manifest=manifest, fold_index=2),
        validation_sources_authority_relative_path_v1(manifest=manifest, fold_index=2),
        validation_environment_registry_relative_path_v1(
            manifest=manifest, fold_index=2
        ),
        validation_primary_trace_relative_path_v1(
            manifest=manifest,
            fold_index=2,
            checkpoint_authority_receipt_sha256=_digest("checkpoint"),
        ),
        validation_cost_ladder_relative_path_v1(
            manifest=manifest,
            fold_index=2,
            checkpoint_authority_receipt_sha256=_digest("checkpoint"),
        ),
        validation_fixed_control_relative_path_v1(
            manifest=manifest,
            fold_index=2,
        ),
    )
    assert len(set(paths)) == 7
    assert all(manifest.semantic_receipt_sha256 in path for path in paths)
    assert all("fold2" in path for path in paths)

    source_parameters = inspect.signature(
        materialize_massive_adaptive_rl_validation_sources_authority_v1
    ).parameters
    registry_parameters = inspect.signature(
        materialize_massive_adaptive_rl_validation_environment_registry_v1
    ).parameters
    assert "artifact_id" not in source_parameters
    assert "checkpoint" not in source_parameters
    assert "forecast_archive" not in source_parameters
    assert "artifact_id" not in registry_parameters
    assert "environment" not in registry_parameters
    assert "cost_basis_points" not in registry_parameters

    for materializer in (
        materialize_massive_adaptive_rl_policy_trace_authority_v1,
        materialize_massive_adaptive_rl_cost_ladder_authority_v1,
        materialize_massive_adaptive_rl_fixed_control_validation_authority_v1,
    ):
        parameters = inspect.signature(materializer).parameters
        assert "validation_environment_registry" in parameters
        assert "validation_environment_authority" not in parameters
        assert "validation_environment_authorities" not in parameters


def test_generic_validation_inputs_are_integrity_only(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="generic-validation-inputs"
    )
    sources = _generic_sources(manifest)
    sources.validate()
    source_relative = validation_sources_authority_relative_path_v1(
        manifest=manifest, fold_index=0
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(sources.semantic_unsigned())),
        root=tmp_path,
        relative_payload_path=source_relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_DATASET,
        source_object_key=source_relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=sources.semantic_receipt_sha256,
        committed_at_ms=1,
    )
    generic_sources = parse_massive_adaptive_rl_validation_sources_authority_v1(
        root=tmp_path,
        loaded_source=load_massive_source_bundle(
            root=tmp_path,
            relative_payload_path=source_relative,
            verified_at_ms=2,
        ),
    )
    assert generic_sources.source_transaction_verified
    assert not generic_sources.runtime_inputs_replayed
    assert not generic_sources.development_stage_authorized

    registry = _generic_registry(manifest, sources)
    registry.validate()
    registry_relative = validation_environment_registry_relative_path_v1(
        manifest=manifest, fold_index=0
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(registry.semantic_unsigned())),
        root=tmp_path,
        relative_payload_path=registry_relative,
        dataset_id=(MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_DATASET),
        source_object_key=registry_relative,
        requested_at_ms=3,
        downloaded_at_ms=3,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=registry.semantic_receipt_sha256,
        committed_at_ms=3,
    )
    generic_registry = parse_massive_adaptive_rl_validation_environment_registry_v1(
        root=tmp_path,
        loaded_source=load_massive_source_bundle(
            root=tmp_path,
            relative_payload_path=registry_relative,
            verified_at_ms=4,
        ),
    )
    assert generic_registry.source_transaction_verified
    assert generic_registry.source_receipt_sha256 == (
        generic_registry._loaded_source.receipt.receipt_sha256  # noqa: SLF001
    )
    assert generic_registry.source_transaction_receipt_sha256 == (
        generic_registry._loaded_source.commit.receipt_sha256  # noqa: SLF001
    )
    assert generic_registry.source_transaction_committed_at_ms == 3
    assert not generic_registry.runtime_environments_replayed
    assert not generic_registry.development_stage_authorized
    assert (
        generic_registry.environment_authority(20.0).transaction_cost_basis_points
        == 20.0
    )
    with pytest.raises(MassiveAdaptiveRLValidationInputsV1Error, match="cost rung"):
        generic_registry.environment_authority(30.0)
    with pytest.raises(MassiveAdaptiveRLValidationInputsV1Error, match="runtime"):
        generic_registry.build_environments()


def test_validation_registry_rejects_a_second_economic_context() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-context-substitution"
    )
    sources = _generic_sources(manifest)
    registry = _generic_registry(manifest, sources)
    changed = replace(
        registry.environment_authorities[-1],
        validation_context_receipt_sha256=_digest("alternative-context"),
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    malformed = replace(
        registry,
        environment_authorities=(*registry.environment_authorities[:-1], changed),
        environment_authority_receipts=(
            *registry.environment_authority_receipts[:-1],
            changed.semantic_receipt_sha256,
        ),
        environment_authority_inventory_sha256=semantic_sha256(
            (
                *registry.environment_authority_receipts[:-1],
                changed.semantic_receipt_sha256,
            )
        ),
        semantic_receipt_sha256="0" * 64,
    )
    malformed = replace(
        malformed,
        semantic_receipt_sha256=semantic_sha256(malformed.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLValidationInputsV1Error, match="differs"):
        malformed.validate()


def test_validation_input_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    ):
        assert len(value) == 64
