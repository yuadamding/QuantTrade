"""Canonical, create-only inputs for adaptive-RL inner validation.

Validation outcomes are meaningful only after the package has committed one
forecast lineage and one economic tape for a fold.  This module makes that
ordering executable.  It deterministically selects the latest causal
supervised lineage from the completed four-fold fit, constructs the complete
target-free validation tensor, plan, forecast, and chronology at canonical
paths, and then persists a registry that creates fresh mutable environments
for the registered 10/20/40-basis-point ladder.

Neither authority accepts a forecast, calibration, environment, cost,
artifact identifier, feature, action origin, or context origin from its
caller.  The predictor roots are selected from the dependency-closed runtime
source reconstruction; clocks, contexts, and reconciled decision roots are
rebuilt by the package before anything is published.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
import json
import math
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
    authorize_massive_adaptive_forecast_archive_v2,
    materialize_massive_adaptive_forecast_archive_v2,
    parse_massive_adaptive_forecast_archive_v2,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
    build_massive_adaptive_inference_plan_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
    build_massive_adaptive_rl_chronology_authority_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLSupervisedLineageSourcesV1,
    MassiveAdaptiveRLRuntimeSourcesV1,
    MassiveAdaptiveRLValidationOriginInputsV1,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-sources-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-validation-sources-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SCHEMA,
            "payload": "canonical-json-validation-source-receipt-graph",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "training": "exact-completed-four-fold-fit",
        "lineage": "latest-causal-supervised-source-fold",
        "tensor": "exact-model-context-plus-complete-126-session-validation",
        "forecast": "single-target-free-cpu-float32-replay",
        "calibration": "checkpoint-specific-selected-lineage",
        "paths": "manifest-and-fold-derived-create-only",
        "caller_checkpoint_or_calibration": False,
        "caller_forecast_or_plan": False,
        "predictor_roots": "persisted-runtime-source-reconstruction-only",
        "caller_feature_action_or_context_roots": False,
        "caller_artifact_id": False,
        "outcome_access": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)

MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-environment-registry-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_DATASET = (
    "massive-adaptive-rl-validation-environment-registry-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SCHEMA),
            "payload": "canonical-json-validation-environment-registry",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "persisted-validation-sources-authority-v1",
        "economics": "runtime-source-graph-owned",
        "costs_basis_points": (10.0, 20.0, 40.0),
        "shared_context": "cost-excluded-identical",
        "mutable_environment": "fresh-construction-only",
        "caller_environment_or_cost": False,
        "caller_artifact_id": False,
        "outcome_access": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)


class MassiveAdaptiveRLValidationInputsV1Error(ValueError):
    """Canonical validation inputs are absent, ambiguous, or mismatched."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _json_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            f"{name} must be a finite number"
        )
    return float(value)


def _source_transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "canonical validation source transaction is incomplete"
        )
    return all(present)


def validation_generation_key_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if fold_index not in manifest.base_manifest.base_manifest.fold_indices:
        raise MassiveAdaptiveRLValidationInputsV1Error("validation fold index differs")
    return f"v4-{manifest.semantic_receipt_sha256}-fold{fold_index}"


def validation_decision_tensor_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return f"massive-adaptive/decision-tensor-v1/{key}-validation-inputs.json"


def validation_forecast_archive_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return f"massive-adaptive/forecast-archive-v2/{key}-validation-forecast.pt"


def validation_sources_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return f"massive-adaptive/rl-validation-inputs-v1/{key}/validation-sources.json"


def validation_environment_registry_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return f"massive-adaptive/rl-validation-inputs-v1/{key}/environment-registry.json"


def validation_primary_trace_artifact_id_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipt_sha256: str,
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    checkpoint = _digest(
        "validation checkpoint authority", checkpoint_authority_receipt_sha256
    )
    return f"{key}-checkpoint-{checkpoint}-primary"


def validation_primary_trace_relative_path_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipt_sha256: str,
) -> str:
    artifact = validation_primary_trace_artifact_id_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipt_sha256=checkpoint_authority_receipt_sha256,
    )
    return f"massive-adaptive/rl-policy-trace-authority-v1/{artifact}.json"


def validation_cost_ladder_artifact_id_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipt_sha256: str,
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    checkpoint = _digest(
        "validation checkpoint authority", checkpoint_authority_receipt_sha256
    )
    return f"{key}-checkpoint-{checkpoint}-cost-ladder"


def validation_cost_ladder_relative_path_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipt_sha256: str,
) -> str:
    artifact = validation_cost_ladder_artifact_id_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipt_sha256=checkpoint_authority_receipt_sha256,
    )
    return f"massive-adaptive/rl-cost-ladder-authority-v1/{artifact}.json"


def validation_fixed_control_artifact_id_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return f"{key}-fc06-primary"


def validation_fixed_control_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    artifact = validation_fixed_control_artifact_id_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    return f"massive-adaptive/rl-fixed-control-validation-authority-v1/{artifact}.json"


@dataclass(frozen=True, slots=True)
class _ValidationSourcesRuntimeV1:
    manifest: MassiveAdaptiveRLExperimentManifestV4
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1
    origin_inputs: MassiveAdaptiveRLValidationOriginInputsV1
    features: tuple[MassiveProfitabilityOriginFeaturesV3, ...]
    action_origins: tuple[MassiveAdaptiveOriginAuthorityV1, ...]
    context_origins: tuple[MassiveAdaptiveContextOriginAuthorityV1, ...]
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...]
    decision_tensor: MassiveAdaptiveDecisionTensorV1
    inference_plan: MassiveAdaptiveInferencePlanV1
    forecast_archive: MassiveAdaptiveForecastArchiveV2
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1


@dataclass(frozen=True, slots=True)
class _ValidationSourcesFactsV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    validation_origin_inputs_receipt_sha256: str
    fold_index: int
    supervised_lineage_receipt_sha256: str
    supervised_training_window_receipt_sha256: str
    supervised_checkpoint_choice_receipt_sha256: str
    supervised_checkpoint_receipt_sha256: str
    supervised_checkpoint_source_receipt_sha256: str
    supervised_model_state_receipt_sha256: str
    supervised_model_spec_receipt_sha256: str
    calibration_receipt_sha256: str
    validation_decision_tensor_receipt_sha256: str
    validation_decision_tensor_source_receipt_sha256: str
    validation_inference_plan_receipt_sha256: str
    validation_forecast_archive_receipt_sha256: str
    validation_forecast_source_receipt_sha256: str
    validation_chronology_authority_receipt_sha256: str
    validation_tensor_session_dates: tuple[str, ...]
    validation_decision_session_dates: tuple[str, ...]
    validation_full_decision_root_inventory_sha256: str
    validation_origin_decision_root_inventory_sha256: str
    validation_context_origin_inventory_sha256: str
    source_data_qualified: bool


def _expected_validation_tensor_dates(
    *,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fold_index: int,
    lineage: MassiveAdaptiveRLSupervisedLineageSourcesV1,
) -> tuple[str, ...]:
    split = runtime_sources.split_plan
    role_dates = split.outer_folds[fold_index].inner_validation_session_dates
    candidate_dates = split.candidate_session_dates
    maximum_context = min(
        lineage.model_spec.maximum_context_sessions,
        MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    )
    start = candidate_dates.index(role_dates[0])
    stop = candidate_dates.index(role_dates[-1]) + 1
    context_start = start - maximum_context + 1
    if context_start < 0:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation tensor causal context is unavailable"
        )
    return candidate_dates[context_start:stop]


def _validation_source_roots(
    *,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    lineage: MassiveAdaptiveRLSupervisedLineageSourcesV1,
    fold_index: int,
) -> tuple[
    MassiveAdaptiveRLValidationOriginInputsV1,
    tuple[MassiveProfitabilityOriginFeaturesV3, ...],
    tuple[MassiveAdaptiveOriginAuthorityV1, ...],
    tuple[MassiveAdaptiveContextOriginAuthorityV1, ...],
    tuple[MassiveAdaptiveDecisionRootV1, ...],
]:
    origin_inputs = runtime_sources.validation_origin_inputs(fold_index)
    origin_inputs.validate()
    feature_rows = origin_inputs.features
    action_rows = origin_inputs.action_origins
    context_rows = origin_inputs.context_origins
    roots = origin_inputs.decision_roots
    expected_dates = _expected_validation_tensor_dates(
        runtime_sources=runtime_sources,
        fold_index=fold_index,
        lineage=lineage,
    )
    dates = tuple(row.decision_session_date for row in feature_rows)
    if (
        dates != expected_dates
        or tuple(row.decision_session_date for row in action_rows) != expected_dates
        or tuple(row.decision_session_date for row in context_rows) != expected_dates
        or tuple(row.decision_session_date for row in roots) != expected_dates
        or origin_inputs.tensor_session_dates != expected_dates
        or origin_inputs.replay_dependency_index_receipt_sha256
        != runtime_sources.replay_dependency_index_receipt_sha256
        or any(
            row.daily_input_authority_semantic_receipt_sha256
            != runtime_sources.daily_input_authority.semantic_receipt_sha256
            for row in feature_rows
        )
        or any(
            row.session_authority_receipt_sha256
            != runtime_sources.session_authority.receipt_sha256
            for row in action_rows
        )
        or any(
            row.session_authority_receipt_sha256
            != runtime_sources.session_authority.receipt_sha256
            or row.identity_authority_receipt_sha256
            != runtime_sources.identity_authority.receipt_sha256
            for row in context_rows
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source roots do not cover the canonical source tape"
        )
    expected_roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=action,
            features=feature,
        )
        for feature, action, context in zip(
            feature_rows, action_rows, context_rows, strict=True
        )
    )
    if roots != expected_roots:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation decision roots do not replay from runtime sources"
        )
    return origin_inputs, feature_rows, action_rows, context_rows, roots


def _validation_sources_facts(
    runtime: _ValidationSourcesRuntimeV1,
) -> _ValidationSourcesFactsV1:
    manifest = runtime.manifest
    four_fold = runtime.four_fold_fit_authority
    runtime_sources = runtime.runtime_sources
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(four_fold) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(runtime_sources) is not MassiveAdaptiveRLRuntimeSourcesV1
        or type(runtime.origin_inputs) is not MassiveAdaptiveRLValidationOriginInputsV1
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source authority root type differs"
        )
    manifest.validate()
    four_fold.validate()
    runtime_sources.validate()
    runtime.origin_inputs.validate()
    fold_index = runtime.inference_plan.fold_index
    if fold_index not in range(4):
        raise MassiveAdaptiveRLValidationInputsV1Error("validation source fold differs")
    fold_fit = four_fold.fold_fit(fold_index)
    lineage = runtime_sources.fold(fold_index).supervised_lineage(fold_index)
    lineage.validate()
    runtime.decision_tensor.validate()
    runtime.inference_plan.validate()
    runtime.forecast_archive.validate()
    runtime.chronology_authority.validate()
    for decision_root in runtime.decision_roots:
        decision_root.validate()
    for context_origin in runtime.context_origins:
        context_origin.validate()
    expected_dates = _expected_validation_tensor_dates(
        runtime_sources=runtime_sources,
        fold_index=fold_index,
        lineage=lineage,
    )
    validation_dates = runtime_sources.split_plan.outer_folds[
        fold_index
    ].inner_validation_session_dates
    expected_plan = build_massive_adaptive_inference_plan_v1(
        decision_tensor=runtime.decision_tensor,
        decision_roots=runtime.decision_roots,
        split_plan=runtime_sources.split_plan,
        fold_index=fold_index,
        inference_role="inner_validation",
        model_spec=lineage.model_spec,
    )
    expected_chronology = build_massive_adaptive_rl_chronology_authority_v1(
        training_forecast_authority=fold_fit.training_forecast_authority,
        validation_inference_plan=expected_plan,
        split_plan=runtime_sources.split_plan,
    )
    tensor_relative = validation_decision_tensor_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    forecast_relative = validation_forecast_archive_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    context_by_date = {
        row.decision_session_date: row for row in runtime.context_origins
    }
    origin_contexts = tuple(context_by_date[date] for date in validation_dates)
    runtime_witness = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if (
        manifest.base_manifest.semantic_receipt_sha256
        != four_fold.manifest_v3_receipt_sha256
        or manifest.base_manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or four_fold.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
        or runtime.origin_inputs.fold_index != fold_index
        or runtime.origin_inputs.replay_dependency_index_receipt_sha256
        != runtime_sources.replay_dependency_index_receipt_sha256
        or runtime.origin_inputs.features != runtime.features
        or runtime.origin_inputs.action_origins != runtime.action_origins
        or runtime.origin_inputs.context_origins != runtime.context_origins
        or runtime.origin_inputs.decision_roots != runtime.decision_roots
        or runtime_witness is None
        or four_fold.runtime_graph_witness_receipt_sha256 != runtime_witness
        or not four_fold.development_stage_authorized
        or not fold_fit.development_stage_authorized
        or tuple(row.decision_session_date for row in runtime.features)
        != expected_dates
        or tuple(row.decision_session_date for row in runtime.action_origins)
        != expected_dates
        or tuple(row.decision_session_date for row in runtime.context_origins)
        != expected_dates
        or runtime.decision_tensor.decision_session_dates != expected_dates
        or runtime.decision_tensor.feature_semantic_receipts
        != tuple(row.semantic_receipt_sha256 for row in runtime.features)
        or runtime.decision_tensor.action_origin_receipts
        != tuple(row.semantic_receipt_sha256 for row in runtime.action_origins)
        or runtime.inference_plan != expected_plan
        or runtime.chronology_authority != expected_chronology
        or runtime.forecast_archive.fold_index != fold_index
        or runtime.forecast_archive.checkpoint_receipt_sha256
        != lineage.selected_checkpoint.semantic_receipt_sha256
        or runtime.forecast_archive.model_state_receipt_sha256
        != lineage.selected_checkpoint.model_state_receipt_sha256
        or runtime.forecast_archive.training_window_plan_receipt_sha256
        != lineage.training_window.semantic_receipt_sha256
        or runtime.forecast_archive.inference_tensor_receipt_sha256
        != runtime.decision_tensor.semantic_receipt_sha256
        or runtime.forecast_archive.inference_plan_receipt_sha256
        != runtime.inference_plan.semantic_receipt_sha256
        or runtime.forecast_archive.inference_full_decision_root_inventory_sha256
        != runtime.inference_plan.full_decision_root_inventory_sha256
        or runtime.forecast_archive.inference_origin_decision_root_inventory_sha256
        != runtime.inference_plan.origin_decision_root_inventory_sha256
        or runtime.forecast_archive.origin_session_dates != validation_dates
        or not runtime.forecast_archive.runtime_forecasts_replayed
        or not runtime.forecast_archive.development_forecast_authorized
        or runtime.decision_tensor.loaded_source.payload_relative_path
        != tensor_relative
        or runtime.forecast_archive.loaded_source.payload_relative_path
        != forecast_relative
        or tuple(row.decision_session_date for row in runtime.decision_roots)
        != expected_dates
        or tuple(row.decision_session_date for row in runtime.context_origins)
        != expected_dates
        or any(
            root.feature_semantic_receipt_sha256 != feature.semantic_receipt_sha256
            or root.action_origin_receipt_sha256 != action.semantic_receipt_sha256
            or root.context_origin_receipt_sha256 != context.semantic_receipt_sha256
            for root, feature, action, context in zip(
                runtime.decision_roots,
                runtime.features,
                runtime.action_origins,
                runtime.context_origins,
                strict=True,
            )
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source lineage or canonical artifact differs"
        )
    source_qualified = bool(
        runtime_sources.source_data_qualified
        and runtime.origin_inputs.source_data_qualified
        and lineage.source_data_qualified
        and all(row.source_inputs_data_qualified for row in runtime.features)
        and all(
            row.action_identity_source_data_qualified for row in runtime.action_origins
        )
        and all(row.source_data_qualified for row in runtime.context_origins)
        and runtime.decision_tensor.committed_source_data_qualified
        and runtime.decision_tensor.runtime_source_replayed
        and runtime.inference_plan.source_data_qualified
        and runtime.forecast_archive.committed_source_data_qualified
        and all(row.source_data_qualified for row in runtime.decision_roots)
        and all(row.source_data_qualified for row in origin_contexts)
        and runtime.chronology_authority.source_data_qualified
    )
    return _ValidationSourcesFactsV1(
        experiment_id=manifest.experiment_id,
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        four_fold_fit_authority_receipt_sha256=four_fold.semantic_receipt_sha256,
        fold_fit_authority_receipt_sha256=fold_fit.semantic_receipt_sha256,
        runtime_sources_receipt_sha256=runtime_sources.semantic_receipt_sha256,
        runtime_graph_witness_receipt_sha256=runtime_witness,
        validation_origin_inputs_receipt_sha256=(
            runtime.origin_inputs.semantic_receipt_sha256
        ),
        fold_index=fold_index,
        supervised_lineage_receipt_sha256=lineage.semantic_receipt_sha256,
        supervised_training_window_receipt_sha256=(
            lineage.training_window.semantic_receipt_sha256
        ),
        supervised_checkpoint_choice_receipt_sha256=(
            lineage.checkpoint_choice.semantic_receipt_sha256
        ),
        supervised_checkpoint_receipt_sha256=(
            lineage.selected_checkpoint.semantic_receipt_sha256
        ),
        supervised_checkpoint_source_receipt_sha256=(
            lineage.selected_checkpoint.loaded_source.receipt.receipt_sha256
        ),
        supervised_model_state_receipt_sha256=(
            lineage.selected_checkpoint.model_state_receipt_sha256
        ),
        supervised_model_spec_receipt_sha256=lineage.model_spec.receipt_sha256,
        calibration_receipt_sha256=lineage.calibration.semantic_receipt_sha256,
        validation_decision_tensor_receipt_sha256=(
            runtime.decision_tensor.semantic_receipt_sha256
        ),
        validation_decision_tensor_source_receipt_sha256=(
            runtime.decision_tensor.loaded_source.receipt.receipt_sha256
        ),
        validation_inference_plan_receipt_sha256=(
            runtime.inference_plan.semantic_receipt_sha256
        ),
        validation_forecast_archive_receipt_sha256=(
            runtime.forecast_archive.semantic_receipt_sha256
        ),
        validation_forecast_source_receipt_sha256=(
            runtime.forecast_archive.loaded_source.receipt.receipt_sha256
        ),
        validation_chronology_authority_receipt_sha256=(
            runtime.chronology_authority.semantic_receipt_sha256
        ),
        validation_tensor_session_dates=expected_dates,
        validation_decision_session_dates=validation_dates,
        validation_full_decision_root_inventory_sha256=semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in runtime.decision_roots)
        ),
        validation_origin_decision_root_inventory_sha256=semantic_sha256(
            tuple(
                row.decision_root_receipt_sha256 for row in runtime.inference_plan.rows
            )
        ),
        validation_context_origin_inventory_sha256=semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in origin_contexts)
        ),
        source_data_qualified=source_qualified,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationSourcesAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    validation_origin_inputs_receipt_sha256: str
    fold_index: int
    supervised_lineage_receipt_sha256: str
    supervised_training_window_receipt_sha256: str
    supervised_checkpoint_choice_receipt_sha256: str
    supervised_checkpoint_receipt_sha256: str
    supervised_checkpoint_source_receipt_sha256: str
    supervised_model_state_receipt_sha256: str
    supervised_model_spec_receipt_sha256: str
    calibration_receipt_sha256: str
    validation_decision_tensor_receipt_sha256: str
    validation_decision_tensor_source_receipt_sha256: str
    validation_inference_plan_receipt_sha256: str
    validation_forecast_archive_receipt_sha256: str
    validation_forecast_source_receipt_sha256: str
    validation_chronology_authority_receipt_sha256: str
    validation_tensor_session_dates: tuple[str, ...]
    validation_decision_session_dates: tuple[str, ...]
    validation_full_decision_root_inventory_sha256: str
    validation_origin_decision_root_inventory_sha256: str
    validation_context_origin_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_inputs_replayed: bool = False
    development_validation_inputs_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SCHEMA
    _runtime: _ValidationSourcesRuntimeV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            **{
                name: getattr(self, name)
                for name in _ValidationSourcesFactsV1.__dataclass_fields__
            },
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_inputs_replayed
            and self.development_validation_inputs_authorized
            and self.source_data_qualified
        )

    @property
    def runtime_forecast_archive(self) -> MassiveAdaptiveForecastArchiveV2:
        self.validate()
        if self._runtime is None:
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation forecast runtime is absent"
            )
        return self._runtime.forecast_archive

    @property
    def runtime_inference_plan(self) -> MassiveAdaptiveInferencePlanV1:
        self.validate()
        if self._runtime is None:
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation inference runtime is absent"
            )
        return self._runtime.inference_plan

    @property
    def runtime_chronology_authority(self) -> MassiveAdaptiveRLChronologyAuthorityV1:
        self.validate()
        if self._runtime is None:
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation chronology runtime is absent"
            )
        return self._runtime.chronology_authority

    def validate(self) -> None:
        if self._loaded_source is not None:
            self._loaded_source.validate()
        runtime_present = self._runtime is not None
        if runtime_present:
            assert self._runtime is not None
            expected = _validation_sources_facts(self._runtime)
            if any(
                getattr(self, name) != getattr(expected, name)
                for name in _ValidationSourcesFactsV1.__dataclass_fields__
            ):
                raise MassiveAdaptiveRLValidationInputsV1Error(
                    "validation source runtime differs"
                )
        expected_authorized = bool(runtime_present and self.source_data_qualified)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or not self.validation_tensor_session_dates
            or not self.validation_decision_session_dates
            or any(
                not isinstance(value, str) or not value
                for value in (
                    *self.validation_tensor_session_dates,
                    *self.validation_decision_session_dates,
                )
            )
            or self.validation_tensor_session_dates
            != tuple(sorted(set(self.validation_tensor_session_dates)))
            or self.validation_decision_session_dates
            != tuple(sorted(set(self.validation_decision_session_dates)))
            or not set(self.validation_decision_session_dates).issubset(
                self.validation_tensor_session_dates
            )
            or not isinstance(self.source_data_qualified, bool)
            or not isinstance(self.runtime_inputs_replayed, bool)
            or not isinstance(self.development_validation_inputs_authorized, bool)
            or not isinstance(self.profitability_reporting_authorized, bool)
            or not isinstance(self.outer_evaluation_authorized, bool)
            or not isinstance(self.lockbox_access_authorized, bool)
            or self.runtime_inputs_replayed != runtime_present
            or self.development_validation_inputs_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation source authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation source transaction differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("validation source authority", value)
        _digest("validation source authority", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _build_validation_sources_authority(
    runtime: _ValidationSourcesRuntimeV1,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    facts = _validation_sources_facts(runtime)
    provisional = MassiveAdaptiveRLValidationSourcesAuthorityV1(
        **asdict(facts),
        semantic_receipt_sha256="0" * 64,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=facts.source_data_qualified,
        _runtime=runtime,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _validation_sources_body(
    authority: MassiveAdaptiveRLValidationSourcesAuthorityV1,
) -> dict[str, object]:
    return authority.semantic_unsigned()


def _load_canonical_decision_tensor(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    features: tuple[MassiveProfitabilityOriginFeaturesV3, ...],
    action_origins: tuple[MassiveAdaptiveOriginAuthorityV1, ...],
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveDecisionTensorV1:
    relative = validation_decision_tensor_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    if _source_transaction_exists(root=root, relative=relative):
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        )
        return authorize_massive_adaptive_decision_tensor_v1(
            root=root,
            tensor=parse_massive_adaptive_decision_tensor_v1(
                root=root,
                loaded_source=loaded,
            ),
            features=features,
            action_origins=action_origins,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "canonical validation decision tensor is absent"
        )
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return materialize_massive_adaptive_decision_tensor_v1(
        root=root,
        artifact_id=f"{key}-validation-inputs",
        features=features,
        action_origins=action_origins,
        committed_at_ms=committed_at_ms,
    )


def _load_canonical_forecast_archive(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    lineage: MassiveAdaptiveRLSupervisedLineageSourcesV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...],
    inference_plan: MassiveAdaptiveInferencePlanV1,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveForecastArchiveV2:
    relative = validation_forecast_archive_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    if _source_transaction_exists(root=root, relative=relative):
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        )
        return authorize_massive_adaptive_forecast_archive_v2(
            root=root,
            archive=parse_massive_adaptive_forecast_archive_v2(
                root=root,
                loaded_source=loaded,
            ),
            checkpoint=lineage.selected_checkpoint,
            training_window_plan=lineage.training_window,
            inference_tensor=decision_tensor,
            inference_decision_roots=decision_roots,
            inference_plan=inference_plan,
            model_spec=lineage.model_spec,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "canonical validation forecast archive is absent"
        )
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return materialize_massive_adaptive_forecast_archive_v2(
        root=root,
        artifact_id=f"{key}-validation-forecast",
        checkpoint=lineage.selected_checkpoint,
        training_window_plan=lineage.training_window,
        inference_tensor=decision_tensor,
        inference_decision_roots=decision_roots,
        inference_plan=inference_plan,
        model_spec=lineage.model_spec,
        committed_at_ms=committed_at_ms,
    )


def _validation_sources_runtime(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool,
) -> _ValidationSourcesRuntimeV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(runtime_sources) is not MassiveAdaptiveRLRuntimeSourcesV1
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source root authority type differs"
        )
    manifest.validate()
    four_fold_fit_authority.validate()
    runtime_sources.validate()
    if (
        fold_index not in manifest.base_manifest.base_manifest.fold_indices
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.experiment_id != four_fold_fit_authority.experiment_id
        or manifest.base_manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or manifest.base_manifest.semantic_receipt_sha256
        != four_fold_fit_authority.manifest_v3_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source manifest or training lineage differs"
        )
    lineage = runtime_sources.fold(fold_index).supervised_lineage(fold_index)
    (
        origin_inputs,
        feature_rows,
        action_rows,
        context_rows,
        decision_roots,
    ) = _validation_source_roots(
        runtime_sources=runtime_sources,
        lineage=lineage,
        fold_index=fold_index,
    )
    tensor = _load_canonical_decision_tensor(
        root=root,
        manifest=manifest,
        fold_index=fold_index,
        features=feature_rows,
        action_origins=action_rows,
        committed_at_ms=committed_at_ms,
        allow_materialize=allow_materialize,
    )
    inference_plan = build_massive_adaptive_inference_plan_v1(
        decision_tensor=tensor,
        decision_roots=decision_roots,
        split_plan=runtime_sources.split_plan,
        fold_index=fold_index,
        inference_role="inner_validation",
        model_spec=lineage.model_spec,
    )
    forecast = _load_canonical_forecast_archive(
        root=root,
        manifest=manifest,
        fold_index=fold_index,
        lineage=lineage,
        decision_tensor=tensor,
        decision_roots=decision_roots,
        inference_plan=inference_plan,
        committed_at_ms=committed_at_ms,
        allow_materialize=allow_materialize,
    )
    chronology = build_massive_adaptive_rl_chronology_authority_v1(
        training_forecast_authority=(
            four_fold_fit_authority.fold_fit(fold_index).training_forecast_authority
        ),
        validation_inference_plan=inference_plan,
        split_plan=runtime_sources.split_plan,
    )
    runtime = _ValidationSourcesRuntimeV1(
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
        origin_inputs=origin_inputs,
        features=feature_rows,
        action_origins=action_rows,
        context_origins=context_rows,
        decision_roots=decision_roots,
        decision_tensor=tensor,
        inference_plan=inference_plan,
        forecast_archive=forecast,
        chronology_authority=chronology,
    )
    _validation_sources_facts(runtime)
    return runtime


def _parse_validation_sources_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source authority is not canonical JSON"
        )
    body = dict(value)
    for name in (
        "validation_tensor_session_dates",
        "validation_decision_session_dates",
    ):
        body[name] = tuple(cast(Sequence[str], body[name]))
    return body


def parse_massive_adaptive_rl_validation_sources_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    body = _parse_validation_sources_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLValidationSourcesAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_inputs_replayed=False,
        development_validation_inputs_authorized=False,
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_sources_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    relative = validation_sources_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=verified_at_ms,
    )
    return parse_massive_adaptive_rl_validation_sources_authority_v1(
        root=root,
        loaded_source=loaded,
    )


def authorize_massive_adaptive_rl_validation_sources_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    authority.validate()
    relative = validation_sources_authority_relative_path_v1(
        manifest=manifest,
        fold_index=authority.fold_index,
    )
    if authority._loaded_source is None or (
        authority._loaded_source.payload_relative_path != relative
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source authority path differs"
        )
    runtime = _validation_sources_runtime(
        root=root,
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
        fold_index=authority.fold_index,
        committed_at_ms=authority._loaded_source.verified_at_ms,
        allow_materialize=False,
    )
    expected = _build_validation_sources_authority(runtime)
    if authority.semantic_unsigned() != expected.semantic_unsigned():
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source authority does not replay"
        )
    result = replace(
        authority,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=authority.source_data_qualified,
        _runtime=runtime,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_validation_sources_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fold_index: int,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    relative = validation_sources_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    if _source_transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "canonical validation source authority already exists"
        )
    runtime = _validation_sources_runtime(
        root=root,
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
        fold_index=fold_index,
        committed_at_ms=committed_at_ms,
        allow_materialize=True,
    )
    authority = _build_validation_sources_authority(runtime)
    body = _validation_sources_body(authority)
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-VALIDATION-SOURCES-V1-{manifest.experiment_id}-"
            f"FOLD{fold_index}"
        ),
    )
    generic = load_massive_adaptive_rl_validation_sources_authority_v1(
        root=root,
        manifest=manifest,
        fold_index=fold_index,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_validation_sources_authority_v1(
        root=root,
        authority=generic,
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
    )


def prepare_or_resume_massive_adaptive_rl_validation_sources_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fold_index: int,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
    """Create the canonical validation lineage, or strictly replay it."""

    relative = validation_sources_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    if not _source_transaction_exists(root=root, relative=relative):
        return materialize_massive_adaptive_rl_validation_sources_authority_v1(
            root=root,
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources=runtime_sources,
            fold_index=fold_index,
            committed_at_ms=committed_at_ms,
        )
    return authorize_massive_adaptive_rl_validation_sources_authority_v1(
        root=root,
        authority=load_massive_adaptive_rl_validation_sources_authority_v1(
            root=root,
            manifest=manifest,
            fold_index=fold_index,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationEnvironmentAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    fold_index: int
    transaction_cost_basis_points: float
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    decision_root_inventory_sha256: str
    context_origin_inventory_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_archive_receipt_sha256: str
    compiler_config_receipt_sha256: str
    initial_capital: float
    maximum_fill_participation: float
    validation_context_receipt_sha256: str
    environment_source_inventory_sha256: str
    economic_compatibility_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "semantic_receipt_sha256"
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or not isinstance(self.transaction_cost_basis_points, float)
            or self.transaction_cost_basis_points not in (10.0, 20.0, 40.0)
            or not isinstance(self.initial_capital, float)
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0.0
            or not isinstance(self.maximum_fill_participation, float)
            or not 0.0 < self.maximum_fill_participation <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment authority differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("validation environment authority", value)
        _digest("validation environment authority", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def validate_environment(
        self, environment: MassiveAdaptiveProfitabilityEnvV1
    ) -> None:
        environment.forecast_archive.validate()
        environment.calibration.validate()
        environment.inference_plan.validate()
        plan_dates = tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        if (
            environment.transaction_cost_basis_points
            != self.transaction_cost_basis_points
            or environment.forecast_archive.semantic_receipt_sha256
            != self.forecast_archive_receipt_sha256
            or environment.inference_plan.semantic_receipt_sha256
            != self.inference_plan_receipt_sha256
            or environment.calibration.semantic_receipt_sha256
            != self.calibration_receipt_sha256
            or semantic_sha256(
                tuple(
                    environment.roots[date].semantic_receipt_sha256
                    for date in plan_dates
                )
            )
            != self.decision_root_inventory_sha256
            or semantic_sha256(
                tuple(
                    environment.contexts[date].semantic_receipt_sha256
                    for date in plan_dates
                )
            )
            != self.context_origin_inventory_sha256
            or environment.daily_input_authority.semantic_receipt_sha256
            != self.daily_input_authority_receipt_sha256
            or environment.fill_source.semantic_receipt_sha256
            != self.fill_source_receipt_sha256
            or environment.identity_authority.receipt_sha256
            != self.identity_authority_receipt_sha256
            or environment.economic_event_archive is None
            or environment.economic_event_archive.receipt_sha256
            != self.economic_event_archive_receipt_sha256
            or environment.compiler_config.receipt_sha256
            != self.compiler_config_receipt_sha256
            or environment.initial_capital != self.initial_capital
            or environment.maximum_fill_participation != self.maximum_fill_participation
            or environment.validation_context_receipt_sha256
            != self.validation_context_receipt_sha256
            or environment.source_inventory_sha256
            != self.environment_source_inventory_sha256
            or environment.economic_compatibility_receipt_sha256
            != self.economic_compatibility_receipt_sha256
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment differs from its authority"
            )


def _validation_environment(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    transaction_cost_basis_points: float,
) -> MassiveAdaptiveProfitabilityEnvV1:
    validation_sources.validate()
    if validation_sources._runtime is None:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation source runtime is absent"
        )
    runtime = validation_sources._runtime
    plan = runtime.inference_plan
    plan_dates = tuple(row.decision_session_date for row in plan.rows)
    roots = {row.decision_session_date: row for row in runtime.decision_roots}
    contexts = {row.decision_session_date: row for row in runtime.context_origins}
    economics = manifest.base_manifest.base_manifest
    return MassiveAdaptiveProfitabilityEnvV1(
        forecast_archive=runtime.forecast_archive,
        calibration=runtime_sources.fold(validation_sources.fold_index)
        .supervised_lineage(validation_sources.fold_index)
        .calibration,
        inference_plan=plan,
        decision_roots=tuple(roots[date] for date in plan_dates),
        context_origins=tuple(contexts[date] for date in plan_dates),
        fill_source=runtime_sources.fill_source,
        daily_input_authority=runtime_sources.daily_input_authority,
        identity_authority=runtime_sources.identity_authority,
        economic_event_archive=runtime_sources.economic_event_archive,
        initial_capital=economics.primary_capital,
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=economics.maximum_fill_participation,
    )


def _validation_environment_source_qualified(
    *,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> bool:
    plan_dates = tuple(
        row.decision_session_date for row in environment.inference_plan.rows
    )
    required_fill_dates = {
        row.next_session_date for row in environment.inference_plan.rows
    }
    daily_dates = {
        row.source_session_date
        for row in runtime_sources.daily_input_authority.sessions
    }
    security_ids = set(validation_sources.runtime_forecast_archive.security_ids)
    return bool(
        validation_sources.development_stage_authorized
        and runtime_sources.source_data_qualified
        and runtime_sources.daily_input_authority.source_transport_qualified
        and runtime_sources.daily_input_authority.daily_input_data_qualified
        and runtime_sources.fill_source.source_paths_replayed
        and runtime_sources.fill_source.source_data_qualified
        and set(plan_dates) <= daily_dates
        and required_fill_dates <= daily_dates
        and required_fill_dates <= set(runtime_sources.fill_source.session_dates)
        and security_ids
        <= set(runtime_sources.daily_input_authority.supported_security_ids)
        and security_ids <= set(runtime_sources.fill_source.supported_security_ids)
    )


def _validation_environment_authority(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLValidationEnvironmentAuthorityV1:
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if runtime_receipt is None:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment runtime witness is absent"
        )
    plan_dates = tuple(
        row.decision_session_date for row in environment.inference_plan.rows
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "validation_sources_authority_receipt_sha256": (
            validation_sources.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "fold_index": validation_sources.fold_index,
        "transaction_cost_basis_points": (environment.transaction_cost_basis_points),
        "forecast_archive_receipt_sha256": (
            environment.forecast_archive.semantic_receipt_sha256
        ),
        "inference_plan_receipt_sha256": (
            environment.inference_plan.semantic_receipt_sha256
        ),
        "calibration_receipt_sha256": (environment.calibration.semantic_receipt_sha256),
        "decision_root_inventory_sha256": semantic_sha256(
            tuple(
                environment.roots[date].semantic_receipt_sha256 for date in plan_dates
            )
        ),
        "context_origin_inventory_sha256": semantic_sha256(
            tuple(
                environment.contexts[date].semantic_receipt_sha256
                for date in plan_dates
            )
        ),
        "daily_input_authority_receipt_sha256": (
            runtime_sources.daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_receipt_sha256": (
            runtime_sources.fill_source.semantic_receipt_sha256
        ),
        "identity_authority_receipt_sha256": (
            runtime_sources.identity_authority.receipt_sha256
        ),
        "economic_event_archive_receipt_sha256": (
            runtime_sources.economic_event_archive.receipt_sha256
        ),
        "compiler_config_receipt_sha256": environment.compiler_config.receipt_sha256,
        "initial_capital": environment.initial_capital,
        "maximum_fill_participation": environment.maximum_fill_participation,
        "validation_context_receipt_sha256": (
            environment.validation_context_receipt_sha256
        ),
        "environment_source_inventory_sha256": environment.source_inventory_sha256,
        "economic_compatibility_receipt_sha256": (
            environment.economic_compatibility_receipt_sha256
        ),
        "source_data_qualified": _validation_environment_source_qualified(
            validation_sources=validation_sources,
            runtime_sources=runtime_sources,
            environment=environment,
        ),
    }
    provisional = MassiveAdaptiveRLValidationEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    result.validate_environment(environment)
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    fold_index: int
    cost_basis_points: tuple[float, ...]
    environment_authorities: tuple[
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1, ...
    ]
    environment_authority_receipts: tuple[str, ...]
    environment_authority_inventory_sha256: str
    validation_context_receipt_sha256: str
    initial_capital: float
    maximum_fill_participation: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_environments_replayed: bool = False
    development_validation_environments_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SCHEMA
    _manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v4_receipt_sha256": self.manifest_v4_receipt_sha256,
            "training_manifest_v3_receipt_sha256": (
                self.training_manifest_v3_receipt_sha256
            ),
            "validation_sources_authority_receipt_sha256": (
                self.validation_sources_authority_receipt_sha256
            ),
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "fold_index": self.fold_index,
            "cost_basis_points": self.cost_basis_points,
            "environment_authorities": tuple(
                row.payload() for row in self.environment_authorities
            ),
            "environment_authority_receipts": self.environment_authority_receipts,
            "environment_authority_inventory_sha256": (
                self.environment_authority_inventory_sha256
            ),
            "validation_context_receipt_sha256": (
                self.validation_context_receipt_sha256
            ),
            "initial_capital": self.initial_capital,
            "maximum_fill_participation": self.maximum_fill_participation,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        """Return the persisted source-object receipt for the registry."""

        if self._loaded_source is None:
            return None
        return self._loaded_source.receipt.receipt_sha256

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        """Return the immutable commit receipt for the persisted registry."""

        if self._loaded_source is None:
            return None
        return self._loaded_source.commit.receipt_sha256

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        """Return when the canonical registry transaction was committed."""

        if self._loaded_source is None:
            return None
        return self._loaded_source.commit.committed_at_ms

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_environments_replayed
            and self.development_validation_environments_authorized
            and self.source_data_qualified
            and self._validation_sources is not None
            and self._validation_sources.development_stage_authorized
        )

    def validate(self) -> None:
        if self._loaded_source is not None:
            self._loaded_source.validate()
        for row in self.environment_authorities:
            row.validate()
        runtime_values = (
            self._manifest,
            self._validation_sources,
            self._runtime_sources,
        )
        runtime_present = all(value is not None for value in runtime_values)
        if any(value is not None for value in runtime_values) != runtime_present:
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment registry runtime is partial"
            )
        if runtime_present:
            assert self._manifest is not None
            assert self._validation_sources is not None
            assert self._runtime_sources is not None
            expected = _build_validation_environment_registry(
                manifest=self._manifest,
                validation_sources=self._validation_sources,
                runtime_sources=self._runtime_sources,
            )
            if self.semantic_unsigned() != expected.semantic_unsigned():
                raise MassiveAdaptiveRLValidationInputsV1Error(
                    "validation environment registry runtime differs"
                )
        expected_authorized = bool(runtime_present and self.source_data_qualified)
        contexts = {
            row.validation_context_receipt_sha256
            for row in self.environment_authorities
        }
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or any(
                type(row) is not MassiveAdaptiveRLValidationEnvironmentAuthorityV1
                for row in self.environment_authorities
            )
            or self.cost_basis_points != (10.0, 20.0, 40.0)
            or any(type(value) is not float for value in self.cost_basis_points)
            or tuple(
                row.transaction_cost_basis_points
                for row in self.environment_authorities
            )
            != self.cost_basis_points
            or self.environment_authority_receipts
            != tuple(
                row.semantic_receipt_sha256 for row in self.environment_authorities
            )
            or self.environment_authority_receipts
            != tuple(dict.fromkeys(self.environment_authority_receipts))
            or self.environment_authority_inventory_sha256
            != semantic_sha256(self.environment_authority_receipts)
            or contexts != {self.validation_context_receipt_sha256}
            or any(
                row.experiment_id != self.experiment_id
                or row.manifest_v4_receipt_sha256 != self.manifest_v4_receipt_sha256
                or row.training_manifest_v3_receipt_sha256
                != self.training_manifest_v3_receipt_sha256
                or row.validation_sources_authority_receipt_sha256
                != self.validation_sources_authority_receipt_sha256
                or row.runtime_sources_receipt_sha256
                != self.runtime_sources_receipt_sha256
                or row.runtime_graph_witness_receipt_sha256
                != self.runtime_graph_witness_receipt_sha256
                or row.fold_index != self.fold_index
                or row.initial_capital != self.initial_capital
                or row.maximum_fill_participation != self.maximum_fill_participation
                for row in self.environment_authorities
            )
            or self.source_data_qualified
            != bool(
                self.environment_authorities
                and all(
                    row.source_data_qualified for row in self.environment_authorities
                )
            )
            or not isinstance(self.initial_capital, float)
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0.0
            or not isinstance(self.maximum_fill_participation, float)
            or not 0.0 < self.maximum_fill_participation <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or not isinstance(self.runtime_environments_replayed, bool)
            or not isinstance(self.development_validation_environments_authorized, bool)
            or not isinstance(self.profitability_reporting_authorized, bool)
            or not isinstance(self.outer_evaluation_authorized, bool)
            or not isinstance(self.lockbox_access_authorized, bool)
            or self.runtime_environments_replayed != runtime_present
            or self.development_validation_environments_authorized
            != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment registry differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment registry source transaction differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("validation environment registry", value)
        _digest("validation environment registry", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def build_environments(
        self,
    ) -> Mapping[float, MassiveAdaptiveProfitabilityEnvV1]:
        """Create a fresh mutable environment for each registered cost rung."""

        self.validate()
        if (
            self._manifest is None
            or self._validation_sources is None
            or self._runtime_sources is None
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment registry runtime is absent"
            )
        result: dict[float, MassiveAdaptiveProfitabilityEnvV1] = {}
        for authority in self.environment_authorities:
            environment = _validation_environment(
                manifest=self._manifest,
                validation_sources=self._validation_sources,
                runtime_sources=self._runtime_sources,
                transaction_cost_basis_points=(authority.transaction_cost_basis_points),
            )
            authority.validate_environment(environment)
            result[authority.transaction_cost_basis_points] = environment
        if tuple(result) != self.cost_basis_points:
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment registry coverage differs"
            )
        return result

    def environment_authority(
        self, transaction_cost_basis_points: float
    ) -> MassiveAdaptiveRLValidationEnvironmentAuthorityV1:
        """Return the canonical authority for one registered cost rung."""

        self.validate()
        if (
            type(transaction_cost_basis_points) is not float
            or transaction_cost_basis_points not in self.cost_basis_points
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment cost rung is absent"
            )
        index = self.cost_basis_points.index(transaction_cost_basis_points)
        result = self.environment_authorities[index]
        if (
            result.transaction_cost_basis_points != transaction_cost_basis_points
            or result.semantic_receipt_sha256
            != self.environment_authority_receipts[index]
        ):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment authority differs"
            )
        return result


def _build_validation_environment_registry(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(validation_sources) is not MassiveAdaptiveRLValidationSourcesAuthorityV1
        or type(runtime_sources) is not MassiveAdaptiveRLRuntimeSourcesV1
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment registry root type differs"
        )
    manifest.validate()
    validation_sources.validate()
    runtime_sources.validate()
    fold_index = validation_sources.fold_index
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    economics = manifest.base_manifest.base_manifest
    if (
        runtime_receipt is None
        or not validation_sources.development_stage_authorized
        or validation_sources.experiment_id != manifest.experiment_id
        or validation_sources.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_sources.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
        or validation_sources.runtime_graph_witness_receipt_sha256 != runtime_receipt
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment registry lineage differs"
        )
    authorities = tuple(
        _validation_environment_authority(
            manifest=manifest,
            validation_sources=validation_sources,
            runtime_sources=runtime_sources,
            environment=_validation_environment(
                manifest=manifest,
                validation_sources=validation_sources,
                runtime_sources=runtime_sources,
                transaction_cost_basis_points=cost,
            ),
        )
        for cost in economics.cost_ladder_basis_points
    )
    context_receipts = {row.validation_context_receipt_sha256 for row in authorities}
    if len(context_receipts) != 1:
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation cost ladder does not share one context"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "validation_sources_authority_receipt_sha256": (
            validation_sources.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "fold_index": fold_index,
        "cost_basis_points": tuple(economics.cost_ladder_basis_points),
        "environment_authorities": authorities,
        "environment_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in authorities
        ),
        "environment_authority_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in authorities)
        ),
        "validation_context_receipt_sha256": next(iter(context_receipts)),
        "initial_capital": float(economics.primary_capital),
        "maximum_fill_participation": float(economics.maximum_fill_participation),
        "source_data_qualified": all(row.source_data_qualified for row in authorities),
    }
    provisional = MassiveAdaptiveRLValidationEnvironmentRegistryV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_environments_replayed=True,
        development_validation_environments_authorized=bool(
            body["source_data_qualified"]
        ),
        _manifest=manifest,
        _validation_sources=validation_sources,
        _runtime_sources=runtime_sources,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    # Avoid recursive runtime rebuilding while checking the constructor result.
    structural = replace(
        result,
        runtime_environments_replayed=False,
        development_validation_environments_authorized=False,
        _manifest=None,
        _validation_sources=None,
        _runtime_sources=None,
    )
    structural.validate()
    for authority, cost in zip(
        result.environment_authorities,
        result.cost_basis_points,
        strict=True,
    ):
        authority.validate_environment(
            _validation_environment(
                manifest=manifest,
                validation_sources=validation_sources,
                runtime_sources=runtime_sources,
                transaction_cost_basis_points=cost,
            )
        )
    return result


def _parse_validation_registry_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment registry is not canonical JSON"
        )
    body = dict(value)
    body["cost_basis_points"] = tuple(
        _json_float("validation cost", value)
        for value in cast(Sequence[object], body["cost_basis_points"])
    )
    body["environment_authority_receipts"] = tuple(
        cast(Sequence[str], body["environment_authority_receipts"])
    )
    authorities: list[MassiveAdaptiveRLValidationEnvironmentAuthorityV1] = []
    for value in cast(Sequence[object], body["environment_authorities"]):
        if not isinstance(value, Mapping):
            raise MassiveAdaptiveRLValidationInputsV1Error(
                "validation environment authority payload is malformed"
            )
        payload = dict(value)
        for name in (
            "transaction_cost_basis_points",
            "initial_capital",
            "maximum_fill_participation",
        ):
            payload[name] = _json_float(name, payload[name])
        authority = MassiveAdaptiveRLValidationEnvironmentAuthorityV1(
            **payload  # type: ignore[arg-type]
        )
        authority.validate()
        authorities.append(authority)
    body["environment_authorities"] = tuple(authorities)
    body["initial_capital"] = _json_float(
        "validation initial capital", body["initial_capital"]
    )
    body["maximum_fill_participation"] = _json_float(
        "validation maximum fill participation",
        body["maximum_fill_participation"],
    )
    return body


def parse_massive_adaptive_rl_validation_environment_registry_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    body = _parse_validation_registry_body(root=root, loaded_source=loaded_source)
    provisional = MassiveAdaptiveRLValidationEnvironmentRegistryV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_environments_replayed=False,
        development_validation_environments_authorized=False,
        _loaded_source=loaded_source,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_environment_registry_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    relative = validation_environment_registry_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    return parse_massive_adaptive_rl_validation_environment_registry_v1(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_validation_environment_registry_v1(
    *,
    root: str | Path,
    registry: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    del root  # All nested source transactions were replayed by their authorities.
    registry.validate()
    relative = validation_environment_registry_relative_path_v1(
        manifest=manifest,
        fold_index=registry.fold_index,
    )
    if registry._loaded_source is None or (
        registry._loaded_source.payload_relative_path != relative
    ):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment registry path differs"
        )
    expected = _build_validation_environment_registry(
        manifest=manifest,
        validation_sources=validation_sources,
        runtime_sources=runtime_sources,
    )
    if registry.semantic_unsigned() != expected.semantic_unsigned():
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "validation environment registry does not replay"
        )
    result = replace(
        registry,
        runtime_environments_replayed=True,
        development_validation_environments_authorized=(registry.source_data_qualified),
        _manifest=manifest,
        _validation_sources=validation_sources,
        _runtime_sources=runtime_sources,
    )
    # Runtime validation would recurse through the package-owned rebuild.  The
    # exact equality above and fresh-environment replay below are the witness.
    for authority, environment in zip(
        result.environment_authorities,
        result.build_environments().values(),
        strict=True,
    ):
        authority.validate_environment(environment)
    return result


def materialize_massive_adaptive_rl_validation_environment_registry_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    relative = validation_environment_registry_relative_path_v1(
        manifest=manifest,
        fold_index=validation_sources.fold_index,
    )
    if _source_transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationInputsV1Error(
            "canonical validation environment registry already exists"
        )
    built = _build_validation_environment_registry(
        manifest=manifest,
        validation_sources=validation_sources,
        runtime_sources=runtime_sources,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(built.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=(MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_DATASET),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=built.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-VALIDATION-ENVIRONMENT-REGISTRY-V1-"
            f"{manifest.experiment_id}-FOLD{validation_sources.fold_index}"
        ),
    )
    return authorize_massive_adaptive_rl_validation_environment_registry_v1(
        root=root,
        registry=load_massive_adaptive_rl_validation_environment_registry_v1(
            root=root,
            manifest=manifest,
            fold_index=validation_sources.fold_index,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        validation_sources=validation_sources,
        runtime_sources=runtime_sources,
    )


def prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
    """Create the canonical economic registry, or strictly replay it."""

    relative = validation_environment_registry_relative_path_v1(
        manifest=manifest,
        fold_index=validation_sources.fold_index,
    )
    if not _source_transaction_exists(root=root, relative=relative):
        return materialize_massive_adaptive_rl_validation_environment_registry_v1(
            root=root,
            manifest=manifest,
            validation_sources=validation_sources,
            runtime_sources=runtime_sources,
            committed_at_ms=committed_at_ms,
        )
    return authorize_massive_adaptive_rl_validation_environment_registry_v1(
        root=root,
        registry=load_massive_adaptive_rl_validation_environment_registry_v1(
            root=root,
            manifest=manifest,
            fold_index=validation_sources.fold_index,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        validation_sources=validation_sources,
        runtime_sources=runtime_sources,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLValidationEnvironmentAuthorityV1",
    "MassiveAdaptiveRLValidationEnvironmentRegistryV1",
    "MassiveAdaptiveRLValidationInputsV1Error",
    "MassiveAdaptiveRLValidationSourcesAuthorityV1",
    "authorize_massive_adaptive_rl_validation_environment_registry_v1",
    "authorize_massive_adaptive_rl_validation_sources_authority_v1",
    "load_massive_adaptive_rl_validation_environment_registry_v1",
    "load_massive_adaptive_rl_validation_sources_authority_v1",
    "materialize_massive_adaptive_rl_validation_environment_registry_v1",
    "materialize_massive_adaptive_rl_validation_sources_authority_v1",
    "parse_massive_adaptive_rl_validation_environment_registry_v1",
    "parse_massive_adaptive_rl_validation_sources_authority_v1",
    "prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1",
    "prepare_or_resume_massive_adaptive_rl_validation_sources_v1",
    "validation_decision_tensor_relative_path_v1",
    "validation_cost_ladder_artifact_id_v1",
    "validation_cost_ladder_relative_path_v1",
    "validation_environment_registry_relative_path_v1",
    "validation_fixed_control_artifact_id_v1",
    "validation_fixed_control_relative_path_v1",
    "validation_forecast_archive_relative_path_v1",
    "validation_generation_key_v1",
    "validation_primary_trace_artifact_id_v1",
    "validation_primary_trace_relative_path_v1",
    "validation_sources_authority_relative_path_v1",
]
