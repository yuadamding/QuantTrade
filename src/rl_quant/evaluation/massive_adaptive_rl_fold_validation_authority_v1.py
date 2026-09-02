"""Persist and replay the complete evidence set for one validation fold.

This authority is the only authorizing bridge from validation economics to
Manifest-V4 policy selection.  It accepts replayed, create-only trace
authorities; it never accepts candidate summaries or caller-provided metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    MassiveAdaptiveRLCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_validation_authority_v1 import (
    MassiveAdaptiveRLFixedControlValidationAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MassiveAdaptiveRLPolicyTraceAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    MassiveAdaptiveRLValidationSourcesAuthorityV1,
    validation_cost_ladder_relative_path_v1,
    validation_fixed_control_relative_path_v1,
    validation_generation_key_v1,
    validation_primary_trace_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)


MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-validation-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fold-validation-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
        "payload": "canonical-json-replayed-validation-evidence-inventory",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "candidate_population": "exact-fold-fit-checkpoint-authority-inventory",
        "primary_trace": "persisted-checkpoint-replayed-20bp",
        "cost_ladder": "persisted-primary-target-replay-10-20-40bp",
        "fixed_control": "persisted-fit-selected-fc06-20bp",
        "validation_sources": "single-canonical-create-only-authority-v1",
        "validation_environment": "single-canonical-create-only-registry-v1",
        "shared_tape": ("dates-forecast-plan-calibration-economic-sources-capital"),
        "caller_candidates": False,
        "caller_metrics": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)


class MassiveAdaptiveRLFoldValidationAuthorityV1Error(ValueError):
    """The persisted validation evidence graph differs or is incomplete."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def fold_validation_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    key = validation_generation_key_v1(manifest=manifest, fold_index=fold_index)
    return (
        f"massive-adaptive/rl-fold-validation-authority-v1/{key}-fold-validation.json"
    )


@dataclass(frozen=True, slots=True)
class _ValidationEvidenceFactsV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    fold_fit_authority_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str
    validation_environment_registry_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    expected_checkpoint_authority_receipts: tuple[str, ...]
    primary_trace_authority_receipts: tuple[str, ...]
    cost_ladder_authority_receipts: tuple[str, ...]
    fixed_control_validation_authority_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fc06_action_receipt_sha256: str
    validation_context_receipt_sha256: str
    validation_decision_session_dates: tuple[str, ...]
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    economic_source_inventory_sha256: str
    initial_capital: float
    validation_tape_receipt_sha256: str
    candidate_evidence_inventory_sha256: str
    source_data_qualified: bool


@dataclass(frozen=True, slots=True)
class _SharedValidationTapeFactsV1:
    fold_index: int
    training_forecast_authority_receipt_sha256: str
    validation_decision_session_dates: tuple[str, ...]
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    economic_source_inventory_sha256: str
    initial_capital: float
    validation_context_receipt_sha256: str
    validation_tape_receipt_sha256: str
    nested_source_data_qualified: bool


def _shared_validation_tape_facts_v1(
    *,
    primary: tuple[MassiveAdaptiveRLPolicyTraceAuthorityV1, ...],
    ladders: tuple[MassiveAdaptiveRLCostLadderAuthorityV1, ...],
    fixed: MassiveAdaptiveRLFixedControlValidationAuthorityV1,
) -> _SharedValidationTapeFactsV1:
    primary_runtime = tuple(row.runtime_trace for row in primary)
    ladder_runtime = tuple(row.runtime_ladder for row in ladders)
    fixed_runtime = fixed.runtime_evaluation
    if (
        not primary
        or len(primary) != len(ladders)
        or any(row is None for row in primary_runtime)
        or any(row is None for row in ladder_runtime)
        or fixed_runtime is None
        or any(not row.runtime_trace_replayed for row in primary)
        or any(not row.runtime_ladder_replayed for row in ladders)
        or not fixed.runtime_evaluation_replayed
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation evidence was not computationally replayed"
        )
    traces: list[MassiveAdaptiveRLPolicyTraceV1] = []
    for primary_authority, ladder_authority, trace_runtime, ladder in zip(
        primary,
        ladders,
        primary_runtime,
        ladder_runtime,
        strict=True,
    ):
        assert trace_runtime is not None
        assert ladder is not None
        if (
            primary_authority.evaluation_role != "inner_validation"
            or ladder_authority.evaluation_role != "inner_validation"
            or trace_runtime != ladder.primary
            or primary_authority.policy_trace_receipt_sha256
            != ladder_authority.primary_trace_receipt_sha256
        ):
            raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
                "fold-validation primary trace and cost ladder differ"
            )
        traces.extend(
            (
                ladder.primary.policy_trace,
                ladder.low_cost_trace,
                ladder.high_cost_trace,
            )
        )
    assert fixed_runtime is not None
    fixed_trace = fixed_runtime.policy_trace
    trace_rows = tuple(traces)
    context_receipts = {
        *(row.validation_context_receipt_sha256 for row in primary),
        *(row.validation_context_receipt_sha256 for row in ladders),
        fixed.validation_context_receipt_sha256,
    }
    common = {
        (
            row.fold_index,
            row.training_forecast_authority_receipt_sha256,
            row.decision_session_dates,
            row.forecast_archive_receipt_sha256,
            row.inference_plan_receipt_sha256,
            row.calibration_receipt_sha256,
            row.initial_capital,
        )
        for row in (*trace_rows, fixed_trace)
    }
    if (
        len(common) != 1
        or len(context_receipts) != 1
        or any(row.evaluation_role != "inner_validation" for row in trace_rows)
        or fixed_trace.evaluation_role != "inner_validation"
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation traces do not share one economic tape"
        )
    (
        fold_index,
        training_forecast_receipt,
        dates,
        forecast_receipt,
        inference_receipt,
        calibration_receipt,
        initial_capital,
    ) = next(iter(common))
    context_receipt = next(iter(context_receipts))
    economic_inventory = semantic_sha256(
        tuple(
            row.economic_source_inventory_sha256 for row in (*trace_rows, fixed_trace)
        )
    )
    tape_receipt = semantic_sha256(
        (
            dates,
            forecast_receipt,
            inference_receipt,
            calibration_receipt,
            initial_capital,
            context_receipt,
        )
    )
    return _SharedValidationTapeFactsV1(
        fold_index=fold_index,
        training_forecast_authority_receipt_sha256=training_forecast_receipt,
        validation_decision_session_dates=dates,
        forecast_archive_receipt_sha256=forecast_receipt,
        inference_plan_receipt_sha256=inference_receipt,
        calibration_receipt_sha256=calibration_receipt,
        economic_source_inventory_sha256=economic_inventory,
        initial_capital=initial_capital,
        validation_context_receipt_sha256=context_receipt,
        validation_tape_receipt_sha256=tape_receipt,
        nested_source_data_qualified=bool(
            all(row.development_policy_evaluation_authorized for row in primary)
            and all(row.development_policy_selection_authorized for row in ladders)
            and fixed.development_stage_authorized
            and all(row.source_data_qualified for row in (*trace_rows, fixed_trace))
        ),
    )


def _validate_canonical_environment_bindings_v1(
    *,
    primary: tuple[MassiveAdaptiveRLPolicyTraceAuthorityV1, ...],
    ladders: tuple[MassiveAdaptiveRLCostLadderAuthorityV1, ...],
    fixed: MassiveAdaptiveRLFixedControlValidationAuthorityV1,
    registry: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
) -> None:
    environment_by_cost = {
        row.transaction_cost_basis_points: row
        for row in registry.environment_authorities
    }
    primary_environment = environment_by_cost[20.0]
    environment_authorities = (
        environment_by_cost[10.0],
        primary_environment,
        environment_by_cost[40.0],
    )
    environment_authority_receipts = tuple(
        row.semantic_receipt_sha256 for row in environment_authorities
    )
    environment_source_inventories = tuple(
        row.environment_source_inventory_sha256 for row in environment_authorities
    )
    economic_compatibility_receipts = tuple(
        row.economic_compatibility_receipt_sha256 for row in environment_authorities
    )
    registry_receipt = registry.semantic_receipt_sha256
    registry_source_receipt = registry.source_receipt_sha256
    registry_commit_receipt = registry.source_transaction_receipt_sha256
    registry_committed_at_ms = registry.source_transaction_committed_at_ms
    if (
        registry_source_receipt is None
        or registry_commit_receipt is None
        or registry_committed_at_ms is None
        or any(
            not row.runtime_validation_environment_replayed
            or not row.runtime_validation_environment_registry_replayed
            or row.validation_sources_authority_receipt_sha256
            != registry.validation_sources_authority_receipt_sha256
            or row.validation_environment_registry_receipt_sha256
            != registry_receipt
            or row.validation_environment_registry_source_receipt_sha256
            != registry_source_receipt
            or row.validation_environment_registry_commit_receipt_sha256
            != registry_commit_receipt
            or row.validation_environment_registry_committed_at_ms
            != registry_committed_at_ms
            or row.validation_environment_authority_receipt_sha256
            != primary_environment.semantic_receipt_sha256
            or row.environment_source_inventory_sha256
            != primary_environment.environment_source_inventory_sha256
            or row.economic_compatibility_receipt_sha256
            != primary_environment.economic_compatibility_receipt_sha256
            for row in primary
        )
        or any(
            not row.runtime_validation_environments_replayed
            or not row.runtime_validation_environment_registry_replayed
            or row.validation_sources_authority_receipt_sha256
            != registry.validation_sources_authority_receipt_sha256
            or row.validation_environment_registry_receipt_sha256
            != registry_receipt
            or row.validation_environment_registry_source_receipt_sha256
            != registry_source_receipt
            or row.validation_environment_registry_commit_receipt_sha256
            != registry_commit_receipt
            or row.validation_environment_registry_committed_at_ms
            != registry_committed_at_ms
            or row.validation_environment_authority_receipts
            != environment_authority_receipts
            or row.environment_source_inventory_sha256s
            != environment_source_inventories
            or row.economic_compatibility_receipt_sha256s
            != economic_compatibility_receipts
            for row in ladders
        )
        or not fixed.runtime_validation_environment_replayed
        or not fixed.runtime_validation_environment_registry_replayed
        or fixed.validation_sources_authority_receipt_sha256
        != registry.validation_sources_authority_receipt_sha256
        or fixed.validation_environment_registry_receipt_sha256 != registry_receipt
        or fixed.validation_environment_registry_source_receipt_sha256
        != registry_source_receipt
        or fixed.validation_environment_registry_commit_receipt_sha256
        != registry_commit_receipt
        or fixed.validation_environment_registry_committed_at_ms
        != registry_committed_at_ms
        or fixed.validation_environment_authority_receipt_sha256
        != primary_environment.semantic_receipt_sha256
        or fixed.environment_source_inventory_sha256
        != primary_environment.environment_source_inventory_sha256
        or fixed.economic_compatibility_receipt_sha256
        != primary_environment.economic_compatibility_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation traces do not match canonical environments"
        )


def validate_massive_adaptive_rl_shared_validation_tape_v1(
    *,
    primary_trace_authorities: Sequence[MassiveAdaptiveRLPolicyTraceAuthorityV1],
    cost_ladder_authorities: Sequence[MassiveAdaptiveRLCostLadderAuthorityV1],
    fixed_control_validation_authority: (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ),
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = None,
) -> str:
    """Return the tape receipt only for exact, replayed, shared trace evidence."""

    primary = tuple(primary_trace_authorities)
    ladders = tuple(cost_ladder_authorities)
    if any(type(row) is not MassiveAdaptiveRLPolicyTraceAuthorityV1 for row in primary):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation primary trace authority type differs"
        )
    if any(type(row) is not MassiveAdaptiveRLCostLadderAuthorityV1 for row in ladders):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation cost-ladder authority type differs"
        )
    if type(fixed_control_validation_authority) is not (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation FC06 authority type differs"
        )
    for primary_row in primary:
        primary_row.validate()
    for ladder_row in ladders:
        ladder_row.validate()
    fixed_control_validation_authority.validate()
    if validation_environment_registry is not None:
        if type(validation_environment_registry) is not (
            MassiveAdaptiveRLValidationEnvironmentRegistryV1
        ):
            raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
                "fold-validation environment registry type differs"
            )
        validation_environment_registry.validate()
        _validate_canonical_environment_bindings_v1(
            primary=primary,
            ladders=ladders,
            fixed=fixed_control_validation_authority,
            registry=validation_environment_registry,
        )
    return _shared_validation_tape_facts_v1(
        primary=primary,
        ladders=ladders,
        fixed=fixed_control_validation_authority,
    ).validation_tape_receipt_sha256


def _validation_evidence_facts_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1,
    validation_sources_authority: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    validation_environment_registry: (MassiveAdaptiveRLValidationEnvironmentRegistryV1),
    primary_trace_authorities: Sequence[MassiveAdaptiveRLPolicyTraceAuthorityV1],
    cost_ladder_authorities: Sequence[MassiveAdaptiveRLCostLadderAuthorityV1],
    fixed_control_validation_authority: (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ),
) -> _ValidationEvidenceFactsV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(fold_fit_authority) is not MassiveAdaptiveRLFoldFitAuthorityV1
        or type(validation_sources_authority)
        is not MassiveAdaptiveRLValidationSourcesAuthorityV1
        or type(validation_environment_registry)
        is not MassiveAdaptiveRLValidationEnvironmentRegistryV1
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation root authority type differs"
        )
    primary = tuple(primary_trace_authorities)
    ladders = tuple(cost_ladder_authorities)
    if any(type(row) is not MassiveAdaptiveRLPolicyTraceAuthorityV1 for row in primary):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation primary trace authority type differs"
        )
    if any(type(row) is not MassiveAdaptiveRLCostLadderAuthorityV1 for row in ladders):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation cost-ladder authority type differs"
        )
    if type(fixed_control_validation_authority) is not (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation FC06 authority type differs"
        )
    manifest.validate()
    fold_fit_authority.validate()
    validation_sources_authority.validate()
    validation_environment_registry.validate()
    chronology_authority = validation_sources_authority.runtime_chronology_authority
    chronology_authority.validate()
    for primary_row in primary:
        primary_row.validate()
    for ladder_row in ladders:
        ladder_row.validate()
    fixed_control_validation_authority.validate()
    expected = fold_fit_authority.candidate_checkpoint_authority_receipts
    if (
        manifest.base_manifest.semantic_receipt_sha256
        != fold_fit_authority.manifest_v3_receipt_sha256
        or manifest.experiment_id != fold_fit_authority.experiment_id
        or not validation_sources_authority.development_stage_authorized
        or not validation_environment_registry.development_stage_authorized
        or validation_sources_authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_sources_authority.fold_fit_authority_receipt_sha256
        != fold_fit_authority.semantic_receipt_sha256
        or validation_environment_registry.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_environment_registry.validation_sources_authority_receipt_sha256
        != validation_sources_authority.semantic_receipt_sha256
        or validation_environment_registry.fold_index
        != fold_fit_authority.outer_fold_index
        or chronology_authority.fold_index != fold_fit_authority.outer_fold_index
        or chronology_authority.training_forecast_authority_receipt_sha256
        != fold_fit_authority.training_forecast_authority.semantic_receipt_sha256
        or len(expected) != fold_fit_authority.outer_fold_index + 1
        or len(primary) != len(expected)
        or len(ladders) != len(expected)
        or tuple(row.checkpoint_authority_receipt_sha256 for row in primary) != expected
        or tuple(row.checkpoint_authority_receipt_sha256 for row in ladders) != expected
        or any(
            row.loaded_source.payload_relative_path
            != validation_primary_trace_relative_path_v1(
                manifest=manifest,
                fold_index=fold_fit_authority.outer_fold_index,
                checkpoint_authority_receipt_sha256=(
                    row.checkpoint_authority_receipt_sha256
                ),
            )
            for row in primary
        )
        or any(
            row.loaded_source.payload_relative_path
            != validation_cost_ladder_relative_path_v1(
                manifest=manifest,
                fold_index=fold_fit_authority.outer_fold_index,
                checkpoint_authority_receipt_sha256=(
                    row.checkpoint_authority_receipt_sha256
                ),
            )
            for row in ladders
        )
        or fixed_control_validation_authority.loaded_source.payload_relative_path
        != validation_fixed_control_relative_path_v1(
            manifest=manifest,
            fold_index=fold_fit_authority.outer_fold_index,
        )
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation candidate coverage or training lineage differs"
        )
    shared = _shared_validation_tape_facts_v1(
        primary=primary,
        ladders=ladders,
        fixed=fixed_control_validation_authority,
    )
    fixed_runtime = fixed_control_validation_authority.runtime_evaluation
    assert fixed_runtime is not None
    trace_values: list[MassiveAdaptiveRLPolicyTraceV1] = []
    for authority in ladders:
        ladder = authority.runtime_ladder
        assert ladder is not None
        trace_values.extend(
            (
                ladder.primary.policy_trace,
                ladder.low_cost_trace,
                ladder.high_cost_trace,
            )
        )
    trace_rows = tuple(trace_values)
    _validate_canonical_environment_bindings_v1(
        primary=primary,
        ladders=ladders,
        fixed=fixed_control_validation_authority,
        registry=validation_environment_registry,
    )
    if (
        shared.fold_index != fold_fit_authority.outer_fold_index
        or fixed_runtime.fixed_control_fit_authority_receipt_sha256
        != fold_fit_authority.fixed_control_fit_authority_receipt_sha256
        or fixed_runtime.fixed_control_selection_authority_receipt_sha256
        != fold_fit_authority.fixed_control_selection_authority_receipt_sha256
        or shared.training_forecast_authority_receipt_sha256
        != fold_fit_authority.training_forecast_authority.semantic_receipt_sha256
        or shared.validation_decision_session_dates
        != chronology_authority.rl_validation_origin_dates
        or shared.inference_plan_receipt_sha256
        != chronology_authority.validation_inference_plan_receipt_sha256
        or shared.validation_decision_session_dates
        != validation_sources_authority.validation_decision_session_dates
        or shared.forecast_archive_receipt_sha256
        != validation_sources_authority.validation_forecast_archive_receipt_sha256
        or shared.inference_plan_receipt_sha256
        != validation_sources_authority.validation_inference_plan_receipt_sha256
        or shared.calibration_receipt_sha256
        != validation_sources_authority.calibration_receipt_sha256
        or shared.validation_context_receipt_sha256
        != validation_environment_registry.validation_context_receipt_sha256
        or shared.initial_capital != validation_environment_registry.initial_capital
        or len(trace_rows) != 3 * len(ladders)
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation traces do not share one economic tape"
        )
    primary_receipts = tuple(row.semantic_receipt_sha256 for row in primary)
    ladder_receipts = tuple(row.semantic_receipt_sha256 for row in ladders)
    candidate_inventory = semantic_sha256(
        tuple(zip(expected, primary_receipts, ladder_receipts, strict=True))
    )
    source_qualified = bool(
        fold_fit_authority.development_stage_authorized
        and validation_sources_authority.development_stage_authorized
        and validation_environment_registry.development_stage_authorized
        and chronology_authority.development_policy_selection_authorized
        and shared.nested_source_data_qualified
    )
    return _ValidationEvidenceFactsV1(
        experiment_id=manifest.experiment_id,
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        fold_index=fold_fit_authority.outer_fold_index,
        fold_fit_authority_receipt_sha256=(fold_fit_authority.semantic_receipt_sha256),
        four_fold_fit_authority_receipt_sha256=(
            validation_sources_authority.four_fold_fit_authority_receipt_sha256
        ),
        validation_sources_authority_receipt_sha256=(
            validation_sources_authority.semantic_receipt_sha256
        ),
        validation_environment_registry_receipt_sha256=(
            validation_environment_registry.semantic_receipt_sha256
        ),
        chronology_authority_receipt_sha256=(
            chronology_authority.semantic_receipt_sha256
        ),
        expected_checkpoint_authority_receipts=expected,
        primary_trace_authority_receipts=primary_receipts,
        cost_ladder_authority_receipts=ladder_receipts,
        fixed_control_validation_authority_receipt_sha256=(
            fixed_control_validation_authority.semantic_receipt_sha256
        ),
        fixed_control_fit_authority_receipt_sha256=(
            fixed_runtime.fixed_control_fit_authority_receipt_sha256
        ),
        fixed_control_selection_authority_receipt_sha256=(
            fixed_runtime.fixed_control_selection_authority_receipt_sha256
        ),
        selected_fc06_action_receipt_sha256=(
            fixed_runtime.selected_action_receipt_sha256
        ),
        validation_context_receipt_sha256=(shared.validation_context_receipt_sha256),
        validation_decision_session_dates=(shared.validation_decision_session_dates),
        forecast_archive_receipt_sha256=(shared.forecast_archive_receipt_sha256),
        inference_plan_receipt_sha256=shared.inference_plan_receipt_sha256,
        calibration_receipt_sha256=shared.calibration_receipt_sha256,
        economic_source_inventory_sha256=(shared.economic_source_inventory_sha256),
        initial_capital=shared.initial_capital,
        validation_tape_receipt_sha256=shared.validation_tape_receipt_sha256,
        candidate_evidence_inventory_sha256=candidate_inventory,
        source_data_qualified=source_qualified,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldValidationAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    fold_fit_authority_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str
    validation_environment_registry_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    expected_checkpoint_authority_receipts: tuple[str, ...]
    primary_trace_authority_receipts: tuple[str, ...]
    cost_ladder_authority_receipts: tuple[str, ...]
    fixed_control_validation_authority_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fc06_action_receipt_sha256: str
    validation_context_receipt_sha256: str
    validation_decision_session_dates: tuple[str, ...]
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    economic_source_inventory_sha256: str
    initial_capital: float
    validation_tape_receipt_sha256: str
    candidate_evidence_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    runtime_fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    runtime_validation_sources_authority: (
        MassiveAdaptiveRLValidationSourcesAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    runtime_validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = field(
        default=None,
        compare=False,
        repr=False,
    )
    runtime_primary_trace_authorities: (
        tuple[MassiveAdaptiveRLPolicyTraceAuthorityV1, ...] | None
    ) = field(default=None, compare=False, repr=False)
    runtime_cost_ladder_authorities: (
        tuple[MassiveAdaptiveRLCostLadderAuthorityV1, ...] | None
    ) = field(default=None, compare=False, repr=False)
    runtime_fixed_control_validation_authority: (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    runtime_validation_replayed: bool = False
    development_validation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("runtime_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "loaded_source",
                "development_validation_authorized",
            }
        }

    @property
    def source_transaction_verified(self) -> bool:
        return True

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_validation_replayed
            and self.development_validation_authorized
            and self.source_data_qualified
        )

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime_values = (
            self.runtime_manifest,
            self.runtime_fold_fit_authority,
            self.runtime_validation_sources_authority,
            self.runtime_validation_environment_registry,
            self.runtime_primary_trace_authorities,
            self.runtime_cost_ladder_authorities,
            self.runtime_fixed_control_validation_authority,
        )
        runtime = all(value is not None for value in runtime_values)
        if any(value is not None for value in runtime_values) != runtime:
            raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
                "fold-validation runtime is partial"
            )
        if runtime:
            assert self.runtime_manifest is not None
            assert self.runtime_fold_fit_authority is not None
            assert self.runtime_validation_sources_authority is not None
            assert self.runtime_validation_environment_registry is not None
            assert self.runtime_primary_trace_authorities is not None
            assert self.runtime_cost_ladder_authorities is not None
            assert self.runtime_fixed_control_validation_authority is not None
            facts = _validation_evidence_facts_v1(
                manifest=self.runtime_manifest,
                fold_fit_authority=self.runtime_fold_fit_authority,
                validation_sources_authority=(
                    self.runtime_validation_sources_authority
                ),
                validation_environment_registry=(
                    self.runtime_validation_environment_registry
                ),
                primary_trace_authorities=self.runtime_primary_trace_authorities,
                cost_ladder_authorities=self.runtime_cost_ladder_authorities,
                fixed_control_validation_authority=(
                    self.runtime_fixed_control_validation_authority
                ),
            )
            if self.semantic_unsigned() != _authority_body(facts=facts):
                raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
                    "fold-validation runtime evidence differs"
                )
        expected_authorized = bool(runtime and self.source_data_qualified)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.expected_checkpoint_authority_receipts) != self.fold_index + 1
            or len(self.primary_trace_authority_receipts)
            != len(self.expected_checkpoint_authority_receipts)
            or len(self.cost_ladder_authority_receipts)
            != len(self.expected_checkpoint_authority_receipts)
            or not self.validation_decision_session_dates
            or self.validation_decision_session_dates
            != tuple(sorted(set(self.validation_decision_session_dates)))
            or not isinstance(self.initial_capital, float)
            or self.initial_capital <= 0.0
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_validation_replayed != runtime
            or self.development_validation_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
                "fold-validation authority differs"
            )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.training_manifest_v3_receipt_sha256,
            self.fold_fit_authority_receipt_sha256,
            self.four_fold_fit_authority_receipt_sha256,
            self.validation_sources_authority_receipt_sha256,
            self.validation_environment_registry_receipt_sha256,
            self.chronology_authority_receipt_sha256,
            *self.expected_checkpoint_authority_receipts,
            *self.primary_trace_authority_receipts,
            *self.cost_ladder_authority_receipts,
            self.fixed_control_validation_authority_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_fc06_action_receipt_sha256,
            self.validation_context_receipt_sha256,
            self.forecast_archive_receipt_sha256,
            self.inference_plan_receipt_sha256,
            self.calibration_receipt_sha256,
            self.economic_source_inventory_sha256,
            self.validation_tape_receipt_sha256,
            self.candidate_evidence_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("fold-validation authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _authority_body(*, facts: _ValidationEvidenceFactsV1) -> dict[str, object]:
    return {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
        **{
            descriptor.name: getattr(facts, descriptor.name)
            for descriptor in fields(facts)
        },
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
        ),
    }


def _load_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation payload is not canonical JSON"
        )
    body = dict(value)
    for name in (
        "expected_checkpoint_authority_receipts",
        "primary_trace_authority_receipts",
        "cost_ladder_authority_receipts",
        "validation_decision_session_dates",
    ):
        body[name] = tuple(body[name])
    return body


def parse_massive_adaptive_rl_fold_validation_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFoldValidationAuthorityV1:
    body = _load_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLFoldValidationAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded_source,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fold_validation_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFoldValidationAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1,
    validation_sources_authority: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    validation_environment_registry: (MassiveAdaptiveRLValidationEnvironmentRegistryV1),
    primary_trace_authorities: Sequence[MassiveAdaptiveRLPolicyTraceAuthorityV1],
    cost_ladder_authorities: Sequence[MassiveAdaptiveRLCostLadderAuthorityV1],
    fixed_control_validation_authority: (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ),
) -> MassiveAdaptiveRLFoldValidationAuthorityV1:
    parsed = parse_massive_adaptive_rl_fold_validation_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    expected_relative = fold_validation_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_fit_authority.outer_fold_index,
    )
    if authority.loaded_source.payload_relative_path != expected_relative:
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation authority path differs"
        )
    primary = tuple(primary_trace_authorities)
    ladders = tuple(cost_ladder_authorities)
    facts = _validation_evidence_facts_v1(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        validation_sources_authority=validation_sources_authority,
        validation_environment_registry=validation_environment_registry,
        primary_trace_authorities=primary,
        cost_ladder_authorities=ladders,
        fixed_control_validation_authority=fixed_control_validation_authority,
    )
    if parsed.semantic_unsigned() != _authority_body(facts=facts):
        raise MassiveAdaptiveRLFoldValidationAuthorityV1Error(
            "fold-validation evidence does not replay"
        )
    result = replace(
        parsed,
        runtime_manifest=manifest,
        runtime_fold_fit_authority=fold_fit_authority,
        runtime_validation_sources_authority=validation_sources_authority,
        runtime_validation_environment_registry=validation_environment_registry,
        runtime_primary_trace_authorities=primary,
        runtime_cost_ladder_authorities=ladders,
        runtime_fixed_control_validation_authority=(fixed_control_validation_authority),
        runtime_validation_replayed=True,
        development_validation_authorized=facts.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fold_validation_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1,
    validation_sources_authority: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    validation_environment_registry: (MassiveAdaptiveRLValidationEnvironmentRegistryV1),
    primary_trace_authorities: Sequence[MassiveAdaptiveRLPolicyTraceAuthorityV1],
    cost_ladder_authorities: Sequence[MassiveAdaptiveRLCostLadderAuthorityV1],
    fixed_control_validation_authority: (
        MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ),
    committed_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationAuthorityV1:
    primary = tuple(primary_trace_authorities)
    ladders = tuple(cost_ladder_authorities)
    facts = _validation_evidence_facts_v1(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        validation_sources_authority=validation_sources_authority,
        validation_environment_registry=validation_environment_registry,
        primary_trace_authorities=primary,
        cost_ladder_authorities=ladders,
        fixed_control_validation_authority=fixed_control_validation_authority,
    )
    body = _authority_body(facts=facts)
    receipt = semantic_sha256(body)
    relative = fold_validation_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_fit_authority.outer_fold_index,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=receipt,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-FOLD-VALIDATION-V1-{manifest.experiment_id}-"
            f"FOLD{fold_fit_authority.outer_fold_index}"
        ),
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fold_validation_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_fold_validation_authority_v1(
            root=root, loaded_source=loaded
        ),
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        validation_sources_authority=validation_sources_authority,
        validation_environment_registry=validation_environment_registry,
        primary_trace_authorities=primary,
        cost_ladder_authorities=ladders,
        fixed_control_validation_authority=fixed_control_validation_authority,
    )


__all__ = [
    "MassiveAdaptiveRLFoldValidationAuthorityV1",
    "MassiveAdaptiveRLFoldValidationAuthorityV1Error",
    "authorize_massive_adaptive_rl_fold_validation_authority_v1",
    "fold_validation_authority_relative_path_v1",
    "materialize_massive_adaptive_rl_fold_validation_authority_v1",
    "parse_massive_adaptive_rl_fold_validation_authority_v1",
    "validate_massive_adaptive_rl_shared_validation_tape_v1",
]
