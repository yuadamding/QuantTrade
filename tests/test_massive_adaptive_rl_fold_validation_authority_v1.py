from __future__ import annotations

import inspect
from io import BytesIO

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256,
    materialize_massive_adaptive_rl_fold_validation_authority_v1,
    parse_massive_adaptive_rl_fold_validation_authority_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MassiveAdaptiveRLPolicySelectionV2Error,
    materialize_massive_adaptive_rl_policy_selection_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    build_massive_adaptive_rl_experiment_manifest_v4,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def test_generic_fold_validation_inventory_cannot_authorize_selection(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="generic-fold-validation"
    )
    checkpoint = _digest("checkpoint-authority")
    primary = _digest("primary-trace-authority")
    ladder = _digest("cost-ladder-authority")
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": 0,
        "fold_fit_authority_receipt_sha256": _digest("fold-fit"),
        "four_fold_fit_authority_receipt_sha256": _digest("four-fold-fit"),
        "validation_sources_authority_receipt_sha256": _digest("validation-sources"),
        "validation_environment_registry_receipt_sha256": _digest(
            "validation-environment-registry"
        ),
        "four_fold_validation_inputs_authority_receipt_sha256": _digest(
            "four-fold-validation-inputs"
        ),
        "four_fold_validation_inputs_source_receipt_sha256": _digest(
            "four-fold-validation-inputs-source"
        ),
        "four_fold_validation_inputs_commit_receipt_sha256": _digest(
            "four-fold-validation-inputs-commit"
        ),
        "four_fold_validation_inputs_committed_at_ms": 0,
        "chronology_authority_receipt_sha256": _digest("chronology"),
        "expected_checkpoint_authority_receipts": (checkpoint,),
        "primary_trace_authority_receipts": (primary,),
        "cost_ladder_authority_receipts": (ladder,),
        "fixed_control_validation_authority_receipt_sha256": _digest("fc06-validation"),
        "fixed_control_fit_authority_receipt_sha256": _digest("fc06-fit"),
        "fixed_control_selection_authority_receipt_sha256": _digest("fc06-selection"),
        "selected_fc06_action_receipt_sha256": _digest("fc06-action"),
        "validation_context_receipt_sha256": _digest("validation-context"),
        "validation_decision_session_dates": ("2024-01-02",),
        "forecast_archive_receipt_sha256": _digest("forecast"),
        "inference_plan_receipt_sha256": _digest("inference-plan"),
        "calibration_receipt_sha256": _digest("calibration"),
        "economic_source_inventory_sha256": _digest("economic-sources"),
        "initial_capital": 10_000_000.0,
        "validation_tape_receipt_sha256": _digest("validation-tape"),
        "candidate_evidence_inventory_sha256": semantic_sha256(
            ((checkpoint, primary, ladder),)
        ),
        "source_data_qualified": True,
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
    receipt = semantic_sha256(body)
    relative = "massive-adaptive/rl-fold-validation-authority-v1/generic.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=receipt,
        committed_at_ms=1,
        request_id="GENERIC-FOLD-VALIDATION",
    )
    loaded = load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path=relative,
        verified_at_ms=2,
    )

    generic = parse_massive_adaptive_rl_fold_validation_authority_v1(
        root=tmp_path,
        loaded_source=loaded,
    )

    assert generic.source_transaction_verified
    assert generic.source_data_qualified
    assert not generic.runtime_validation_replayed
    assert not generic.development_stage_authorized
    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV2Error,
        match="validation evidence is not authorized",
    ):
        materialize_massive_adaptive_rl_policy_selection_authority_v2(
            root=tmp_path,
            artifact_id="selection",
            manifest=manifest,
            validation_authority=generic,
            committed_at_ms=3,
        )


def test_fold_validation_materializer_accepts_evidence_not_candidate_metrics() -> None:
    parameters = inspect.signature(
        materialize_massive_adaptive_rl_fold_validation_authority_v1
    ).parameters
    assert "primary_trace_authorities" in parameters
    assert "cost_ladder_authorities" in parameters
    assert "fixed_control_validation_authority" in parameters
    assert "validation_sources_authority" in parameters
    assert "validation_environment_registry" in parameters
    assert "chronology_authority" not in parameters
    assert "artifact_id" not in parameters
    assert "candidates" not in parameters
    assert "metrics" not in parameters
