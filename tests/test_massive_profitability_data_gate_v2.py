from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA,
    MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256,
    MassiveProfitabilityDataGateV2,
    MassiveProfitabilityDataGateV2Error,
    build_massive_profitability_data_gate_v2,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_lockbox_target_seal_v1 import (
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_DATASET,
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SPEC_SHA256,
    parse_massive_profitability_lockbox_target_seal_v1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

_ENTITLEMENT = "e" * 64
_INPUT_SCHEMAS = tuple(
    sorted(
        (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        )
    )
)


def _nonpassing_gate() -> MassiveProfitabilityDataGateV2:
    digest = semantic_sha256("gate-v2-test")
    provisional = MassiveProfitabilityDataGateV2(
        coverage_semantic_receipt_sha256=digest,
        archive_freeze_semantic_receipt_sha256=digest,
        accounting_freeze_semantic_receipt_sha256=digest,
        origin_plan_semantic_receipt_sha256=digest,
        daily_input_authority_semantic_receipt_sha256=digest,
        fill_source_authority_semantic_receipt_sha256=digest,
        terminal_authority_semantic_receipt_sha256=digest,
        lockbox_seal_semantic_receipt_sha256=digest,
        gated_session_dates=(),
        outer_test_session_dates=(),
        excluded_lockbox_session_dates=(),
        feature_receipts=(),
        target_receipts=(),
        date_support_gates=(),
        input_schemas=_INPUT_SCHEMAS,
        source_transport_qualified=False,
        rank_bar_data_qualified=False,
        exact_frozen_acquisition_complete=False,
        exact_accounting_freeze_complete=False,
        exact_origin_plan_membership_complete=False,
        exact_feature_cutoff_complete=False,
        exact_source_staleness_complete=False,
        exact_64_session_rectangles_complete=False,
        fill_source_complete=False,
        economic_accounting_data_qualified=False,
        terminal_accounting_complete=False,
        common_model_support_complete=False,
        package_rematerialization_complete=False,
        future_mutation_invariance_complete=False,
        lockbox_targets_sealed_and_excluded=False,
        legacy_generations_rejected=True,
        data_gate_passed=False,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        component_audit_inventory_sha256=digest,
        audit_receipt_sha256="0" * 64,
        development_training_authorized=False,
        outer_prediction_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
        schema=MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "component_audit_inventory_sha256": digest,
            }
        ),
    )
    result.validate()
    return result


def test_data_gate_v2_has_no_caller_attestation_parameters() -> None:
    parameters = inspect.signature(build_massive_profitability_data_gate_v2).parameters
    assert "rematerialized_feature_receipts" not in parameters
    assert "rematerialized_target_receipts" not in parameters
    assert "future_mutation_results" not in parameters
    assert "component_results" not in parameters
    assert _nonpassing_gate().development_training_authorized is False


def test_data_gate_v2_requires_the_exact_generation_inventory() -> None:
    gate = _nonpassing_gate()
    changed = replace(gate, input_schemas=gate.input_schemas[:-1])
    semantic = semantic_sha256(changed.semantic_unsigned())
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "component_audit_inventory_sha256": (
                    changed.component_audit_inventory_sha256
                ),
            }
        ),
    )
    with pytest.raises(
        MassiveProfitabilityDataGateV2Error,
        match="identity, dates, or authorization differs",
    ):
        changed.validate()


def test_public_lockbox_artifact_contains_commitments_only(tmp_path: Path) -> None:
    dates = tuple(
        (date(2020, 1, 1) + timedelta(days=offset)).isoformat()
        for offset in range(252)
    )
    targets = tuple(semantic_sha256(("lockbox-target", day)) for day in dates)
    payload = {
        "schema": MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        "lockbox_session_dates": dates,
        "archive_freeze_semantic_receipt_sha256": semantic_sha256("archive"),
        "accounting_freeze_semantic_receipt_sha256": semantic_sha256("accounting"),
        "target_semantic_receipts": targets,
        "target_inventory_sha256": semantic_sha256(targets),
        "sealed_blob_physical_sha256": semantic_sha256("physical"),
        "sealed_blob_source_receipt_sha256": semantic_sha256("source"),
        "sealed_blob_commit_receipt_sha256": semantic_sha256("commit"),
        "sealed_target_count": 252,
        "public_commitment_only": True,
        "separate_permission_root_verified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SHA256
        ),
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    relative = "massive-profitability-lockbox-target-seal-v1/test.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_ENTITLEMENT,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    seal = parse_massive_profitability_lockbox_target_seal_v1(
        root=tmp_path, loaded_source=loaded
    )
    raw = read_loaded_massive_source_bytes(root=tmp_path, loaded_source=loaded)
    assert seal.public_commitment_only is True
    assert seal.lockbox_access_authorized is False
    assert b"simple_returns" not in raw
    assert b'"rows"' not in raw
