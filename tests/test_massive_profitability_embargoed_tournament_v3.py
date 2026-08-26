from __future__ import annotations

import inspect
from dataclasses import asdict, replace
from datetime import date, timedelta
from io import BytesIO

import pytest
import torch

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v3 import (
    materialize_massive_profitability_evaluation_plan_v3,
)
from rl_quant.evaluation.massive_profitability_evaluation_source_bundle_v4 import (
    MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_DATASET,
    MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA,
    MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SPEC_SHA256,
    MassiveProfitabilityEvaluationSourceDateV4,
    parse_massive_profitability_evaluation_source_bundle_v4,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v3 import (
    materialize_massive_profitability_outer_evidence_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    adapt_massive_profitability_training_fold_v2,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    build_massive_profitability_date_tensor_v1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentDatasetV2,
    MassiveProfitabilityTournamentV2Error,
)


def _dates(count: int) -> tuple[str, ...]:
    start = date(2010, 1, 1)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(count))


def test_training_fold_v2_preserves_the_embargoed_phase_receipt() -> None:
    dates = _dates(1_134)
    body = {
        "fold_index": 0,
        "fit_session_dates": dates[:756],
        "inner_purge_session_dates": dates[756:819],
        "inner_validation_session_dates": dates[819:945],
        "outer_purge_session_dates": dates[945:1008],
        "outer_test_session_dates": dates[1008:1134],
        "fit_inventory_sha256": semantic_sha256(dates[:756]),
        "inner_validation_inventory_sha256": semantic_sha256(dates[819:945]),
        "outer_test_inventory_sha256": semantic_sha256(dates[1008:1134]),
    }
    source = MassiveProfitabilityOuterFoldPlanV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    source.validate()
    adapted = adapt_massive_profitability_training_fold_v2(source)

    assert asdict(adapted) == asdict(source)
    assert adapted.receipt_sha256 == source.receipt_sha256


def test_dataset_v2_marks_embargo_rows_maturation_only() -> None:
    dates = _dates(3)
    rows = tuple(
        build_massive_profitability_date_tensor_v1(
            decision_session_date=session_date,
            security_ids=("SEC0", "SEC1"),
            bars_values=torch.zeros((2, 19)),
            bars_valid=torch.ones((2, 19), dtype=torch.bool),
            tape_values=torch.zeros((2, 15)),
            tape_valid=torch.ones((2, 15), dtype=torch.bool),
            target_values=torch.zeros((2, 4)),
            target_valid=torch.ones((2, 4), dtype=torch.bool),
            feature_semantic_receipt_sha256=semantic_sha256((session_date, "feature")),
            target_semantic_receipt_sha256=semantic_sha256((session_date, "target")),
        )
        for session_date in dates
    )
    body = {
        "dates": tuple(row.source_array_sha256 for row in rows),
        "data_gate": "a" * 64,
        "phase_plan": "b" * 64,
        "entry_session_dates": dates[:2],
        "maturation_only_session_dates": dates[2:],
    }
    dataset = MassiveProfitabilityTournamentDatasetV2(
        dates=rows,
        data_gate_semantic_receipt_sha256="a" * 64,
        phase_plan_semantic_receipt_sha256="b" * 64,
        entry_session_dates=dates[:2],
        maturation_only_session_dates=dates[2:],
        dataset_receipt_sha256=semantic_sha256(body),
    )
    dataset.validate()

    with pytest.raises(MassiveProfitabilityTournamentV2Error):
        replace(dataset, entry_session_dates=dates).validate()


def test_v3_entry_points_require_new_tournament_and_source_bundle_roots() -> None:
    plan = inspect.signature(materialize_massive_profitability_evaluation_plan_v3)
    evidence = inspect.signature(materialize_massive_profitability_outer_evidence_v3)
    assert "tournament_plan" in plan.parameters
    assert "runtime_sources" in evidence.parameters
    assert "evaluation_source_bundle" in evidence.parameters
    assert "features" not in evidence.parameters
    assert "target_accounting" not in evidence.parameters


def test_generic_source_bundle_v4_reload_is_nonauthorizing(tmp_path) -> None:
    digest = "a" * 64
    row_body = {
        "decision_session_date": "2024-01-02",
        "origin_receipt_sha256": digest,
        "feature_semantic_receipt_sha256": digest,
        "target_semantic_receipt_sha256": digest,
        "feature_accounting_semantic_receipt_sha256": digest,
        "target_accounting_semantic_receipt_sha256": digest,
        "frozen_economic_coverage_semantic_receipt_sha256": digest,
        "feature_scoped_economic_coverage_semantic_receipt_sha256": digest,
        "target_scoped_economic_coverage_semantic_receipt_sha256": digest,
        "economic_coverage_audit_receipt_sha256": digest,
    }
    row = MassiveProfitabilityEvaluationSourceDateV4(
        **row_body, receipt_sha256=semantic_sha256(row_body)  # type: ignore[arg-type]
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA,
        "data_gate_semantic_receipt_sha256": digest,
        "phase_plan_semantic_receipt_sha256": digest,
        "evaluation_plan_semantic_receipt_sha256": digest,
        "tournament_plan_receipt_sha256": digest,
        "accounting_freeze_semantic_receipt_sha256": digest,
        "origin_plan_semantic_receipt_sha256": digest,
        "daily_input_authority_semantic_receipt_sha256": digest,
        "fill_source_authority_semantic_receipt_sha256": digest,
        "terminal_authority_semantic_receipt_sha256": digest,
        "terminal_accounting_mode": "conservative-lower-bound",
        "outer_test_session_dates": (row.decision_session_date,),
        "rows": (asdict(row),),
        "row_inventory_sha256": semantic_sha256((row.receipt_sha256,)),
        "committed_source_bundle_data_qualified": True,
        "committed_outer_evaluation_authorized": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = "massive-profitability/evaluation-source-bundle-v4/generic.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=digest,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    parsed = parse_massive_profitability_evaluation_source_bundle_v4(
        root=tmp_path, loaded_source=loaded
    )

    assert parsed.committed_source_bundle_data_qualified is True
    assert parsed.source_bundle_data_qualified is False
    assert parsed.outer_evaluation_authorized is False
