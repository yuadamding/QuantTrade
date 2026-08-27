from __future__ import annotations

import inspect
from dataclasses import asdict, fields, replace
from io import BytesIO

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v6 import (
    materialize_massive_profitability_evaluation_plan_v6,
)
from rl_quant.evaluation.massive_profitability_evaluation_source_bundle_v7 import (
    MassiveProfitabilityEvaluationRuntimeSourcesV7,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v6 import (
    materialize_massive_profitability_outer_evidence_v6,
)
from rl_quant.evaluation.massive_profitability_prediction_replay_authority_v1 import (
    MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET,
    MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA,
    MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityPredictionReplayAuthorityV1Error,
    MassiveProfitabilityPredictionReplayRowV1,
    parse_massive_profitability_prediction_replay_authority_v1,
    resolve_massive_profitability_prediction_replay_authority_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

_SETTINGS = ("MV00", "MV02", "MV04", "MV04-SHUFFLE")


def _row(
    *, fold_index: int, setting_id: str
) -> MassiveProfitabilityPredictionReplayRowV1:
    digest = semantic_sha256((fold_index, setting_id))
    checkpoint_receipts = (
        ()
        if setting_id == "MV00"
        else tuple(semantic_sha256((digest, "checkpoint", seed)) for seed in range(5))
    )
    run_receipts = (
        ()
        if setting_id == "MV00"
        else tuple(semantic_sha256((digest, "run", seed)) for seed in range(5))
    )
    per_seed = tuple(
        semantic_sha256((digest, "rows", seed))
        for seed in ((0,) if setting_id == "MV00" else range(5))
    )
    body = {
        "fold_index": fold_index,
        "setting_id": setting_id,
        "prediction_semantic_receipt_sha256": digest,
        "prediction_source_receipt_sha256": semantic_sha256((digest, "source")),
        "prediction_payload_relative_path": (
            f"massive-profitability/predictions-v3/f{fold_index}-{setting_id}.json"
        ),
        "prediction_verified_at_ms": 1,
        "dataset_semantic_receipt_sha256": "a" * 64,
        "dataset_source_receipt_sha256": "b" * 64,
        "data_gate_semantic_receipt_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "tournament_plan_receipt_sha256": "e" * 64,
        "tournament_plan_source_receipt_sha256": "f" * 64,
        "training_replay_authority_semantic_receipt_sha256": "1" * 64,
        "fold_receipt_sha256": semantic_sha256((fold_index, "fold")),
        "prediction_row_inventory_sha256": semantic_sha256((digest, "inventory")),
        "feature_inventory_sha256": semantic_sha256((digest, "features")),
        "checkpoint_source_receipts": checkpoint_receipts,
        "trained_run_receipts": run_receipts,
        "per_seed_row_inventory_receipts": per_seed,
        "committed_prediction_semantic_receipt_sha256": digest,
        "replayed_prediction_semantic_receipt_sha256": digest,
        "replay_success": True,
    }
    result = MassiveProfitabilityPredictionReplayRowV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def test_generic_prediction_replay_authority_reload_is_nonauthorizing(
    tmp_path,
) -> None:
    rows = tuple(
        _row(fold_index=fold_index, setting_id=setting_id)
        for fold_index in range(4)
        for setting_id in _SETTINGS
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA,
        "dataset_semantic_receipt_sha256": "a" * 64,
        "dataset_source_receipt_sha256": "b" * 64,
        "data_gate_semantic_receipt_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "tournament_plan_receipt_sha256": "e" * 64,
        "tournament_plan_source_receipt_sha256": "f" * 64,
        "training_replay_authority_semantic_receipt_sha256": "1" * 64,
        "training_replay_authority_source_receipt_sha256": "2" * 64,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "committed_prediction_replay_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = "massive-profitability/prediction-replay-authority-v1/generic.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256="a" * 64,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    parsed = parse_massive_profitability_prediction_replay_authority_v1(
        root=tmp_path, loaded_source=loaded
    )

    assert len(parsed.rows) == 16
    assert parsed.committed_prediction_replay_qualified is True
    assert parsed.runtime_predictions_replayed is False
    assert parsed.outer_evaluation_authorized is False
    with pytest.raises(MassiveProfitabilityPredictionReplayAuthorityV1Error):
        replace(parsed, rows=parsed.rows[:-1]).validate()
    with pytest.raises(MassiveProfitabilityPredictionReplayAuthorityV1Error):
        replace(
            parsed.rows[0],
            replayed_prediction_semantic_receipt_sha256="9" * 64,
        ).validate()


def test_final_evaluation_boundary_resolves_predictions_from_authority() -> None:
    plan = inspect.signature(materialize_massive_profitability_evaluation_plan_v6)
    evidence = inspect.signature(materialize_massive_profitability_outer_evidence_v6)
    resolver = inspect.signature(
        resolve_massive_profitability_prediction_replay_authority_v1
    )
    runtime_fields = {
        field.name for field in fields(MassiveProfitabilityEvaluationRuntimeSourcesV7)
    }

    assert "prediction_replay_authority" in plan.parameters
    assert "predictions" not in plan.parameters
    assert "prediction_replay_authority" in runtime_fields
    assert "training_replay_authority" in runtime_fields
    assert "predictions" not in evidence.parameters
    assert "MassiveProfitabilityEvaluationSourceBundleV7" in str(
        evidence.parameters["evaluation_source_bundle"].annotation
    )
    assert {
        "replay_authority",
        "training_replay_authority",
        "dataset",
        "data_gate",
        "phase_plan",
        "features",
        "targets",
        "tournament_plan",
    } <= resolver.parameters.keys()
