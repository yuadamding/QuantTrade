from __future__ import annotations

import inspect
from dataclasses import asdict, fields, replace
from io import BytesIO

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v5 import (
    materialize_massive_profitability_evaluation_plan_v5,
)
from rl_quant.evaluation.massive_profitability_evaluation_source_bundle_v6 import (
    MassiveProfitabilityEvaluationRuntimeSourcesV6,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v5 import (
    materialize_massive_profitability_outer_evidence_v5,
)
from rl_quant.evaluation.massive_profitability_training_replay_authority_v1 import (
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET,
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA,
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityTrainingReplayAuthorityV1Error,
    MassiveProfitabilityTrainingReplayRowV1,
    parse_massive_profitability_training_replay_authority_v1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
)


def _row(*, fold_index: int, setting_id: str, seed: int):
    digest = semantic_sha256((fold_index, setting_id, seed))
    body = {
        "fold_index": fold_index,
        "setting_id": setting_id,
        "seed": seed,
        "dataset_semantic_receipt_sha256": "a" * 64,
        "dataset_source_receipt_sha256": "b" * 64,
        "data_gate_semantic_receipt_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "tournament_plan_receipt_sha256": "e" * 64,
        "tournament_plan_source_receipt_sha256": "f" * 64,
        "fold_receipt_sha256": semantic_sha256((fold_index, "fold")),
        "training_config_receipt_sha256": "1" * 64,
        "training_runtime_receipt_sha256": "2" * 64,
        "epoch_trace_receipt_sha256": "3" * 64,
        "checkpoint_v3_source_receipt_sha256": digest,
        "checkpoint_v3_payload_relative_path": (
            f"massive-profitability/model-checkpoint-v3/"
            f"f{fold_index}-{setting_id}-{seed}.json"
        ),
        "checkpoint_v3_verified_at_ms": 1,
        "committed_run_v3_semantic_receipt_sha256": digest,
        "replayed_run_v3_semantic_receipt_sha256": digest,
        "run_v2_receipt_sha256": semantic_sha256((digest, "run-v2")),
        "checkpoint_v2_source_receipt_sha256": semantic_sha256(
            (digest, "checkpoint-v2")
        ),
        "replay_success": True,
    }
    return MassiveProfitabilityTrainingReplayRowV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )


def test_generic_training_replay_authority_reload_is_nonauthorizing(tmp_path) -> None:
    rows = tuple(
        _row(fold_index=fold_index, setting_id=setting_id, seed=seed)
        for fold_index in range(4)
        for setting_id in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
        for seed in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA,
        "dataset_semantic_receipt_sha256": "a" * 64,
        "dataset_source_receipt_sha256": "b" * 64,
        "data_gate_semantic_receipt_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "tournament_plan_receipt_sha256": "e" * 64,
        "tournament_plan_source_receipt_sha256": "f" * 64,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "committed_root_replay_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = "massive-profitability/training-replay-authority-v1/generic.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256="a" * 64,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    parsed = parse_massive_profitability_training_replay_authority_v1(
        root=tmp_path, loaded_source=loaded
    )

    assert len(parsed.rows) == 60
    assert parsed.committed_root_replay_qualified is True
    assert parsed.runtime_root_replayed is False
    assert parsed.outer_evaluation_authorized is False
    with pytest.raises(MassiveProfitabilityTrainingReplayAuthorityV1Error):
        replace(parsed.rows[0], replayed_run_v3_semantic_receipt_sha256="9" * 64).validate()


def test_final_evaluation_boundary_requires_root_replay_authority() -> None:
    plan = inspect.signature(materialize_massive_profitability_evaluation_plan_v5)
    evidence = inspect.signature(materialize_massive_profitability_outer_evidence_v5)
    runtime_fields = {field.name for field in fields(MassiveProfitabilityEvaluationRuntimeSourcesV6)}

    assert "training_replay_authority" in plan.parameters
    assert "training_checkpoints" not in plan.parameters
    assert "dataset" in plan.parameters
    assert "features" in plan.parameters
    assert "targets" in plan.parameters
    assert {
        "training_replay_authority",
        "dataset",
        "tournament_plan",
    } <= runtime_fields
    assert "MassiveProfitabilityEvaluationSourceBundleV6" in str(
        evidence.parameters["evaluation_source_bundle"].annotation
    )
