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
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityPredictionRowV1,
)
from rl_quant.evaluation.massive_profitability_predictions_v3 import (
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET,
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA,
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_SPEC_SHA256,
    parse_massive_profitability_outer_predictions_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_DATASET,
    MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA,
    MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SPEC_SHA256,
    parse_massive_profitability_tournament_dataset_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    adapt_massive_profitability_training_fold_v2,
)
from rl_quant.evaluation.massive_profitability_training_v3 import (
    train_and_publish_massive_profitability_fold_v3,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
    MASSIVE_PROFITABILITY_HORIZONS_V1,
    MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
    MassiveProfitabilityTabularModelV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MassiveProfitabilityNormalizationV1,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
    build_massive_profitability_date_tensor_v1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentDatasetV2,
    MassiveProfitabilityTournamentV2Error,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    MassiveProfitabilityTrainedRunV2Error,
    bind_massive_profitability_trained_run_v2,
    parse_massive_profitability_model_checkpoint_v2,
    publish_massive_profitability_model_checkpoint_v2,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256


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
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
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
        **row_body,
        receipt_sha256=semantic_sha256(row_body),  # type: ignore[arg-type]
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


def test_dataset_v3_generic_reload_has_no_tensors_or_training_authority(
    tmp_path,
) -> None:
    digest = "a" * 64
    dates = _dates(3)
    features = tuple(semantic_sha256(("feature", value)) for value in dates)
    targets = tuple(semantic_sha256(("target", value)) for value in dates)
    feature_audits = tuple(semantic_sha256(("feature-audit", value)) for value in dates)
    target_audits = tuple(semantic_sha256(("target-audit", value)) for value in dates)
    tensors = tuple(semantic_sha256(("tensor", value)) for value in dates)
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA,
        "data_gate_semantic_receipt_sha256": digest,
        "data_gate_audit_receipt_sha256": "b" * 64,
        "data_gate_snapshot_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "phase_plan_source_receipt_sha256": "e" * 64,
        "feature_receipts": features,
        "target_receipts": targets,
        "tensor_source_array_receipts": tensors,
        "entry_session_dates": dates[:2],
        "maturation_only_session_dates": dates[2:],
        "dataset_v2_receipt_sha256": "f" * 64,
        "feature_inventory_sha256": semantic_sha256(features),
        "target_inventory_sha256": semantic_sha256(targets),
        "tensor_inventory_sha256": semantic_sha256(tensors),
        "committed_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "feature_audit_receipts": feature_audits,
        "target_audit_receipts": target_audits,
        "semantic_receipt_sha256": semantic_receipt,
        "audit_receipt_sha256": semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "data_gate_audit_receipt_sha256": "b" * 64,
                "phase_plan_source_receipt_sha256": "e" * 64,
                "feature_audit_receipts": feature_audits,
                "target_audit_receipts": target_audits,
            }
        ),
    }
    relative = "massive-profitability/tournament-dataset-v3/generic.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=digest,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    parsed = parse_massive_profitability_tournament_dataset_v3(
        root=tmp_path, loaded_source=loaded
    )

    assert parsed.committed_data_qualified is True
    assert parsed.runtime_dataset is None
    assert parsed.runtime_data_qualified is False
    assert parsed.development_training_authorized is False
    assert parsed.outer_prediction_authorized is False


def test_prediction_v3_generic_reload_cannot_authorize_fabricated_scores(
    tmp_path,
) -> None:
    digest = "a" * 64
    dates = _dates(126)
    feature_receipts = tuple(
        semantic_sha256(("prediction-feature", value)) for value in dates
    )
    rows = []
    for session_date, feature_receipt in zip(dates, feature_receipts, strict=True):
        body = {
            "decision_session_date": session_date,
            "security_id": "SEC0",
            "mean": (1.0, 1.0, 1.0, 1.0),
            "downside_quantile": (0.0, 0.0, 0.0, 0.0),
            "median": (1.0, 1.0, 1.0, 1.0),
            "upside_quantile": (2.0, 2.0, 2.0, 2.0),
            "scale": (1.0, 1.0, 1.0, 1.0),
            "feature_semantic_receipt_sha256": feature_receipt,
        }
        rows.append(
            MassiveProfitabilityPredictionRowV1(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
        )
    row_tuple = tuple(rows)
    per_seed = semantic_sha256(tuple(row.receipt_sha256 for row in row_tuple))
    semantic = {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA,
        "setting_id": "MV00",
        "fold_index": 0,
        "seed_inventory": (0,),
        "ensemble": False,
        "outer_test_session_dates": dates,
        "rows": tuple(asdict(row) for row in row_tuple),
        "feature_receipts": feature_receipts,
        "checkpoint_source_receipts": (),
        "trained_run_receipts": (),
        "per_seed_row_inventory_receipts": (per_seed,),
        "dataset_semantic_receipt_sha256": digest,
        "dataset_source_receipt_sha256": "b" * 64,
        "dataset_v2_receipt_sha256": "c" * 64,
        "tournament_plan_receipt_sha256": "d" * 64,
        "tournament_plan_source_receipt_sha256": "e" * 64,
        "phase_plan_semantic_receipt_sha256": "f" * 64,
        "fold_receipt_sha256": "1" * 64,
        "row_inventory_sha256": per_seed,
        "committed_prediction_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = "massive-profitability/predictions-v3/fabricated.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=digest,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    parsed = parse_massive_profitability_outer_predictions_v3(
        root=tmp_path, loaded_source=loaded
    )

    assert parsed.committed_prediction_qualified is True
    assert parsed.runtime_prediction_replayed is False
    assert parsed.outer_prediction_authorized is False


def test_v3_training_and_run_contracts_require_exact_dataset_identity() -> None:
    training = inspect.signature(train_and_publish_massive_profitability_fold_v3)
    binding = inspect.signature(bind_massive_profitability_trained_run_v2)

    assert "dataset" in training.parameters
    assert "data_gate" in training.parameters
    assert "features" in training.parameters
    assert "targets" in training.parameters
    assert "dataset_semantic_receipt_sha256" in binding.parameters
    assert "dataset_source_receipt_sha256" in binding.parameters
    assert "dataset_v2_receipt_sha256" in binding.parameters


def test_checkpoint_v2_round_trip_binds_dataset_source_and_tensors(tmp_path) -> None:
    fit_inventory = "1" * 64
    training_config_receipt = MassiveProfitabilityTrainingConfigV1().receipt_sha256
    normalization_body = {
        "bars_mean": (0.0,) * MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
        "bars_scale": (1.0,) * MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
        "bars_observed": (True,) * MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
        "tape_mean": (0.0,) * MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
        "tape_scale": (1.0,) * MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
        "tape_observed": (True,) * MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
        "target_median": (0.0,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
        "target_scale": (1.0,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
        "fit_session_inventory_sha256": fit_inventory,
    }
    normalization = MassiveProfitabilityNormalizationV1(
        **normalization_body,
        receipt_sha256=semantic_sha256(normalization_body),  # type: ignore[arg-type]
    )
    model = MassiveProfitabilityTabularModelV1(setting_id="MV02")
    state = tuple(
        sorted(
            (
                (name, value.detach().to(dtype=torch.float32, device="cpu").clone())
                for name, value in model.state_dict().items()
            ),
            key=lambda item: item[0],
        )
    )
    model_hash = state_dict_sha256(dict(state))
    run_body = {
        "setting_id": "MV02",
        "fold_index": 0,
        "seed": 0,
        "best_epoch": 0,
        "completed_epochs": 1,
        "validation_rank_ic": (0.1,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
        "normalization_receipt_sha256": normalization.receipt_sha256,
        "model_state_sha256": model_hash,
        "fit_inventory_sha256": fit_inventory,
        "validation_inventory_sha256": "2" * 64,
        "training_source_receipt_sha256": "3" * 64,
        "tournament_plan_receipt_sha256": "4" * 64,
        "training_config_receipt_sha256": training_config_receipt,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    base = MassiveProfitabilityTrainedRunV1(
        setting_id="MV02",
        fold_index=0,
        seed=0,
        best_epoch=0,
        completed_epochs=1,
        validation_rank_ic=(0.1,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
        normalization=normalization,
        model_state_sha256=model_hash,
        model_state=state,
        fit_inventory_sha256=fit_inventory,
        validation_inventory_sha256="2" * 64,
        training_source_receipt_sha256="3" * 64,
        tournament_plan_receipt_sha256="4" * 64,
        training_config_receipt_sha256=training_config_receipt,
        run_receipt_sha256=semantic_sha256(run_body),
        outer_prediction_authorized=True,
    )
    run = bind_massive_profitability_trained_run_v2(
        run_v1=base,
        dataset_semantic_receipt_sha256="6" * 64,
        dataset_source_receipt_sha256="7" * 64,
        dataset_v2_receipt_sha256="8" * 64,
        tournament_plan_receipt_sha256="4" * 64,
        tournament_plan_source_receipt_sha256="9" * 64,
        phase_plan_semantic_receipt_sha256="a" * 64,
        fold_receipt_sha256="b" * 64,
    )
    checkpoint = publish_massive_profitability_model_checkpoint_v2(
        root=tmp_path, artifact_id="bound", run=run, committed_at_ms=1
    )
    reloaded = parse_massive_profitability_model_checkpoint_v2(
        root=tmp_path, loaded_source=checkpoint.loaded_source
    )

    assert reloaded.run.run_receipt_sha256 == run.run_receipt_sha256
    assert reloaded.run.dataset_source_receipt_sha256 == "7" * 64
    assert reloaded.run.run_v1.model_state_sha256 == model_hash
    with pytest.raises(MassiveProfitabilityTrainedRunV2Error):
        replace(reloaded.run, dataset_source_receipt_sha256="c" * 64).validate()
