from __future__ import annotations

import inspect

import torch

from rl_quant.evaluation.massive_profitability_evaluation_plan_v4 import (
    materialize_massive_profitability_evaluation_plan_v4,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v4 import (
    materialize_massive_profitability_outer_evidence_v4,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
    MASSIVE_PROFITABILITY_HORIZONS_V1,
    MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
    MassiveProfitabilityTabularModelV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_profitability_tournament_v1 import (
    MassiveProfitabilityNormalizationV1,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    bind_massive_profitability_trained_run_v2,
    publish_massive_profitability_model_checkpoint_v2,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    authorize_massive_profitability_model_checkpoint_v3,
    bind_massive_profitability_trained_run_v3,
    load_massive_profitability_prediction_checkpoint_v2_from_v3,
    parse_massive_profitability_model_checkpoint_v3,
    publish_massive_profitability_model_checkpoint_v3,
)
from rl_quant.training.massive_profitability_training_replay_v3 import (
    MassiveProfitabilityTrainingEpochV3,
    massive_profitability_training_runtime_v3,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256


def _manual_run_v2():
    fit_inventory = "1" * 64
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
                name,
                value.detach().to(dtype=torch.float32, device="cpu").clone(),
            )
            for name, value in model.state_dict().items()
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
        "training_config_receipt_sha256": MassiveProfitabilityTrainingConfigV1().receipt_sha256,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    run_v1 = MassiveProfitabilityTrainedRunV1(
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
        training_config_receipt_sha256=MassiveProfitabilityTrainingConfigV1().receipt_sha256,
        run_receipt_sha256=semantic_sha256(run_body),
        outer_prediction_authorized=True,
    )
    return bind_massive_profitability_trained_run_v2(
        run_v1=run_v1,
        dataset_semantic_receipt_sha256="6" * 64,
        dataset_source_receipt_sha256="7" * 64,
        dataset_v2_receipt_sha256="8" * 64,
        tournament_plan_receipt_sha256="4" * 64,
        tournament_plan_source_receipt_sha256="9" * 64,
        phase_plan_semantic_receipt_sha256="a" * 64,
        fold_receipt_sha256="b" * 64,
    )


def test_checkpoint_v3_generic_reload_is_nonauthorizing(tmp_path) -> None:
    run_v2 = _manual_run_v2()
    checkpoint_v2 = publish_massive_profitability_model_checkpoint_v2(
        root=tmp_path,
        artifact_id="replayed-compat",
        run=run_v2,
        committed_at_ms=1,
    )
    epoch_body = {
        "epoch": 0,
        "validation_rank_ic": run_v2.run_v1.validation_rank_ic,
        "validation_mean_rank_ic": 0.1,
        "model_state_sha256": run_v2.run_v1.model_state_sha256,
        "selected_as_best": True,
    }
    epoch = MassiveProfitabilityTrainingEpochV3(
        **epoch_body,
        receipt_sha256=semantic_sha256(epoch_body),  # type: ignore[arg-type]
    )
    replayed_run = bind_massive_profitability_trained_run_v3(
        run_v2=run_v2,
        checkpoint_v2_source_receipt_sha256=(
            checkpoint_v2.loaded_source.receipt_sha256
        ),
        checkpoint_v2_payload_relative_path=(
            checkpoint_v2.loaded_source.payload_relative_path
        ),
        checkpoint_v2_verified_at_ms=checkpoint_v2.loaded_source.verified_at_ms,
        training_runtime=massive_profitability_training_runtime_v3(),
        epoch_trace=(epoch,),
    )
    checkpoint = publish_massive_profitability_model_checkpoint_v3(
        root=tmp_path,
        artifact_id="replayed",
        run=replayed_run,
        committed_at_ms=1,
    )
    generic = parse_massive_profitability_model_checkpoint_v3(
        root=tmp_path, loaded_source=checkpoint.loaded_source
    )

    assert generic.run.committed_training_qualified is True
    assert generic.run.runtime_training_replayed is False
    assert generic.run.outer_prediction_authorized is False

    promoted = authorize_massive_profitability_model_checkpoint_v3(
        root=tmp_path,
        checkpoint=generic,
        replayed_run=replayed_run,
    )
    assert promoted.run.runtime_training_replayed is True
    assert promoted.run.outer_prediction_authorized is True
    prediction_checkpoint = (
        load_massive_profitability_prediction_checkpoint_v2_from_v3(
            root=tmp_path, checkpoint=promoted
        )
    )
    assert prediction_checkpoint.run.run_receipt_sha256 == run_v2.run_receipt_sha256


def test_evaluation_v4_accepts_only_replayed_prediction_v3_boundary() -> None:
    plan = inspect.signature(materialize_massive_profitability_evaluation_plan_v4)
    evidence = inspect.signature(materialize_massive_profitability_outer_evidence_v4)

    assert "MassiveProfitabilityOuterPredictionsV3" in str(
        plan.parameters["predictions"].annotation
    )
    assert "training_checkpoints" in plan.parameters
    assert "MassiveProfitabilityOuterPredictionsV3" in str(
        evidence.parameters["predictions"].annotation
    )
    assert "MassiveProfitabilityEvaluationSourceBundleV5" in str(
        evidence.parameters["evaluation_source_bundle"].annotation
    )
