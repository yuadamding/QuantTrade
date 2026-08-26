from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectError
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    parse_massive_profitability_outer_predictions_v1,
    publish_massive_profitability_mv00_outer_predictions_v1,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v1 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SOURCE_SHA256,
    adapt_massive_profitability_training_fold_v1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
    MassiveProfitabilityTabularModelV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256,
    MassiveProfitabilityDateTensorV1,
    MassiveProfitabilityTournamentDatasetV1,
    MassiveProfitabilityTournamentPlanV1,
    MassiveProfitabilityTrainingConfigV1,
    _tensor_sha256,
    fit_massive_profitability_normalization_v1,
    massive_profitability_tape_permutation_v1,
    parse_massive_profitability_model_checkpoint_v1,
    publish_massive_profitability_model_checkpoint_v1,
    train_massive_profitability_fold_v1,
)

_DIGEST = "a" * 64


def _date_tensor(index: int, *, tape_shift: float = 0.0) -> MassiveProfitabilityDateTensorV1:
    session_date = f"d{index:04d}"
    security_ids = ("SEC-A", "SEC-B")
    bars = torch.zeros((2, len(BARS_MIN_V2_FIELDS)), dtype=torch.float32)
    tape = torch.zeros((2, len(TAPE_MIN_V2_FIELDS)), dtype=torch.float32)
    bars[:, 0] = torch.tensor((index / 1000.0, -index / 1000.0))
    bars[:, 6] = torch.tensor((0.25, -0.25))
    bars[:, 7] = torch.tensor((0.50, -0.50))
    tape[:, 0] = torch.tensor((1.0 + tape_shift, -1.0 - tape_shift))
    tape[:, 4] = torch.tensor((0.40 + tape_shift, -0.40 - tape_shift))
    bars_valid = torch.ones_like(bars, dtype=torch.bool)
    tape_valid = torch.ones_like(tape, dtype=torch.bool)
    target = torch.tensor(
        (
            (0.01 + index * 1e-6, 0.02, 0.03, 0.04),
            (-0.01 - index * 1e-6, -0.02, -0.03, -0.04),
        ),
        dtype=torch.float32,
    )
    target_valid = torch.ones_like(target, dtype=torch.bool)
    feature_receipt = semantic_sha256(("feature", session_date, tape_shift))
    target_receipt = semantic_sha256(("target", session_date))
    identity = {
        "decision_session_date": session_date,
        "security_ids": security_ids,
        "bars_values": _tensor_sha256(bars),
        "bars_valid": _tensor_sha256(bars_valid),
        "tape_values": _tensor_sha256(tape),
        "tape_valid": _tensor_sha256(tape_valid),
        "target_values": _tensor_sha256(target),
        "target_valid": _tensor_sha256(target_valid),
        "feature_receipt": feature_receipt,
        "target_receipt": target_receipt,
    }
    result = MassiveProfitabilityDateTensorV1(
        decision_session_date=session_date,
        security_ids=security_ids,
        bars_values=bars,
        bars_valid=bars_valid,
        tape_values=tape,
        tape_valid=tape_valid,
        target_values=target,
        target_valid=target_valid,
        feature_semantic_receipt_sha256=feature_receipt,
        target_semantic_receipt_sha256=target_receipt,
        source_array_sha256=semantic_sha256(identity),
    )
    result.validate()
    return result


def _dataset(indices: tuple[int, ...]) -> MassiveProfitabilityTournamentDatasetV1:
    rows = tuple(_date_tensor(index) for index in indices)
    body = {
        "dates": tuple(row.source_array_sha256 for row in rows),
        "data_gate": _DIGEST,
        "phase_plan": "b" * 64,
    }
    result = MassiveProfitabilityTournamentDatasetV1(
        dates=rows,
        data_gate_semantic_receipt_sha256=_DIGEST,
        phase_plan_semantic_receipt_sha256="b" * 64,
        dataset_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _fold() -> MassiveProfitabilityOuterFoldPlanV1:
    fit = tuple(f"d{index:04d}" for index in range(756))
    inner_purge = tuple(f"d{index:04d}" for index in range(756, 819))
    validation = tuple(f"d{index:04d}" for index in range(819, 945))
    outer_purge = tuple(f"d{index:04d}" for index in range(945, 1008))
    outer = tuple(f"d{index:04d}" for index in range(1008, 1134))
    body = {
        "fold_index": 0,
        "fit_session_dates": fit,
        "inner_purge_session_dates": inner_purge,
        "inner_validation_session_dates": validation,
        "outer_purge_session_dates": outer_purge,
        "outer_test_session_dates": outer,
        "fit_inventory_sha256": semantic_sha256(fit),
        "inner_validation_inventory_sha256": semantic_sha256(validation),
        "outer_test_inventory_sha256": semantic_sha256(outer),
    }
    result = MassiveProfitabilityOuterFoldPlanV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _plan(fold: MassiveProfitabilityOuterFoldPlanV1) -> MassiveProfitabilityTournamentPlanV1:
    body = {
        "data_gate_semantic_receipt_sha256": _DIGEST,
        "phase_plan_semantic_receipt_sha256": "b" * 64,
        "fold_receipts": (fold.receipt_sha256, "c" * 64, "d" * 64, "e" * 64),
        "settings": MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
        "development_seeds": MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
        "confirmation_seeds": MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        "specification_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
        "input_adapter_source_sha256": (
            MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SOURCE_SHA256
        ),
        "development_training_authorized": True,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "schema": "rl-quant.massive-profitability-tournament-v1",
    }
    result = MassiveProfitabilityTournamentPlanV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def test_tabular_distribution_is_ordered_and_mv02_ignores_tape() -> None:
    torch.manual_seed(7)
    model = MassiveProfitabilityTabularModelV1(setting_id="MV02")
    model.eval()
    bars = torch.randn(2, 3, len(BARS_MIN_V2_FIELDS))
    tape = torch.randn(2, 3, len(TAPE_MIN_V2_FIELDS))
    bars_valid = torch.ones_like(bars, dtype=torch.bool)
    tape_valid = torch.ones_like(tape, dtype=torch.bool)
    staleness = torch.zeros((2, 3, 1))
    first = model(
        bars_values=bars,
        bars_valid=bars_valid,
        tape_values=tape,
        tape_valid=tape_valid,
        source_staleness=staleness,
    )
    second = model(
        bars_values=bars,
        bars_valid=bars_valid,
        tape_values=tape * 100.0,
        tape_valid=tape_valid,
        source_staleness=staleness,
    )
    assert first.mean.shape == (2, 3, 4)
    assert torch.equal(first.mean, second.mean)
    assert bool((first.downside_quantile <= first.median).all())
    assert bool((first.median <= first.upside_quantile).all())
    assert bool((first.scale > 0.0).all())


def test_real_tape_branch_changes_output_and_shuffle_is_target_independent() -> None:
    torch.manual_seed(11)
    model = MassiveProfitabilityTabularModelV1(setting_id="MV04")
    model.eval()
    bars = torch.zeros(1, 4, len(BARS_MIN_V2_FIELDS))
    tape = torch.zeros(1, 4, len(TAPE_MIN_V2_FIELDS))
    valid_bars = torch.ones_like(bars, dtype=torch.bool)
    valid_tape = torch.ones_like(tape, dtype=torch.bool)
    tape[0, :, 0] = torch.arange(4, dtype=torch.float32)
    first = model(
        bars_values=bars,
        bars_valid=valid_bars,
        tape_values=tape,
        tape_valid=valid_tape,
        source_staleness=torch.zeros(1, 4, 1),
    )
    second = model(
        bars_values=bars,
        bars_valid=valid_bars,
        tape_values=tape + 3.0,
        tape_valid=valid_tape,
        source_staleness=torch.zeros(1, 4, 1),
    )
    assert not torch.equal(first.mean, second.mean)
    security_ids = ("A", "B", "C", "D")
    permutation = massive_profitability_tape_permutation_v1(
        decision_session_date="2024-01-02", security_ids=security_ids
    )
    assert permutation.tolist() != list(range(4))
    assert sorted(permutation.tolist()) == list(range(4))
    assert torch.equal(
        permutation,
        massive_profitability_tape_permutation_v1(
            decision_session_date="2024-01-02", security_ids=security_ids
        ),
    )


def test_fit_only_normalization_ignores_nonfit_mutation() -> None:
    baseline = _dataset((0, 1, 2))
    changed_row = _date_tensor(2, tape_shift=100.0)
    changed_body = {
        "dates": (
            baseline.dates[0].source_array_sha256,
            baseline.dates[1].source_array_sha256,
            changed_row.source_array_sha256,
        ),
        "data_gate": _DIGEST,
        "phase_plan": "b" * 64,
    }
    changed = replace(
        baseline,
        dates=(baseline.dates[0], baseline.dates[1], changed_row),
        dataset_receipt_sha256=semantic_sha256(changed_body),
    )
    first = fit_massive_profitability_normalization_v1(
        dataset=baseline, fit_session_dates=("d0000", "d0001")
    )
    second = fit_massive_profitability_normalization_v1(
        dataset=changed, fit_session_dates=("d0000", "d0001")
    )
    assert first.receipt_sha256 == second.receipt_sha256


def test_small_training_is_deterministic_but_nonauthorizing(tmp_path: Path) -> None:
    fold = _fold()
    plan = _plan(fold)
    dataset = _dataset(tuple(range(756)) + tuple(range(819, 945)))
    config = MassiveProfitabilityTrainingConfigV1(
        maximum_epochs=2,
        early_stopping_patience=1,
        complete_dates_per_batch=756,
    )
    first = train_massive_profitability_fold_v1(
        dataset=dataset,
        tournament_plan=plan,
        fold=adapt_massive_profitability_training_fold_v1(fold),
        setting_id="MV02",
        seed=0,
        config=config,
    )
    second = train_massive_profitability_fold_v1(
        dataset=dataset,
        tournament_plan=plan,
        fold=adapt_massive_profitability_training_fold_v1(fold),
        setting_id="MV02",
        seed=0,
        config=config,
    )
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.run_receipt_sha256 == second.run_receipt_sha256
    assert first.outer_prediction_authorized is False
    assert first.profitability_reporting_authorized is False
    checkpoint = publish_massive_profitability_model_checkpoint_v1(
        root=tmp_path,
        artifact_id="mv02-fold0-seed0",
        run=first,
        committed_at_ms=500,
    )
    reopened = parse_massive_profitability_model_checkpoint_v1(
        root=tmp_path, loaded_source=checkpoint.loaded_source
    )
    assert reopened.run.model_state_sha256 == first.model_state_sha256
    with pytest.raises(MassiveSourceObjectError):
        publish_massive_profitability_model_checkpoint_v1(
            root=tmp_path,
            artifact_id="mv02-fold0-seed0",
            run=first,
            committed_at_ms=501,
        )


def test_mv00_outer_predictions_are_create_only_and_round_trip(tmp_path: Path) -> None:
    fold = _fold()
    plan = _plan(fold)
    dataset = _dataset(tuple(range(756)) + tuple(range(1008, 1134)))
    artifact = publish_massive_profitability_mv00_outer_predictions_v1(
        root=tmp_path,
        artifact_id="mv00-fold0",
        dataset=dataset,
        tournament_plan=plan,
        fold=fold,
        committed_at_ms=1000,
    )
    loaded = parse_massive_profitability_outer_predictions_v1(
        root=tmp_path, loaded_source=artifact.loaded_source
    )
    assert loaded.semantic_receipt_sha256 == artifact.semantic_receipt_sha256
    assert loaded.profitability_reporting_authorized is False
    assert loaded.lockbox_access_authorized is False
    with pytest.raises(MassiveSourceObjectError):
        publish_massive_profitability_mv00_outer_predictions_v1(
            root=tmp_path,
            artifact_id="mv00-fold0",
            dataset=dataset,
            tournament_plan=plan,
            fold=fold,
            committed_at_ms=1001,
        )


def test_tournament_plan_and_runs_never_authorize_profit_reporting() -> None:
    plan = _plan(_fold())
    payload = asdict(plan)
    assert payload["profitability_reporting_authorized"] is False
    assert payload["lockbox_access_authorized"] is False
    assert payload["reinforcement_learning_authorized"] is False
