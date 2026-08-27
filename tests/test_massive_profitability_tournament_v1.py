from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectError
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityPredictionRowV1,
    build_massive_profitability_recovery_predictions_for_test_v1,
    parse_massive_profitability_outer_predictions_v1,
    publish_massive_profitability_mv00_outer_predictions_v1,
)
from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveProfitabilityResidualInputRowV1,
    MassiveProfitabilityResidualScoreRowV1,
    MassiveProfitabilityResidualScoresV1,
    MassiveProfitabilitySelectedTranchePositionV1,
    MassiveProfitabilitySelectedTranchesV1,
    build_massive_profitability_residual_scores_v1,
    evaluate_massive_profitability_fixed_tranches_v1,
)
from rl_quant.evaluation.massive_profitability_inference_v1 import (
    MassiveProfitabilityHorizonRiskScalingV1,
    build_massive_profitability_composite_pnl_v1,
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
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
    MassiveProfitabilityTabularModelV1,
    massive_profitability_mv00_scores_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256,
    MassiveProfitabilityDateTensorV1,
    MassiveProfitabilityTournamentDatasetV1,
    MassiveProfitabilityTournamentPlanV1,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
    _tensor_sha256,
    fit_massive_profitability_normalization_v1,
    massive_profitability_tape_permutation_v1,
    parse_massive_profitability_model_checkpoint_v1,
    publish_massive_profitability_model_checkpoint_v1,
    train_massive_profitability_fold_v1,
)

_DIGEST = "a" * 64


def _date_tensor(
    index: int, *, tape_shift: float = 0.0
) -> MassiveProfitabilityDateTensorV1:
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


def _planted_date_tensor(
    index: int, *, asset_count: int = 16
) -> MassiveProfitabilityDateTensorV1:
    session_date = f"d{index:04d}"
    security_ids = tuple(f"SEC-{asset:03d}" for asset in range(asset_count))
    bars = torch.zeros((asset_count, len(BARS_MIN_V2_FIELDS)), dtype=torch.float32)
    tape = torch.zeros((asset_count, len(TAPE_MIN_V2_FIELDS)), dtype=torch.float32)
    bar_signal = torch.tensor(
        tuple(
            (((asset * 7 + index * 3) % 17) - 8) / 8.0 for asset in range(asset_count)
        ),
        dtype=torch.float32,
    )
    tape_signal = torch.tensor(
        tuple(
            (((asset * 11 + index * 5) % 19) - 9) / 9.0 for asset in range(asset_count)
        ),
        dtype=torch.float32,
    )
    bars[:, 0] = bar_signal
    tape[:, 0] = tape_signal
    bars_valid = torch.ones_like(bars, dtype=torch.bool)
    tape_valid = torch.ones_like(tape, dtype=torch.bool)
    alpha = 0.50 * bar_signal + 0.50 * tape_signal
    target = torch.stack((alpha, alpha * 1.2, alpha * 1.5, alpha * 2.0), dim=-1)
    target_valid = torch.ones_like(target, dtype=torch.bool)
    feature_receipt = semantic_sha256(("planted-feature", session_date))
    target_receipt = semantic_sha256(("planted-target", session_date))
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


def _planted_alpha(index: int, asset: int) -> float:
    bar_signal = (((asset * 7 + index * 3) % 17) - 8) / 8.0
    tape_signal = (((asset * 11 + index * 5) % 19) - 9) / 9.0
    return 0.50 * bar_signal + 0.50 * tape_signal


def _planted_target_authority(
    *, index: int, asset_count: int = 16
) -> MassiveProfitabilityTargetAccountingAuthorityV2:
    decision_session_date = f"d{index:04d}"
    session_dates = tuple(f"d{index + offset:04d}" for offset in range(64))
    paths = []
    for asset in range(asset_count):
        security_id = f"SEC-{asset:03d}"
        economic_return = 0.05 * _planted_alpha(index, asset)
        values = (100.0,) + (100.0 * (1.0 + economic_return),) * 63
        path_body = {
            "security_id": security_id,
            "economic_at_ms": tuple(1_000 + offset for offset in range(64)),
            "available_at_ms": tuple(1_000 + offset for offset in range(64)),
            "values": values,
            "valid": (True,) * 64,
            "terminal": (False,) * 64,
            "mark_kinds": ("market",) * 64,
            "mark_receipts": tuple(
                semantic_sha256((decision_session_date, security_id, offset))
                for offset in range(64)
            ),
            "unresolved_terminal_fallback_session_offset": None,
            "conservative_total_loss_fallback": False,
        }
        paths.append(
            MassiveProfitabilityTargetEconomicPathRowV1(
                **path_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(path_body),
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in paths))
    semantic = {
        "schema": "rl-quant.massive-profitability-target-accounting-authority-v2",
        "origin_receipt_sha256": semantic_sha256(
            ("planted-origin", decision_session_date)
        ),
        "origin_plan_semantic_receipt_sha256": "b" * 64,
        "decision_session_date": decision_session_date,
        "session_dates": session_dates,
        "rows": tuple(asdict(row) for row in paths),
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "daily_input_authority_semantic_receipt_sha256": "c" * 64,
        "fill_source_authority_semantic_receipt_sha256": "d" * 64,
        "terminal_authority_semantic_receipt_sha256": "e" * 64,
        "economic_coverage_semantic_receipt_sha256": "f" * 64,
        "scoped_economic_event_inventory_sha256": semantic_sha256(()),
        "row_inventory_sha256": row_inventory,
        "fill_sources_qualified": True,
        "economic_values_data_qualified": True,
        "terminal_accounting_complete": True,
        "conservative_total_loss_target_count": 0,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    economic_audit = semantic_sha256(("planted-economic-audit", semantic_receipt))
    result = MassiveProfitabilityTargetAccountingAuthorityV2(
        **{**semantic, "rows": tuple(paths)},  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_receipt,
        economic_archive_audit_receipt_sha256=economic_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_archive_audit_receipt_sha256": economic_audit,
            }
        ),
    )
    result.validate()
    return result


def _planted_residual_scores(
    *,
    setting_id: str,
    scores: dict[tuple[str, str], tuple[float, ...]],
) -> MassiveProfitabilityResidualScoresV1:
    inputs = []
    for (session_date, security_id), raw_scores in sorted(scores.items()):
        body = {
            "decision_session_date": session_date,
            "security_id": security_id,
            "raw_scores": raw_scores,
            # Constant causal exposures make this a pure numerical test of the
            # frozen residual operator without planting another return signal.
            "exposures": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "trailing_63_session_adv": 100_000_000.0,
            "prediction_row_receipt_sha256": semantic_sha256(
                ("planted-prediction", setting_id, session_date, security_id)
            ),
            "feature_row_receipt_sha256": semantic_sha256(
                ("planted-feature-row", session_date, security_id)
            ),
            "feature_accounting_row_inventory_sha256": semantic_sha256(
                ("planted-accounting", session_date)
            ),
        }
        inputs.append(
            MassiveProfitabilityResidualInputRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    result = build_massive_profitability_residual_scores_v1(
        setting_id=setting_id,
        fold_index=0,
        evaluation_plan_semantic_receipt_sha256=semantic_sha256(
            "planted-nonauthorizing-evaluation-plan"
        ),
        prediction_semantic_receipt_sha256=semantic_sha256(
            ("planted-prediction-inventory", setting_id)
        ),
        rows=inputs,
    )
    assert result.outer_evaluation_authorized is False
    return result


def _planted_selected_tranches(
    *,
    residual: MassiveProfitabilityResidualScoresV1,
    target_accounting: tuple[MassiveProfitabilityTargetAccountingAuthorityV2, ...],
) -> MassiveProfitabilitySelectedTranchesV1:
    paths = {
        (authority.decision_session_date, row.security_id): row
        for authority in target_accounting
        for row in authority.rows
    }
    target_inventory = semantic_sha256(
        tuple(sorted(row.semantic_receipt_sha256 for row in target_accounting))
    )
    grouped: dict[str, list[MassiveProfitabilityResidualScoreRowV1]] = {}
    for row in residual.rows:
        grouped.setdefault(row.decision_session_date, []).append(row)
    positions = []
    horizons = (1, 5, 21, 63)
    for session_date, raw_rows in sorted(grouped.items()):
        rows = list(raw_rows)
        tail_count = len(rows) // 5
        for horizon_index, horizon in enumerate(horizons):
            sides = (
                (
                    "short",
                    sorted(
                        rows,
                        key=lambda row: (
                            row.residual_scores[horizon_index],
                            row.security_id,
                        ),
                    )[:tail_count],
                ),
                (
                    "long",
                    sorted(
                        rows,
                        key=lambda row: (
                            -row.residual_scores[horizon_index],
                            row.security_id,
                        ),
                    )[:tail_count],
                ),
            )
            for side, selected_rows in sides:
                for tail_rank, score in enumerate(selected_rows, start=1):
                    path = paths[(session_date, score.security_id)]
                    weight = (0.5 if side == "long" else -0.5) / (horizon * tail_count)
                    body = {
                        "decision_session_date": session_date,
                        "security_id": score.security_id,
                        "horizon_sessions": horizon,
                        "side": side,
                        "tail_rank": tail_rank,
                        "signed_entry_weight": weight,
                        "residual_score": score.residual_scores[horizon_index],
                        "trailing_63_session_adv": score.trailing_63_session_adv,
                        "residual_score_row_receipt_sha256": score.receipt_sha256,
                        "target_path_receipt_sha256": path.receipt_sha256,
                        "unresolved_terminal_fallback_session_offset": None,
                    }
                    positions.append(
                        MassiveProfitabilitySelectedTranchePositionV1(
                            **body,  # type: ignore[arg-type]
                            receipt_sha256=semantic_sha256(body),
                        )
                    )
    ordered = tuple(
        sorted(
            positions,
            key=lambda row: (
                row.decision_session_date,
                row.horizon_sessions,
                row.side,
                row.tail_rank,
            ),
        )
    )
    position_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in ordered))
    body = {
        "setting_id": residual.setting_id,
        "fold_index": residual.fold_index,
        "positions": tuple(asdict(row) for row in ordered),
        "residual_scores_semantic_receipt_sha256": residual.semantic_receipt_sha256,
        "target_accounting_inventory_sha256": target_inventory,
        "position_inventory_sha256": position_inventory,
        "path_support_complete": True,
        "direction_safe_terminal_support_complete": True,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilitySelectedTranchesV1(
        setting_id=residual.setting_id,
        fold_index=residual.fold_index,
        positions=ordered,
        residual_scores_semantic_receipt_sha256=residual.semantic_receipt_sha256,
        target_accounting_inventory_sha256=target_inventory,
        position_inventory_sha256=position_inventory,
        path_support_complete=True,
        direction_safe_terminal_support_complete=True,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def _planted_composite_mean_net_20bp(
    *,
    setting_id: str,
    scores: dict[tuple[str, str], tuple[float, ...]],
    target_accounting: tuple[MassiveProfitabilityTargetAccountingAuthorityV2, ...],
) -> tuple[float, tuple[float, float, float]]:
    residual = _planted_residual_scores(setting_id=setting_id, scores=scores)
    selected = _planted_selected_tranches(
        residual=residual, target_accounting=target_accounting
    )
    pnl = evaluate_massive_profitability_fixed_tranches_v1(
        selected=selected, target_accounting=target_accounting
    )
    scaling_body = {
        "fold_index": 0,
        "horizon_volatility": (1.0, 1.0, 1.0, 1.0),
        "horizon_weights": (0.25, 0.25, 0.25, 0.25),
        "fit_source_receipt_sha256": semantic_sha256("planted-fixed-horizon-weights"),
    }
    scaling = MassiveProfitabilityHorizonRiskScalingV1(
        fold_index=0,
        horizon_volatility=(1.0, 1.0, 1.0, 1.0),
        horizon_weights=(0.25, 0.25, 0.25, 0.25),
        fit_source_receipt_sha256=semantic_sha256("planted-fixed-horizon-weights"),
        receipt_sha256=semantic_sha256(scaling_body),
    )
    composite = build_massive_profitability_composite_pnl_v1(
        pnl=pnl, risk_scaling=scaling
    )
    means = tuple(
        sum(row.net_returns[index] for row in composite.rows) / len(composite.rows)
        for index in range(3)
    )
    assert composite.profitability_reporting_authorized is False
    assert composite.lockbox_access_authorized is False
    return means[1], (means[0], means[1], means[2])


def _planted_dataset(
    indices: tuple[int, ...],
) -> MassiveProfitabilityTournamentDatasetV1:
    rows = tuple(_planted_date_tensor(index) for index in indices)
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


def _average_outer_rank_ic(
    *,
    scores: dict[tuple[str, str], tuple[float, ...]],
    dataset: MassiveProfitabilityTournamentDatasetV1,
    session_dates: tuple[str, ...],
) -> float:
    by_date = dataset.by_date()
    values: list[float] = []
    for session_date in session_dates:
        row = by_date[session_date]
        prediction = torch.tensor(
            [scores[(session_date, security_id)][0] for security_id in row.security_ids]
        )
        target = row.target_values[:, 0]
        prediction_rank = torch.argsort(
            torch.argsort(prediction, stable=True), stable=True
        ).float()
        target_rank = torch.argsort(
            torch.argsort(target, stable=True), stable=True
        ).float()
        values.append(
            float(torch.corrcoef(torch.stack((prediction_rank, target_rank)))[0, 1])
        )
    return sum(values) / len(values)


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
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _plan(
    fold: MassiveProfitabilityOuterFoldPlanV1,
) -> MassiveProfitabilityTournamentPlanV1:
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
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
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


def test_planted_incremental_tape_alpha_recovers_through_net_pnl_canary() -> None:
    fold_source = _fold()
    fold = adapt_massive_profitability_training_fold_v1(fold_source)
    plan = _plan(fold_source)
    dataset = _planted_dataset(
        tuple(range(756)) + tuple(range(819, 945)) + tuple(range(1008, 1134))
    )
    config = MassiveProfitabilityTrainingConfigV1(
        learning_rate=2e-2,
        maximum_epochs=6,
        early_stopping_patience=2,
        complete_dates_per_batch=189,
    )
    ensemble_scores: dict[str, dict[tuple[str, str], tuple[float, ...]]] = {}
    recovery_runs: dict[tuple[str, int], MassiveProfitabilityTrainedRunV1] = {}
    recovery_prediction_rows: dict[
        tuple[str, int], tuple[MassiveProfitabilityPredictionRowV1, ...]
    ] = {}
    for setting_id in ("MV02", "MV04", "MV04-SHUFFLE"):
        seed_scores: list[dict[tuple[str, str], tuple[float, ...]]] = []
        for seed in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1:
            run = train_massive_profitability_fold_v1(
                dataset=dataset,
                tournament_plan=plan,
                fold=fold,
                setting_id=setting_id,
                seed=seed,
                config=config,
            )
            assert run.outer_prediction_authorized is False
            rows = build_massive_profitability_recovery_predictions_for_test_v1(
                dataset=dataset,
                tournament_plan=plan,
                fold=fold_source,
                run=run,
            )
            recovery_runs[(setting_id, seed)] = run
            recovery_prediction_rows[(setting_id, seed)] = rows
            seed_scores.append(
                {(row.decision_session_date, row.security_id): row.mean for row in rows}
            )
        keys = tuple(seed_scores[0])
        ensemble_scores[setting_id] = {
            key: tuple(
                sum(values[key][horizon] for values in seed_scores) / len(seed_scores)
                for horizon in range(4)
            )
            for key in keys
        }

    replayed_run = train_massive_profitability_fold_v1(
        dataset=dataset,
        tournament_plan=plan,
        fold=fold,
        setting_id="MV04",
        seed=0,
        config=config,
    )
    committed_run = recovery_runs[("MV04", 0)]
    assert replayed_run.run_receipt_sha256 == committed_run.run_receipt_sha256
    assert replayed_run.model_state_sha256 == committed_run.model_state_sha256
    assert tuple(name for name, _ in replayed_run.model_state) == tuple(
        name for name, _ in committed_run.model_state
    )
    assert all(
        torch.equal(replayed, committed)
        for (_, replayed), (_, committed) in zip(
            replayed_run.model_state, committed_run.model_state, strict=True
        )
    )
    assert (
        build_massive_profitability_recovery_predictions_for_test_v1(
            dataset=dataset,
            tournament_plan=plan,
            fold=fold_source,
            run=replayed_run,
        )
        == recovery_prediction_rows[("MV04", 0)]
    )

    mv00_scores: dict[tuple[str, str], tuple[float, ...]] = {}
    mapping = dataset.by_date()
    for session_date in fold_source.outer_test_session_dates:
        row = mapping[session_date]
        score = massive_profitability_mv00_scores_v1(
            bars_values=row.bars_values, bars_valid=row.bars_valid
        )
        mv00_scores.update(
            {
                (session_date, security_id): tuple(
                    float(value) for value in score[index]
                )
                for index, security_id in enumerate(row.security_ids)
            }
        )
    rank_ic = {
        setting_id: _average_outer_rank_ic(
            scores=scores,
            dataset=dataset,
            session_dates=fold_source.outer_test_session_dates,
        )
        for setting_id, scores in ensemble_scores.items()
    }
    rank_ic["MV00"] = _average_outer_rank_ic(
        scores=mv00_scores,
        dataset=dataset,
        session_dates=fold_source.outer_test_session_dates,
    )
    assert rank_ic["MV04"] > rank_ic["MV02"] > rank_ic["MV00"], rank_ic
    assert abs(rank_ic["MV04-SHUFFLE"] - rank_ic["MV02"]) < 0.15
    assert rank_ic["MV04"] - rank_ic["MV04-SHUFFLE"] > 0.25

    target_accounting = tuple(
        _planted_target_authority(index=index) for index in range(1008, 1134)
    )
    pnl_scores = {**ensemble_scores, "MV00": mv00_scores}
    pnl = {
        setting_id: _planted_composite_mean_net_20bp(
            setting_id=setting_id,
            scores=scores,
            target_accounting=target_accounting,
        )
        for setting_id, scores in pnl_scores.items()
    }
    mean_net_20bp = {setting_id: value[0] for setting_id, value in pnl.items()}
    assert mean_net_20bp["MV04"] > mean_net_20bp["MV02"] > mean_net_20bp["MV00"], (
        mean_net_20bp
    )
    assert abs(mean_net_20bp["MV04-SHUFFLE"] - mean_net_20bp["MV02"]) < 1e-4
    assert mean_net_20bp["MV04"] - mean_net_20bp["MV04-SHUFFLE"] > 1e-4
    for _, cost_ladder in pnl.values():
        assert cost_ladder[0] > cost_ladder[1] > cost_ladder[2]

    reversed_mv04 = {
        key: tuple(-value for value in scores)
        for key, scores in ensemble_scores["MV04"].items()
    }
    reversed_net, _ = _planted_composite_mean_net_20bp(
        setting_id="MV04",
        scores=reversed_mv04,
        target_accounting=target_accounting,
    )
    assert reversed_net < 0.0 < mean_net_20bp["MV04"]


def test_tournament_plan_and_runs_never_authorize_profit_reporting() -> None:
    plan = _plan(_fold())
    payload = asdict(plan)
    assert payload["profitability_reporting_authorized"] is False
    assert payload["lockbox_access_authorized"] is False
    assert payload["reinforcement_learning_authorized"] is False
