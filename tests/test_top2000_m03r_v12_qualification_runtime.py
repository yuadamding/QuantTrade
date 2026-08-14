from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import rl_quant.training.top2000_m03r_v12_runtime as sleeve_runtime
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
    resolve_m03r_v12_setting,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import M03RV12LoadedCheckpoint
from rl_quant.training.top2000_m03r_v12_objective import (
    M03RV12PredictiveBatch,
)
from rl_quant.training.top2000_m03r_v12_policy import M03RV12HeadIdentity
from rl_quant.training.top2000_m03r_v12_qualification_runtime import (
    M03RV12FoldQualificationLineage,
    M03RV12QualificationRuntimeError,
    build_m03r_v12_fold_qualification_lineage,
    qualify_m03r_v12_round_trip_candidate,
)
from rl_quant.training.top2000_m03r_v12_runtime import M03RV12SimpleSleeveTrace
from rl_quant.training.top2000_m03r_v12_selection import (
    build_m03r_v12_bootstrap_plan,
    build_m03r_v12_fold_evidence,
)


def _head(setting_index: int) -> M03RV12HeadIdentity:
    return M03RV12HeadIdentity(
        setting_id=M03R_V12_SETTING_IDS[setting_index],
        selected_alpha_horizon=3,
        economic_mean_head_state_sha256="6" * 64,
        economic_scale_head_state_sha256="7" * 64,
        rank_score_head_state_sha256="8" * 64,
    )


def _lineage(fold_index: int) -> M03RV12FoldQualificationLineage:
    checkpoint_sha = f"{fold_index + 1:x}" * 64
    evidence = build_m03r_v12_fold_evidence(
        setting_index=1,
        fold_index=fold_index,
        horizon_sessions=3,
        score_session_index=torch.arange(
            fold_index * 100, fold_index * 100 + 40, dtype=torch.int64
        ),
        gross_active_return=torch.full((40,), 0.001, dtype=torch.float64),
        policy_one_way_turnover=torch.full((40,), 0.012, dtype=torch.float64),
        benchmark_one_way_turnover=torch.full((40,), 0.010, dtype=torch.float64),
        top_bottom_spread=torch.full((40,), 0.002, dtype=torch.float64),
        requested_to_executed_retention=torch.full((40,), 0.8, dtype=torch.float64),
        mean_spearman_rank_ic=0.03,
        median_spearman_rank_ic=0.02,
        positive_ic_date_fraction=0.65,
        mean_prediction_cross_sectional_std=0.02,
        mean_target_cross_sectional_std=0.02,
        checkpoint_file_sha256=checkpoint_sha,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
    )
    loaded = M03RV12LoadedCheckpoint(
        setting_index=1,
        setting_id=M03R_V12_SETTING_IDS[1],
        fold_index=fold_index,
        completed_updates=64,
        selected_horizon_sessions=3,
        model_state_sha256="c" * 64,
        checkpoint_file_sha256=checkpoint_sha,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
        source_array_sha256="d" * 64,
        asset_axis_sha256="e" * 64,
        head_identity=_head(1),
        protocol_sha256=M03R_V12_PROTOCOL_SHA256,
    )
    result = M03RV12FoldQualificationLineage(
        loaded_checkpoint=loaded,
        fold_evidence=evidence,
        evaluation_trace_sha256=f"{fold_index + 7:x}" * 64,
        qualification_source_array_sha256="d" * 64,
        qualification_asset_axis_sha256="e" * 64,
        qualification_residual_operator_root_sha256="b" * 64,
    )
    result.validate()
    return result


def _qualification_batch_and_trace() -> tuple[
    M03RV12LoadedCheckpoint,
    M03RV12PredictiveBatch,
    M03RV12SimpleSleeveTrace,
]:
    dates, assets = 3, 4
    prediction = torch.zeros(
        (dates, assets, len(M03R_V12_HORIZONS)), dtype=torch.float64
    )
    target = torch.zeros_like(prediction)
    for date in range(dates):
        values = torch.tensor([0.0, -0.02, 0.01, 0.03]) + date * 0.001
        prediction[date] = values.unsqueeze(-1)
        target[date] = values.unsqueeze(-1) * 0.5
    valid = torch.ones_like(prediction, dtype=torch.bool)
    valid[:, 0] = False
    operator_receipts = tuple(
        f"{index + 1:x}" * 64 for index in range(dates * len(M03R_V12_HORIZONS))
    )
    batch = M03RV12PredictiveBatch(
        predicted_mean=prediction,
        predicted_log_scale=torch.full_like(prediction, -4.0),
        predicted_rank_score=prediction.clone(),
        target_log_return=target,
        valid=valid,
        origin_indices=torch.tensor([20, 21, 22]),
        split="qualification",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=20,
        split_stop_exclusive=100,
        source_array_sha256="d" * 64,
        asset_axis_sha256="e" * 64,
        exposure_receipt_sha256="f" * 64,
        setting=resolve_m03r_v12_setting(1),
        residual_operator_receipt_sha256=operator_receipts,
        available_risky_asset_count=(3,) * len(operator_receipts),
        factor_qualified_risky_asset_count=(3,) * len(operator_receipts),
        effective_design_rank=(2,) * len(operator_receipts),
        weighted_residual_degrees_of_freedom=(1,) * len(operator_receipts),
        residual_operators=tuple(
            SimpleNamespace(receipt_sha256=value) for value in operator_receipts
        ),  # type: ignore[arg-type]
    )
    loaded = M03RV12LoadedCheckpoint(
        setting_index=1,
        setting_id=M03R_V12_SETTING_IDS[1],
        fold_index=0,
        completed_updates=64,
        selected_horizon_sessions=3,
        model_state_sha256="c" * 64,
        checkpoint_file_sha256="1" * 64,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
        source_array_sha256="d" * 64,
        asset_axis_sha256="e" * 64,
        head_identity=_head(1),
    )
    arrays = (
        torch.full((dates,), 0.001, dtype=torch.float64),
        torch.zeros(dates, dtype=torch.float64),
        torch.full((dates,), 0.012, dtype=torch.float64),
        torch.full((dates,), 0.010, dtype=torch.float64),
        torch.full((dates, assets), 0.25, dtype=torch.float64),
        torch.full((dates, assets), 0.25, dtype=torch.float64),
        torch.full((dates,), 0.8, dtype=torch.float64),
    )
    provisional = M03RV12SimpleSleeveTrace(
        setting_index=1,
        setting_id=M03R_V12_SETTING_IDS[1],
        fold_index=0,
        selected_horizon_sessions=3,
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        checkpoint_model_state_sha256=loaded.model_state_sha256,
        source_receipt_sha256=loaded.source_array_sha256,
        asset_axis_sha256=loaded.asset_axis_sha256,
        risk_state_sha256="2" * 64,
        risk_manifest_sha256="3" * 64,
        signal_operator_receipt_sha256=tuple(
            operator_receipts[
                date * len(M03R_V12_HORIZONS) + M03R_V12_HORIZONS.index(3)
            ]
            for date in range(dates)
        ),
        state_start_index=20,
        policy_gross_returns=arrays[0],
        benchmark_gross_returns=arrays[1],
        policy_one_way_turnover=arrays[2],
        benchmark_one_way_turnover=arrays[3],
        requested_weight_trace=arrays[4],
        projected_weight_trace=arrays[5],
        requested_to_executed_retention=arrays[6],
        array_sha256=tuple(sleeve_runtime._tensor_sha256(value) for value in arrays),
        trace_sha256="0" * 64,
    )
    trace = replace(
        provisional,
        trace_sha256=sleeve_runtime._sha256(provisional.unsigned_payload()),
    )
    trace.validate()
    return loaded, batch, trace


def test_round_trip_candidate_requires_six_exact_checkpoint_lineages() -> None:
    lineages = tuple(_lineage(fold) for fold in range(6))
    bootstrap = build_m03r_v12_bootstrap_plan(
        tuple(row.fold_evidence.score_session_index for row in lineages),
        bootstrap_seed=17,
    )
    result = qualify_m03r_v12_round_trip_candidate(lineages, bootstrap)
    assert result.qualification.passed
    assert len(result.fold_lineage_sha256) == 6


def test_round_trip_candidate_rejects_rehashed_evidence_for_other_checkpoint() -> None:
    lineage = _lineage(0)
    drifted = replace(
        lineage,
        fold_evidence=replace(
            lineage.fold_evidence,
            checkpoint_file_sha256="f" * 64,
        ),
    )
    with pytest.raises(M03RV12QualificationRuntimeError, match="reloaded checkpoint"):
        drifted.validate()


def test_fold_builder_derives_metrics_from_exact_batch_and_v12_trace() -> None:
    loaded, batch, trace = _qualification_batch_and_trace()
    result = build_m03r_v12_fold_qualification_lineage(loaded, batch, trace)
    assert result.evaluation_trace_sha256 == trace.trace_sha256
    assert result.fold_evidence.mean_spearman_rank_ic == pytest.approx(1.0)
    assert result.fold_evidence.positive_ic_date_fraction == 1.0
    assert result.qualification_residual_operator_root_sha256 == (
        result.fold_evidence.residual_operator_root_sha256
    )
    assert result.qualification_residual_operator_root_sha256 != (
        loaded.residual_operator_root_sha256
    )


def test_fold_builder_rejects_trace_using_other_operator_inventory() -> None:
    loaded, batch, trace = _qualification_batch_and_trace()
    drifted = replace(
        trace,
        signal_operator_receipt_sha256=(
            "9" * 64,
            *trace.signal_operator_receipt_sha256[1:],
        ),
        trace_sha256="0" * 64,
    )
    drifted = replace(
        drifted,
        trace_sha256=sleeve_runtime._sha256(drifted.unsigned_payload()),
    )
    with pytest.raises(M03RV12QualificationRuntimeError, match="drifted"):
        build_m03r_v12_fold_qualification_lineage(loaded, batch, drifted)
