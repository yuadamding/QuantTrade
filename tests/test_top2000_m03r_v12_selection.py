from __future__ import annotations

from dataclasses import replace

import torch

from rl_quant.training.top2000_m03r_v12_selection import (
    build_m03r_v12_bootstrap_plan,
    build_m03r_v12_fold_evidence,
    qualify_m03r_v12_predictive_candidate,
)


def _fold(fold: int, *, incremental_turnover: float = 0.002) -> object:
    length = 40
    start = fold * 100
    return build_m03r_v12_fold_evidence(
        setting_index=1,
        fold_index=fold,
        horizon_sessions=3,
        score_session_index=torch.arange(start, start + length),
        gross_active_return=torch.full((length,), 0.001, dtype=torch.float64),
        policy_one_way_turnover=torch.full(
            (length,), 0.01 + incremental_turnover, dtype=torch.float64
        ),
        benchmark_one_way_turnover=torch.full((length,), 0.01, dtype=torch.float64),
        top_bottom_spread=torch.full((length,), 0.002, dtype=torch.float64),
        requested_to_executed_retention=torch.full(
            (length,), 0.80, dtype=torch.float64
        ),
        mean_spearman_rank_ic=0.03,
        median_spearman_rank_ic=0.02,
        positive_ic_date_fraction=0.65,
        mean_prediction_cross_sectional_std=0.02,
        mean_target_cross_sectional_std=0.02,
        checkpoint_file_sha256=f"{fold + 1:x}" * 64,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
    )


def test_joint_block_gate_and_aggregate_break_even_pass_known_process() -> None:
    folds = tuple(_fold(fold) for fold in range(6))
    plan = build_m03r_v12_bootstrap_plan(
        tuple(row.score_session_index for row in folds),  # type: ignore[attr-defined]
        bootstrap_seed=17,
    )
    result = qualify_m03r_v12_predictive_candidate(folds, plan)  # type: ignore[arg-type]
    assert result.passed
    assert result.gross_active_return_lcb > 0.0
    assert result.net_active_return_10bp_lcb > 0.0
    assert result.break_even_category == "finite-positive"
    assert result.break_even_one_way_cost_basis_points is not None
    assert result.break_even_one_way_cost_basis_points > 10.0


def test_break_even_uses_aggregate_sign_cases_not_fold_ratio_average() -> None:
    folds = tuple(_fold(fold, incremental_turnover=-0.001) for fold in range(6))
    plan = build_m03r_v12_bootstrap_plan(
        tuple(row.score_session_index for row in folds),  # type: ignore[attr-defined]
        bootstrap_seed=23,
    )
    result = qualify_m03r_v12_predictive_candidate(folds, plan)  # type: ignore[arg-type]
    assert result.break_even_category == "favorable-cost-dominance"
    assert result.break_even_one_way_cost_basis_points is None
    weak = tuple(
        replace(
            row,
            gross_active_return=-row.gross_active_return,
            array_sha256=(
                row.array_sha256[0],
                row.array_sha256[1],
                row.array_sha256[2],
                row.array_sha256[3],
                row.array_sha256[4],
                row.array_sha256[5],
            ),
        )
        for row in folds
    )
    # Rebuild instead of accepting a rehashed mutation through the typed constructor.
    negative = tuple(
        build_m03r_v12_fold_evidence(
            setting_index=row.setting_index,
            fold_index=row.fold_index,
            horizon_sessions=row.horizon_sessions,
            score_session_index=row.score_session_index,
            gross_active_return=-row.gross_active_return,
            policy_one_way_turnover=row.policy_one_way_turnover,
            benchmark_one_way_turnover=row.benchmark_one_way_turnover,
            top_bottom_spread=row.top_bottom_spread,
            requested_to_executed_retention=row.requested_to_executed_retention,
            mean_spearman_rank_ic=row.mean_spearman_rank_ic,
            median_spearman_rank_ic=row.median_spearman_rank_ic,
            positive_ic_date_fraction=row.positive_ic_date_fraction,
            mean_prediction_cross_sectional_std=(
                row.mean_prediction_cross_sectional_std
            ),
            mean_target_cross_sectional_std=row.mean_target_cross_sectional_std,
            checkpoint_file_sha256=row.checkpoint_file_sha256,
            episode_schedule_sha256=row.episode_schedule_sha256,
            residual_operator_root_sha256=row.residual_operator_root_sha256,
        )
        for row in folds
    )
    del weak
    failed = qualify_m03r_v12_predictive_candidate(negative, plan)
    assert not failed.passed
    assert failed.break_even_category == "no-positive-break-even"
