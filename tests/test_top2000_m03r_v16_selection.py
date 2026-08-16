from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    M03RV16PredictiveQualification,
    M03RV16SelectionError,
    build_m03r_v16_bootstrap_plan,
    build_m03r_v16_panel_decision,
    load_m03r_v16_panel_decision,
    write_m03r_v16_panel_decision,
    _draw_indices,
)


def _chronologies() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    geometries = render_m03r_v16_fold_geometries(1001)
    decisions = tuple(
        torch.arange(
            geometry.qualification_origin_start_inclusive,
            geometry.qualification_origin_stop_exclusive,
            dtype=torch.int64,
        )
        for geometry in geometries
    )
    executions = tuple(
        torch.arange(
            geometry.qualification_origin_start_inclusive,
            geometry.qualification_origin_stop_exclusive + 29,
            dtype=torch.int64,
        )
        for geometry in geometries
    )
    return decisions, executions


def _passing_qualification(setting_index: int) -> M03RV16PredictiveQualification:
    primary = setting_index == 2
    return M03RV16PredictiveQualification(
        setting_index=setting_index,
        setting_id=M03R_V16_SETTING_IDS[setting_index],
        fold_trace_sha256=tuple(f"{index:x}" * 64 for index in range(1, 6)),
        terminal_checkpoint_authority_sha256=tuple(
            f"{index:x}" * 64 for index in range(6, 11)
        ),
        qualified_score_authority_sha256=tuple(
            f"{index:x}" * 64 for index in range(11, 16)
        ),
        panel_schedule_sha256="a" * 64,
        bootstrap_plan_sha256="b" * 64,
        mean_projected_rank_ic=0.025,
        positive_mean_ic_fold_count=5,
        positive_spread_fold_count=5,
        annualized_gross_active_return=0.04,
        annualized_net_active_return_10bp=0.02,
        gross_active_lcb_by_block=(0.01, 0.008, 0.006),
        net_10bp_active_lcb_by_block=(0.005, 0.004, 0.003),
        spread_lcb_by_block=(0.001, 0.0008, 0.0006),
        break_even_category="finite-positive",
        break_even_one_way_cost_basis_points=12.0,
        absolute_policy_break_even_one_way_cost_basis_points=8.0,
        median_risk_projection_retention=0.9,
        minimum_fold_median_risk_projection_retention=0.8,
        median_weighted_cohort_age=15.0,
        gates_passed=True,
        primary_hypothesis_passed=primary,
        three_seed_confirmation_may_be_minted=primary,
    )


def _adequacy_receipts() -> tuple[str, ...]:
    return tuple(f"{index + 20:064x}" for index in range(5))


def test_v16_bootstrap_plan_is_fold_bounded_and_deterministic() -> None:
    decisions, executions = _chronologies()
    first = build_m03r_v16_bootstrap_plan(decisions, executions)
    second = build_m03r_v16_bootstrap_plan(decisions, executions)
    assert first == second
    assert first.block_sessions == (42, 30, 63)
    assert first.decision_fold_lengths == (63,) * 5
    assert first.execution_fold_lengths == (92,) * 5
    assert first.diagnostic_draw_sha256_by_block != first.economic_draw_sha256_by_block


def test_v16_bootstrap_is_nonwrapping_and_block63_resamples_folds() -> None:
    lengths = (63,) * 5
    draws = _draw_indices(lengths, block_sessions=42, stream=1)
    for destination in range(5):
        row = draws[0, destination * 63 : (destination + 1) * 63]
        assert len({int(value) // 63 for value in row}) == 1
        assert bool((row[1:42] == row[:41] + 1).all())
        assert bool((row[43:] == row[42:-1] + 1).all())
    full_fold = _draw_indices(lengths, block_sessions=63, stream=1)
    synthetic_fold_means = torch.arange(5, dtype=torch.float64).repeat_interleave(63)
    distribution = synthetic_fold_means[full_fold].mean(dim=1)
    assert float(distribution.std()) > 0.0


def test_v16_bootstrap_rejects_overlapping_execution_support() -> None:
    decisions, executions = _chronologies()
    rows = list(executions)
    rows[1] = torch.arange(int(rows[0][-1]), int(rows[0][-1]) + 92)
    with pytest.raises(M03RV16SelectionError, match="ordered and disjoint"):
        build_m03r_v16_bootstrap_plan(decisions, tuple(rows))


def test_v16_only_primary_r2_can_authorize_three_seed_confirmation() -> None:
    control = _passing_qualification(0)
    control.validate()
    assert control.gates_passed is True
    assert control.primary_hypothesis_passed is False
    assert control.three_seed_confirmation_may_be_minted is False

    primary = _passing_qualification(2)
    primary.validate()
    assert primary.primary_hypothesis_passed is True
    assert primary.three_seed_confirmation_may_be_minted is True
    assert primary.economic_generation_may_be_minted is False
    assert primary.reinforcement_learning_authorized is False


def test_v16_one_seed_gate_cannot_authorize_economic_or_rl_training() -> None:
    primary = _passing_qualification(2)
    with pytest.raises(M03RV16SelectionError, match="qualification"):
        replace(primary, economic_generation_may_be_minted=True).validate()
    with pytest.raises(M03RV16SelectionError, match="qualification"):
        replace(primary, reinforcement_learning_authorized=True).validate()


def test_v16_panel_decision_round_trip_preserves_primary_only_rule(
    tmp_path: Path,
) -> None:
    decisions, executions = _chronologies()
    bootstrap = build_m03r_v16_bootstrap_plan(decisions, executions)
    qualifications = tuple(
        replace(
            _passing_qualification(index),
            bootstrap_plan_sha256=bootstrap.receipt_sha256,
        )
        for index in range(3)
    )
    decision = build_m03r_v16_panel_decision(
        qualifications,
        bootstrap,
        primary_training_adequacy="adequate",
        primary_training_adequacy_receipt_sha256=_adequacy_receipts(),
    )
    assert decision.primary_hypothesis_passed is True
    assert decision.next_research_action == "three-seed-predictive-confirmation"
    assert decision.economic_generation_may_be_minted is False
    assert decision.reinforcement_learning_authorized is False
    path = tmp_path / "panel-decision.json"
    file_sha = write_m03r_v16_panel_decision(
        path, decision, qualifications, bootstrap
    )
    assert (
        load_m03r_v16_panel_decision(
            path,
            expected_file_sha256=file_sha,
            qualifications=qualifications,
            bootstrap=bootstrap,
        )
        == decision
    )


def test_v16_failed_primary_ends_daily_target_tuning() -> None:
    decisions, executions = _chronologies()
    bootstrap = build_m03r_v16_bootstrap_plan(decisions, executions)
    controls = tuple(
        replace(
            _passing_qualification(index),
            bootstrap_plan_sha256=bootstrap.receipt_sha256,
        )
        for index in range(2)
    )
    failed_primary = replace(
        _passing_qualification(2),
        bootstrap_plan_sha256=bootstrap.receipt_sha256,
        mean_projected_rank_ic=0.0,
        positive_mean_ic_fold_count=0,
        positive_spread_fold_count=0,
        annualized_gross_active_return=-0.01,
        annualized_net_active_return_10bp=-0.02,
        gross_active_lcb_by_block=(-0.01, -0.01, -0.01),
        net_10bp_active_lcb_by_block=(-0.02, -0.02, -0.02),
        spread_lcb_by_block=(-0.001, -0.001, -0.001),
        break_even_category="no-positive-break-even",
        break_even_one_way_cost_basis_points=None,
        gates_passed=False,
        primary_hypothesis_passed=False,
        three_seed_confirmation_may_be_minted=False,
    )
    decision = build_m03r_v16_panel_decision(
        (*controls, failed_primary),
        bootstrap,
        primary_training_adequacy="adequate",
        primary_training_adequacy_receipt_sha256=_adequacy_receipts(),
    )
    assert decision.next_research_action == "ordered-five-minute-representation"
    assert decision.daily_target_or_loss_tuning_authorized is False


def test_v16_inconclusive_fit_routes_to_a_fresh_longer_training_protocol() -> None:
    decisions, executions = _chronologies()
    bootstrap = build_m03r_v16_bootstrap_plan(decisions, executions)
    qualifications = tuple(
        replace(
            _passing_qualification(index),
            bootstrap_plan_sha256=bootstrap.receipt_sha256,
        )
        for index in range(3)
    )
    decision = build_m03r_v16_panel_decision(
        qualifications,
        bootstrap,
        primary_training_adequacy="inconclusive-undertrained",
        primary_training_adequacy_receipt_sha256=_adequacy_receipts(),
    )
    assert decision.next_research_action == "longer-training-protocol"
