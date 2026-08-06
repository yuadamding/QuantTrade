from __future__ import annotations

from itertools import pairwise

import torch

from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DevelopmentPolicy,
    render_top2000_m03r_v7_development_folds,
    top2000_m03r_v7_persistence_penalty,
)


def test_top2000_development_folds_are_six_expanding_disjoint_score_windows() -> None:
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    assert len(folds) == 6
    assert [fold.training_state_stop_exclusive for fold in folds] == [
        378,
        471,
        564,
        657,
        750,
        843,
    ]
    assert all(
        fold.validation_decision_stop_exclusive
        - fold.validation_decision_start
        == 63
        for fold in folds
    )
    assert all(
        left.validation_decision_stop_exclusive
        <= right.validation_decision_start
        for left, right in pairwise(folds)
    )


def test_all_twelve_development_policy_routes_are_compact_and_distinctly_bound() -> None:
    parameter_counts: list[int] = []
    for setting_id in M03R_TOP2000_DEV_SETTING_IDS:
        policy = Top2000M03RV7DevelopmentPolicy(
            setting_id,
            token_dim=512,
            raw_stock_chunk=512,
        )
        assert policy.setting.setting_id == setting_id
        assert all(policy.core._raw_day_mask(252))
        parameter_counts.append(sum(value.numel() for value in policy.parameters()))
    assert len(parameter_counts) == 12
    assert max(parameter_counts) < 7_000_000


def test_fixed_hazard_ablation_reaches_native_fixed_prior_route() -> None:
    policy = Top2000M03RV7DevelopmentPolicy(
        "A08-fixed-exit-hazard-top2000-dev-v1",
        token_dim=16,
        raw_stock_chunk=32,
    )
    assert policy.core.config.hold30_fixed_hazard_residual == 0.0


def test_persistence_penalty_is_proportional_and_mature_sales_cannot_dilute() -> None:
    young = torch.zeros(61, dtype=torch.float64, requires_grad=True)
    young = young.clone()
    young[0] = 0.01
    mature = torch.zeros(61, dtype=torch.float64, requires_grad=True)
    mature = mature.clone()
    mature[45] = 0.99

    young_only = top2000_m03r_v7_persistence_penalty(
        young,
        coefficient_basis_points=5.0,
        warmup_multiplier=1.0,
        valid_decision_session_count=1,
    )
    mixed = top2000_m03r_v7_persistence_penalty(
        young + mature,
        coefficient_basis_points=5.0,
        warmup_multiplier=1.0,
        valid_decision_session_count=1,
    )
    full = top2000_m03r_v7_persistence_penalty(
        young * 100.0,
        coefficient_basis_points=5.0,
        warmup_multiplier=1.0,
        valid_decision_session_count=1,
    )

    assert torch.allclose(mixed, young_only)
    assert torch.allclose(full, young_only * 100.0)
    young_gradient, mature_gradient = torch.autograd.grad(
        mixed,
        (young, mature),
    )
    assert young_gradient[0] > 0
    assert mature_gradient[45] == 0
