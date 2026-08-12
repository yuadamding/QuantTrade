from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution.cost_aware_active_policy_v2 import (
    M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS,
    M03RV9CostAwareActivePolicyError,
    build_cost_aware_active_proposal_v2,
)


def _inputs() -> dict[str, torch.Tensor]:
    anchor = torch.tensor(
        [[0.20, 0.10, 0.25, 0.25, 0.20]],
        dtype=torch.float64,
    )
    return {
        "post_exit_derisk_anchor_weights": anchor,
        "benchmark_weights": anchor.clone(),
        "selected_alpha_mean": torch.tensor(
            [[0.0, 0.100, 0.040, 0.020, -0.040]],
            dtype=torch.float64,
        ),
        "selected_alpha_scale": torch.full_like(anchor, 0.010),
        "one_way_cost": torch.full_like(anchor, 0.002),
        "learned_release": torch.tensor(
            [[0.0, 0.10, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        "explicit_derisk_amount": torch.tensor([0.02], dtype=torch.float64),
        "held_mask": torch.tensor([[False, True, True, True, True]]),
        "trade_mask": torch.ones_like(anchor, dtype=torch.bool),
        "risk_asset_caps": torch.tensor(
            [[1.0, 0.40, 0.40, 0.40, 0.40]],
            dtype=torch.float64,
        ),
        "signal_confidence": torch.tensor([0.8], dtype=torch.float64),
        "covariance_factor": torch.zeros((1, 5, 2), dtype=torch.float64),
        "specific_variance": torch.full_like(anchor, 1.0e-4),
    }


def test_learned_exit_proceeds_replace_risk_but_not_explicit_derisk() -> None:
    inputs = _inputs()
    result = build_cost_aware_active_proposal_v2(**inputs)

    assert result.learned_exit_proceeds.item() == pytest.approx(0.10)
    assert result.replacement_budget.item() == pytest.approx(0.08)
    assert result.replacement_used.item() == pytest.approx(0.08)
    assert result.replacement_one_way_turnover.item() == pytest.approx(0.08)
    assert result.replacement_anchor_weights[0, 0].item() == pytest.approx(0.12)
    assert result.replacement_buy_weights.sum().item() == pytest.approx(0.08)
    assert result.total_post_exit_one_way_turnover.item() == pytest.approx(
        result.replacement_used.item()
        + result.requested_incremental_one_way_turnover.item()
    )
    assert result.proximal_iterations == M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS


def test_same_step_repurchase_is_forbidden_then_allowed_next_decision() -> None:
    inputs = _inputs()
    first = build_cost_aware_active_proposal_v2(**inputs)
    assert not first.same_step_buy_mask[0, 1]
    assert first.replacement_buy_weights[0, 1].item() == 0.0
    assert first.reallocation_buy_weights[0, 1].item() == 0.0

    second_inputs = {
        **inputs,
        "post_exit_derisk_anchor_weights": first.requested_weights.detach(),
        "benchmark_weights": first.requested_weights.detach().clone(),
        "learned_release": torch.zeros_like(inputs["learned_release"]),
        "explicit_derisk_amount": torch.zeros_like(inputs["explicit_derisk_amount"]),
    }
    second = build_cost_aware_active_proposal_v2(**second_inputs)
    assert second.same_step_buy_mask[0, 1]
    assert second.reallocation_buy_weights[0, 1].item() > 0.0


def test_probability_gate_and_portfolio_cost_suppress_untradeable_edge() -> None:
    inputs = _inputs()
    low_cost = build_cost_aware_active_proposal_v2(**inputs)
    high_cost = build_cost_aware_active_proposal_v2(
        **{
            **inputs,
            "one_way_cost": torch.full_like(inputs["one_way_cost"], 0.20),
        }
    )

    assert low_cost.replacement_used.item() > 0.0
    assert high_cost.replacement_used.item() == 0.0
    assert high_cost.requested_incremental_one_way_turnover.item() == 0.0
    assert high_cost.buy_gate[:, 1:].max().item() == 0.0


def test_optional_reallocation_is_zero_sum_bounded_and_differentiable() -> None:
    inputs = _inputs()
    mean = inputs["selected_alpha_mean"].clone().requires_grad_(True)
    result = build_cost_aware_active_proposal_v2(
        **{**inputs, "selected_alpha_mean": mean}
    )

    torch.testing.assert_close(
        result.reallocation_buy_weights.sum(-1),
        result.reallocation_sell_weights.sum(-1),
    )
    torch.testing.assert_close(
        result.requested_weights.sum(-1),
        torch.ones(1, dtype=torch.float64),
    )
    assert result.requested_incremental_one_way_turnover.item() <= 0.016 + 1.0e-12
    objective = (
        result.requested_weights
        * torch.tensor([[0.0, 0.0, 2.0, 0.5, -1.0]], dtype=torch.float64)
    ).sum()
    objective.backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad[:, 1:].abs().sum().item() > 0.0


def test_malformed_scale_risk_and_mutated_schema_fail_closed() -> None:
    inputs = _inputs()
    with pytest.raises(M03RV9CostAwareActivePolicyError, match="contract"):
        build_cost_aware_active_proposal_v2(
            **{
                **inputs,
                "selected_alpha_scale": torch.zeros_like(
                    inputs["selected_alpha_scale"]
                ),
            }
        )
    with pytest.raises(M03RV9CostAwareActivePolicyError, match="risk tensors"):
        build_cost_aware_active_proposal_v2(
            **{
                **inputs,
                "covariance_factor": torch.zeros((1, 4, 2), dtype=torch.float64),
            }
        )
    result = build_cost_aware_active_proposal_v2(**inputs)
    with pytest.raises(M03RV9CostAwareActivePolicyError, match="reconciliation"):
        replace(result, schema="drifted").validate()
