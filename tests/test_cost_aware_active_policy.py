from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution.cost_aware_active_policy import (
    M03RV8CostAwareActivePolicyError,
    build_cost_aware_active_proposal,
)
from rl_quant.execution.hold30 import centered_benchmark_tilt


def _inputs() -> dict[str, torch.Tensor]:
    anchor = torch.tensor(
        [[0.10, 0.25, 0.25, 0.20, 0.20]],
        dtype=torch.float64,
    )
    return {
        "hazard_anchor_weights": anchor,
        "benchmark_weights": anchor.clone(),
        "expected_active_alpha": torch.tensor(
            [[0.0, 0.030, -0.020, 0.010, -0.010]],
            dtype=torch.float64,
        ),
        "uncertainty": torch.full_like(anchor, 0.002),
        "one_way_cost": torch.full_like(anchor, 0.002),
        "signal_confidence": torch.tensor([0.8], dtype=torch.float64),
        "held_mask": torch.tensor([[False, True, True, False, False]]),
        "trade_mask": torch.ones_like(anchor, dtype=torch.bool),
        "risk_asset_caps": torch.tensor(
            [[1.0, 0.40, 0.40, 0.40, 0.40]],
            dtype=torch.float64,
        ),
    }


def test_zero_confidence_preserves_the_hazard_anchor_exactly() -> None:
    inputs = _inputs()
    inputs["signal_confidence"] = torch.zeros(1, dtype=torch.float64)
    result = build_cost_aware_active_proposal(**inputs)

    assert torch.equal(result.requested_weights, inputs["hazard_anchor_weights"])
    assert torch.equal(result.requested_delta, torch.zeros_like(result.requested_delta))
    assert result.requested_incremental_one_way_turnover.item() == 0.0
    assert result.allowed_incremental_one_way_turnover.item() == 0.0


def test_cost_and_uncertainty_gate_shrinks_incremental_trading() -> None:
    inputs = _inputs()
    low_hurdle = build_cost_aware_active_proposal(**inputs)
    high_hurdle = build_cost_aware_active_proposal(
        **{
            **inputs,
            "one_way_cost": torch.full_like(inputs["one_way_cost"], 0.050),
            "uncertainty": torch.full_like(inputs["uncertainty"], 0.050),
        }
    )

    assert high_hurdle.requested_incremental_one_way_turnover.item() < (
        low_hurdle.requested_incremental_one_way_turnover.item()
    )
    assert high_hurdle.no_trade_gate[:, 1:].max().item() < (
        low_hurdle.no_trade_gate[:, 1:].max().item()
    )


def test_retention_hysteresis_has_a_lower_hurdle_than_new_entry() -> None:
    inputs = _inputs()
    inputs["expected_active_alpha"] = torch.tensor(
        [[0.0, 0.010, -0.010, 0.010, -0.010]],
        dtype=torch.float64,
    )
    inputs["held_mask"] = torch.tensor([[False, True, True, False, False]])
    result = build_cost_aware_active_proposal(
        **inputs,
        entry_hurdle_multiplier=1.0,
        retention_hurdle_multiplier=0.5,
    )

    assert result.centered_expected_active_alpha[0, 1].item() == pytest.approx(
        result.centered_expected_active_alpha[0, 3].item()
    )
    assert result.no_trade_gate[0, 1] > result.no_trade_gate[0, 3]
    assert result.no_trade_gate[0, 2] > result.no_trade_gate[0, 4]


def test_distinct_extreme_alpha_rankings_do_not_collapse_to_one_book() -> None:
    inputs = _inputs()
    common = {
        **inputs,
        "uncertainty": torch.zeros_like(inputs["uncertainty"]),
        "one_way_cost": torch.zeros_like(inputs["one_way_cost"]),
        "signal_confidence": torch.ones(1, dtype=torch.float64),
    }
    first_alpha = torch.tensor(
        [[0.0, 100.0, 50.0, -50.0, -100.0]],
        dtype=torch.float64,
    )
    second_alpha = torch.tensor(
        [[0.0, 50.0, 100.0, -50.0, -100.0]],
        dtype=torch.float64,
    )
    first = build_cost_aware_active_proposal(
        **{**common, "expected_active_alpha": first_alpha}
    )
    second = build_cost_aware_active_proposal(
        **{**common, "expected_active_alpha": second_alpha}
    )

    # The v7 score adapter clips both positive risky scores to +2 and therefore
    # loses their ordering. This is the local analogue of the Phase-0 collapse.
    old_first = centered_benchmark_tilt(
        first_alpha,
        inputs["benchmark_weights"],
        inputs["trade_mask"],
    )
    old_second = centered_benchmark_tilt(
        second_alpha,
        inputs["benchmark_weights"],
        inputs["trade_mask"],
    )
    assert torch.equal(old_first, old_second)

    assert not torch.equal(first.requested_weights, second.requested_weights)
    assert first.requested_weights[0, 1] > first.requested_weights[0, 2]
    assert second.requested_weights[0, 2] > second.requested_weights[0, 1]
    assert first.requested_incremental_one_way_turnover.item() == pytest.approx(0.02)
    assert second.requested_incremental_one_way_turnover.item() == pytest.approx(0.02)


def test_request_is_a_bounded_long_only_zero_sum_reallocation() -> None:
    inputs = _inputs()
    result = build_cost_aware_active_proposal(**inputs)

    torch.testing.assert_close(
        result.buy_weights.sum(-1),
        result.sell_weights.sum(-1),
    )
    torch.testing.assert_close(
        result.requested_delta.sum(-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.requested_weights.sum(-1),
        torch.ones(1, dtype=torch.float64),
    )
    assert bool((result.requested_weights >= 0.0).all())
    assert result.requested_incremental_one_way_turnover.item() <= 0.8 * 0.02 + 1.0e-12
    assert bool(
        (
            result.requested_weights[:, 1:] <= inputs["risk_asset_caps"][:, 1:] + 2.0e-7
        ).all()
    )


def test_alpha_path_has_finite_nonzero_gradients() -> None:
    inputs = _inputs()
    alpha = inputs["expected_active_alpha"].clone().requires_grad_(True)
    result = build_cost_aware_active_proposal(
        **{**inputs, "expected_active_alpha": alpha}
    )
    value = (
        result.requested_weights
        * torch.tensor([[0.0, 2.0, -1.0, 0.5, -0.25]], dtype=torch.float64)
    ).sum()
    value.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert alpha.grad[:, 1:].abs().sum().item() > 0.0


def test_malformed_hurdles_and_mutated_receipt_are_rejected() -> None:
    inputs = _inputs()
    with pytest.raises(
        M03RV8CostAwareActivePolicyError,
        match="retention hurdle",
    ):
        build_cost_aware_active_proposal(
            **inputs,
            entry_hurdle_multiplier=0.5,
            retention_hurdle_multiplier=1.0,
        )
    with pytest.raises(M03RV8CostAwareActivePolicyError, match="cost"):
        build_cost_aware_active_proposal(
            **{**inputs, "one_way_cost": -inputs["one_way_cost"]}
        )

    result = build_cost_aware_active_proposal(**inputs)
    with pytest.raises(M03RV8CostAwareActivePolicyError, match="schema"):
        replace(result, schema="drifted").validate()
