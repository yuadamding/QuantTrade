from __future__ import annotations

import torch

from rl_quant.execution.cost_aware_active_policy_v3 import (
    build_cost_aware_active_proposal_v3,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
)


def _inputs(mean_scale: float) -> dict[str, object]:
    anchor = torch.tensor([[0.20, 0.20, 0.20, 0.20, 0.20]], dtype=torch.float64)
    return {
        "post_exit_derisk_anchor_weights": anchor,
        "benchmark_weights": anchor.clone(),
        "selected_alpha_mean": torch.tensor(
            [[0.0, 2.0, 1.0, -1.0, -2.0]], dtype=torch.float64
        )
        * mean_scale,
        "selected_alpha_scale": torch.full_like(anchor, 0.01),
        "one_way_cost": torch.zeros_like(anchor),
        "learned_release": torch.zeros_like(anchor),
        "explicit_derisk_amount": torch.zeros(1, dtype=torch.float64),
        "held_mask": torch.tensor([[False, True, True, True, True]]),
        "trade_mask": torch.ones_like(anchor, dtype=torch.bool),
        "risk_asset_caps": torch.full_like(anchor, 0.80),
        "signal_confidence": torch.ones(1, dtype=torch.float64),
        "covariance_factor": torch.zeros((1, 5, 2), dtype=torch.float64),
        "specific_variance": torch.zeros_like(anchor),
        "selected_horizon_sessions": 21,
        "research_contract_sha256": M03R_V11_PROTOCOL_SHA256,
    }


def test_weak_signals_produce_vanishing_not_full_budget_turnover() -> None:
    turnovers = []
    for scale in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5):
        result = build_cost_aware_active_proposal_v3(**_inputs(scale))  # type: ignore[arg-type]
        turnovers.append(result.requested_incremental_one_way_turnover.item())
        assert (
            result.requested_incremental_one_way_turnover
            <= torch.minimum(result.desired_buy_mass, result.desired_sell_mass)
            + 1.0e-12
        )
    assert turnovers == sorted(turnovers, reverse=True)
    assert turnovers[-1] < turnovers[0] / 100.0
    assert turnovers[-1] < 1.0e-4


def test_soft_probability_gate_preserves_gradient_and_same_step_guard() -> None:
    inputs = _inputs(0.01)
    mean = inputs["selected_alpha_mean"].clone().requires_grad_(True)  # type: ignore[union-attr]
    release = torch.zeros_like(mean)
    release[0, 1] = 0.05
    anchor = inputs["post_exit_derisk_anchor_weights"].clone()  # type: ignore[union-attr]
    anchor[0, 0] += 0.05
    anchor[0, 1] -= 0.05
    result = build_cost_aware_active_proposal_v3(
        **{
            **inputs,
            "selected_alpha_mean": mean,
            "learned_release": release,
            "post_exit_derisk_anchor_weights": anchor,
        }  # type: ignore[arg-type]
    )
    assert not result.same_step_buy_mask[0, 1]
    assert result.reallocation_buy_weights[0, 1].item() == 0.0
    result.requested_weights[0, 2].backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum().item() > 0.0
