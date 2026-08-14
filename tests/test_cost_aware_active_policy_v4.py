from __future__ import annotations

import torch

from rl_quant.execution.cost_aware_active_policy_v4 import (
    build_cost_aware_active_proposal_v4,
)


def _inputs(mean_scale: float) -> dict[str, object]:
    anchor = torch.full((1, 9), 1.0 / 9.0, dtype=torch.float64)
    signal = (
        torch.tensor(
            [[0.0, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0]],
            dtype=torch.float64,
        )
        * mean_scale
    )
    return {
        "post_exit_derisk_anchor_weights": anchor,
        "benchmark_weights": anchor.clone(),
        "selected_alpha_mean": signal,
        "selected_alpha_scale": torch.full_like(anchor, 0.01),
        "one_way_cost": torch.full_like(anchor, 0.001),
        "learned_release": torch.zeros_like(anchor),
        "explicit_derisk_amount": torch.zeros(1, dtype=torch.float64),
        "held_mask": torch.ones_like(anchor, dtype=torch.bool),
        "trade_mask": torch.ones_like(anchor, dtype=torch.bool),
        "risk_asset_caps": torch.full_like(anchor, 0.50),
        "signal_confidence": torch.ones(1, dtype=torch.float64),
        "covariance_factor": torch.zeros((1, 9, 2), dtype=torch.float64),
        "specific_variance": torch.zeros_like(anchor),
        "selected_horizon_sessions": 3,
    }


def test_v12_turnover_utilization_is_nonsaturating_and_monotone() -> None:
    rows = [
        build_cost_aware_active_proposal_v4(**_inputs(scale))  # type: ignore[arg-type]
        for scale in (0.0, 0.001, 0.003, 0.01)
    ]
    utilization = [row.turnover_utilization.item() for row in rows]
    assert all(row.selected_horizon_sessions == 3 for row in rows)
    turnover = [
        row.proposal.requested_incremental_one_way_turnover.item() for row in rows
    ]
    assert utilization == sorted(utilization)
    assert turnover == sorted(turnover)
    assert utilization[0] == 0.0
    assert turnover[0] == 0.0
    assert 0.0 < utilization[-1] < 1.0
    assert rows[-1].proposal.allowed_incremental_one_way_turnover.item() < 0.02


def test_v12_utilization_preserves_gradient_to_economic_mean() -> None:
    inputs = _inputs(0.003)
    mean = inputs["selected_alpha_mean"].clone().requires_grad_(True)  # type: ignore[union-attr]
    result = build_cost_aware_active_proposal_v4(
        **{**inputs, "selected_alpha_mean": mean}  # type: ignore[arg-type]
    )
    result.proposal.requested_weights[0, 1].backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum().item() > 0.0
