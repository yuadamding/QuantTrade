from __future__ import annotations

import pytest
import torch

from rl_quant.execution.hold30 import (
    build_h2_hold30_action,
    build_holding_action,
    build_scalar_gate_hold30_action,
    capped_waterfill,
    centered_benchmark_tilt,
)
from rl_quant.models.daily_policy import HOLD30_HAZARD_MAX, HOLD30_HAZARD_MIN


def _ledger_from_weights(weights: torch.Tensor, age: int = 12) -> torch.Tensor:
    ledger = weights.new_zeros((*weights.shape, 61))
    ledger[..., age] = weights
    return ledger


def _common(weights: torch.Tensor):
    mask = torch.ones_like(weights, dtype=torch.bool)
    caps = torch.ones_like(weights)
    gross = torch.ones(weights.shape[0], dtype=weights.dtype)
    benchmark = weights.clone()
    return benchmark, mask, caps, gross


def test_h2_finite_neutral_action_is_exact_hold() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    benchmark, mask, caps, gross = _common(weights)
    result = build_h2_hold30_action(
        weights,
        _ledger_from_weights(weights),
        entry_scores=torch.tensor([[4.0, -3.0, 2.0, 1.0]], dtype=torch.float64),
        hazard_residual=torch.full_like(weights, HOLD30_HAZARD_MIN),
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        benchmark_weights=benchmark,
        trade_mask=mask,
        risk_asset_caps=caps,
        risk_gross_max=gross,
    )
    assert torch.equal(result.target_weights, weights)
    assert torch.equal(result.constructed_delta, torch.zeros_like(weights))
    assert torch.equal(result.proposed_release, torch.zeros_like(weights))
    assert result.constructed_turnover.item() == 0.0


def test_new_generic_action_defaults_to_calibrated_target_three() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    benchmark, mask, caps, gross = _common(weights)
    inputs = {
        "repaired_weights": weights,
        "age_notional": _ledger_from_weights(weights, age=1),
        "entry_scores": torch.zeros_like(weights),
        "hazard_residual": torch.zeros_like(weights),
        "exposure_residual": torch.zeros(1, dtype=torch.float64),
        "benchmark_weights": benchmark,
        "trade_mask": mask,
        "risk_asset_caps": caps,
        "risk_gross_max": gross,
    }
    generic = build_holding_action(**inputs)
    legacy = build_h2_hold30_action(**inputs)
    assert float(generic.proposed_release[:, 1:].sum()) > 100.0 * float(
        legacy.proposed_release[:, 1:].sum()
    )


def test_hold_envelope_does_not_force_an_out_of_band_book_to_rebalance() -> None:
    weights = torch.tensor([[0.50, 0.20, 0.15, 0.15]], dtype=torch.float64)
    benchmark = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    mask = torch.ones_like(weights, dtype=torch.bool)
    caps = torch.ones_like(weights)
    result = build_h2_hold30_action(
        weights,
        _ledger_from_weights(weights, age=45),
        entry_scores=torch.zeros_like(weights),
        hazard_residual=torch.full_like(weights, HOLD30_HAZARD_MIN),
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        benchmark_weights=benchmark,
        trade_mask=mask,
        risk_asset_caps=caps,
        risk_gross_max=torch.ones(1, dtype=torch.float64),
    )
    assert torch.equal(result.target_weights, weights)


def test_strong_reversal_can_exit_young_position_without_same_name_reentry() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01, 0.0]], dtype=torch.float64)
    benchmark = torch.tensor([[0.97, 0.0, 0.01, 0.01, 0.01]], dtype=torch.float64)
    mask = torch.ones_like(weights, dtype=torch.bool)
    caps = torch.ones_like(weights)
    hazard = torch.full_like(weights, HOLD30_HAZARD_MIN)
    hazard[:, 1] = HOLD30_HAZARD_MAX
    result = build_h2_hold30_action(
        weights,
        _ledger_from_weights(weights, age=1),
        entry_scores=torch.zeros_like(weights),
        hazard_residual=hazard,
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        benchmark_weights=benchmark,
        trade_mask=mask,
        risk_asset_caps=caps,
        risk_gross_max=torch.ones(1, dtype=torch.float64),
    )
    # The finite +12 endpoint liquidates more than 90% of this one-session
    # cohort immediately; the normalized sigmoid clock is deliberately soft,
    # not a hard lock or a forced all-or-nothing sale.
    assert result.target_weights[0, 1] < 0.001
    assert result.target_weights[0, 4] > 0.009
    assert result.constructed_turnover.item() <= 0.10 + 1e-12


def test_scalar_gate_zero_is_exact_hold_after_fill_time_repair() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    benchmark, mask, caps, gross = _common(weights)
    result = build_scalar_gate_hold30_action(
        weights,
        target_logits=torch.tensor([[0.0, 8.0, -8.0, 1.0]], dtype=torch.float64),
        gate=torch.zeros(1, dtype=torch.float64),
        benchmark_weights=benchmark,
        trade_mask=mask,
        risk_asset_caps=caps,
        risk_gross_max=gross,
    )
    assert torch.equal(result.target_weights, weights)
    assert result.constructed_turnover.item() == 0.0


def test_scalar_gate_risky_exposure_respects_a_nonzero_cash_index() -> None:
    cash_index = 50
    weights = torch.full((1, 101), 0.01, dtype=torch.float64)
    weights[:, cash_index] = 0.0
    benchmark = weights.clone()
    mask = torch.ones_like(weights, dtype=torch.bool)
    logits = torch.zeros_like(weights)
    logits[:, cash_index] = -20.0
    result = build_scalar_gate_hold30_action(
        weights,
        target_logits=logits,
        gate=torch.ones(1, dtype=torch.float64),
        benchmark_weights=benchmark,
        trade_mask=mask,
        risk_asset_caps=torch.ones_like(weights),
        risk_gross_max=torch.ones(1, dtype=torch.float64),
        cash_index=cash_index,
    )
    assert result.desired_risky_exposure.item() == pytest.approx(1.0)
    assert result.target_weights.sum().item() == pytest.approx(1.0)
    assert result.target_weights[0, cash_index].item() == pytest.approx(0.0, abs=1e-8)


def test_centered_tilt_cannot_allocate_to_zero_benchmark_or_masked_asset() -> None:
    scores = torch.tensor([[0.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    benchmark = torch.tensor([[0.97, 0.01, 0.0, 0.02]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    direction = centered_benchmark_tilt(scores, benchmark, mask)
    assert torch.equal(direction[0, [0, 2, 3]], torch.zeros(3, dtype=torch.float64))
    assert direction[0, 1].item() == pytest.approx(1.0)


def test_waterfill_interior_gradient_matches_finite_difference() -> None:
    requested = torch.tensor([0.012], dtype=torch.float64)
    direction = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    capacity = torch.tensor([[0.004, 0.020, 0.020]], dtype=torch.float64)
    allocated, effective = capped_waterfill(requested, direction, capacity)
    objective = (
        allocated * torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float64)
    ).sum()
    objective.backward()
    analytic = direction.grad.detach().clone()

    epsilon = 1e-6
    numeric = torch.zeros_like(direction)
    for column in range(direction.shape[1]):
        plus = direction.detach().clone()
        minus = direction.detach().clone()
        plus[0, column] += epsilon
        minus[0, column] -= epsilon
        plus_value = (
            capped_waterfill(requested, plus, capacity)[0]
            * torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float64)
        ).sum()
        minus_value = (
            capped_waterfill(requested, minus, capacity)[0]
            * torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float64)
        ).sum()
        numeric[0, column] = (plus_value - minus_value) / (2 * epsilon)
    assert effective.item() == pytest.approx(0.012)
    assert torch.allclose(analytic, numeric, atol=2e-6, rtol=2e-4)


def test_waterfill_exact_cap_tie_conserves_requested_mass() -> None:
    requested = torch.tensor([0.5], dtype=torch.float64)
    direction = torch.tensor([[0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    capacity = torch.tensor([[0.25, 1.0]], dtype=torch.float64)
    allocated, effective = capped_waterfill(requested, direction, capacity)
    torch.testing.assert_close(
        allocated, torch.tensor([[0.25, 0.25]], dtype=torch.float64)
    )
    torch.testing.assert_close(allocated.sum(-1), effective)
    allocated.sum().backward()
    assert direction.grad is not None
    assert torch.isfinite(direction.grad).all()
