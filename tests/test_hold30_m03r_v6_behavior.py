"""Behavioral qualification for nonbinding M03R v6 persistence."""

from __future__ import annotations

import pytest
import torch

from rl_quant.execution.hold30_m03r_soft_persistence_v6 import (
    m03r_v6_release_cohorts,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN,
    M03R_SETTINGS_BY_ID,
)


def _one_position(age: int, notional: float = 1.0) -> torch.Tensor:
    ledger = torch.zeros((1, 1, 61), dtype=torch.float64)
    ledger[0, 0, age] = notional
    return ledger


def test_age_two_adverse_reversal_can_exit_fully_without_an_age_mask() -> None:
    ledger = _one_position(2)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.tensor([[12.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(result.discretionary_release_by_age, ledger)
    torch.testing.assert_close(
        result.remaining_notional_by_age, torch.zeros_like(ledger)
    )


def test_age_45_winner_has_no_day_30_expiry_and_can_continue() -> None:
    ledger = _one_position(45)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.tensor([[-8.0]], dtype=torch.float64),
    )
    assert float(result.discretionary_release_by_asset) < 0.002
    assert float(result.remaining_notional_by_age[0, 0, 45]) > 0.998


def test_weak_young_signal_is_almost_carry_without_exact_hold_atom() -> None:
    ledger = _one_position(2)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.zeros((1, 1), dtype=torch.float64),
        exact_hold_decision_st=None,
    )
    assert float(result.discretionary_release_by_asset) < 0.001
    assert float(result.remaining_notional_by_age[0, 0, 2]) > 0.999
    assert not M03R_DESIGN.active_risk.zero_confidence_forces_benchmark_derisk


def test_optional_exact_hold_atom_suppresses_only_discretionary_release() -> None:
    ledger = _one_position(5)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.tensor([[12.0]], dtype=torch.float64),
        exact_hold_decision_st=torch.ones((1, 1), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.discretionary_release_by_age,
        torch.zeros_like(ledger),
    )
    torch.testing.assert_close(result.remaining_notional_by_age, ledger)
    assert (
        M03R_SETTINGS_BY_ID["A11-no-exact-hold-atom"].exact_hold_action_supported
        is False
    )


def test_hard_risk_repair_ignores_hold_preference_and_is_cause_separate() -> None:
    ledger = _one_position(2)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.full((1, 1), -12.0, dtype=torch.float64),
        exact_hold_decision_st=torch.ones((1, 1), dtype=torch.float64),
        forced_exit_fraction=torch.ones((1, 1), dtype=torch.float64),
    )
    torch.testing.assert_close(result.forced_release_by_age, ledger)
    torch.testing.assert_close(
        result.discretionary_release_by_age,
        torch.zeros_like(ledger),
    )
    torch.testing.assert_close(
        result.remaining_notional_by_age,
        torch.zeros_like(ledger),
    )


@pytest.mark.parametrize("age", (0, 2, 10, 20, 29, 30, 45, 60))
def test_no_age_has_a_discretionary_sell_prohibition(age: int) -> None:
    ledger = _one_position(age)
    result = m03r_v6_release_cohorts(
        ledger,
        torch.tensor([[12.0]], dtype=torch.float64),
    )
    assert float(result.discretionary_release_by_asset) == pytest.approx(1.0)
