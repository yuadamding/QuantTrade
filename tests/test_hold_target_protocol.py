from __future__ import annotations

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.models.daily_policy import hold30_release_hazard
from rl_quant.protocol.hold_target import (
    DEFAULT_HOLD_TARGET_SPEC,
    LEGACY_HOLD30_TARGET_SPEC,
    HoldTargetSpec,
    HoldTargetProtocolError,
    hold_release_hazard,
    hold_survival_weights,
)


def _legacy_reference(age: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    age = age.to(device=residual.device, dtype=residual.dtype)
    beta = -2.0 + (age.clamp(min=0.0, max=60.0) - 30.0) / 4.0
    bounded = residual.clamp(min=-12.0, max=12.0)
    p_min = torch.sigmoid((beta - 12.0).clamp(min=-20.0, max=20.0))
    release = torch.sigmoid((beta + bounded).clamp(min=-20.0, max=20.0))
    return (release - p_min) / (1.0 - p_min)


def test_generic_holding_default_is_soft_three_sessions() -> None:
    assert DEFAULT_HOLD_TARGET_SPEC.target_sessions == 3
    assert not DEFAULT_HOLD_TARGET_SPEC.hard_minimum_hold
    assert DEFAULT_HOLD_TARGET_SPEC.expected_neutral_hold_sessions == pytest.approx(
        3.0, abs=1.0e-6
    )
    ages = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    hazards = hold_release_hazard(
        ages,
        torch.zeros_like(ages),
        hold_spec=DEFAULT_HOLD_TARGET_SPEC,
    )
    assert bool((hazards[:3] > 0.0).all())
    assert bool((hazards < 1.0).all())


@pytest.mark.parametrize("target", (1, 3, 5, 21, 30, 60))
def test_generic_holding_target_is_semantically_calibrated(target: int) -> None:
    spec = HoldTargetSpec(target_sessions=target)
    assert spec.expected_neutral_hold_sessions == pytest.approx(target, abs=1.0e-6)
    assert hold_survival_weights(hold_spec=spec).sum().item() == pytest.approx(
        target, abs=1.0e-6
    )
    ages = torch.arange(1, spec.age_cap_sessions + 1, dtype=torch.float64)
    hazards = hold_release_hazard(
        ages,
        torch.zeros_like(ages),
        hold_spec=spec,
    )
    survival = torch.cumprod(
        torch.cat((torch.ones(1, dtype=torch.float64), 1.0 - hazards[:-1])),
        dim=0,
    )
    assert float(survival.sum()) == pytest.approx(target, abs=1.0e-6)


def test_legacy_hold30_wrapper_preserves_the_frozen_numerical_clock() -> None:
    ages = torch.arange(0, 61, dtype=torch.float64)
    for residual_value in (-12.0, -3.0, 0.0, 4.0, 12.0):
        residual = torch.full_like(ages, residual_value)
        expected = _legacy_reference(ages, residual)
        generic = hold_release_hazard(
            ages,
            residual,
            hold_spec=LEGACY_HOLD30_TARGET_SPEC,
        )
        wrapper = hold30_release_hazard(ages, residual)
        assert torch.equal(generic, expected)
        assert torch.equal(wrapper, expected)
    neutral_runtime = hold30_release_hazard(ages[1:], torch.zeros_like(ages[1:]))
    assert tuple(neutral_runtime.tolist()) == pytest.approx(
        LEGACY_HOLD30_TARGET_SPEC.neutral_release_hazards,
        abs=1.0e-15,
    )


def test_holding_target_does_not_change_age_cap_or_legacy_identity() -> None:
    assert DEFAULT_HOLD_TARGET_SPEC.age_cap_sessions == 60
    assert LEGACY_HOLD30_TARGET_SPEC.age_cap_sessions == 60
    assert LEGACY_HOLD30_TARGET_SPEC.target_sessions == 30
    assert DEFAULT_HOLD_TARGET_SPEC.receipt_sha256 != (
        LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    )


@pytest.mark.parametrize(
    "spec",
    (
        HoldTargetSpec(target_sessions=True),
        HoldTargetSpec(target_sessions=0),
        HoldTargetSpec(target_sessions=61),
        HoldTargetSpec(hard_minimum_hold=True),
        HoldTargetSpec(target_sessions=3, prior_family="legacy-hold30-v1"),
    ),
)
def test_invalid_holding_target_specs_fail_closed(spec: HoldTargetSpec) -> None:
    with pytest.raises(HoldTargetProtocolError):
        spec.validate()


def test_generic_three_session_penalty_is_soft_and_zero_at_maturity() -> None:
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    young = CohortLedger.from_weights(
        weights, cash_index=0, initial_age=0, track_initial_units=True
    )
    _sold, young_accounting = young.trade_to_holding_target(
        target,
        cause=TurnoverCause.DISCRETIONARY,
        hold_spec=DEFAULT_HOLD_TARGET_SPEC,
    )
    assert float(young_accounting.early_exit_notional) > 0.0

    mature = young
    zero_returns = torch.zeros_like(weights)
    for _ in range(3):
        mature = mature.age_and_drift(zero_returns)
    _sold, mature_accounting = mature.trade_to_holding_target(
        target,
        cause=TurnoverCause.DISCRETIONARY,
        hold_spec=DEFAULT_HOLD_TARGET_SPEC,
    )
    assert float(mature_accounting.early_exit_notional) == 0.0
