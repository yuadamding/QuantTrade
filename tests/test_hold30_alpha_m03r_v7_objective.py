from __future__ import annotations

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_CANONICAL_SETTING_ID,
)
from rl_quant.training.hold30_alpha_m03r_v7 import (
    M03RV7ExitNotionalByAge,
    M03RV7TrainingPlan,
    M03RV7TrainingProgress,
    m03r_v7_soft_persistence_objective,
)

P00 = "P00-no-soft-persistence-v7"
P10 = "P10-soft-persistence-10bp-v7"


def _progress(setting_id: str, *, completed: int = 10) -> M03RV7TrainingProgress:
    return M03RV7TrainingProgress(
        plan=M03RV7TrainingPlan(setting_id=setting_id, total_optimizer_steps=100),
        completed_optimizer_steps=completed,
    )


def _exits(
    discretionary: torch.Tensor,
    *,
    forced: torch.Tensor | None = None,
    valid_sessions: int = 1,
) -> M03RV7ExitNotionalByAge:
    zero = torch.zeros_like(discretionary)
    return M03RV7ExitNotionalByAge(
        discretionary_policy=discretionary,
        other_forced=zero if forced is None else forced,
        unavailable=zero,
        risk_repair=zero,
        corporate_action=zero,
        terminal=zero,
        valid_decision_session_count=valid_sessions,
    )


def _age_vector(*, young: float = 0.0, mature: float = 0.0) -> torch.Tensor:
    rows = torch.zeros(61, dtype=torch.float64)
    rows[0] = young
    rows[30] = mature
    return rows


def test_v7_setting_identity_controls_zero_five_and_ten_bp_coefficients() -> None:
    exits = _exits(_age_vector(young=1.0))
    p00, d00 = m03r_v7_soft_persistence_objective(exits, _progress(P00))
    canonical, d05 = m03r_v7_soft_persistence_objective(
        exits,
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    p10, d10 = m03r_v7_soft_persistence_objective(exits, _progress(P10))

    assert p00.item() == 0.0
    assert canonical.item() == pytest.approx(5.0e-4)
    assert p10.item() == pytest.approx(1.0e-3)
    assert (d00.coefficient_basis_points, d05.coefficient_basis_points, d10.coefficient_basis_points) == (0.0, 5.0, 10.0)


def test_v7_penalty_is_proportional_to_young_sold_nav() -> None:
    small, _ = m03r_v7_soft_persistence_objective(
        _exits(_age_vector(young=0.01)),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    full, _ = m03r_v7_soft_persistence_objective(
        _exits(_age_vector(young=1.0)),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    assert small.item() == pytest.approx(0.01 * full.item())


def test_mature_sales_neither_dilute_penalty_nor_receive_gradient() -> None:
    sold = _age_vector(young=0.01, mature=0.99).requires_grad_()
    penalty, _ = m03r_v7_soft_persistence_objective(
        _exits(sold),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    penalty.backward()

    young_only, _ = m03r_v7_soft_persistence_objective(
        _exits(_age_vector(young=0.01)),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    assert penalty.item() == pytest.approx(young_only.item())
    assert sold.grad is not None
    assert sold.grad[0].item() > 0.0
    assert sold.grad[30].item() == 0.0


def test_forced_sales_are_exempt_and_valid_session_count_is_only_denominator() -> None:
    young = _age_vector(young=0.20)
    forced = _age_vector(young=1.0)
    one, _ = m03r_v7_soft_persistence_objective(
        _exits(young, forced=forced, valid_sessions=1),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    four, diagnostics = m03r_v7_soft_persistence_objective(
        _exits(young, forced=100.0 * forced, valid_sessions=4),
        _progress(M03R_V7_CANONICAL_SETTING_ID),
    )
    assert four.item() == pytest.approx(one.item() / 4.0)
    assert diagnostics.valid_decision_session_count == 4


def test_warmup_is_bound_to_training_plan_updates() -> None:
    exits = _exits(_age_vector(young=1.0))
    start, _ = m03r_v7_soft_persistence_objective(
        exits,
        _progress(M03R_V7_CANONICAL_SETTING_ID, completed=0),
    )
    halfway, _ = m03r_v7_soft_persistence_objective(
        exits,
        _progress(M03R_V7_CANONICAL_SETTING_ID, completed=5),
    )
    complete, _ = m03r_v7_soft_persistence_objective(
        exits,
        _progress(M03R_V7_CANONICAL_SETTING_ID, completed=10),
    )
    assert start.item() == 0.0
    assert halfway.item() == pytest.approx(0.5 * complete.item())


def test_v7_objective_rejects_misaligned_age_support_and_auxiliary_identity() -> None:
    bad = torch.zeros(60, dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly 61"):
        _exits(bad)
    with pytest.raises(ValueError, match="unknown primary"):
        M03RV7TrainingPlan(
            setting_id="M01-benchmark-subtraction-v7",
            total_optimizer_steps=4,
        )
