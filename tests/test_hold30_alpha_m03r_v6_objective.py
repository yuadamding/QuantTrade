"""Deterministic tests for the isolated M03R v6 soft-persistence loss."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_SOFT_PERSISTENCE
from rl_quant.training.hold30_alpha_m03r_v6 import (
    M03RV6ExitNotionalByAge,
    M03RV6ObjectiveError,
    M03RV6SoftPersistenceConfig,
    M03RV6SoftPersistenceDiagnostics,
    M03RV6TrainingPlan,
    M03RV6TrainingProgress,
    m03r_v6_early_exit_age_weight,
    m03r_v6_gradient_norm_telemetry,
    m03r_v6_soft_persistence_objective,
)

AGE_BUCKETS = 61
FULL_PROGRESS = M03RV6TrainingProgress(
    completed_optimizer_steps=100,
    training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
)


def _zeros() -> torch.Tensor:
    return torch.zeros(AGE_BUCKETS, dtype=torch.float64)


def _exits(
    discretionary_policy: torch.Tensor | None = None,
    *,
    valid_decision_session_count: int = 1,
) -> M03RV6ExitNotionalByAge:
    return M03RV6ExitNotionalByAge(
        discretionary_policy=(
            _zeros() if discretionary_policy is None else discretionary_policy
        ),
        other_forced=_zeros(),
        unavailable=_zeros(),
        risk_repair=_zeros(),
        corporate_action=_zeros(),
        terminal=_zeros(),
        valid_decision_session_count=valid_decision_session_count,
    )


def _objective(
    exits: M03RV6ExitNotionalByAge,
    progress: M03RV6TrainingProgress = FULL_PROGRESS,
    config: M03RV6SoftPersistenceConfig | None = None,
) -> tuple[torch.Tensor, M03RV6SoftPersistenceDiagnostics]:
    return m03r_v6_soft_persistence_objective(
        exits,
        progress,
        config,
    )


def test_quadratic_one_sided_age_weight_regression() -> None:
    cases = (
        (0, 1.0),
        (2, (28.0 / 30.0) ** 2),
        (10, (20.0 / 30.0) ** 2),
        (15, 0.25),
        (20, (10.0 / 30.0) ** 2),
        (30, 0.0),
        (45, 0.0),
    )
    for age, expected in cases:
        weight = m03r_v6_early_exit_age_weight(age)
        assert math.isfinite(weight)
        assert weight == pytest.approx(expected, abs=1e-15)


def test_loss_uses_exact_age_weight_and_zeroes_only_the_penalty_after_30() -> None:
    for age in (0, 10, 15, 20, 30, 45):
        learned = _zeros()
        learned[age] = 1.0
        loss, diagnostics = _objective(_exits(learned))
        expected = 5.0e-4 * m03r_v6_early_exit_age_weight(age)
        assert float(loss) == pytest.approx(expected, abs=1e-15)
        assert float(diagnostics.early_exit_penalty_paid) == pytest.approx(
            expected,
            abs=1e-15,
        )
        if age >= 30:
            assert float(loss) == 0.0


def test_non_discretionary_exit_causes_are_exempt() -> None:
    for excluded_field in (
        "other_forced",
        "unavailable",
        "risk_repair",
        "corporate_action",
        "terminal",
    ):
        exits = _exits()
        excluded = _zeros()
        excluded[0] = 1.0
        exits = replace(exits, **{excluded_field: excluded})

        loss, diagnostics = _objective(exits)
        assert float(loss) == 0.0
        assert float(diagnostics.early_exit_penalty_paid) == 0.0
        assert float(diagnostics.total_discretionary_exit_notional) == 0.0


def test_penalty_is_proportional_to_young_sale_notional() -> None:
    small = _zeros()
    small[0] = 0.01
    full = _zeros()
    full[0] = 1.0

    small_loss, _ = _objective(_exits(small))
    full_loss, _ = _objective(_exits(full))

    assert float(small_loss) == pytest.approx(0.01 * float(full_loss))


def test_mature_sales_cannot_dilute_the_young_exit_penalty() -> None:
    learned = _zeros()
    learned[0] = 0.25
    learned[30] = 0.75
    loss, diagnostics = _objective(_exits(learned))
    young_only = _zeros()
    young_only[0] = 0.25
    young_only_loss, _ = _objective(_exits(young_only))

    assert float(loss) == pytest.approx(5.0e-4 * 0.25)
    assert float(loss) == pytest.approx(float(young_only_loss))
    torch.testing.assert_close(
        diagnostics.discretionary_exit_notional_by_age,
        learned,
    )
    assert float(diagnostics.total_discretionary_exit_notional) == pytest.approx(1.0)


def test_mature_sales_have_zero_persistence_gradient() -> None:
    learned = _zeros()
    learned[0] = 0.25
    learned[45] = 0.75
    learned.requires_grad_(True)
    loss, _ = _objective(_exits(learned))
    loss.backward()

    assert learned.grad is not None
    assert float(learned.grad[0]) > 0.0
    assert float(learned.grad[45]) == 0.0


def test_aggregated_notional_is_normalized_by_valid_decision_sessions() -> None:
    learned = _zeros()
    learned[0] = 0.50
    one_session_loss, _ = _objective(_exits(learned))
    ten_session_loss, diagnostics = _objective(
        _exits(learned, valid_decision_session_count=10),
    )

    assert float(ten_session_loss) == pytest.approx(float(one_session_loss) / 10.0)
    assert diagnostics.valid_decision_session_count == 10
    assert float(diagnostics.weighted_early_exit_notional) == pytest.approx(0.50)
    assert float(
        diagnostics.weighted_early_exit_notional_per_valid_session
    ) == pytest.approx(0.05)


def test_empty_discretionary_sale_is_finite_and_exactly_zero() -> None:
    loss, diagnostics = _objective(_exits())
    assert bool(torch.isfinite(loss))
    assert float(loss) == 0.0
    assert float(diagnostics.weighted_early_exit_notional) == 0.0
    assert float(diagnostics.early_exit_penalty_paid) == 0.0


def test_canonical_v6_freezes_five_basis_points() -> None:
    learned = _zeros()
    learned[10] = 0.4
    loss, diagnostics = _objective(_exits(learned))
    assert float(loss) > 0.0
    assert diagnostics.coefficient_bp_per_unit_at_age_zero == 5.0
    for coefficient in (2.0, 10.0):
        with pytest.raises(M03RV6ObjectiveError, match="new protocol generation"):
            M03RV6SoftPersistenceConfig(
                early_exit_penalty_bp_per_unit_at_age_zero=coefficient
            )


def test_penalty_uses_the_frozen_ten_percent_linear_warmup() -> None:
    learned = _zeros()
    learned[0] = 1.0
    penalties = []
    for completed in (0, 5, 10, 50):
        loss, diagnostics = _objective(
            _exits(learned),
            M03RV6TrainingProgress(
                completed_optimizer_steps=completed,
                training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
            ),
        )
        penalties.append(float(loss))
        assert diagnostics.warmup_multiplier == pytest.approx(
            min(1.0, completed / 10.0)
        )
    assert penalties[0] == 0.0
    assert penalties[1] == pytest.approx(0.5 * penalties[2])
    assert penalties[3] == pytest.approx(penalties[2])


def test_early_sales_are_not_hard_masked_and_retain_gradients() -> None:
    learned = _zeros()
    learned[0] = 0.2
    learned[45] = 0.8
    learned.requires_grad_(True)
    before = learned.detach().clone()
    loss, diagnostics = _objective(_exits(learned))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CUDA initialization: CUDA unknown error",
            category=UserWarning,
        )
    torch.autograd.backward(loss)

    torch.testing.assert_close(learned.detach(), before)
    torch.testing.assert_close(
        diagnostics.discretionary_exit_notional_by_age,
        before,
    )
    assert float(loss.detach()) > 0.0
    assert learned.grad is not None
    assert bool(torch.isfinite(learned.grad).all())
    assert float(learned.grad[0]) > 0.0
    assert M03R_SOFT_PERSISTENCE.minimum_holding_period_sessions is None
    assert not M03R_SOFT_PERSISTENCE.sell_mask_before_preference_horizon


def test_objective_rejects_wrong_age_support_and_invalid_session_count() -> None:
    with pytest.raises(M03RV6ObjectiveError, match="exactly 61 age bins"):
        _exits(torch.zeros(31, dtype=torch.float64))
    with pytest.raises(M03RV6ObjectiveError, match="exactly 61 age bins"):
        _exits(torch.zeros(62, dtype=torch.float64))
    with pytest.raises(M03RV6ObjectiveError, match="positive integer"):
        _exits(valid_decision_session_count=0)


def test_gradient_norm_telemetry_does_not_accumulate_or_combine_losses() -> None:
    parameter = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    hold_loss = (2.0 * parameter).sum()
    economic_loss = (3.0 * parameter).sum()
    telemetry = m03r_v6_gradient_norm_telemetry(
        hold_loss,
        economic_loss,
        (parameter,),
    )
    assert telemetry.hold_gradient_l2_norm == pytest.approx(2.0)
    assert telemetry.economic_gradient_l2_norm == pytest.approx(3.0)
    assert telemetry.holding_to_economic_gradient_norm_ratio == pytest.approx(2.0 / 3.0)
    assert parameter.grad is None
    torch.testing.assert_close(economic_loss, torch.tensor(6.0, dtype=torch.float64))


def test_v6_objective_rejects_cross_generation_identity_and_unfrozen_coefficients() -> (
    None
):
    with pytest.raises(M03RV6ObjectiveError, match="identity drifted"):
        M03RV6SoftPersistenceConfig(
            protocol_generation="prelockbox-hold30-active-alpha-m03r-v5"
        )
    with pytest.raises(M03RV6ObjectiveError, match="new protocol generation"):
        M03RV6SoftPersistenceConfig(early_exit_penalty_bp_per_unit_at_age_zero=7.5)


def test_total_optimizer_steps_are_content_bound_by_the_training_plan() -> None:
    short = M03RV6TrainingPlan(total_optimizer_steps=100)
    long = M03RV6TrainingPlan(total_optimizer_steps=200)
    assert short.receipt_sha256 != long.receipt_sha256
    assert short.canonical_payload()["total_optimizer_steps"] == 100
    with pytest.raises(M03RV6ObjectiveError, match="positive total"):
        M03RV6TrainingPlan(total_optimizer_steps=0)
