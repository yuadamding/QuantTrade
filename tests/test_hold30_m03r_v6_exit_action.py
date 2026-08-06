"""Reachability and gradients for the isolated M03R v6 exact-exit surface."""

from __future__ import annotations

import warnings

import torch

from rl_quant.execution.hold30_exit_v6 import build_m03r_v6_exit_release
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_CONTINUOUS_ACTION_INDEX,
    M03R_V6_EXIT_ACTION_INDEX,
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitActionHead,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_AGE_LEDGER_BIN_COUNT
from rl_quant.training.hold30_alpha_m03r_v6 import (
    M03RV6ExitNotionalByAge,
    M03RV6TrainingPlan,
    M03RV6TrainingProgress,
    m03r_v6_soft_persistence_objective,
)


def _hidden() -> torch.Tensor:
    return torch.zeros((1, 3, 4), dtype=torch.float64)


def _available() -> torch.Tensor:
    # Asset zero is CASH; asset two is unavailable.
    return torch.tensor([[True, True, False]], dtype=torch.bool)


def _ledger() -> torch.Tensor:
    ledger = torch.zeros((1, 3, M03R_AGE_LEDGER_BIN_COUNT), dtype=torch.float64)
    ledger[0, 1, 2] = 0.30
    ledger[0, 1, 45] = 0.20
    return ledger


def _head_with_action(action_index: int) -> M03RV6ExitActionHead:
    head = M03RV6ExitActionHead(hidden_dim=4).to(dtype=torch.float64)
    with torch.no_grad():
        head.action_logits.bias.fill_(-8.0)
        head.action_logits.bias[action_index] = 8.0
    return head


def test_actual_model_exit_output_releases_every_cohort_exactly() -> None:
    head = _head_with_action(M03R_V6_EXIT_ACTION_INDEX)
    action = head(_hidden(), _available())
    assert action.exit_decision_st[0, 1].item() == 1.0
    assert action.hold_decision_st[0, 0].item() == 1.0
    assert action.hold_decision_st[0, 2].item() == 1.0
    assert torch.equal(
        action.decision_st.detach().sum(dim=-1),
        torch.ones((1, 3), dtype=torch.float64),
    )

    ledger = _ledger()
    release = build_m03r_v6_exit_release(
        ledger,
        torch.zeros((1, 3), dtype=torch.float64),
        action,
    )
    torch.testing.assert_close(
        release.discretionary_release_by_age[0, 1],
        ledger[0, 1],
        atol=0.0,
        rtol=0.0,
    )
    assert release.discretionary_release[0, 1].item() == 0.50
    assert bool((release.effective_release_fraction_by_age[0, 1] == 1.0).all())
    assert release.discretionary_release[:, (0, 2)].sum().item() == 0.0


def test_exact_hold_and_continuous_actions_remain_distinct() -> None:
    ledger = _ledger()
    hazard = torch.zeros((1, 3), dtype=torch.float64)
    hold = build_m03r_v6_exit_release(
        ledger,
        hazard,
        _head_with_action(M03R_V6_HOLD_ACTION_INDEX)(_hidden(), _available()),
    )
    assert hold.discretionary_release.sum().item() == 0.0
    assert bool((hold.effective_release_fraction_by_age == 0.0).all())

    continuous = build_m03r_v6_exit_release(
        ledger,
        hazard,
        _head_with_action(M03R_V6_CONTINUOUS_ACTION_INDEX)(
            _hidden(),
            _available(),
        ),
    )
    assert 0.0 < continuous.discretionary_release[0, 1].item() < 0.50
    assert bool(
        (
            continuous.effective_release_fraction_by_age[0, 1]
            == continuous.continuous_release_fraction_by_age[0, 1]
        ).all()
    )


def test_a11_removes_only_hold_while_exit_remains_exact_and_trainable() -> None:
    head = M03RV6ExitActionHead(
        hidden_dim=4,
        allow_exact_hold_atom=False,
    ).to(dtype=torch.float64)
    initial = head(_hidden(), _available())
    risky = initial.risky_available
    assert bool(
        (initial.soft_probabilities[..., M03R_V6_HOLD_ACTION_INDEX][risky] == 0.0).all()
    )
    assert bool((initial.continuous_decision_st[risky] == 1.0).all())

    with torch.no_grad():
        head.action_logits.bias.fill_(-8.0)
        head.action_logits.bias[M03R_V6_EXIT_ACTION_INDEX] = 8.0
    exit_action = head(_hidden(), _available())
    assert bool((exit_action.exit_decision_st[risky] == 1.0).all())
    exit_probability = exit_action.soft_probabilities[..., M03R_V6_EXIT_ACTION_INDEX][
        risky
    ].sum()
    head.zero_grad(set_to_none=True)
    exit_probability.backward()
    gradient = head.action_logits.bias.grad
    assert gradient is not None
    assert gradient[M03R_V6_EXIT_ACTION_INDEX].item() > 0.0
    assert gradient[M03R_V6_CONTINUOUS_ACTION_INDEX].item() < 0.0
    assert gradient[M03R_V6_HOLD_ACTION_INDEX].item() == 0.0


def test_young_adverse_position_can_optimize_into_the_exact_exit_atom() -> None:
    head = M03RV6ExitActionHead(hidden_dim=4).to(dtype=torch.float64)
    hidden = _hidden()
    available = _available()
    ledger = torch.zeros((1, 3, M03R_AGE_LEDGER_BIN_COUNT), dtype=torch.float64)
    ledger[0, 1, 2] = 1.0
    hazard = torch.zeros((1, 3), dtype=torch.float64)
    zeros = torch.zeros(M03R_AGE_LEDGER_BIN_COUNT, dtype=torch.float64)
    progress = M03RV6TrainingProgress(
        completed_optimizer_steps=100,
        training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
    )

    initial = head(hidden, available)
    assert initial.continuous_decision_st[0, 1].item() == 1.0
    first_exit_gradient = 0.0
    for step in range(6):
        action = head(hidden, available)
        release = build_m03r_v6_exit_release(ledger, hazard, action)
        discretionary_by_age = release.discretionary_release_by_age.sum(dim=(0, 1))
        persistence, _diagnostics = m03r_v6_soft_persistence_objective(
            M03RV6ExitNotionalByAge(
                discretionary_policy=discretionary_by_age,
                other_forced=zeros,
                unavailable=zeros,
                risk_repair=zeros,
                corporate_action=zeros,
                terminal=zeros,
                valid_decision_session_count=1,
            ),
            progress,
        )
        # A five-percent avoided adverse return dominates the soft early-exit
        # preference. Minimization should make the exact EXIT atom reachable.
        economic_loss = 0.05 * (1.0 - release.discretionary_release.sum())
        loss = economic_loss + persistence
        head.zero_grad(set_to_none=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA initialization: CUDA unknown error",
                category=UserWarning,
            )
            torch.autograd.backward(loss)
        gradient = head.action_logits.bias.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        if step == 0:
            first_exit_gradient = float(gradient[M03R_V6_EXIT_ACTION_INDEX])
        with torch.no_grad():
            for parameter in head.parameters():
                assert parameter.grad is not None
                parameter.add_(parameter.grad, alpha=-5.0)

    assert first_exit_gradient < 0.0
    optimized = head(hidden, available)
    assert optimized.exit_decision_st[0, 1].item() == 1.0
    final_release = build_m03r_v6_exit_release(ledger, hazard, optimized)
    assert final_release.discretionary_release[0, 1].item() == 1.0
