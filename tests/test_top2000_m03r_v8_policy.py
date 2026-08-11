from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitAction,
    straight_through_m03r_v6_exit_action,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    M03R_V8_FIXED_HAZARD_SOURCE_SETTING_ID,
    M03RV8PolicyError,
    Top2000M03RV8DevelopmentPolicy,
    apply_m03r_v8_exit_action_temperature,
)


def _exit_action(logits: torch.Tensor) -> M03RV6ExitAction:
    risky = torch.tensor([[False, True, True]])
    soft, decision = straight_through_m03r_v6_exit_action(logits)
    hold = torch.zeros_like(soft)
    hold[..., M03R_V6_HOLD_ACTION_INDEX] = 1.0
    return M03RV6ExitAction(
        logits=torch.where(risky.unsqueeze(-1), logits, torch.zeros_like(logits)),
        soft_probabilities=torch.where(risky.unsqueeze(-1), soft, hold),
        decision_st=torch.where(risky.unsqueeze(-1), decision, hold),
        risky_available=risky,
        exact_hold_atom_enabled=True,
    )


def _policy(setting_index: int) -> Top2000M03RV8DevelopmentPolicy:
    return Top2000M03RV8DevelopmentPolicy(
        setting_index,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_temperature_one_preserves_identity_and_softened_temperature_is_exact() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 2.0, -0.5], [1.5, 0.2, -0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    action = _exit_action(logits)

    assert apply_m03r_v8_exit_action_temperature(action, temperature=1.0) is action
    softened = apply_m03r_v8_exit_action_temperature(action, temperature=1.5)

    assert torch.equal(softened.decision_st.detach(), action.decision_st.detach())
    assert torch.equal(
        softened.soft_probabilities[:, 0], action.soft_probabilities[:, 0]
    )
    assert (
        softened.soft_probabilities[0, 1, M03R_V6_HOLD_ACTION_INDEX]
        < (action.soft_probabilities[0, 1, M03R_V6_HOLD_ACTION_INDEX])
    )
    assert not torch.equal(
        softened.soft_probabilities[:, 1:], action.soft_probabilities[:, 1:]
    )
    softened.decision_st[..., M03R_V6_HOLD_ACTION_INDEX].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_unfrozen_action_temperature_fails_closed() -> None:
    action = _exit_action(torch.zeros((1, 3, 3), dtype=torch.float64))
    for value in (0.0, 1.25, float("nan")):
        with pytest.raises(M03RV8PolicyError, match="temperature"):
            apply_m03r_v8_exit_action_temperature(action, temperature=value)


def test_v8_rows_bind_only_the_reviewed_reference_or_fixed_hazard_source() -> None:
    for setting_index in range(8):
        policy = _policy(setting_index)
        assert policy.source_setting_id == (
            M03R_V8_FIXED_HAZARD_SOURCE_SETTING_ID
            if setting_index == 6
            else M03R_TOP2000_DEV_REFERENCE_SETTING_ID
        )
        assert policy.source_policy.setting.setting_id == policy.source_setting_id
        assert len(policy.protocol_sha256) == 64


def test_four_horizon_distribution_masks_cash_and_backpropagates() -> None:
    policy = _policy(0)
    state = torch.randn((2, 5, 16), requires_grad=True)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, False, True, True]]
    )
    output = policy.alpha_pretraining_distribution(state, available)

    assert output.predicted_mean.shape == (2, 5, 4)
    assert output.predicted_log_scale.shape == (2, 5, 4)
    assert torch.equal(output.predicted_mean[:, 0], torch.zeros((2, 4)))
    assert torch.equal(output.predicted_log_scale[:, 0], torch.zeros((2, 4)))
    assert torch.equal(output.predicted_mean[1, 2], torch.zeros(4))
    (output.predicted_mean.sum() + output.predicted_log_scale.sum()).backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert policy.alpha_log_scale_head.weight.grad is not None


def test_softened_row_changes_only_exit_action_surrogate_for_equal_model_state() -> (
    None
):
    reference = _policy(0)
    softened = _policy(3)
    softened.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for policy in (reference, softened):
            head = policy.source_policy.core.alpha_head
            assert head is not None and head.exit_action_head_v6 is not None
            head.exit_action_head_v6.action_logits.weight.zero_()
            head.exit_action_head_v6.action_logits.bias.copy_(
                torch.tensor([0.1, 2.0, -0.5])
            )

    state = torch.randn((1, 5, 16))
    weights = torch.tensor([[0.96, 0.01, 0.01, 0.01, 0.01]])
    available = torch.ones((1, 5), dtype=torch.bool)
    age = torch.zeros((1, 5, 5))
    reference_intent = reference.hold30_intent(state, weights, available, age)
    softened_intent = softened.hold30_intent(state, weights, available, age)

    assert reference_intent.exit_action_v6 is not None
    assert softened_intent.exit_action_v6 is not None
    assert torch.equal(reference_intent.entry_scores, softened_intent.entry_scores)
    assert torch.equal(
        reference_intent.hazard_residual, softened_intent.hazard_residual
    )
    assert torch.equal(
        reference_intent.exit_action_v6.decision_st.detach(),
        softened_intent.exit_action_v6.decision_st.detach(),
    )
    assert not torch.equal(
        reference_intent.exit_action_v6.soft_probabilities,
        softened_intent.exit_action_v6.soft_probabilities,
    )


def test_distribution_tamper_revalidates() -> None:
    policy = _policy(0)
    output = policy.alpha_pretraining_distribution(
        torch.randn((1, 3, 16)), torch.ones((1, 3), dtype=torch.bool)
    )
    with pytest.raises(M03RV8PolicyError, match="distribution"):
        replace(
            output,
            predicted_log_scale=torch.full_like(
                output.predicted_log_scale, float("nan")
            ),
        ).validate()
