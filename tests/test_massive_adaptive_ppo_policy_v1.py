from __future__ import annotations

from inspect import signature

import torch

from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
    MassiveAdaptivePPOActorCriticV1,
)


def test_mixed_action_distribution_respects_registered_support() -> None:
    distribution = MassiveAdaptiveBoundedControlDistributionV1(
        mean=torch.zeros((32, 10)),
        log_std=torch.full((32, 10), -1.5),
    )
    action = distribution.sample()

    assert action.shape == (32, 10)
    assert torch.all(action.abs() < 1.0)
    assert torch.isfinite(distribution.log_prob(action)).all()
    assert torch.isfinite(distribution.entropy()).all()


def test_actor_critic_is_small_stateless_and_neutral_initialized() -> None:
    model = MassiveAdaptivePPOActorCriticV1(observation_dim=90)
    output = model({"adaptive_state": torch.zeros((4, 90))})
    mode = output.distribution.mode()

    assert output.value.shape == (4,)
    assert output.recurrent_state == {}
    assert torch.equal(mode, torch.zeros((4, 10)))
    assert model.actor is not model.critic
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_policy_surface_contains_no_duration_control() -> None:
    forbidden = ("age", "duration", "persistence", "hazard", "scheduled_exit")
    assert all(
        not any(fragment in parameter for fragment in forbidden)
        for parameter in signature(
            MassiveAdaptiveBoundedControlDistributionV1.__init__
        ).parameters
    )
