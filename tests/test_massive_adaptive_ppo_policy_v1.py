from __future__ import annotations

from inspect import signature

import pytest
import torch

from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
    MassiveAdaptivePPOActorCriticV1,
    build_seeded_massive_adaptive_ppo_model_v1,
    massive_adaptive_ppo_initial_model_state_receipt_v1,
    massive_adaptive_ppo_model_state_receipt_v1,
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


def test_seeded_model_initialization_is_scoped_and_deterministic() -> None:
    torch.manual_seed(111)
    ambient_rng_before_first = torch.get_rng_state().clone()
    first = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
    assert torch.equal(torch.get_rng_state(), ambient_rng_before_first)

    torch.manual_seed(222)
    ambient_rng_before_second = torch.get_rng_state().clone()
    second = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
    assert torch.equal(torch.get_rng_state(), ambient_rng_before_second)

    different = build_seeded_massive_adaptive_ppo_model_v1(seed=18)
    first_receipt = massive_adaptive_ppo_model_state_receipt_v1(first)
    assert first_receipt == massive_adaptive_ppo_initial_model_state_receipt_v1(
        seed=17
    )
    assert first_receipt == massive_adaptive_ppo_model_state_receipt_v1(second)
    assert first_receipt != massive_adaptive_ppo_model_state_receipt_v1(different)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


@pytest.mark.parametrize("seed", (-1, True, 1.5))
def test_seeded_model_initialization_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_seeded_massive_adaptive_ppo_model_v1(seed=seed)  # type: ignore[arg-type]


def test_policy_surface_contains_no_duration_control() -> None:
    forbidden = ("age", "duration", "persistence", "hazard", "scheduled_exit")
    assert all(
        not any(fragment in parameter for fragment in forbidden)
        for parameter in signature(
            MassiveAdaptiveBoundedControlDistributionV1.__init__
        ).parameters
    )
