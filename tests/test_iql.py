from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from rl_quant.rl import MaskedDirichlet, ObservationBatch
from rl_quant.rl.iql import (
    IQLConfig,
    ImplicitQLearning,
    RegimeMixtureIQLActorCritic,
    VectorIQLActorCritic,
)
from rl_quant.rl.replay import ReplayBatch, TransitionReplayBuffer
from rl_quant.rl.robust import AdverseRewardTransform, ObservationAffineTransform, UncertaintyAbstention


def _batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    *,
    action_masks: torch.Tensor | None = None,
    executed_actions: torch.Tensor | None = None,
) -> ReplayBatch:
    count = observations.shape[0]
    return ReplayBatch(
        observations={"state": observations},
        actions=actions,
        rewards=rewards,
        next_observations={"state": torch.zeros_like(observations)},
        discounts=torch.zeros(count),
        terminated=torch.ones(count, dtype=torch.bool),
        truncated=torch.zeros(count, dtype=torch.bool),
        executed_actions=actions if executed_actions is None else executed_actions,
        action_masks=action_masks,
        next_action_masks=action_masks,
    )


def test_iql_worst_case_target_uses_adverse_transformed_environment() -> None:
    model = VectorIQLActorCritic(
        observation_key="state",
        observation_dim=2,
        action_dim=1,
        hidden_dims=(8,),
    )
    algorithm = ImplicitQLearning(
        model,
        transforms=(
            ObservationAffineTransform(keys=("state",), scale=-1.0, transform_current=False),
            AdverseRewardTransform(0.25),
        ),
    )
    batch = _batch(torch.randn(5, 2), torch.randn(5, 1), torch.ones(5))
    batch = replace(
        batch,
        next_observations={"state": torch.ones_like(batch.observations["state"])},
    )
    target, spread = algorithm._conservative_target(batch)
    torch.testing.assert_close(target, torch.full((5,), 0.75))
    torch.testing.assert_close(spread, torch.full((5,), 0.25))

    current_only = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=2,
            action_dim=1,
            hidden_dims=(8,),
        ),
        transforms=(
            ObservationAffineTransform(
                keys=("state",),
                scale=-1.0,
                transform_current=True,
                transform_next=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="current-observation-only"):
        current_only._conservative_target(batch)


def test_builtin_robust_transforms_reject_changed_field_overflow() -> None:
    maximum = torch.finfo(torch.float32).max
    batch = _batch(
        torch.full((1, 1), maximum),
        torch.zeros(1, 1),
        torch.full((1,), -maximum),
    )

    with pytest.raises(ValueError, match="Observation transform.*non-finite"):
        ObservationAffineTransform(
            keys=("state",),
            scale=2.0,
            transform_current=True,
            transform_next=False,
        )(batch)
    with pytest.raises(ValueError, match="Adverse reward transform.*non-finite"):
        AdverseRewardTransform(maximum)(batch)


def test_iql_robust_transform_cannot_reorder_decision_rows() -> None:
    class ReverseRows:
        def __call__(self, batch: ReplayBatch) -> ReplayBatch:
            indices = torch.arange(batch.batch_size - 1, -1, -1)
            return batch.index(indices)

    model = VectorIQLActorCritic(
        observation_key="state",
        observation_dim=1,
        action_dim=1,
        hidden_dims=(8,),
    )
    algorithm = ImplicitQLearning(model, transforms=(ReverseRows(),))
    batch = _batch(
        torch.tensor([[0.0], [1.0]]),
        torch.zeros(2, 1),
        torch.tensor([1.0, 100.0]),
    )

    with pytest.raises(ValueError, match="row identity and order"):
        algorithm._conservative_target(batch)
    with pytest.raises(TypeError, match="checkpoint-stable"):
        algorithm.state_dict()


def test_iql_without_robust_transforms_does_not_revalidate_trusted_batch(monkeypatch) -> None:
    model = VectorIQLActorCritic(
        observation_key="state",
        observation_dim=1,
        action_dim=1,
        hidden_dims=(8,),
    )
    algorithm = ImplicitQLearning(model)
    batch = _batch(torch.zeros(4, 1), torch.zeros(4, 1), torch.ones(4))

    def fail_revalidation(self: ReplayBatch, _validate_values: bool) -> None:
        del self, _validate_values
        raise AssertionError("trusted target path reconstructed ReplayBatch")

    monkeypatch.setattr(ReplayBatch, "__post_init__", fail_revalidation)
    target, spread = algorithm._conservative_target(batch)

    assert target.shape == (4,)
    torch.testing.assert_close(spread, torch.zeros(4))


def test_iql_update_metrics_and_checkpoint_are_complete() -> None:
    torch.manual_seed(3)
    model = VectorIQLActorCritic(
        observation_key="state",
        observation_dim=3,
        action_dim=2,
        hidden_dims=(16, 16),
    )
    config = IQLConfig(
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        value_learning_rate=1e-3,
        critic_uncertainty_penalty=0.5,
    )
    algorithm = ImplicitQLearning(model, config, transforms=(AdverseRewardTransform(0.01),))
    batch = _batch(torch.randn(32, 3), torch.randn(32, 2), torch.randn(32))
    metrics = algorithm.update(batch)
    for name in (
        "critic_loss",
        "value_loss",
        "actor_loss",
        "critic_uncertainty_mean",
        "transform_target_spread_mean",
        "critic_grad_norm",
    ):
        assert math.isfinite(float(metrics[name])), name
    assert metrics["robust_transform_count"] == 1

    restored = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=3,
            action_dim=2,
            hidden_dims=(16, 16),
        ),
        config,
        transforms=(AdverseRewardTransform(0.01),),
    )
    restored.load_state_dict(algorithm.state_dict())
    observation = ObservationBatch(tensors={"state": torch.randn(4, 3)})
    torch.testing.assert_close(
        restored.act(observation, deterministic=True).action,
        algorithm.act(observation, deterministic=True).action,
    )
    assert restored.update_count == 1


def test_iql_learns_planted_offline_continuous_bandit() -> None:
    torch.manual_seed(5)
    count = 1024
    context = torch.where(torch.rand(count, 1) > 0.5, 0.8, -0.8)
    behavior = torch.empty(count, 1).uniform_(-1.0, 1.0)
    reward = -(behavior - context).square().squeeze(-1)
    replay = TransitionReplayBuffer(capacity=count)
    replay.add(_batch(context, behavior, reward))
    model = VectorIQLActorCritic(
        observation_key="state",
        observation_dim=1,
        action_dim=1,
        hidden_dims=(32, 32),
    )
    algorithm = ImplicitQLearning(
        model,
        IQLConfig(
            actor_learning_rate=3e-3,
            critic_learning_rate=3e-3,
            value_learning_rate=3e-3,
            expectile=0.8,
            advantage_temperature=5.0,
            target_tau=0.02,
        ),
    )
    evaluation = ObservationBatch(tensors={"state": torch.tensor([[-0.8], [0.8]])})
    before = algorithm.act(evaluation, deterministic=True).action.squeeze(-1)
    for _ in range(250):
        algorithm.update(replay.sample(128))
    after = algorithm.act(evaluation, deterministic=True).action.squeeze(-1)
    target = torch.tensor([-0.8, 0.8])
    assert (after - target).square().mean() < (before - target).square().mean()
    assert (after - target).abs().max() < 0.35


def test_dirichlet_iql_respects_active_simplex_and_uncertainty_abstention() -> None:
    torch.manual_seed(7)
    count = 24
    observations = torch.randn(count, 3)
    masks = torch.ones(count, 3, dtype=torch.bool)
    masks[::2, 2] = False
    behavior = MaskedDirichlet(torch.ones(count, 3), masks).sample()
    rewards = behavior[:, 1] - 0.1 * behavior[:, 2]
    batch = _batch(observations, behavior, rewards, action_masks=masks)
    algorithm = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=3,
            action_dim=3,
            hidden_dims=(16,),
            action_kind="dirichlet",
        ),
        IQLConfig(
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            value_learning_rate=1e-3,
        ),
    )
    metrics = algorithm.update(batch)
    assert math.isfinite(float(metrics["actor_loss"]))
    observation = ObservationBatch(tensors={"state": observations[:2]}, action_mask=masks[:2])
    action = algorithm.act(observation, deterministic=True)
    torch.testing.assert_close(action.action.sum(-1), torch.ones(2))
    assert action.action[0, 2] == 0

    overlay = UncertaintyAbstention(threshold=0.1, fallback_action=torch.tensor([1.0, 0.0, 0.0]))
    result = overlay.apply(action, torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(result.action.action[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result.action.action[1], action.action[1])
    assert result.abstained.tolist() == [True, False]
    assert result.action.log_prob is None


def test_iql_uses_executed_actions_and_smooths_sparse_simplex_only_for_likelihood() -> None:
    observations = torch.randn(8, 3)
    requested = torch.full((8, 3), 1.0 / 3.0)
    executed = torch.zeros_like(requested)
    executed[:, 0] = 1.0
    masks = torch.ones_like(executed, dtype=torch.bool)
    batch = _batch(
        observations,
        requested,
        torch.zeros(8),
        action_masks=masks,
        executed_actions=executed,
    )
    algorithm = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=3,
            action_dim=3,
            hidden_dims=(16,),
            action_kind="dirichlet",
        ),
        IQLConfig(
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            value_learning_rate=1e-3,
            simplex_behavior_smoothing=1e-5,
        ),
    )

    training_actions, used_executed = algorithm._training_actions(batch)
    torch.testing.assert_close(training_actions, executed)
    assert used_executed
    metrics = algorithm.update(batch)
    assert metrics["critic_uses_executed_actions"] == 1
    assert float(metrics["behavior_smoothing_l1_mean"]) > 0
    assert float(metrics["action_projection_l1_mean"]) > 0


@pytest.mark.parametrize(
    "invalid_action, message",
    [
        (torch.tensor([-0.1, 1.1]), "negative"),
        (torch.tensor([0.2, 0.2]), "sum to one"),
    ],
)
def test_dirichlet_iql_rejects_corrupt_logged_simplex_before_any_update(
    invalid_action: torch.Tensor,
    message: str,
) -> None:
    observations = torch.randn(4, 2)
    actions = invalid_action.repeat(4, 1)
    masks = torch.ones_like(actions, dtype=torch.bool)
    algorithm = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=2,
            action_dim=2,
            hidden_dims=(8,),
            action_kind="dirichlet",
        )
    )
    before = {name: value.detach().clone() for name, value in algorithm.model.state_dict().items()}

    with pytest.raises(ValueError, match=message):
        algorithm.update(_batch(observations, actions, torch.zeros(4), action_masks=masks))

    assert algorithm.update_count == 0
    for name, value in algorithm.model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_sparse_simplex_without_smoothing_fails_before_any_optimizer_step() -> None:
    observations = torch.randn(4, 2)
    actions = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    masks = torch.ones_like(actions, dtype=torch.bool)
    algorithm = ImplicitQLearning(
        VectorIQLActorCritic(
            observation_key="state",
            observation_dim=2,
            action_dim=2,
            hidden_dims=(8,),
            action_kind="dirichlet",
        ),
        IQLConfig(simplex_behavior_smoothing=0.0),
    )
    before = {name: value.detach().clone() for name, value in algorithm.model.state_dict().items()}

    with pytest.raises(ValueError, match="strictly positive"):
        algorithm.update(_batch(observations, actions, torch.zeros(4), action_masks=masks))

    assert algorithm.update_count == 0
    for name, value in algorithm.model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_regime_mixture_iql_routes_specialists_on_the_masked_simplex() -> None:
    torch.manual_seed(13)
    observations = torch.randn(12, 4)
    masks = torch.ones(12, 3, dtype=torch.bool)
    masks[::3, 2] = False
    executed = torch.zeros(12, 3)
    executed[:, 0] = 1.0
    batch = _batch(
        observations,
        torch.full((12, 3), 1.0 / 3.0),
        torch.zeros(12),
        action_masks=masks,
        executed_actions=executed,
    )
    algorithm = ImplicitQLearning(
        RegimeMixtureIQLActorCritic(
            observation_key="state",
            observation_dim=4,
            action_dim=3,
            num_experts=3,
            hidden_dims=(16,),
            router_hidden_dim=8,
            action_kind="dirichlet",
        ),
        IQLConfig(
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            value_learning_rate=1e-3,
            router_balance_coefficient=0.1,
            router_entropy_coefficient=0.01,
        ),
    )

    metrics = algorithm.update(batch)
    assert math.isfinite(float(metrics["router_entropy"]))
    assert math.isfinite(float(metrics["router_balance_loss"]))
    action = algorithm.act(
        ObservationBatch(tensors={"state": observations[:3]}, action_mask=masks[:3])
    )
    torch.testing.assert_close(action.action.sum(dim=-1), torch.ones(3))
    assert torch.equal(action.action[~masks[:3]], torch.zeros_like(action.action[~masks[:3]]))
    assert action.extras["router_probabilities"].shape == (3, 3)
    assert action.extras["routed_expert"].shape == (3,)
