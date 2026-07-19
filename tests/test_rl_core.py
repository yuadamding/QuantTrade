from __future__ import annotations

import pytest
import torch

from rl_quant.rl import (
    ActionBatch,
    ActionSpec,
    ObservationBatch,
    OnPolicyTrajectoryBuffer,
    RewardComponents,
    TensorSpec,
    TransitionBatch,
)


def _transition(
    step: int,
    *,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    discount: torch.Tensor,
) -> TransitionBatch:
    batch_size = reward.shape[0]
    observation = ObservationBatch(
        tensors={"state": torch.full((batch_size, 2), float(step))},
        action_mask=torch.ones((batch_size, 2), dtype=torch.bool),
        episode_start=torch.full((batch_size,), step == 0, dtype=torch.bool),
    )
    next_observation = ObservationBatch(
        tensors={"state": torch.full((batch_size, 2), float(step + 1))},
        action_mask=torch.ones((batch_size, 2), dtype=torch.bool),
        episode_start=torch.zeros(batch_size, dtype=torch.bool),
    )
    action = ActionBatch(
        action=torch.tensor([[1.0, 0.0]]).expand(batch_size, -1).clone(),
        log_prob=torch.full((batch_size,), -0.5),
    )
    zeros = torch.zeros_like(reward)
    rewards = RewardComponents(
        gross_return=reward,
        execution_cost=zeros,
        impact_cost=zeros,
        risk_penalty=zeros,
        constraint_penalty=zeros,
        liquidation_cost=zeros,
    )
    return TransitionBatch(
        observation=observation,
        action=action,
        executed_action=action.action,
        rewards=rewards,
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
        discount=discount,
    )


def test_specs_and_batches_fail_closed_and_support_device_transfer() -> None:
    spec = TensorSpec(shape=(3,), dtype=torch.float32, low=0.0, high=1.0)
    action_spec = ActionSpec(spec, simplex=True, cash_index=0)
    action_spec.validate(torch.tensor([[0.2, 0.3, 0.5]]))
    with pytest.raises(ValueError, match="sum to one"):
        action_spec.validate(torch.tensor([[0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError, match="dtype"):
        spec.validate(torch.ones((1, 3), dtype=torch.float64))

    observation = ObservationBatch(
        tensors={"market": torch.ones((2, 3, 4))},
        action_mask=torch.ones((2, 3), dtype=torch.bool),
        episode_start=torch.tensor([True, False]),
    )
    moved = observation.to(torch.device("cpu"))
    assert moved.batch_size == 2
    assert moved.action_mask is not None and moved.action_mask.dtype == torch.bool
    assert moved.detach().tensors["market"].grad_fn is None

    with pytest.raises(ValueError, match="batch size"):
        ObservationBatch(tensors={"a": torch.ones(2, 1), "b": torch.ones(3, 1)})


def test_reward_ledger_is_additive_and_rejects_negative_costs() -> None:
    rewards = RewardComponents(
        gross_return=torch.tensor([0.10]),
        execution_cost=torch.tensor([0.01]),
        impact_cost=torch.tensor([0.02]),
        risk_penalty=torch.tensor([0.03]),
        constraint_penalty=torch.tensor([0.01]),
        liquidation_cost=torch.tensor([0.01]),
        extras={"bonus": torch.tensor([0.005])},
    )
    torch.testing.assert_close(rewards.total, torch.tensor([0.025]))
    with pytest.raises(ValueError, match="nonnegative"):
        RewardComponents(
            gross_return=torch.tensor([0.0]),
            execution_cost=torch.tensor([-0.1]),
            impact_cost=torch.tensor([0.0]),
            risk_penalty=torch.tensor([0.0]),
            constraint_penalty=torch.tensor([0.0]),
            liquidation_cost=torch.tensor([0.0]),
        )


def test_gae_bootstraps_truncation_but_not_true_termination() -> None:
    transition = _transition(
        0,
        reward=torch.tensor([1.0, 1.0]),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
        discount=torch.tensor([0.0, 0.9]),
    )
    buffer = OnPolicyTrajectoryBuffer(horizon=1, num_envs=2)
    buffer.add(
        transition,
        value=torch.zeros(2),
        next_value=torch.tensor([999.0, 10.0]),
    )
    buffer.compute_gae(gae_lambda=0.95)
    batch = buffer.as_batch()
    torch.testing.assert_close(batch.advantages[0], torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(batch.returns[0], torch.tensor([1.0, 10.0]))


def test_rollout_can_begin_mid_episode_without_resetting_recurrent_state() -> None:
    transition = _transition(
        5,
        reward=torch.ones(1),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([False]),
        discount=torch.tensor([0.9]),
    )
    buffer = OnPolicyTrajectoryBuffer(horizon=1, num_envs=1)
    buffer.add(transition, value=torch.zeros(1), next_value=torch.zeros(1))
    buffer.compute_gae(gae_lambda=0.95)
    assert buffer.as_batch().episode_start.tolist() == [[False]]


def test_recurrent_sequences_use_burn_in_without_crossing_episode_boundaries() -> None:
    buffer = OnPolicyTrajectoryBuffer(horizon=4, num_envs=1)
    for step in range(4):
        terminal = step == 3
        transition = _transition(
            step,
            reward=torch.ones(1),
            terminated=torch.tensor([terminal]),
            truncated=torch.tensor([False]),
            discount=torch.tensor([0.0 if terminal else 0.9]),
        )
        buffer.add(
            transition,
            value=torch.zeros(1),
            next_value=torch.zeros(1),
            recurrent_state={"hidden": torch.full((1, 3), float(step))},
        )
    buffer.compute_gae(gae_lambda=1.0)
    sequences = buffer.recurrent_sequences(sequence_length=2, burn_in=1)

    assert sequences.rewards.shape == (2, 3)
    assert sequences.valid_mask.tolist() == [[False, True, True], [True, True, True]]
    assert sequences.loss_mask.tolist() == [[False, True, True], [False, True, True]]
    assert sequences.time_indices.tolist() == [[-1, 0, 1], [1, 2, 3]]
    assert sequences.episode_start.tolist() == [[False, True, False], [False, False, False]]
    torch.testing.assert_close(
        sequences.initial_recurrent_state["hidden"],
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    assert set(sequences.reward_components) == {
        "gross_return",
        "execution_cost",
        "impact_cost",
        "risk_penalty",
        "constraint_penalty",
        "liquidation_cost",
    }
    assert sequences.discounts.shape == sequences.rewards.shape


def test_recurrent_sequences_split_at_a_truncation() -> None:
    buffer = OnPolicyTrajectoryBuffer(horizon=4, num_envs=1)
    for step in range(4):
        truncated = step == 1
        terminated = step == 3
        transition = _transition(
            step,
            reward=torch.ones(1),
            terminated=torch.tensor([terminated]),
            truncated=torch.tensor([truncated]),
            discount=torch.tensor([0.0 if terminated else 0.9]),
        )
        buffer.add(transition, value=torch.zeros(1), next_value=torch.zeros(1))
    buffer.compute_gae(gae_lambda=1.0)
    sequences = buffer.recurrent_sequences(sequence_length=4, burn_in=1)

    assert sequences.num_sequences == 2
    assert sequences.time_indices.tolist() == [[-1, 0, 1, -1, -1], [-1, 2, 3, -1, -1]]
    # No state or advantage from the second episode may enter the first sequence.
    assert sequences.valid_mask.sum(dim=1).tolist() == [2, 2]
