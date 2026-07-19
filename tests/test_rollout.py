from __future__ import annotations

from typing import Mapping

import pytest
import torch

from rl_quant.rl import (
    ActionBatch,
    ActionSpec,
    ObservationBatch,
    OnPolicyRolloutCoordinator,
    PPOConfig,
    RecurrentActorCritic,
    RecurrentPPO,
    RewardComponents,
    TensorSpec,
    TransitionBatch,
)


class _CountingAgent:
    """Tiny recurrent actor-critic whose values expose state-timing mistakes."""

    def __init__(self) -> None:
        self.environment: _ToyVectorEnvironment | None = None
        self.value_events: list[tuple[float, float, int]] = []

    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        return {
            "hidden": torch.zeros(
                (observation.batch_size, 1),
                dtype=observation.tensors["state"].dtype,
                device=observation.device,
            )
        }

    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
    ) -> ActionBatch:
        del deterministic
        if recurrent_state is None:
            raise AssertionError("The coordinator must provide recurrent state.")
        hidden = recurrent_state["hidden"]
        return ActionBatch(
            action=torch.zeros(observation.batch_size, dtype=torch.long, device=observation.device),
            log_prob=torch.full(
                (observation.batch_size,),
                -0.25,
                dtype=observation.tensors["state"].dtype,
                device=observation.device,
            ),
            recurrent_state={"hidden": hidden + 1.0},
        )

    def value(
        self,
        observation: ObservationBatch,
        *,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if recurrent_state is None:
            raise AssertionError("The coordinator must provide recurrent state.")
        state = observation.tensors["state"].squeeze(-1)
        hidden = recurrent_state["hidden"].squeeze(-1)
        reset_count = -1 if self.environment is None else self.environment.reset_count
        self.value_events.append((float(state[0].item()), float(hidden[0].item()), reset_count))
        return state + hidden


class _ToyVectorEnvironment:
    """Two synchronous episodes: one terminates while the other truncates."""

    def __init__(self, *, partial_done: bool = False) -> None:
        self.action_spec = ActionSpec(
            TensorSpec(shape=(), dtype=torch.long, low=0, high=1),
            kind="discrete",
        )
        self._batch_size = 2
        self.partial_done = partial_done
        self.reset_count = 0
        self._step = 0
        self._done = True
        self._observation: ObservationBatch | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _make_observation(self, step: int, *, episode_start: bool) -> ObservationBatch:
        return ObservationBatch(
            tensors={"state": torch.full((self.batch_size, 1), float(step))},
            episode_start=torch.full((self.batch_size,), episode_start, dtype=torch.bool),
        )

    def reset(self) -> tuple[ObservationBatch, Mapping[str, torch.Tensor]]:
        self.reset_count += 1
        self._step = 0
        self._done = False
        self._observation = self._make_observation(0, episode_start=True)
        return self._observation, {}

    def step(self, action: ActionBatch | torch.Tensor) -> TransitionBatch:
        if self._observation is None or self._done:
            raise RuntimeError("reset required")
        action_batch = action if isinstance(action, ActionBatch) else ActionBatch(action=action)
        self.action_spec.validate(action_batch.action)
        old_observation = self._observation
        self._step += 1
        if self.partial_done and self._step == 1:
            terminated = torch.tensor([True, False])
            truncated = torch.tensor([False, False])
        else:
            completed = self._step == 3
            terminated = torch.tensor([completed, False])
            truncated = torch.tensor([False, completed])
        done = terminated | truncated
        self._done = bool(done.all().item())
        self._observation = self._make_observation(self._step, episode_start=False)
        reward = torch.ones(self.batch_size)
        zeros = torch.zeros_like(reward)
        rewards = RewardComponents(
            gross_return=reward,
            execution_cost=zeros,
            impact_cost=zeros,
            risk_penalty=zeros,
            constraint_penalty=zeros,
            liquidation_cost=zeros,
        )
        discount = torch.full_like(reward, 0.9)
        discount = torch.where(terminated, torch.zeros_like(discount), discount)
        return TransitionBatch(
            observation=old_observation,
            action=action_batch,
            executed_action=action_batch.action,
            rewards=rewards,
            next_observation=self._observation,
            terminated=terminated,
            truncated=truncated,
            discount=discount,
        )


def test_rollout_carries_recurrent_and_episode_state_and_bootstraps_only_truncation() -> None:
    environment = _ToyVectorEnvironment()
    agent = _CountingAgent()
    agent.environment = environment
    coordinator = OnPolicyRolloutCoordinator(
        environment,
        agent,
        horizon=2,
        gae_lambda=1.0,
    )

    first = coordinator.collect()
    first_batch = first.buffer.as_batch()
    assert environment.reset_count == 1
    assert first_batch.episode_start.tolist() == [[True, True], [False, False]]
    torch.testing.assert_close(
        first_batch.recurrent_states["hidden"],
        torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]]]),
    )
    torch.testing.assert_close(first.continuation.recurrent_state["hidden"], torch.full((2, 1), 2.0))
    torch.testing.assert_close(first.continuation.running_episode_returns, torch.tensor([2.0, 2.0]))
    assert first.continuation.running_episode_lengths.tolist() == [2, 2]
    assert first.metrics.episodes_completed == 0
    assert first.metrics.mean_episode_return is None

    second = coordinator.collect(first.continuation)
    second_batch = second.buffer.as_batch()
    # The first step consumes the carried state; the second follows a real reset.
    torch.testing.assert_close(
        second_batch.recurrent_states["hidden"],
        torch.tensor([[[2.0], [2.0]], [[0.0], [0.0]]]),
    )
    assert second_batch.episode_start.tolist() == [[False, False], [True, True]]
    torch.testing.assert_close(second_batch.values[0], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(second_batch.next_values[0], torch.tensor([0.0, 6.0]))
    torch.testing.assert_close(second_batch.advantages[0], torch.tensor([-3.0, 2.4]))

    # V(next) at state=3, hidden=3 was evaluated while reset_count was still 1.
    assert (3.0, 3.0, 1) in agent.value_events
    assert environment.reset_count == 2
    assert second.metrics.environment_steps == 2
    assert second.metrics.transitions == 4
    assert second.metrics.episodes_completed == 2
    assert second.metrics.episodes_terminated == 1
    assert second.metrics.episodes_truncated == 1
    assert second.metrics.mean_episode_return == pytest.approx(3.0)
    assert second.metrics.mean_episode_length == pytest.approx(3.0)
    torch.testing.assert_close(second.continuation.running_episode_returns, torch.ones(2))
    assert second.continuation.running_episode_lengths.tolist() == [1, 1]
    torch.testing.assert_close(second.continuation.recurrent_state["hidden"], torch.ones((2, 1)))


def test_rollout_fails_clearly_when_only_part_of_batch_finishes() -> None:
    environment = _ToyVectorEnvironment(partial_done=True)
    coordinator = OnPolicyRolloutCoordinator(
        environment,
        _CountingAgent(),
        horizon=1,
        gae_lambda=0.95,
    )

    with pytest.raises(RuntimeError, match="Only part.*whole-batch reset"):
        coordinator.collect()


def test_toy_environment_rollout_trains_with_recurrent_ppo() -> None:
    torch.manual_seed(7)
    environment = _ToyVectorEnvironment()
    model = RecurrentActorCritic(
        observation_key="state",
        observation_dim=1,
        hidden_dim=8,
        action_dim=2,
        action_kind="categorical",
    )
    algorithm = RecurrentPPO(
        model,
        PPOConfig(epochs=1, minibatch_sequences=2),
    )
    coordinator = OnPolicyRolloutCoordinator(
        environment,
        algorithm,
        horizon=2,
        gae_lambda=0.95,
    )

    rollout = coordinator.collect()
    sequences = rollout.buffer.recurrent_sequences(sequence_length=2)
    update_metrics = algorithm.update(sequences)

    assert rollout.buffer.full
    assert sequences.initial_recurrent_state["hidden"].shape == (2, 8)
    assert torch.isfinite(torch.tensor(float(update_metrics["loss"])))
