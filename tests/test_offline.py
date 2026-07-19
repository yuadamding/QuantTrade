from __future__ import annotations

from typing import Any, Mapping

import pytest
import torch

from rl_quant.envs import HistoricalMarketData, PortfolioConstraints, VectorPortfolioEnv
from rl_quant.rl import (
    ActionBatch,
    Algorithm,
    ObservationBatch,
    OfflineTrainer,
    OfflineTrainingConfig,
    ReplayBatch,
    ReplayRolloutCollector,
    TransitionReplayBuffer,
)


class _StockBehavior:
    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
    ) -> ActionBatch:
        del deterministic, recurrent_state
        action = torch.zeros((observation.batch_size, 2), device=observation.device)
        action[:, 1] = 1.0
        return ActionBatch(action=action, log_prob=torch.zeros(observation.batch_size))


class _CountingAlgorithm(Algorithm):
    def __init__(self) -> None:
        self.updates = 0
        self.sampled_rewards: list[tuple[float, ...]] = []

    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
    ) -> ActionBatch:
        del deterministic, recurrent_state
        return ActionBatch(action=torch.zeros(observation.batch_size, 1))

    def update(self, batch: Any) -> Mapping[str, float | int | torch.Tensor]:
        assert isinstance(batch, ReplayBatch)
        self.updates += 1
        self.sampled_rewards.append(tuple(float(value) for value in batch.rewards.tolist()))
        return {"reward_mean": batch.rewards.mean(), "updates": self.updates}

    def state_dict(self) -> Mapping[str, Any]:
        return {"updates": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.updates = int(state["updates"])


def _environment() -> VectorPortfolioEnv:
    returns = torch.tensor([[[0.0, 0.10], [0.0, -0.05], [0.0, 0.02]]])
    return VectorPortfolioEnv(
        HistoricalMarketData(
            features={"signal": torch.zeros(1, 4, 1)},
            asset_returns=returns,
            availability=torch.ones(1, 4, 2, dtype=torch.bool),
        ),
        constraints=PortfolioConstraints(max_turnover=0.25),
    )


def test_replay_collector_preserves_continuation_and_executed_action_identity() -> None:
    replay = TransitionReplayBuffer(capacity=8)
    collector = ReplayRolloutCollector(_environment(), _StockBehavior())

    first = collector.collect(replay, environment_steps=2)
    assert first.continuation.running_episode_lengths.tolist() == [2]
    assert first.metrics.episodes_completed == 0
    assert len(replay) == 2
    assert first.metrics.requested_execution_l1_mean > 0

    second = collector.collect(replay, environment_steps=1, continuation=first.continuation)
    assert second.metrics.episodes_completed == 1
    assert second.metrics.episodes_terminated == 1
    assert second.continuation.running_episode_lengths.tolist() == [0]
    stored = replay.all()
    assert stored.executed_actions is not None
    assert not torch.equal(stored.actions[0], stored.executed_actions[0])
    assert stored.terminated.tolist() == [False, False, True]


def test_offline_trainer_is_seeded_and_aggregates_scalar_metrics() -> None:
    replay = TransitionReplayBuffer(capacity=8)
    ReplayRolloutCollector(_environment(), _StockBehavior()).collect(replay, environment_steps=3)
    algorithm = _CountingAlgorithm()
    callbacks: list[int] = []
    summary = OfflineTrainer(algorithm, replay).fit(
        OfflineTrainingConfig(updates=4, batch_size=2, seed=17),
        on_update=lambda update, _metrics: callbacks.append(update),
    )

    assert summary.updates == 4
    assert algorithm.updates == 4
    assert callbacks == [1, 2, 3, 4]
    assert summary.last_metrics["updates"] == 4.0
    assert summary.mean_metrics["updates"] == 2.5


def test_offline_trainer_checkpoint_resumes_exact_sample_sequence() -> None:
    replay = TransitionReplayBuffer(capacity=8)
    ReplayRolloutCollector(_environment(), _StockBehavior()).collect(replay, environment_steps=3)
    first_algorithm = _CountingAlgorithm()
    first = OfflineTrainer(first_algorithm, replay)
    config = OfflineTrainingConfig(updates=2, batch_size=2, seed=23)
    first.fit(config)
    checkpoint = first.state_dict()

    first_algorithm.sampled_rewards.clear()
    first.fit(config)
    expected = list(first_algorithm.sampled_rewards)

    restored_algorithm = _CountingAlgorithm()
    restored = OfflineTrainer(restored_algorithm, replay)
    restored.load_state_dict(checkpoint)
    restored.fit(config)

    assert restored_algorithm.sampled_rewards == expected
    assert restored.state_dict()["updates_completed"] == 4


def test_offline_trainer_rejects_malformed_sampling_rng_state() -> None:
    replay = TransitionReplayBuffer(capacity=8)
    trainer = OfflineTrainer(_CountingAlgorithm(), replay)

    with pytest.raises(ValueError, match="seed/generator_state"):
        trainer.load_state_dict(
            {
                "seed": 23,
                "updates_completed": 0,
                "replay_device": "cpu",
                "generator_state": torch.zeros(4, dtype=torch.float32),
            }
        )
