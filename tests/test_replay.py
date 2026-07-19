from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.rl import (
    ActionBatch,
    ObservationBatch,
    RewardComponents,
    TransitionBatch,
    align_replay_batches,
)
from rl_quant.rl.replay import ReplayBatch, TransitionReplayBuffer


def _transition(start: int, count: int = 3) -> TransitionBatch:
    ids = torch.arange(start, start + count)
    observation = ObservationBatch(
        tensors={"state": torch.stack((ids.float(), ids.float() + 0.5), dim=-1)},
        action_mask=torch.ones(count, 2, dtype=torch.bool),
        episode_start=ids.remainder(5) == 0,
    )
    next_observation = ObservationBatch(
        tensors={"state": observation.tensors["state"] + 1.0},
        action_mask=torch.ones(count, 2, dtype=torch.bool),
        episode_start=ids.remainder(5) == 4,
    )
    action = ActionBatch(
        action=torch.stack((torch.ones(count), torch.zeros(count)), dim=-1),
        log_prob=-ids.float(),
    )
    zeros = torch.zeros(count)
    terminal = ids.remainder(5) == 4
    return TransitionBatch(
        observation=observation,
        action=action,
        executed_action=action.action,
        rewards=RewardComponents(
            gross_return=ids.float() / 100.0,
            execution_cost=zeros,
            impact_cost=zeros,
            risk_penalty=zeros,
            constraint_penalty=zeros,
            liquidation_cost=zeros,
        ),
        next_observation=next_observation,
        terminated=terminal,
        truncated=torch.zeros(count, dtype=torch.bool),
        discount=torch.where(terminal, zeros, torch.full((count,), 0.99)),
        info={"row_id": ids},
    )


def test_replay_batch_preserves_requested_execution_masks_and_terminal_semantics() -> None:
    batch = ReplayBatch.from_transition(_transition(2))
    assert batch.batch_size == 3
    assert batch.executed_actions is not None
    assert batch.action_masks is not None and batch.next_action_masks is not None
    assert batch.behavior_log_probs is not None
    assert batch.episode_starts is not None and batch.next_episode_starts is not None
    assert batch.episode_starts.tolist() == [False, False, False]
    assert batch.next_episode_starts.tolist() == [False, False, True]
    assert set(batch.reward_components) == {
        "gross_return",
        "execution_cost",
        "impact_cost",
        "risk_penalty",
        "constraint_penalty",
        "liquidation_cost",
    }
    assert batch.extras["row_id"].tolist() == [2, 3, 4]
    assert batch.terminated.tolist() == [False, False, True]
    assert batch.discounts.tolist()[-1] == 0.0

    with pytest.raises(ValueError, match="zero discount"):
        ReplayBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            discounts=torch.ones(3),
            terminated=batch.terminated,
            truncated=batch.truncated,
        )


def test_circular_replay_wraps_chronologically_and_samples_schema_locked_batches() -> None:
    replay = TransitionReplayBuffer(capacity=5)
    replay.add(_transition(0, 3))
    replay.add(_transition(3, 4))
    assert len(replay) == 5
    assert replay.all().extras["row_id"].tolist() == [2, 3, 4, 5, 6]

    generator = torch.Generator().manual_seed(9)
    sampled = replay.sample(4, replacement=False, generator=generator)
    assert sampled.batch_size == 4
    assert sampled.actions.shape == (4, 2)
    assert sampled.observations["state"].shape == (4, 2)
    assert sampled.episode_starts is not None
    assert sampled.next_episode_starts is not None

    changed = ReplayBatch.from_transition(_transition(8, 1))
    changed = ReplayBatch(
        observations={"different": changed.observations["state"]},
        actions=changed.actions,
        rewards=changed.rewards,
        next_observations={"different": changed.next_observations["state"]},
        discounts=changed.discounts,
        terminated=changed.terminated,
        truncated=changed.truncated,
        episode_starts=changed.episode_starts,
        next_episode_starts=changed.next_episode_starts,
        executed_actions=changed.executed_actions,
        action_masks=changed.action_masks,
        next_action_masks=changed.next_action_masks,
        behavior_log_probs=changed.behavior_log_probs,
        reward_components=changed.reward_components,
        extras=changed.extras,
    )
    with pytest.raises(ValueError, match="schema changed"):
        replay.add(changed)


def test_replay_checkpoint_round_trip_and_oversized_add_keeps_latest_rows() -> None:
    replay = TransitionReplayBuffer(capacity=4)
    replay.add(_transition(0, 6))
    assert replay.all().extras["row_id"].tolist() == [2, 3, 4, 5]

    restored = TransitionReplayBuffer(capacity=4)
    restored.load_state_dict(replay.state_dict())
    batch = restored.all()
    assert batch.extras["row_id"].tolist() == [2, 3, 4, 5]
    torch.testing.assert_close(batch.rewards, replay.all().rewards)


def test_replay_revalidates_mutable_external_batches_and_partial_cursor() -> None:
    batch = ReplayBatch.from_transition(_transition(0, 2))
    batch.rewards[0] = float("nan")
    replay = TransitionReplayBuffer(capacity=4)
    with pytest.raises(ValueError, match="rewards must be finite"):
        replay.add(batch)
    assert len(replay) == 0

    source = TransitionReplayBuffer(capacity=4)
    source.add(_transition(0, 2))
    malformed = source.state_dict()
    malformed["cursor"] = 3
    with pytest.raises(ValueError, match="cursor equal to size"):
        replay.load_state_dict(malformed)
    assert len(replay) == 0


def test_replay_alignment_uses_exact_decision_identity_not_row_position() -> None:
    first = ReplayBatch.from_transition(_transition(10, 3))
    first = replace(
        first,
        extras={
            **first.extras,
            "environment_index": torch.tensor([0, 0, 0]),
            "decision_id": torch.tensor([100, 101, 102]),
        },
    )
    permutation = torch.tensor([2, 0, 1])
    shuffled = first.index(permutation)
    aligned_first, aligned_second = align_replay_batches((first, shuffled))

    torch.testing.assert_close(aligned_second.rewards, aligned_first.rewards)
    torch.testing.assert_close(aligned_second.decision_keys(), aligned_first.decision_keys())

    mismatched = replace(
        shuffled,
        extras={**shuffled.extras, "decision_id": torch.tensor([999, 100, 101])},
    )
    with pytest.raises(ValueError, match="differs"):
        align_replay_batches((first, mismatched))

    duplicated = replace(
        first,
        extras={**first.extras, "decision_id": torch.tensor([100, 100, 102])},
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicated.decision_keys()
