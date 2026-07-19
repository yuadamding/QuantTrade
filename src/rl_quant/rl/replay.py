"""Typed transition replay for offline and off-policy reinforcement learning.

The older foundation replay buffers store anonymous field dictionaries.  This
module keeps the richer general-RL contract intact: requested and executed
actions, termination versus truncation, action masks, behavior likelihoods,
reward decomposition, and named observations all survive sampling.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Mapping

import torch

from rl_quant.rl.types import TransitionBatch


def _mapping_to(
    values: Mapping[str, torch.Tensor],
    device: torch.device | str,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=non_blocking)
        for name, value in values.items()
    }


def _mapping_index(
    values: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in values.items()}


@dataclass(frozen=True)
class ReplayBatch:
    """A validated batch of offline/off-policy transitions.

    All tensors share one leading transition dimension and device.  Discounts
    already include the algorithm's gamma convention; a true termination must
    have zero discount, while a truncation may retain a bootstrap discount.
    """

    observations: Mapping[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: Mapping[str, torch.Tensor]
    discounts: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_starts: torch.Tensor | None = None
    next_episode_starts: torch.Tensor | None = None
    executed_actions: torch.Tensor | None = None
    action_masks: torch.Tensor | None = None
    next_action_masks: torch.Tensor | None = None
    behavior_log_probs: torch.Tensor | None = None
    reward_components: Mapping[str, torch.Tensor] = field(default_factory=dict)
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)
    _validate_values: InitVar[bool] = True

    def __post_init__(self, _validate_values: bool) -> None:
        if not self.observations:
            raise ValueError("ReplayBatch observations cannot be empty.")
        if set(self.observations) != set(self.next_observations):
            raise ValueError("Current and next observation keys must match.")
        first = next(iter(self.observations.values()))
        if first.ndim == 0 or first.shape[0] <= 0:
            raise ValueError("ReplayBatch needs a non-empty leading transition dimension.")
        batch_size, device = first.shape[0], first.device

        for name, value in self.observations.items():
            next_value = self.next_observations[name]
            if value.ndim == 0 or value.shape[0] != batch_size or value.device != device:
                raise ValueError(f"observations[{name!r}] must have leading size {batch_size} on {device}.")
            if next_value.shape != value.shape or next_value.dtype != value.dtype or next_value.device != device:
                raise ValueError(f"next_observations[{name!r}] must match its current observation schema.")
            for label, tensor in (("observation", value), ("next observation", next_value)):
                if _validate_values and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                    raise ValueError(f"{label} {name!r} must be finite; store missingness explicitly.")

        if self.actions.ndim == 0 or self.actions.shape[0] != batch_size or self.actions.device != device:
            raise ValueError(f"actions must have leading size {batch_size} on {device}.")
        if _validate_values and self.actions.is_floating_point() and not bool(torch.isfinite(self.actions).all().item()):
            raise ValueError("actions must be finite.")
        if self.executed_actions is not None:
            if (
                self.executed_actions.shape != self.actions.shape
                or self.executed_actions.dtype != self.actions.dtype
                or self.executed_actions.device != device
            ):
                raise ValueError("executed_actions must exactly match the requested action schema.")
            if _validate_values and self.executed_actions.is_floating_point() and not bool(
                torch.isfinite(self.executed_actions).all().item()
            ):
                raise ValueError("executed_actions must be finite.")

        vectors = {
            "rewards": self.rewards,
            "discounts": self.discounts,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if self.behavior_log_probs is not None:
            vectors["behavior_log_probs"] = self.behavior_log_probs
        if self.episode_starts is not None:
            vectors["episode_starts"] = self.episode_starts
        if self.next_episode_starts is not None:
            vectors["next_episode_starts"] = self.next_episode_starts
        for name, value in vectors.items():
            if value.shape != (batch_size,) or value.device != device:
                raise ValueError(f"{name} must have shape ({batch_size},) on {device}.")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise ValueError("terminated and truncated must be boolean.")
        for name, optional_value in (
            ("episode_starts", self.episode_starts),
            ("next_episode_starts", self.next_episode_starts),
        ):
            if optional_value is not None and optional_value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean.")
        if _validate_values and bool((self.terminated & self.truncated).any().item()):
            raise ValueError("A replay transition cannot be both terminated and truncated.")
        for name, value in (("rewards", self.rewards), ("discounts", self.discounts)):
            if not value.is_floating_point() or (
                _validate_values and not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"{name} must be finite floating point.")
        if _validate_values and bool(((self.discounts < 0) | (self.discounts > 1)).any().item()):
            raise ValueError("discounts must lie in [0, 1].")
        if _validate_values and bool((self.discounts[self.terminated] != 0).any().item()):
            raise ValueError("True terminations must have zero discount.")
        if self.behavior_log_probs is not None and (
            not self.behavior_log_probs.is_floating_point()
            or (_validate_values and not bool(torch.isfinite(self.behavior_log_probs).all().item()))
        ):
            raise ValueError("behavior_log_probs must be finite floating point.")

        for name, mask in (("action_masks", self.action_masks), ("next_action_masks", self.next_action_masks)):
            if mask is None:
                continue
            if mask.ndim < 2 or mask.shape[0] != batch_size or mask.dtype != torch.bool or mask.device != device:
                raise ValueError(f"{name} must be bool with leading size {batch_size} on {device}.")

        for group_name, values in (("reward_components", self.reward_components), ("extras", self.extras)):
            for name, value in values.items():
                if value.ndim == 0 or value.shape[0] != batch_size or value.device != device:
                    raise ValueError(f"{group_name}[{name!r}] needs leading size {batch_size} on {device}.")
                if _validate_values and value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{group_name}[{name!r}] must be finite.")

    def validate(self) -> None:
        """Revalidate structure and values at an external trust boundary.

        Tensor payloads remain mutable even though this dataclass is frozen.
        Replay ingestion therefore cannot assume that a batch validated at
        construction has remained unchanged.
        """

        self.__post_init__(True)

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.actions.device

    @classmethod
    def from_transition(cls, transition: TransitionBatch) -> ReplayBatch:
        return cls(
            observations=transition.observation.tensors,
            actions=transition.action.action,
            executed_actions=transition.executed_action,
            rewards=transition.reward,
            next_observations=transition.next_observation.tensors,
            discounts=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            episode_starts=transition.observation.episode_start,
            next_episode_starts=transition.next_observation.episode_start,
            action_masks=transition.observation.action_mask,
            next_action_masks=transition.next_observation.action_mask,
            behavior_log_probs=transition.action.log_prob,
            reward_components=transition.rewards.as_dict(),
            extras=transition.info,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> ReplayBatch:
        move = lambda value: value.to(device=device, non_blocking=non_blocking)  # noqa: E731
        return ReplayBatch(
            observations=_mapping_to(self.observations, device, non_blocking=non_blocking),
            actions=move(self.actions),
            executed_actions=None if self.executed_actions is None else move(self.executed_actions),
            rewards=move(self.rewards),
            next_observations=_mapping_to(self.next_observations, device, non_blocking=non_blocking),
            discounts=move(self.discounts),
            terminated=move(self.terminated),
            truncated=move(self.truncated),
            episode_starts=None if self.episode_starts is None else move(self.episode_starts),
            next_episode_starts=(
                None if self.next_episode_starts is None else move(self.next_episode_starts)
            ),
            action_masks=None if self.action_masks is None else move(self.action_masks),
            next_action_masks=None if self.next_action_masks is None else move(self.next_action_masks),
            behavior_log_probs=None if self.behavior_log_probs is None else move(self.behavior_log_probs),
            reward_components=_mapping_to(self.reward_components, device, non_blocking=non_blocking),
            extras=_mapping_to(self.extras, device, non_blocking=non_blocking),
            _validate_values=False,
        )

    def index(self, indices: torch.Tensor) -> ReplayBatch:
        if indices.dtype != torch.long or indices.device != self.device or indices.ndim != 1:
            raise ValueError("Replay indices must be a one-dimensional long tensor on the batch device.")
        return ReplayBatch(
            observations=_mapping_index(self.observations, indices),
            actions=self.actions[indices],
            executed_actions=None if self.executed_actions is None else self.executed_actions[indices],
            rewards=self.rewards[indices],
            next_observations=_mapping_index(self.next_observations, indices),
            discounts=self.discounts[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            episode_starts=None if self.episode_starts is None else self.episode_starts[indices],
            next_episode_starts=(
                None if self.next_episode_starts is None else self.next_episode_starts[indices]
            ),
            action_masks=None if self.action_masks is None else self.action_masks[indices],
            next_action_masks=None if self.next_action_masks is None else self.next_action_masks[indices],
            behavior_log_probs=None if self.behavior_log_probs is None else self.behavior_log_probs[indices],
            reward_components=_mapping_index(self.reward_components, indices),
            extras=_mapping_index(self.extras, indices),
            _validate_values=False,
        )

    def decision_keys(self) -> torch.Tensor:
        """Return globally unique int64 decision alignment keys.

        Positional alignment is unsafe across seeds, workers, or replay export
        order.  Trading environments can persist stable int64 decision IDs in
        ``extras``; this method fails closed when the identity is absent or not
        unique instead of silently pairing unrelated transitions. Runtime
        environment slots are deliberately excluded because batching layouts
        are not stable identities.
        """

        if "decision_id" not in self.extras:
            raise ValueError("Replay batch lacks the exact decision_id identity field.")
        keys = self.extras["decision_id"]
        if keys.shape != (self.batch_size,) or keys.dtype != torch.long:
            raise ValueError(f"Replay decision_id must be int64 shape ({self.batch_size},).")
        if torch.unique(keys).shape[0] != self.batch_size:
            raise ValueError("Replay decision identity contains duplicate keys.")
        return keys

    def aligned_to(self, reference_keys: torch.Tensor) -> ReplayBatch:
        """Reorder this batch to an exact reference decision-key sequence."""

        if (
            reference_keys.shape != (self.batch_size,)
            or reference_keys.dtype != torch.long
            or reference_keys.device != self.device
        ):
            raise ValueError(
                f"reference_keys must be int64 shape ({self.batch_size},) on {self.device}."
            )
        keys = self.decision_keys()
        reference_rows = [int(item) for item in reference_keys.detach().cpu().tolist()]
        if len(set(reference_rows)) != self.batch_size:
            raise ValueError("Reference decision keys contain duplicates.")
        key_rows = [int(item) for item in keys.detach().cpu().tolist()]
        positions = {key: index for index, key in enumerate(key_rows)}
        if set(positions) != set(reference_rows):
            raise ValueError("Replay decision-key set differs from the reference set.")
        indices = torch.tensor(
            [positions[key] for key in reference_rows],
            dtype=torch.long,
            device=self.device,
        )
        return self.index(indices)


def align_replay_batches(batches: tuple[ReplayBatch, ...]) -> tuple[ReplayBatch, ...]:
    """Align multiple transition traces by exact composite decision identity."""

    if not batches:
        raise ValueError("At least one replay batch is required for alignment.")
    reference = batches[0].decision_keys()
    return (batches[0], *(batch.aligned_to(reference.to(batch.device)) for batch in batches[1:]))


StorageKey = tuple[str, str]


class TransitionReplayBuffer:
    """Schema-locked circular replay with lazy, contiguous tensor allocation.

    The first added batch defines the schema.  Subsequent batches must match it
    exactly, preventing an offline data refresh from silently changing action or
    observation meaning midway through training.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        if pin_memory and self.device.type != "cpu":
            raise ValueError("pin_memory is valid only for CPU replay storage.")
        self.pin_memory = bool(pin_memory)
        self.size = 0
        self.cursor = 0
        self._storage: dict[StorageKey, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.size

    def _flatten(self, batch: ReplayBatch) -> dict[StorageKey, torch.Tensor]:
        values: dict[StorageKey, torch.Tensor] = {
            **{("observation", name): value for name, value in batch.observations.items()},
            **{("next_observation", name): value for name, value in batch.next_observations.items()},
            ("core", "actions"): batch.actions,
            ("core", "rewards"): batch.rewards,
            ("core", "discounts"): batch.discounts,
            ("core", "terminated"): batch.terminated,
            ("core", "truncated"): batch.truncated,
            **{("reward", name): value for name, value in batch.reward_components.items()},
            **{("extra", name): value for name, value in batch.extras.items()},
        }
        optional = {
            "executed_actions": batch.executed_actions,
            "action_masks": batch.action_masks,
            "next_action_masks": batch.next_action_masks,
            "behavior_log_probs": batch.behavior_log_probs,
            "episode_starts": batch.episode_starts,
            "next_episode_starts": batch.next_episode_starts,
        }
        values.update(
            (("optional", name), value)
            for name, value in optional.items()
            if value is not None
        )
        return values

    def _allocate(self, values: Mapping[StorageKey, torch.Tensor]) -> None:
        for key, value in values.items():
            self._storage[key] = torch.empty(
                (self.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
            )

    def add(self, transition: TransitionBatch | ReplayBatch) -> None:
        if isinstance(transition, TransitionBatch):
            batch = ReplayBatch.from_transition(transition)
        elif isinstance(transition, ReplayBatch):
            transition.validate()
            batch = transition
        else:
            raise TypeError("Replay add expects a TransitionBatch or ReplayBatch.")
        values = self._flatten(batch)
        if not self._storage:
            self._allocate(values)
        if set(values) != set(self._storage):
            missing = sorted(set(self._storage) - set(values))
            added = sorted(set(values) - set(self._storage))
            raise ValueError(f"Replay schema changed; missing={missing}, added={added}.")
        for key, value in values.items():
            target = self._storage[key]
            if value.shape[1:] != target.shape[1:] or value.dtype != target.dtype:
                raise ValueError(
                    f"Replay field {key} has schema {tuple(value.shape[1:])}/{value.dtype}; "
                    f"expected {tuple(target.shape[1:])}/{target.dtype}."
                )

        count = batch.batch_size
        if count >= self.capacity:
            values = {key: value[-self.capacity:] for key, value in values.items()}
            count = self.capacity
        first = min(count, self.capacity - self.cursor)
        second = count - first
        for key, target in self._storage.items():
            source = values[key].to(self.device, non_blocking=self.pin_memory)
            target[self.cursor:self.cursor + first].copy_(source[:first])
            if second:
                target[:second].copy_(source[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def _from_indices(
        self,
        indices: torch.Tensor,
        *,
        validate_values: bool = False,
        storage: Mapping[StorageKey, torch.Tensor] | None = None,
    ) -> ReplayBatch:
        source = self._storage if storage is None else storage
        selected = {key: value[indices] for key, value in source.items()}
        group = lambda prefix: {  # noqa: E731
            name: value for (kind, name), value in selected.items() if kind == prefix
        }
        core = group("core")
        optional = group("optional")
        return ReplayBatch(
            observations=group("observation"),
            actions=core["actions"],
            executed_actions=optional.get("executed_actions"),
            rewards=core["rewards"],
            next_observations=group("next_observation"),
            discounts=core["discounts"],
            terminated=core["terminated"],
            truncated=core["truncated"],
            episode_starts=optional.get("episode_starts"),
            next_episode_starts=optional.get("next_episode_starts"),
            action_masks=optional.get("action_masks"),
            next_action_masks=optional.get("next_action_masks"),
            behavior_log_probs=optional.get("behavior_log_probs"),
            reward_components=group("reward"),
            extras=group("extra"),
            _validate_values=validate_values,
        )

    def sample(
        self,
        batch_size: int,
        *,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not replacement and batch_size > self.size:
            raise ValueError("Cannot sample more replay rows than available without replacement.")
        if replacement:
            indices = torch.randint(self.size, (batch_size,), device=self.device, generator=generator)
        else:
            indices = torch.randperm(self.size, device=self.device, generator=generator)[:batch_size]
        return self._from_indices(indices)

    def all(self, *, chronological: bool = True) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot read an empty replay buffer.")
        if chronological and self.size == self.capacity:
            indices = torch.cat(
                (
                    torch.arange(self.cursor, self.capacity, device=self.device),
                    torch.arange(0, self.cursor, device=self.device),
                )
            )
        else:
            indices = torch.arange(self.size, device=self.device)
        return self._from_indices(indices)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "device_type": self.device.type,
            "size": self.size,
            "cursor": self.cursor,
            "storage": {key: value.detach().cpu().clone() for key, value in self._storage.items()},
        }

    def load_state_dict(self, state: Mapping) -> None:
        capacity_value = state.get("capacity")
        if (
            isinstance(capacity_value, bool)
            or not isinstance(capacity_value, int)
            or capacity_value != self.capacity
        ):
            raise ValueError("Replay checkpoint capacity differs from this buffer.")
        size_value, cursor_value = state.get("size"), state.get("cursor")
        if (
            isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or isinstance(cursor_value, bool)
            or not isinstance(cursor_value, int)
        ):
            raise ValueError("Replay checkpoint size/cursor is invalid.")
        size, cursor = size_value, cursor_value
        if not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError("Replay checkpoint size/cursor is invalid.")
        if size < self.capacity and cursor != size:
            raise ValueError("A partially filled replay checkpoint must have cursor equal to size.")
        stored = state.get("storage")
        if not isinstance(stored, Mapping):
            raise ValueError("Replay checkpoint storage is missing.")
        if bool(stored) != bool(size):
            raise ValueError("Replay checkpoint storage is inconsistent with its size.")
        restored: dict[StorageKey, torch.Tensor] = {}
        for key, value in stored.items():
            if not isinstance(key, tuple) or len(key) != 2 or not torch.is_tensor(value):
                raise ValueError("Replay checkpoint contains an invalid storage entry.")
            if value.shape[0] != self.capacity:
                raise ValueError(f"Replay checkpoint field {key} has the wrong capacity.")
            restored[key] = value.to(self.device).clone()
            if self.pin_memory:
                restored[key] = restored[key].pin_memory()
        if size:
            # Checkpoint payloads are untrusted. Pay the full value-validation
            # cost once at restore, then sampled slices can use the trusted fast
            # path just like data accepted through ``add``.
            self._from_indices(
                torch.arange(size, device=self.device),
                validate_values=True,
                storage=restored,
            )
        # Publish only after every checkpoint invariant and active value has
        # passed; a failed restore leaves the existing buffer untouched.
        self._storage = restored
        self.size, self.cursor = size, cursor
