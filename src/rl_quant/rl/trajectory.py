"""Recurrent on-policy trajectory storage with correct terminal bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from rl_quant.rl.types import TransitionBatch


@dataclass(frozen=True)
class TrajectoryBatch:
    """Time-major rollout tensors with shape ``[time, env, ...]``."""

    observations: Mapping[str, torch.Tensor]
    action_masks: torch.Tensor | None
    actions: torch.Tensor
    executed_actions: torch.Tensor
    rewards: torch.Tensor
    reward_components: Mapping[str, torch.Tensor]
    discounts: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_start: torch.Tensor
    recurrent_states: Mapping[str, torch.Tensor]

    @property
    def horizon(self) -> int:
        return self.rewards.shape[0]

    @property
    def num_envs(self) -> int:
        return self.rewards.shape[1]


@dataclass(frozen=True)
class RecurrentSequenceBatch:
    """Fixed-length padded sequences for truncated backpropagation through time.

    ``valid_mask`` includes burn-in observations.  ``loss_mask`` contains only
    learning positions, so recurrent state can warm up without those steps
    contributing to PPO/value losses.
    """

    observations: Mapping[str, torch.Tensor]
    action_masks: torch.Tensor | None
    actions: torch.Tensor
    executed_actions: torch.Tensor
    rewards: torch.Tensor
    reward_components: Mapping[str, torch.Tensor]
    discounts: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_start: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    time_indices: torch.Tensor
    initial_recurrent_state: Mapping[str, torch.Tensor]
    burn_in: int

    @property
    def num_sequences(self) -> int:
        return self.rewards.shape[0]

    @property
    def sequence_width(self) -> int:
        return self.rewards.shape[1]


class OnPolicyTrajectoryBuffer:
    """Fixed-horizon on-policy buffer for vectorized recurrent algorithms.

    Each transition stores both ``value`` and ``next_value``.  A truncated
    transition therefore bootstraps from its real continuation while advantage
    recursion still stops at the reset boundary.  A true termination has zero
    transition discount and never bootstraps.
    """

    def __init__(self, *, horizon: int, num_envs: int) -> None:
        if horizon <= 0 or num_envs <= 0:
            raise ValueError("horizon and num_envs must be positive.")
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self._transitions: list[TransitionBatch] = []
        self._values: list[torch.Tensor] = []
        self._next_values: list[torch.Tensor] = []
        self._log_probs: list[torch.Tensor] = []
        self._recurrent_states: list[dict[str, torch.Tensor]] = []
        self._advantages: torch.Tensor | None = None
        self._returns: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def full(self) -> bool:
        return len(self) == self.horizon

    def clear(self) -> None:
        self._transitions.clear()
        self._values.clear()
        self._next_values.clear()
        self._log_probs.clear()
        self._recurrent_states.clear()
        self._advantages = None
        self._returns = None

    def add(
        self,
        transition: TransitionBatch,
        *,
        value: torch.Tensor,
        next_value: torch.Tensor,
        log_prob: torch.Tensor | None = None,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self.full:
            raise RuntimeError("Trajectory buffer is full; consume or clear it before adding more steps.")
        if transition.observation.batch_size != self.num_envs:
            raise ValueError(
                f"Transition has {transition.observation.batch_size} environments; expected {self.num_envs}."
            )
        device = transition.observation.device
        for name, tensor in (("value", value), ("next_value", next_value)):
            if tensor.shape != (self.num_envs,) or tensor.device != device:
                raise ValueError(f"{name} must have shape ({self.num_envs},) on {device}.")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype.")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} must be finite.")
        chosen_log_prob = transition.action.log_prob if log_prob is None else log_prob
        if chosen_log_prob is None:
            raise ValueError("An on-policy trajectory requires action log probabilities.")
        if chosen_log_prob.shape != (self.num_envs,) or chosen_log_prob.device != device:
            raise ValueError(f"log_prob must have shape ({self.num_envs},) on {device}.")
        if not chosen_log_prob.is_floating_point() or not bool(torch.isfinite(chosen_log_prob).all().item()):
            raise ValueError("log_prob must be finite and floating point.")
        state = {} if recurrent_state is None else dict(recurrent_state)
        for name, tensor in state.items():
            if tensor.ndim == 0 or tensor.shape[0] != self.num_envs or tensor.device != device:
                raise ValueError(f"recurrent_state[{name!r}] needs leading size {self.num_envs} on {device}.")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"recurrent_state[{name!r}] must be finite.")
        if self._transitions:
            first = self._transitions[0]
            if set(first.observation.tensors) != set(transition.observation.tensors):
                raise ValueError("Observation keys changed within one trajectory.")
            for name, value_tensor in transition.observation.tensors.items():
                first_tensor = first.observation.tensors[name]
                if value_tensor.shape[1:] != first_tensor.shape[1:] or value_tensor.dtype != first_tensor.dtype:
                    raise ValueError(f"Observation schema for {name!r} changed within one trajectory.")
            if first.action.action.shape[1:] != transition.action.action.shape[1:]:
                raise ValueError("Action event shape changed within one trajectory.")
            if (first.observation.action_mask is None) != (transition.observation.action_mask is None):
                raise ValueError("action_mask presence changed within one trajectory.")
            if first.observation.action_mask is not None and transition.observation.action_mask is not None:
                if first.observation.action_mask.shape[1:] != transition.observation.action_mask.shape[1:]:
                    raise ValueError("action_mask event shape changed within one trajectory.")
            if set(first.rewards.as_dict()) != set(transition.rewards.as_dict()):
                raise ValueError("Reward component names changed within one trajectory.")
            if set(self._recurrent_states[0]) != set(state):
                raise ValueError("Recurrent-state keys changed within one trajectory.")
            for name, state_tensor in state.items():
                first_state = self._recurrent_states[0][name]
                if state_tensor.shape[1:] != first_state.shape[1:] or state_tensor.dtype != first_state.dtype:
                    raise ValueError(f"Recurrent-state schema for {name!r} changed within one trajectory.")
        self._transitions.append(transition.detach())
        self._values.append(value.detach())
        self._next_values.append(next_value.detach())
        self._log_probs.append(chosen_log_prob.detach())
        self._recurrent_states.append({name: tensor.detach() for name, tensor in state.items()})
        self._advantages = None
        self._returns = None

    def compute_gae(self, *, gae_lambda: float, normalize: bool = False, epsilon: float = 1e-8) -> None:
        if not self._transitions:
            raise RuntimeError("Cannot compute GAE for an empty trajectory.")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must lie in [0, 1]; got {gae_lambda}.")
        rewards = torch.stack([transition.reward for transition in self._transitions])
        discounts = torch.stack([transition.discount for transition in self._transitions])
        terminated = torch.stack([transition.terminated for transition in self._transitions])
        truncated = torch.stack([transition.truncated for transition in self._transitions])
        values = torch.stack(self._values)
        next_values = torch.stack(self._next_values)

        deltas = rewards + discounts * next_values - values
        advantages = torch.zeros_like(deltas)
        running = torch.zeros(self.num_envs, dtype=deltas.dtype, device=deltas.device)
        for step in range(len(self._transitions) - 1, -1, -1):
            # Truncations bootstrap in `deltas`, but cannot carry advantages across
            # the environment reset. Terminations already have zero discount.
            continues_same_episode = ~(terminated[step] | truncated[step])
            running = deltas[step] + discounts[step] * gae_lambda * continues_same_episode * running
            advantages[step] = running
        if normalize:
            scale = advantages.std(unbiased=False)
            advantages = (advantages - advantages.mean()) / scale.clamp_min(epsilon)
        self._advantages = advantages
        self._returns = advantages + values

    def as_batch(self) -> TrajectoryBatch:
        if self._advantages is None or self._returns is None:
            raise RuntimeError("Call compute_gae() before consuming a trajectory.")
        first = self._transitions[0]
        observation_keys = first.observation.tensors.keys()
        observations = {
            name: torch.stack([transition.observation.tensors[name] for transition in self._transitions])
            for name in observation_keys
        }
        masks = [transition.observation.action_mask for transition in self._transitions]
        if all(mask is None for mask in masks):
            action_masks = None
        elif any(mask is None for mask in masks):
            raise ValueError("action_mask presence changed within one trajectory.")
        else:
            action_masks = torch.stack([mask for mask in masks if mask is not None])

        done = torch.stack([transition.done for transition in self._transitions])
        provided_starts = [transition.observation.episode_start for transition in self._transitions]
        episode_start = torch.zeros_like(done)
        episode_start[0] = True if provided_starts[0] is None else provided_starts[0]
        if len(self._transitions) > 1:
            episode_start[1:] = done[:-1]
        for step, provided in enumerate(provided_starts):
            if provided is not None:
                episode_start[step] |= provided

        recurrent_states: dict[str, torch.Tensor] = {}
        if self._recurrent_states and self._recurrent_states[0]:
            recurrent_states = {
                name: torch.stack([state[name] for state in self._recurrent_states])
                for name in self._recurrent_states[0]
            }
        return TrajectoryBatch(
            observations=observations,
            action_masks=action_masks,
            actions=torch.stack([transition.action.action for transition in self._transitions]),
            executed_actions=torch.stack([transition.executed_action for transition in self._transitions]),
            rewards=torch.stack([transition.reward for transition in self._transitions]),
            reward_components={
                name: torch.stack([transition.rewards.as_dict()[name] for transition in self._transitions])
                for name in first.rewards.as_dict()
            },
            discounts=torch.stack([transition.discount for transition in self._transitions]),
            log_probs=torch.stack(self._log_probs),
            values=torch.stack(self._values),
            next_values=torch.stack(self._next_values),
            advantages=self._advantages,
            returns=self._returns,
            terminated=torch.stack([transition.terminated for transition in self._transitions]),
            truncated=torch.stack([transition.truncated for transition in self._transitions]),
            episode_start=episode_start,
            recurrent_states=recurrent_states,
        )

    def recurrent_sequences(self, *, sequence_length: int, burn_in: int = 0) -> RecurrentSequenceBatch:
        """Split each episode into padded learning chunks with optional causal burn-in."""

        if sequence_length <= 0 or burn_in < 0:
            raise ValueError("sequence_length must be positive and burn_in nonnegative.")
        batch = self.as_batch()
        done_cpu = (batch.terminated | batch.truncated).detach().cpu()
        # (env, actual source start, learning start, learning end, left padding)
        windows: list[tuple[int, int, int, int, int]] = []
        for env in range(batch.num_envs):
            episode_start = 0
            episode_ends = [step + 1 for step in range(batch.horizon) if bool(done_cpu[step, env])]
            if not episode_ends or episode_ends[-1] != batch.horizon:
                episode_ends.append(batch.horizon)
            for episode_end in episode_ends:
                for learning_start in range(episode_start, episode_end, sequence_length):
                    learning_end = min(learning_start + sequence_length, episode_end)
                    actual_start = max(episode_start, learning_start - burn_in)
                    left_padding = burn_in - (learning_start - actual_start)
                    windows.append((env, actual_start, learning_start, learning_end, left_padding))
                episode_start = episode_end
        if not windows:
            raise RuntimeError("No recurrent sequences could be built.")

        width = burn_in + sequence_length

        def _padded(source: torch.Tensor, *, fill: float | int | bool = 0) -> torch.Tensor:
            output = torch.full(
                (len(windows), width, *source.shape[2:]),
                fill,
                dtype=source.dtype,
                device=source.device,
            )
            for row, (env, actual_start, _learning_start, learning_end, left_padding) in enumerate(windows):
                count = learning_end - actual_start
                output[row, left_padding : left_padding + count] = source[actual_start:learning_end, env]
            return output

        device = batch.rewards.device
        valid_mask = torch.zeros((len(windows), width), dtype=torch.bool, device=device)
        loss_mask = torch.zeros_like(valid_mask)
        time_indices = torch.full((len(windows), width), -1, dtype=torch.long, device=device)
        for row, (_env, actual_start, learning_start, learning_end, left_padding) in enumerate(windows):
            count = learning_end - actual_start
            valid_mask[row, left_padding : left_padding + count] = True
            learn_count = learning_end - learning_start
            loss_mask[row, burn_in : burn_in + learn_count] = True
            time_indices[row, left_padding : left_padding + count] = torch.arange(
                actual_start, learning_end, device=device
            )

        initial_state: dict[str, torch.Tensor] = {}
        for name, source in batch.recurrent_states.items():
            initial_state[name] = torch.stack(
                [source[actual_start, env] for env, actual_start, *_rest in windows]
            )
        return RecurrentSequenceBatch(
            observations={name: _padded(value) for name, value in batch.observations.items()},
            action_masks=None if batch.action_masks is None else _padded(batch.action_masks),
            actions=_padded(batch.actions),
            executed_actions=_padded(batch.executed_actions),
            rewards=_padded(batch.rewards),
            reward_components={name: _padded(value) for name, value in batch.reward_components.items()},
            discounts=_padded(batch.discounts),
            old_log_probs=_padded(batch.log_probs),
            old_values=_padded(batch.values),
            advantages=_padded(batch.advantages),
            returns=_padded(batch.returns),
            terminated=_padded(batch.terminated),
            truncated=_padded(batch.truncated),
            episode_start=_padded(batch.episode_start),
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            time_indices=time_indices,
            initial_recurrent_state=initial_state,
            burn_in=burn_in,
        )
