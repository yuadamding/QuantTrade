"""Offline replay collection and optimization coordination.

Historical markets are only *partially* offline: future market returns are
logged, while inventory, costs, constraints, and portfolio state depend on the
behavior action.  This module collects those action-dependent transitions
through the same environment ledger used for evaluation, then drives any
replay-based :class:`~rl_quant.rl.algorithm.Algorithm` without embedding market
semantics in the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping

import torch

from rl_quant.rl.algorithm import Actor, Algorithm, MetricValue
from rl_quant.rl.environment import VectorEnvironment
from rl_quant.rl.replay import TransitionReplayBuffer
from rl_quant.rl.types import ObservationBatch


def _copy_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def _validate_state(state: Mapping[str, torch.Tensor], observation: ObservationBatch) -> None:
    for name, value in state.items():
        if not name or value.ndim == 0 or value.shape[0] != observation.batch_size:
            raise ValueError(f"recurrent_state[{name!r}] must have the observation batch dimension.")
        if value.device != observation.device:
            raise ValueError(f"recurrent_state[{name!r}] must be on {observation.device}.")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"recurrent_state[{name!r}] must be finite.")


def _initial_state(actor: Actor, observation: ObservationBatch) -> dict[str, torch.Tensor]:
    initializer = getattr(actor, "initial_recurrent_state", None)
    state = {} if initializer is None else initializer(observation)
    if not isinstance(state, Mapping):
        raise TypeError("initial_recurrent_state must return a tensor mapping.")
    _validate_state(state, observation)
    return _copy_state(state)


@dataclass(frozen=True)
class ReplayCollectionContinuation:
    """State required to continue replay collection without an implicit reset."""

    observation: ObservationBatch
    recurrent_state: Mapping[str, torch.Tensor]
    running_episode_returns: torch.Tensor
    running_episode_lengths: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.observation.batch_size
        if (
            self.running_episode_returns.shape != (batch_size,)
            or self.running_episode_returns.device != self.observation.device
            or not self.running_episode_returns.is_floating_point()
            or not bool(torch.isfinite(self.running_episode_returns).all().item())
        ):
            raise ValueError("running_episode_returns must be a finite floating batch vector.")
        if (
            self.running_episode_lengths.shape != (batch_size,)
            or self.running_episode_lengths.device != self.observation.device
            or self.running_episode_lengths.dtype != torch.long
            or bool((self.running_episode_lengths < 0).any().item())
        ):
            raise ValueError("running_episode_lengths must be a nonnegative long batch vector.")
        _validate_state(self.recurrent_state, self.observation)


@dataclass(frozen=True)
class ReplayCollectionMetrics:
    environment_steps: int
    transitions: int
    episodes_completed: int
    episodes_terminated: int
    episodes_truncated: int
    reward_sum: float
    reward_mean: float
    mean_completed_episode_return: float | None
    requested_execution_l1_mean: float


@dataclass(frozen=True)
class ReplayCollectionResult:
    continuation: ReplayCollectionContinuation
    metrics: ReplayCollectionMetrics


class ReplayRolloutCollector:
    """Collect behavior-policy transitions into a schema-locked replay buffer.

    The collector preserves requested and executed actions separately and uses
    whole-batch reset semantics, matching :class:`VectorEnvironment`.  A partial
    vector completion fails closed because the base environment contract has no
    subset-reset operation.
    """

    def __init__(self, environment: VectorEnvironment, actor: Actor, *, deterministic: bool = False) -> None:
        if environment.batch_size <= 0:
            raise ValueError("environment.batch_size must be positive.")
        self.environment = environment
        self.actor = actor
        self.deterministic = bool(deterministic)
        self._batch_size = int(environment.batch_size)

    @torch.no_grad()
    def initial_continuation(self) -> ReplayCollectionContinuation:
        observation, _info = self.environment.reset()
        if observation.batch_size != self._batch_size:
            raise ValueError("Environment reset returned a different batch size.")
        observation = observation.detach()
        state = _initial_state(self.actor, observation)
        return ReplayCollectionContinuation(
            observation=observation,
            recurrent_state=state,
            running_episode_returns=torch.zeros(
                self._batch_size,
                dtype=next(
                    (value.dtype for value in observation.tensors.values() if value.is_floating_point()),
                    torch.get_default_dtype(),
                ),
                device=observation.device,
            ),
            running_episode_lengths=torch.zeros(
                self._batch_size, dtype=torch.long, device=observation.device
            ),
        )

    @torch.no_grad()
    def collect(
        self,
        replay: TransitionReplayBuffer,
        *,
        environment_steps: int,
        continuation: ReplayCollectionContinuation | None = None,
    ) -> ReplayCollectionResult:
        if not isinstance(replay, TransitionReplayBuffer):
            raise TypeError("replay must be a TransitionReplayBuffer.")
        if isinstance(environment_steps, bool) or not isinstance(environment_steps, int) or environment_steps <= 0:
            raise ValueError("environment_steps must be a positive integer.")
        current = self.initial_continuation() if continuation is None else continuation
        if current.observation.batch_size != self._batch_size:
            raise ValueError("Replay continuation has the wrong batch size.")
        observation = current.observation.detach()
        recurrent_state = _copy_state(current.recurrent_state)
        running_returns = current.running_episode_returns.detach().clone()
        running_lengths = current.running_episode_lengths.detach().clone()

        reward_sum = 0.0
        projection_l1_sum = 0.0
        completed_returns: list[torch.Tensor] = []
        terminated_count = 0
        truncated_count = 0

        for _step in range(environment_steps):
            input_state = _copy_state(recurrent_state)
            action = self.actor.act(
                observation,
                deterministic=self.deterministic,
                recurrent_state=input_state or None,
            )
            if action.batch_size != self._batch_size or action.device != observation.device:
                raise ValueError("Behavior action must match the environment batch and device.")
            _validate_state(action.recurrent_state, observation)
            if recurrent_state and set(action.recurrent_state) != set(recurrent_state):
                raise ValueError("Behavior actor changed its recurrent-state schema.")

            transition = self.environment.step(action)
            if transition.observation.batch_size != self._batch_size:
                raise ValueError("Environment transition changed batch size.")
            if not torch.equal(transition.action.action, action.action):
                raise ValueError(
                    "Environment must preserve the behavior request in transition.action and put "
                    "the authoritative constrained action in transition.executed_action."
                )
            transition = replace(transition, observation=observation, action=action)
            replay.add(transition)

            reward = transition.reward
            if reward.dtype != running_returns.dtype:
                if bool((running_lengths != 0).any().item()):
                    raise ValueError("Environment reward dtype changed within an episode.")
                running_returns = running_returns.to(dtype=reward.dtype)
            running_returns = running_returns + reward
            running_lengths = running_lengths + 1
            reward_sum += float(reward.sum().item())
            projection_l1_sum += float(
                (transition.executed_action - action.action)
                .abs()
                .reshape(self._batch_size, -1)
                .sum(dim=-1)
                .sum()
                .item()
            )
            terminated_count += int(transition.terminated.sum().item())
            truncated_count += int(transition.truncated.sum().item())

            done = transition.done
            any_done, all_done = bool(done.any().item()), bool(done.all().item())
            if any_done and not all_done:
                raise RuntimeError(
                    "Only part of the vector environment completed; the base contract has no subset reset."
                )
            if all_done:
                completed_returns.append(running_returns.detach().clone())
                observation, _info = self.environment.reset()
                if observation.batch_size != self._batch_size:
                    raise ValueError("Environment reset changed batch size.")
                observation = observation.detach()
                recurrent_state = _initial_state(self.actor, observation)
                running_returns = torch.zeros_like(running_returns)
                running_lengths = torch.zeros_like(running_lengths)
            else:
                observation = transition.next_observation.detach()
                recurrent_state = _copy_state(action.recurrent_state)

        transitions = environment_steps * self._batch_size
        completed = sum(values.numel() for values in completed_returns)
        mean_completed = (
            None
            if not completed_returns
            else float(torch.cat(completed_returns).mean().item())
        )
        return ReplayCollectionResult(
            continuation=ReplayCollectionContinuation(
                observation=observation,
                recurrent_state=recurrent_state,
                running_episode_returns=running_returns,
                running_episode_lengths=running_lengths,
            ),
            metrics=ReplayCollectionMetrics(
                environment_steps=environment_steps,
                transitions=transitions,
                episodes_completed=completed,
                episodes_terminated=terminated_count,
                episodes_truncated=truncated_count,
                reward_sum=reward_sum,
                reward_mean=reward_sum / transitions,
                mean_completed_episode_return=mean_completed,
                requested_execution_l1_mean=projection_l1_sum / transitions,
            ),
        )


@dataclass(frozen=True)
class OfflineTrainingConfig:
    updates: int
    batch_size: int
    replacement: bool = True
    seed: int = 0
    update_device: torch.device | str | None = None

    def __post_init__(self) -> None:
        for name, value in (("updates", self.updates), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")


@dataclass(frozen=True)
class OfflineTrainingSummary:
    updates: int
    mean_metrics: Mapping[str, float]
    last_metrics: Mapping[str, float]


def _scalar_metric(name: str, value: MetricValue) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Training metric {name!r} must be scalar.")
        result = float(value.detach().item())
    else:
        result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Training metric {name!r} is non-finite.")
    return result


class OfflineTrainer:
    """Small deterministic update loop with checkpointable replay-sampling state."""

    def __init__(self, algorithm: Algorithm, replay: TransitionReplayBuffer) -> None:
        self.algorithm = algorithm
        self.replay = replay
        self._generator: torch.Generator | None = None
        self._seed: int | None = None
        self._updates_completed = 0

    def _sampling_generator(self, seed: int) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator(device=self.replay.device)
            self._generator.manual_seed(seed)
            self._seed = seed
        elif self._seed != seed:
            raise ValueError(
                f"OfflineTrainer sampling already uses seed {self._seed}; create a new trainer "
                f"to start a distinct seed {seed}."
            )
        return self._generator

    def state_dict(self) -> dict[str, object]:
        """Return sampling continuation state; checkpoint algorithm/replay separately."""

        return {
            "seed": self._seed,
            "updates_completed": self._updates_completed,
            "replay_device": str(self.replay.device),
            "generator_state": (
                None if self._generator is None else self._generator.get_state().detach().clone()
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"seed", "updates_completed", "replay_device", "generator_state"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"OfflineTrainer checkpoint is missing fields: {sorted(missing)}.")
        if state["replay_device"] != str(self.replay.device):
            raise ValueError("OfflineTrainer checkpoint replay device differs from the active replay.")
        updates_value = state["updates_completed"]
        if isinstance(updates_value, bool) or not isinstance(updates_value, int) or updates_value < 0:
            raise ValueError("OfflineTrainer updates_completed must be a nonnegative integer.")
        updates_completed = updates_value
        seed_value = state["seed"]
        generator_state = state["generator_state"]
        if seed_value is None:
            if generator_state is not None or updates_completed:
                raise ValueError("An uninitialized OfflineTrainer checkpoint has inconsistent state.")
            self._generator = None
            self._seed = None
        else:
            if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
                raise ValueError("OfflineTrainer checkpoint seed is invalid.")
            seed = seed_value
            if (
                not torch.is_tensor(generator_state)
                or generator_state.dtype != torch.uint8
                or generator_state.device.type != "cpu"
                or generator_state.ndim != 1
            ):
                raise ValueError("OfflineTrainer checkpoint seed/generator_state is invalid.")
            self._generator = torch.Generator(device=self.replay.device)
            self._generator.set_state(generator_state.detach().clone())
            self._seed = seed
        self._updates_completed = updates_completed

    def fit(
        self,
        config: OfflineTrainingConfig,
        *,
        on_update: Callable[[int, Mapping[str, float]], None] | None = None,
    ) -> OfflineTrainingSummary:
        if len(self.replay) == 0:
            raise RuntimeError("Cannot train from an empty replay buffer.")
        if not config.replacement and config.batch_size > len(self.replay):
            raise ValueError("batch_size exceeds replay size without replacement.")
        generator = self._sampling_generator(config.seed)
        sums: dict[str, float] = {}
        last: dict[str, float] = {}
        metric_schema: set[str] | None = None
        self.algorithm.train(True)
        for update_index in range(config.updates):
            batch = self.replay.sample(
                config.batch_size,
                replacement=config.replacement,
                generator=generator,
            )
            if config.update_device is not None:
                batch = batch.to(config.update_device, non_blocking=self.replay.pin_memory)
            raw_metrics = self.algorithm.update(batch)
            metrics = {name: _scalar_metric(name, value) for name, value in raw_metrics.items()}
            if metric_schema is None:
                metric_schema = set(metrics)
            elif set(metrics) != metric_schema:
                raise ValueError("Algorithm training metric schema changed between offline updates.")
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value
            last = metrics
            self._updates_completed += 1
            if on_update is not None:
                on_update(self._updates_completed, metrics)
        return OfflineTrainingSummary(
            updates=config.updates,
            mean_metrics={name: value / config.updates for name, value in sums.items()},
            last_metrics=last,
        )
