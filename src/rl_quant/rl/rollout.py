"""Domain-neutral synchronous on-policy rollout coordination.

The coordinator owns interaction boundaries, not environment semantics.  In
particular, it keeps the recurrent state that belongs *before* each observation,
evaluates the real next observation before any reset, and lets transition
discounts distinguish true termination from time-limit truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Protocol, runtime_checkable

import torch

from rl_quant.rl.algorithm import RecurrentState
from rl_quant.rl.environment import VectorEnvironment
from rl_quant.rl.trajectory import OnPolicyTrajectoryBuffer
from rl_quant.rl.types import ActionBatch, ObservationBatch


@runtime_checkable
class OnPolicyAgent(Protocol):
    """Actor-critic surface needed by :class:`OnPolicyRolloutCoordinator`.

    ``act`` must return the recurrent state *after* consuming ``observation``.
    ``initial_recurrent_state`` returns the state to use immediately before the
    first observation of a freshly reset batch.
    """

    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: RecurrentState | None = None,
    ) -> ActionBatch: ...

    def value(
        self,
        observation: ObservationBatch,
        *,
        recurrent_state: RecurrentState | None = None,
    ) -> torch.Tensor: ...

    def initial_recurrent_state(self, observation: ObservationBatch) -> RecurrentState: ...


def _copy_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def _validate_state(
    state: Mapping[str, torch.Tensor],
    observation: ObservationBatch,
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> None:
    if reference is not None and set(state) != set(reference):
        raise ValueError(
            "The policy changed recurrent-state keys while acting: "
            f"expected {sorted(reference)}, got {sorted(state)}."
        )
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Recurrent-state names must be non-empty strings.")
        if value.ndim == 0 or value.shape[0] != observation.batch_size:
            raise ValueError(
                f"recurrent_state[{name!r}] needs leading batch size {observation.batch_size}."
            )
        if value.device != observation.device:
            raise ValueError(
                f"recurrent_state[{name!r}] is on {value.device}, expected {observation.device}."
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"recurrent_state[{name!r}] must be finite.")
        if reference is not None:
            expected = reference[name]
            if value.shape[1:] != expected.shape[1:] or value.dtype != expected.dtype:
                raise ValueError(
                    f"The schema of recurrent_state[{name!r}] changed while acting."
                )


def _with_episode_start(observation: ObservationBatch) -> ObservationBatch:
    """Canonicalize a full reset as the start of every vectorized episode."""

    return ObservationBatch(
        tensors=observation.tensors,
        action_mask=observation.action_mask,
        episode_start=torch.ones(
            observation.batch_size,
            dtype=torch.bool,
            device=observation.device,
        ),
    )


def _initial_return_dtype(observation: ObservationBatch) -> torch.dtype:
    for value in observation.tensors.values():
        if value.is_floating_point():
            return value.dtype
    return torch.get_default_dtype()


@dataclass(frozen=True)
class RolloutContinuation:
    """Everything needed to continue collection without resetting an episode."""

    observation: ObservationBatch
    recurrent_state: Mapping[str, torch.Tensor]
    running_episode_returns: torch.Tensor
    running_episode_lengths: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.observation.batch_size
        device = self.observation.device
        if self.running_episode_returns.shape != (batch_size,):
            raise ValueError(f"running_episode_returns must have shape ({batch_size},).")
        if self.running_episode_returns.device != device:
            raise ValueError("running_episode_returns must share the observation device.")
        if not self.running_episode_returns.is_floating_point() or not bool(
            torch.isfinite(self.running_episode_returns).all().item()
        ):
            raise ValueError("running_episode_returns must be finite and floating point.")
        if self.running_episode_lengths.shape != (batch_size,):
            raise ValueError(f"running_episode_lengths must have shape ({batch_size},).")
        if self.running_episode_lengths.device != device or self.running_episode_lengths.dtype != torch.long:
            raise ValueError("running_episode_lengths must be torch.long on the observation device.")
        if bool((self.running_episode_lengths < 0).any().item()):
            raise ValueError("running_episode_lengths cannot be negative.")
        _validate_state(self.recurrent_state, self.observation)


@dataclass(frozen=True)
class RolloutMetrics:
    """Low-cardinality collection diagnostics suitable for training logs."""

    environment_steps: int
    transitions: int
    reward_sum: float
    reward_mean: float
    episodes_completed: int
    episodes_terminated: int
    episodes_truncated: int
    mean_episode_return: float | None
    mean_episode_length: float | None


@dataclass(frozen=True)
class RolloutResult:
    """A GAE-ready trajectory, its continuation, and collection diagnostics."""

    buffer: OnPolicyTrajectoryBuffer
    continuation: RolloutContinuation
    metrics: RolloutMetrics


class OnPolicyRolloutCoordinator:
    """Collect fixed-horizon rollouts from a synchronous vector environment.

    The base :class:`VectorEnvironment` protocol only exposes whole-batch
    ``reset``.  Consequently, this coordinator supports episode completion when
    every environment finishes on the same step and fails clearly if only part
    of the batch finishes.  An asynchronous adapter must define subset-reset
    semantics before it can be used here safely.
    """

    def __init__(
        self,
        environment: VectorEnvironment,
        agent: OnPolicyAgent,
        *,
        horizon: int,
        gae_lambda: float,
        normalize_advantages: bool = False,
        advantage_epsilon: float = 1e-8,
        deterministic: bool = False,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be finite and lie in [0, 1].")
        if not math.isfinite(advantage_epsilon) or advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be finite and positive.")
        if environment.batch_size <= 0:
            raise ValueError("environment.batch_size must be positive.")
        self.environment = environment
        self.agent = agent
        self.horizon = int(horizon)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = bool(normalize_advantages)
        self.advantage_epsilon = float(advantage_epsilon)
        self.deterministic = bool(deterministic)
        self._num_envs = int(environment.batch_size)

    def _validate_observation(self, observation: ObservationBatch) -> None:
        if observation.batch_size != self._num_envs:
            raise ValueError(
                f"Environment returned batch size {observation.batch_size}; expected {self._num_envs}."
            )

    @torch.no_grad()
    def initial_continuation(self) -> RolloutContinuation:
        """Reset the full environment batch and construct fresh recurrent state."""

        observation, _reset_info = self.environment.reset()
        self._validate_observation(observation)
        observation = _with_episode_start(observation).detach()
        recurrent_state = self.agent.initial_recurrent_state(observation)
        _validate_state(recurrent_state, observation)
        return RolloutContinuation(
            observation=observation,
            recurrent_state=_copy_state(recurrent_state),
            running_episode_returns=torch.zeros(
                self._num_envs,
                dtype=_initial_return_dtype(observation),
                device=observation.device,
            ),
            running_episode_lengths=torch.zeros(
                self._num_envs,
                dtype=torch.long,
                device=observation.device,
            ),
        )

    @staticmethod
    def _policy_value(
        agent: OnPolicyAgent,
        observation: ObservationBatch,
        recurrent_state: Mapping[str, torch.Tensor],
        action: ActionBatch | None = None,
    ) -> torch.Tensor:
        # PPO already evaluated V(s) during action selection.  Other agents may
        # omit it, in which case the explicit critic protocol is the fallback.
        if action is not None and "value" in action.extras:
            value = action.extras["value"]
        else:
            value = agent.value(observation, recurrent_state=recurrent_state)
        if value.shape != (observation.batch_size,):
            raise ValueError(
                f"Critic value must have shape ({observation.batch_size},); got {tuple(value.shape)}."
            )
        if value.device != observation.device or not value.is_floating_point():
            raise ValueError("Critic value must be floating point on the observation device.")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("Critic value must be finite.")
        return value

    def _validate_continuation(self, continuation: RolloutContinuation) -> None:
        self._validate_observation(continuation.observation)
        if int(self.environment.batch_size) != self._num_envs:
            raise RuntimeError("environment.batch_size changed after coordinator construction.")

    @torch.no_grad()
    def collect(self, continuation: RolloutContinuation | None = None) -> RolloutResult:
        """Collect one fixed horizon and compute generalized advantage estimates.

        Passing ``None`` starts a fresh environment batch.  Passing the previous
        result's continuation carries observation, recurrent state, and partial
        episode accounting across rollout boundaries without an implicit reset.
        """

        current = self.initial_continuation() if continuation is None else continuation
        self._validate_continuation(current)
        observation = current.observation.detach()
        recurrent_state = _copy_state(current.recurrent_state)
        running_returns = current.running_episode_returns.detach().clone()
        running_lengths = current.running_episode_lengths.detach().clone()

        buffer = OnPolicyTrajectoryBuffer(horizon=self.horizon, num_envs=self._num_envs)
        reward_steps: list[torch.Tensor] = []
        completed_returns: list[torch.Tensor] = []
        completed_lengths: list[torch.Tensor] = []
        terminated_counts: list[torch.Tensor] = []
        truncated_counts: list[torch.Tensor] = []

        for _step in range(self.horizon):
            # Snapshot before act(): mutable/custom agents must not be able to
            # turn the buffer's pre-observation state into a post-state in place.
            input_state = _copy_state(recurrent_state)
            stored_input_state = _copy_state(input_state)
            action = self.agent.act(
                observation,
                deterministic=self.deterministic,
                recurrent_state=input_state,
            )
            if action.batch_size != self._num_envs or action.device != observation.device:
                raise ValueError("Policy action must match the environment batch and observation device.")
            if action.log_prob is None:
                raise ValueError("On-policy collection requires action.log_prob.")
            _validate_state(action.recurrent_state, observation, reference=input_state)
            value = self._policy_value(self.agent, observation, input_state, action)

            transition = self.environment.step(action)
            if transition.observation.batch_size != self._num_envs:
                raise ValueError("Environment transition batch size changed during collection.")
            if transition.observation.device != observation.device:
                raise ValueError("Environment transition device changed during collection.")
            if not torch.equal(transition.action.action, action.action):
                raise ValueError(
                    "transition.action must preserve the requested policy action; "
                    "put constrained or projected actions in transition.executed_action."
                )
            # The local objects are the exact observation/action seen by the
            # policy.  Keeping them authoritative also preserves the canonical
            # episode-start flag even if an adapter rebuilds its transition.
            transition = replace(
                transition,
                observation=observation,
                action=action,
            )
            next_starts = transition.next_observation.episode_start
            if next_starts is not None and bool(next_starts.any().item()):
                raise RuntimeError(
                    "Environment step returned episode_start=True in next_observation. "
                    "The coordinator requires the real pre-reset next observation so truncations can bootstrap."
                )

            done = transition.done
            any_done = bool(done.any().item())
            all_done = bool(done.all().item())
            if any_done and not all_done:
                raise RuntimeError(
                    "Only part of the vector environment completed an episode, but VectorEnvironment "
                    "exposes only whole-batch reset(). Use synchronous episode boundaries or an adapter "
                    "with explicit subset-reset semantics."
                )

            post_state = _copy_state(action.recurrent_state)
            next_value = self._policy_value(
                self.agent,
                transition.next_observation,
                post_state,
            )
            # The transition contract already requires zero terminal discount.
            # Zeroing the estimate too makes the no-bootstrap invariant explicit
            # in stored diagnostics while truncations retain their critic value.
            next_value = torch.where(
                transition.terminated,
                torch.zeros_like(next_value),
                next_value,
            )
            buffer.add(
                transition,
                value=value,
                next_value=next_value,
                log_prob=action.log_prob,
                recurrent_state=stored_input_state,
            )

            reward = transition.reward
            if running_returns.device != reward.device:
                raise ValueError("Reward device changed within an episode.")
            if running_returns.dtype != reward.dtype:
                if bool((running_lengths != 0).any().item()):
                    raise ValueError("Reward dtype changed within an episode.")
                running_returns = running_returns.to(dtype=reward.dtype)
            running_returns = running_returns + reward
            running_lengths = running_lengths + 1
            reward_steps.append(reward.detach())
            terminated_counts.append(transition.terminated.sum().detach())
            truncated_counts.append(transition.truncated.sum().detach())

            if all_done:
                completed_returns.append(running_returns.detach().clone())
                completed_lengths.append(running_lengths.detach().clone())
                # Value estimation above deliberately happens before reset.
                reset_observation, _reset_info = self.environment.reset()
                self._validate_observation(reset_observation)
                observation = _with_episode_start(reset_observation).detach()
                reset_state = self.agent.initial_recurrent_state(observation)
                _validate_state(reset_state, observation)
                recurrent_state = _copy_state(reset_state)
                running_returns = torch.zeros_like(running_returns)
                running_lengths = torch.zeros_like(running_lengths)
            else:
                observation = transition.next_observation.detach()
                recurrent_state = post_state

        buffer.compute_gae(
            gae_lambda=self.gae_lambda,
            normalize=self.normalize_advantages,
            epsilon=self.advantage_epsilon,
        )
        rewards = torch.stack(reward_steps)
        reward_sum = float(rewards.sum().item())
        completed = sum(value.numel() for value in completed_returns)
        if completed_returns:
            episode_returns = torch.cat(completed_returns)
            episode_lengths = torch.cat(completed_lengths)
            mean_episode_return: float | None = float(episode_returns.mean().item())
            mean_episode_length: float | None = float(episode_lengths.float().mean().item())
        else:
            mean_episode_return = None
            mean_episode_length = None

        result_continuation = RolloutContinuation(
            observation=observation.detach(),
            recurrent_state=_copy_state(recurrent_state),
            running_episode_returns=running_returns.detach().clone(),
            running_episode_lengths=running_lengths.detach().clone(),
        )
        metrics = RolloutMetrics(
            environment_steps=self.horizon,
            transitions=self.horizon * self._num_envs,
            reward_sum=reward_sum,
            reward_mean=reward_sum / (self.horizon * self._num_envs),
            episodes_completed=completed,
            episodes_terminated=sum(int(value.item()) for value in terminated_counts),
            episodes_truncated=sum(int(value.item()) for value in truncated_counts),
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
        )
        return RolloutResult(buffer=buffer, continuation=result_continuation, metrics=metrics)
