"""Minimal PPO optimization loop over the adaptive three-book environment."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import random
from typing import Any, Mapping

import numpy as np
import torch

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvStateV1,
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.rl.massive_adaptive_rl_observation_v1 import (
    MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256,
    MassiveAdaptiveRLObservationV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV1,
)

MASSIVE_ADAPTIVE_PPO_V1_SCHEMA = "rl-quant.massive-adaptive-ppo-v1"
MASSIVE_ADAPTIVE_RL_CHECKPOINT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-checkpoint-v1"
)
MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256 = semantic_sha256(
    {
        "bidirectional": "nine-tanh-normal-controls",
        "turnover": "one-beta-control",
        "hard_constraints": "compiler-owned",
    }
)
MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256 = semantic_sha256(
    {
        "optimization": "10000-times-strategy-minus-neutral-log-wealth",
        "reported": "unpenalized-economic-log-wealth",
        "terminal": "liquidation-adjusted",
        "extra-turnover-penalty": False,
    }
)


class MassiveAdaptivePPOV1Error(ValueError):
    """PPO rollout, optimization, or exact-resume state differs."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePPOConfigV1:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    value_clip_range: float = 0.20
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    entropy_coefficient: float = 1.0e-3
    value_coefficient: float = 0.5
    maximum_gradient_norm: float = 0.5
    epochs_per_rollout: int = 4
    rollout_length: int = 126
    minibatch_size: int = 126
    seed: int = 17
    schema: str = MASSIVE_ADAPTIVE_PPO_V1_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PPO_V1_SCHEMA
            or self.gamma != 1.0
            or not 0.0 <= self.gae_lambda <= 1.0
            or min(
                self.clip_range,
                self.value_clip_range,
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.maximum_gradient_norm,
            )
            <= 0.0
            or self.entropy_coefficient < 0.0
            or self.value_coefficient < 0.0
            or min(
                self.epochs_per_rollout,
                self.rollout_length,
                self.minibatch_size,
            )
            <= 0
            or isinstance(self.seed, bool)
            or self.seed < 0
            or any(
                not math.isfinite(value)
                for value in (
                    self.gamma,
                    self.gae_lambda,
                    self.clip_range,
                    self.value_clip_range,
                    self.actor_learning_rate,
                    self.critic_learning_rate,
                    self.entropy_coefficient,
                    self.value_coefficient,
                    self.maximum_gradient_norm,
                )
            )
        ):
            raise MassiveAdaptivePPOV1Error("adaptive PPO configuration differs")
        assert_no_adaptive_hold_semantics(self)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePPORolloutV1:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminated: torch.Tensor
    observation_receipts: tuple[str, ...]
    transition_receipts: tuple[str, ...]
    final_environment_state_receipt_sha256: str

    def validate(self) -> None:
        length = self.rewards.shape[0]
        if (
            self.observations.ndim != 2
            or self.actions.shape != (length, 10)
            or self.old_log_probabilities.shape != (length,)
            or self.old_values.shape != (length,)
            or self.advantages.shape != (length,)
            or self.returns.shape != (length,)
            or self.terminated.shape != (length,)
            or self.terminated.dtype != torch.bool
            or len(self.observation_receipts) != length
            or len(self.transition_receipts) != length
            or any(
                not bool(torch.isfinite(value).all().item())
                for value in (
                    self.observations,
                    self.actions,
                    self.old_log_probabilities,
                    self.old_values,
                    self.rewards,
                    self.advantages,
                    self.returns,
                )
            )
        ):
            raise MassiveAdaptivePPOV1Error("adaptive PPO rollout differs")


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_identity(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), _tensor_receipt(value))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            str(array.dtype),
            tuple(array.shape),
            hashlib.sha256(array.tobytes()).hexdigest(),
        )
    if isinstance(value, Mapping):
        return tuple(
            (type(key).__name__, str(key), _state_identity(item))
            for key, item in sorted(
                value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0]))
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_state_identity(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise MassiveAdaptivePPOV1Error(
        f"unsupported adaptive RL checkpoint value {type(value).__name__}"
    )


def _state_receipt(value: object) -> str:
    return semantic_sha256(_state_identity(value))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCheckpointV1:
    update_index: int
    model_state: dict[str, torch.Tensor]
    actor_optimizer_state: dict[str, Any]
    critic_optimizer_state: dict[str, Any]
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    python_rng_state: tuple[Any, ...]
    numpy_rng_state: tuple[Any, ...]
    minibatch_rng_state: torch.Tensor
    environment_state: MassiveAdaptiveProfitabilityEnvStateV1
    loss_trace: tuple[tuple[float, ...], ...]
    model_state_receipt_sha256: str
    actor_optimizer_state_receipt_sha256: str
    critic_optimizer_state_receipt_sha256: str
    rng_state_receipt_sha256: str
    environment_state_receipt_sha256: str
    loss_trace_receipt_sha256: str
    training_source_inventory_sha256: str
    training_forecast_authority_receipt_sha256: str | None
    ppo_config_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    semantic_receipt_sha256: str
    exact_resume_authorized: bool
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_CHECKPOINT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "update_index": self.update_index,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "actor_optimizer_state_receipt_sha256": self.actor_optimizer_state_receipt_sha256,
            "critic_optimizer_state_receipt_sha256": self.critic_optimizer_state_receipt_sha256,
            "rng_state_receipt_sha256": self.rng_state_receipt_sha256,
            "environment_state_receipt_sha256": self.environment_state_receipt_sha256,
            "loss_trace_receipt_sha256": self.loss_trace_receipt_sha256,
            "training_source_inventory_sha256": self.training_source_inventory_sha256,
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "ppo_config_receipt_sha256": self.ppo_config_receipt_sha256,
            "observation_specification_sha256": self.observation_specification_sha256,
            "action_specification_sha256": self.action_specification_sha256,
            "reward_specification_sha256": self.reward_specification_sha256,
            "exact_resume_authorized": self.exact_resume_authorized,
            "development_rl_training_authorized": (
                self.development_rl_training_authorized
            ),
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        self.environment_state.validate()
        rng_receipt = _state_receipt(
            (
                self.torch_rng_state,
                self.cuda_rng_states,
                self.python_rng_state,
                self.numpy_rng_state,
                self.minibatch_rng_state,
            )
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_CHECKPOINT_V1_SCHEMA
            or self.update_index < 0
            or self.model_state_receipt_sha256 != _state_receipt(self.model_state)
            or self.actor_optimizer_state_receipt_sha256
            != _state_receipt(self.actor_optimizer_state)
            or self.critic_optimizer_state_receipt_sha256
            != _state_receipt(self.critic_optimizer_state)
            or self.rng_state_receipt_sha256 != rng_receipt
            or self.environment_state_receipt_sha256
            != self.environment_state.semantic_receipt_sha256
            or self.loss_trace_receipt_sha256 != semantic_sha256(self.loss_trace)
            or self.observation_specification_sha256
            != MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256
            or self.action_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
            or self.reward_specification_sha256
            != MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
            or not self.exact_resume_authorized
            or not isinstance(self.development_rl_training_authorized, bool)
            or self.development_rl_training_authorized
            != (self.training_forecast_authority_receipt_sha256 is not None)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePPOV1Error("adaptive RL checkpoint differs")
        digests = (
            self.model_state_receipt_sha256,
            self.actor_optimizer_state_receipt_sha256,
            self.critic_optimizer_state_receipt_sha256,
            self.rng_state_receipt_sha256,
            self.environment_state_receipt_sha256,
            self.loss_trace_receipt_sha256,
            self.training_source_inventory_sha256,
            self.ppo_config_receipt_sha256,
            self.observation_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        )
        if self.training_forecast_authority_receipt_sha256 is not None:
            digests = (*digests, self.training_forecast_authority_receipt_sha256)
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise MassiveAdaptivePPOV1Error(
                "adaptive RL checkpoint digest differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


class MassiveAdaptivePPOTrainerV1:
    """Collect economic rollouts and optimize the small bounded controller."""

    def __init__(
        self,
        *,
        environment: MassiveAdaptiveProfitabilityEnvV1,
        model: MassiveAdaptivePPOActorCriticV1,
        config: MassiveAdaptivePPOConfigV1 | None = None,
        device: torch.device | str = "cpu",
        training_forecast_authority: (
            MassiveAdaptiveRLTrainingForecastAuthorityV1 | None
        ) = None,
    ) -> None:
        self.environment = environment
        self.model = model.to(device)
        self.config = config or MassiveAdaptivePPOConfigV1()
        self.config.validate()
        self.device = torch.device(device)
        if training_forecast_authority is not None:
            training_forecast_authority.validate()
            source_forecasts = {
                block.source_forecast_archive_receipt_sha256
                for block in training_forecast_authority.blocks
            }
            if (
                not training_forecast_authority.reinforcement_learning_authorized
                or environment.forecast_archive.semantic_receipt_sha256
                not in source_forecasts
            ):
                raise MassiveAdaptivePPOV1Error(
                    "adaptive PPO environment is outside its RL forecast authority"
                )
        self.training_forecast_authority = training_forecast_authority
        actor_parameters = [
            *self.model.actor.parameters(),
            *self.model.actor_mean.parameters(),
            self.model.actor_log_std,
            *self.model.turnover_alpha.parameters(),
            *self.model.turnover_beta.parameters(),
        ]
        critic_parameters = [
            *self.model.critic.parameters(),
            *self.model.value_head.parameters(),
        ]
        self.actor_optimizer = torch.optim.Adam(
            actor_parameters, lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=self.config.critic_learning_rate
        )
        self.minibatch_rng = torch.Generator(device="cpu")
        self.minibatch_rng.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.update_index = 0
        self.loss_trace: list[tuple[float, ...]] = []
        self._observation: MassiveAdaptiveRLObservationV1 | None = None

    def _observation_tensor(
        self, observation: MassiveAdaptiveRLObservationV1
    ) -> torch.Tensor:
        return torch.tensor(
            observation.values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _ensure_observation(self) -> MassiveAdaptiveRLObservationV1:
        if self._observation is None:
            self._observation, _ = self.environment.reset()
        return self._observation

    def collect_rollout(self, *, steps: int | None = None) -> MassiveAdaptivePPORolloutV1:
        count = self.config.rollout_length if steps is None else steps
        if count <= 0:
            raise MassiveAdaptivePPOV1Error("rollout length must be positive")
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []
        rewards: list[float] = []
        terminated_rows: list[bool] = []
        observation_receipts: list[str] = []
        transition_receipts: list[str] = []
        for _ in range(count):
            observation = self._ensure_observation()
            tensor = self._observation_tensor(observation)
            with torch.no_grad():
                output = self.model({"adaptive_state": tensor})
                sampled = output.distribution.sample()
                log_probability = output.distribution.log_prob(sampled)
            action_values = sampled[0].detach().cpu().tolist()
            action = build_massive_adaptive_rl_action_v1(
                bucket_controls=tuple(float(value) for value in action_values[:7]),
                uncertainty_control=float(action_values[7]),
                risk_control=float(action_values[8]),
                turnover_control=float(action_values[9]),
            )
            next_observation, reward, terminated, truncated, info = (
                self.environment.step(action)
            )
            if truncated:
                raise MassiveAdaptivePPOV1Error(
                    "environment cannot truncate an economic transition"
                )
            if terminated:
                next_value = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                assert next_observation is not None
                with torch.no_grad():
                    next_value = self.model(
                        {"adaptive_state": self._observation_tensor(next_observation)}
                    ).value[0]
            transition = info["transition"]
            observations.append(tensor[0])
            actions.append(sampled[0].detach())
            log_probabilities.append(log_probability[0].detach())
            values.append(output.value[0].detach())
            next_values.append(next_value.detach())
            rewards.append(float(reward))
            terminated_rows.append(terminated)
            observation_receipts.append(observation.semantic_receipt_sha256)
            transition_receipts.append(transition.semantic_receipt_sha256)
            self._observation = next_observation
            if terminated and len(observations) < count:
                self._observation, _ = self.environment.reset()
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        value_tensor = torch.stack(values)
        next_value_tensor = torch.stack(next_values)
        terminated_tensor = torch.tensor(
            terminated_rows, dtype=torch.bool, device=self.device
        )
        advantages = torch.zeros_like(reward_tensor)
        running = torch.zeros((), dtype=torch.float32, device=self.device)
        for index in range(count - 1, -1, -1):
            continuation = 0.0 if terminated_rows[index] else 1.0
            delta = (
                reward_tensor[index]
                + self.config.gamma * next_value_tensor[index] * continuation
                - value_tensor[index]
            )
            running = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * continuation
                * running
            )
            advantages[index] = running
        result = MassiveAdaptivePPORolloutV1(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            old_log_probabilities=torch.stack(log_probabilities),
            old_values=value_tensor,
            rewards=reward_tensor,
            advantages=advantages,
            returns=advantages + value_tensor,
            terminated=terminated_tensor,
            observation_receipts=tuple(observation_receipts),
            transition_receipts=tuple(transition_receipts),
            final_environment_state_receipt_sha256=(
                self.environment.state.semantic_receipt_sha256
            ),
        )
        result.validate()
        return result

    def update(self, rollout: MassiveAdaptivePPORolloutV1) -> dict[str, float]:
        rollout.validate()
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1.0e-8)
        totals = torch.zeros(5, dtype=torch.float64)
        updates = 0
        count = rollout.rewards.shape[0]
        for _ in range(self.config.epochs_per_rollout):
            order = torch.randperm(count, generator=self.minibatch_rng)
            for start in range(0, count, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size].to(
                    device=self.device
                )
                output = self.model(
                    {"adaptive_state": rollout.observations[indices]}
                )
                log_probability = output.distribution.log_prob(rollout.actions[indices])
                ratio = torch.exp(
                    (log_probability - rollout.old_log_probabilities[indices]).clamp(
                        -20.0, 20.0
                    )
                )
                selected_advantages = advantages[indices]
                policy_loss = -torch.minimum(
                    ratio * selected_advantages,
                    ratio.clamp(
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    * selected_advantages,
                ).mean()
                entropy = output.distribution.entropy().mean()
                value_delta = output.value - rollout.old_values[indices]
                clipped_value = rollout.old_values[indices] + value_delta.clamp(
                    -self.config.value_clip_range,
                    self.config.value_clip_range,
                )
                value_loss = 0.5 * torch.maximum(
                    (output.value - rollout.returns[indices]).square(),
                    (clipped_value - rollout.returns[indices]).square(),
                ).mean()
                loss = (
                    policy_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.value_coefficient * value_loss
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.maximum_gradient_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                approximate_kl = (
                    rollout.old_log_probabilities[indices] - log_probability
                ).mean()
                totals += torch.tensor(
                    (
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy.detach()),
                        float(approximate_kl.detach()),
                        float(gradient_norm.detach()),
                    ),
                    dtype=torch.float64,
                )
                updates += 1
        metrics_tensor = totals / updates
        metrics = {
            name: float(metrics_tensor[index])
            for index, name in enumerate(
                ("policy_loss", "value_loss", "entropy", "approximate_kl", "gradient_norm")
            )
        }
        self.update_index += 1
        self.loss_trace.append(tuple(metrics.values()))
        return metrics

    def checkpoint(self) -> MassiveAdaptiveRLCheckpointV1:
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        actor_state = copy.deepcopy(self.actor_optimizer.state_dict())
        critic_state = copy.deepcopy(self.critic_optimizer.state_dict())
        rng = (
            torch.get_rng_state().clone(),
            tuple(value.clone() for value in torch.cuda.get_rng_state_all()),
            random.getstate(),
            np.random.get_state(),
            self.minibatch_rng.get_state().clone(),
        )
        body = {
            "schema": MASSIVE_ADAPTIVE_RL_CHECKPOINT_V1_SCHEMA,
            "update_index": self.update_index,
            "model_state": model_state,
            "actor_optimizer_state": actor_state,
            "critic_optimizer_state": critic_state,
            "torch_rng_state": rng[0],
            "cuda_rng_states": rng[1],
            "python_rng_state": rng[2],
            "numpy_rng_state": rng[3],
            "minibatch_rng_state": rng[4],
            "environment_state": self.environment.state,
            "loss_trace": tuple(self.loss_trace),
            "model_state_receipt_sha256": _state_receipt(model_state),
            "actor_optimizer_state_receipt_sha256": _state_receipt(actor_state),
            "critic_optimizer_state_receipt_sha256": _state_receipt(critic_state),
            "rng_state_receipt_sha256": _state_receipt(rng),
            "environment_state_receipt_sha256": (
                self.environment.state.semantic_receipt_sha256
            ),
            "loss_trace_receipt_sha256": semantic_sha256(tuple(self.loss_trace)),
            "training_source_inventory_sha256": self.environment.source_inventory_sha256,
            "training_forecast_authority_receipt_sha256": None
            if self.training_forecast_authority is None
            else self.training_forecast_authority.semantic_receipt_sha256,
            "ppo_config_receipt_sha256": self.config.receipt_sha256,
            "observation_specification_sha256": MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256,
            "action_specification_sha256": MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256,
            "reward_specification_sha256": MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256,
            "exact_resume_authorized": True,
            "development_rl_training_authorized": (
                self.training_forecast_authority is not None
            ),
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        }
        provisional = MassiveAdaptiveRLCheckpointV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256="0" * 64,
        )
        result = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        )
        result.validate()
        return result

    def restore(self, checkpoint: MassiveAdaptiveRLCheckpointV1) -> None:
        checkpoint.validate()
        expected_training_authority = (
            None
            if self.training_forecast_authority is None
            else self.training_forecast_authority.semantic_receipt_sha256
        )
        if (
            checkpoint.training_source_inventory_sha256
            != self.environment.source_inventory_sha256
            or checkpoint.ppo_config_receipt_sha256 != self.config.receipt_sha256
            or checkpoint.training_forecast_authority_receipt_sha256
            != expected_training_authority
        ):
            raise MassiveAdaptivePPOV1Error(
                "adaptive RL checkpoint and trainer roots differ"
            )
        self.model.load_state_dict(checkpoint.model_state)
        self.actor_optimizer.load_state_dict(checkpoint.actor_optimizer_state)
        self.critic_optimizer.load_state_dict(checkpoint.critic_optimizer_state)
        torch.set_rng_state(checkpoint.torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(checkpoint.cuda_rng_states))
        random.setstate(checkpoint.python_rng_state)
        np.random.set_state(checkpoint.numpy_rng_state)
        self.minibatch_rng.set_state(checkpoint.minibatch_rng_state)
        self.environment.restore(checkpoint.environment_state)
        self.update_index = checkpoint.update_index
        self.loss_trace = list(checkpoint.loss_trace)
        self._observation = self.environment._observation


__all__ = [
    "MassiveAdaptivePPOConfigV1",
    "MassiveAdaptivePPORolloutV1",
    "MassiveAdaptivePPOTrainerV1",
    "MassiveAdaptivePPOV1Error",
    "MassiveAdaptiveRLCheckpointV1",
]
