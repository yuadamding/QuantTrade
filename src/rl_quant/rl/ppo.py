"""Domain-neutral recurrent PPO reference implementation.

The implementation consumes :class:`RecurrentSequenceBatch` directly.  Sequence
minibatches preserve recurrent order, padded/burn-in positions update hidden state
but never enter a loss, and true terminations remain distinct from truncations in
the GAE targets computed by the trajectory buffer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from rl_quant.rl.algorithm import Algorithm, MetricValue, RecurrentState
from rl_quant.rl.trajectory import RecurrentSequenceBatch
from rl_quant.rl.types import ActionBatch, ObservationBatch


class ActionDistribution(Protocol):
    """Small distribution surface required by PPO."""

    @property
    def leading_shape(self) -> torch.Size: ...

    def sample(self) -> torch.Tensor: ...

    def mode(self) -> torch.Tensor: ...

    def log_prob(self, action: torch.Tensor) -> torch.Tensor: ...

    def entropy(self) -> torch.Tensor: ...


class MaskedCategorical:
    """Categorical distribution that assigns exactly zero mass to invalid actions."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if not logits.is_floating_point() or logits.ndim < 2:
            raise ValueError("Categorical logits need floating shape [..., action].")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Categorical logits must be finite.")
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        if mask.shape != logits.shape or mask.dtype != torch.bool or mask.device != logits.device:
            raise ValueError("Categorical mask must be bool and exactly match the logits shape/device.")
        if bool((~mask.any(dim=-1)).any().item()):
            raise ValueError("Every categorical row must contain at least one valid action.")
        self.logits = logits
        self.mask = mask
        # A finite sentinel keeps log-probability/entropy finite under mixed
        # precision while softmax still maps invalid actions to exactly zero.
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        self._distribution = torch.distributions.Categorical(logits=masked_logits)

    @property
    def leading_shape(self) -> torch.Size:
        return self.logits.shape[:-1]

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def mode(self) -> torch.Tensor:
        return self._distribution.logits.argmax(dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.leading_shape or action.dtype != torch.long or action.device != self.logits.device:
            raise ValueError(
                f"Categorical action must be long with shape {tuple(self.leading_shape)} on {self.logits.device}."
            )
        selected_valid = self.mask.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        if not bool(selected_valid.all().item()):
            raise ValueError("A categorical action selected an invalid/masked choice.")
        return self._distribution.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()


class DiagonalNormal:
    """Independent Normal distribution with stable, bounded log standard deviation."""

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        if not mean.is_floating_point() or mean.ndim < 2:
            raise ValueError("Normal means need floating shape [..., action].")
        if log_std.shape != mean.shape or log_std.device != mean.device:
            raise ValueError("Normal mean and log_std must have identical shape/device.")
        if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(log_std).all().item()):
            raise ValueError("Normal parameters must be finite.")
        if not math.isfinite(min_log_std) or not math.isfinite(max_log_std) or min_log_std > max_log_std:
            raise ValueError("Need finite min_log_std <= max_log_std.")
        self.mean = mean
        self.log_std = log_std.clamp(min=min_log_std, max=max_log_std)
        self._distribution = torch.distributions.Normal(mean, self.log_std.exp())

    @property
    def leading_shape(self) -> torch.Size:
        return self.mean.shape[:-1]

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.mean.shape or action.device != self.mean.device or not action.is_floating_point():
            raise ValueError(f"Normal action must be floating with shape {tuple(self.mean.shape)} on {self.mean.device}.")
        if not bool(torch.isfinite(action).all().item()):
            raise ValueError("Normal action must be finite.")
        return self._distribution.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)


class MaskedDirichlet:
    """Dirichlet distribution on each row's active simplex.

    Masked dimensions are excluded from the event measure, always sample as
    exact zero, and must be exact zero in :meth:`log_prob`. Log probability and
    entropy use the closed-form Dirichlet expressions over active dimensions
    only, including the degenerate one-action simplex whose values are both 0.
    """

    def __init__(self, concentration: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if not concentration.is_floating_point() or concentration.ndim < 2:
            raise ValueError("Dirichlet concentration needs floating shape [..., action].")
        if not bool(torch.isfinite(concentration).all().item()) or bool((concentration <= 0).any().item()):
            raise ValueError("Dirichlet concentration must be finite and strictly positive.")
        if mask is None:
            mask = torch.ones_like(concentration, dtype=torch.bool)
        if mask.shape != concentration.shape or mask.dtype != torch.bool or mask.device != concentration.device:
            raise ValueError("Dirichlet mask must be bool and exactly match concentration shape/device.")
        if bool((~mask.any(dim=-1)).any().item()):
            raise ValueError("Every Dirichlet row must contain at least one active action.")
        self.concentration = concentration
        self.mask = mask

    @property
    def leading_shape(self) -> torch.Size:
        return self.concentration.shape[:-1]

    @property
    def _work_dtype(self) -> torch.dtype:
        return (
            torch.float32
            if self.concentration.dtype in (torch.float16, torch.bfloat16)
            else self.concentration.dtype
        )

    def _active_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        concentration = self.concentration.to(dtype=self._work_dtype)
        active = torch.where(self.mask, concentration, torch.ones_like(concentration))
        return concentration, active

    def sample(self) -> torch.Tensor:
        _concentration, active = self._active_parameters()
        gamma = torch.distributions.Gamma(active, torch.ones_like(active)).sample()
        gamma = torch.where(self.mask, gamma, torch.zeros_like(gamma))
        total = gamma.sum(dim=-1, keepdim=True)
        normalized = gamma / total.clamp_min(torch.finfo(gamma.dtype).tiny)
        # Gamma sampling is positive, but the mean is a deterministic safe
        # fallback for an underflowed all-zero row in reduced precision.
        normalized = torch.where(total > 0, normalized, self.mode().to(dtype=gamma.dtype))
        return torch.where(self.mask, normalized, 0.0)

    def mode(self) -> torch.Tensor:
        # The mathematical mode is not interior when any alpha <= 1. The mean
        # is the always-defined deterministic representative used for rollout.
        concentration = self.concentration.to(dtype=self._work_dtype)
        concentration = torch.where(self.mask, concentration, torch.zeros_like(concentration))
        mean = concentration / concentration.sum(dim=-1, keepdim=True)
        return torch.where(self.mask, mean, torch.zeros_like(mean))

    def _validated_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.concentration.shape or action.device != self.concentration.device:
            raise ValueError(
                f"Dirichlet action must have shape {tuple(self.concentration.shape)} on "
                f"{self.concentration.device}."
            )
        if not action.is_floating_point() or not bool(torch.isfinite(action).all().item()):
            raise ValueError("Dirichlet action must be finite and floating point.")
        if bool((action[~self.mask] != 0).any().item()):
            raise ValueError("Masked Dirichlet action dimensions must be exactly zero.")
        if bool((action[self.mask] <= 0).any().item()):
            raise ValueError("Active Dirichlet action dimensions must be strictly positive.")
        sums = action.sum(dim=-1)
        tolerance = 1e-3 if action.dtype in (torch.float16, torch.bfloat16) else 1e-6
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=0.0)):
            raise ValueError("Dirichlet action must sum to one on its active simplex.")
        return action.to(dtype=self._work_dtype)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action_work = self._validated_action(action)
        concentration, active = self._active_parameters()
        active_total = torch.where(self.mask, concentration, 0.0).sum(dim=-1)
        safe_action = torch.where(self.mask, action_work, torch.ones_like(action_work))
        log_normalizer = torch.lgamma(active_total) - torch.where(
            self.mask, torch.lgamma(active), 0.0
        ).sum(dim=-1)
        log_kernel = torch.where(
            self.mask,
            (active - 1.0) * safe_action.log(),
            0.0,
        ).sum(dim=-1)
        return log_normalizer + log_kernel

    def entropy(self) -> torch.Tensor:
        concentration, active = self._active_parameters()
        active_total = torch.where(self.mask, concentration, 0.0).sum(dim=-1)
        active_count = self.mask.sum(dim=-1).to(dtype=self._work_dtype)
        log_beta = torch.where(self.mask, torch.lgamma(active), 0.0).sum(dim=-1) - torch.lgamma(active_total)
        correction = torch.where(
            self.mask,
            (active - 1.0) * torch.digamma(active),
            0.0,
        ).sum(dim=-1)
        return log_beta + (active_total - active_count) * torch.digamma(active_total) - correction


@dataclass(frozen=True)
class PPOModelOutput:
    distribution: ActionDistribution
    value: torch.Tensor
    recurrent_state: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.value.shape != self.distribution.leading_shape:
            raise ValueError(
                f"Critic value shape {tuple(self.value.shape)} does not match distribution leading shape "
                f"{tuple(self.distribution.leading_shape)}."
            )
        if not self.value.is_floating_point() or not bool(torch.isfinite(self.value).all().item()):
            raise ValueError("Critic values must be finite floating-point tensors.")


class PPOActorCritic(nn.Module, ABC):
    """Actor-critic interface used by :class:`RecurrentPPO`."""

    @abstractmethod
    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        action_mask: torch.Tensor | None = None,
        recurrent_state: RecurrentState | None = None,
        episode_start: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        burn_in: int = 0,
    ) -> PPOModelOutput: ...

    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        """Return an empty state for stateless supplied models."""

        del observation
        return {}


class RecurrentActorCritic(PPOActorCritic):
    """Compact GRU actor-critic reference model for vector or sequence inputs.

    A caller names one vector-valued observation field.  Domain-specific encoders
    can instead implement :class:`PPOActorCritic` directly while retaining the
    exact same PPO/trajectory implementation.
    """

    def __init__(
        self,
        *,
        observation_key: str,
        observation_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_kind: str = "categorical",
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_concentration: float = 1e-4,
        max_concentration: float = 1e4,
    ) -> None:
        super().__init__()
        if not observation_key:
            raise ValueError("observation_key cannot be empty.")
        if observation_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim, hidden_dim, and action_dim must be positive.")
        if action_kind not in ("categorical", "normal", "dirichlet"):
            raise ValueError("action_kind must be 'categorical', 'normal', or 'dirichlet'.")
        if not math.isfinite(min_log_std) or not math.isfinite(max_log_std) or min_log_std > max_log_std:
            raise ValueError("Need finite min_log_std <= max_log_std.")
        if (
            not math.isfinite(min_concentration)
            or not math.isfinite(max_concentration)
            or min_concentration <= 0
            or min_concentration > max_concentration
        ):
            raise ValueError("Need 0 < finite min_concentration <= max_concentration.")
        self.observation_key = observation_key
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.action_kind = action_kind
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.input_layer = nn.Linear(observation_dim, hidden_dim)
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        if action_kind == "normal":
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.register_parameter("log_std", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (self.input_layer, self.recurrent, self.actor_head, self.value_head):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.orthogonal_(parameter, gain=math.sqrt(2.0))
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        if self.observation_key not in observation.tensors:
            raise ValueError(f"Missing observation field {self.observation_key!r}.")
        inputs = observation.tensors[self.observation_key]
        if not inputs.is_floating_point():
            raise ValueError(f"Observation {self.observation_key!r} must be floating point.")
        return {
            "hidden": torch.zeros(
                (observation.batch_size, self.hidden_dim),
                dtype=inputs.dtype,
                device=inputs.device,
            )
        }

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        action_mask: torch.Tensor | None = None,
        recurrent_state: RecurrentState | None = None,
        episode_start: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        burn_in: int = 0,
    ) -> PPOModelOutput:
        if self.observation_key not in observations:
            raise ValueError(f"Missing observation field {self.observation_key!r}.")
        inputs = observations[self.observation_key]
        if not inputs.is_floating_point() or inputs.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Observation {self.observation_key!r} needs floating final dimension {self.observation_dim}."
            )
        if inputs.ndim not in (2, 3):
            raise ValueError("RecurrentActorCritic expects [batch, feature] or [batch, time, feature].")
        single_step = inputs.ndim == 2
        sequence = inputs.unsqueeze(1) if single_step else inputs
        batch_size, sequence_length = sequence.shape[:2]
        if burn_in < 0 or burn_in >= sequence_length:
            raise ValueError(f"burn_in must lie in [0, {sequence_length - 1}] for this input.")
        if single_step and burn_in != 0:
            raise ValueError("Single-step policy evaluation cannot use burn-in.")
        device, dtype = sequence.device, sequence.dtype

        if not recurrent_state:
            hidden = torch.zeros((batch_size, self.hidden_dim), dtype=dtype, device=device)
        else:
            if set(recurrent_state) != {"hidden"}:
                raise ValueError("RecurrentActorCritic state must contain exactly the 'hidden' tensor.")
            hidden = recurrent_state["hidden"]
            if hidden.shape != (batch_size, self.hidden_dim) or hidden.device != device or hidden.dtype != dtype:
                raise ValueError(
                    f"hidden state must have shape {(batch_size, self.hidden_dim)} on {device} with {dtype}."
                )
        if episode_start is None:
            starts = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
        else:
            starts = episode_start.unsqueeze(1) if single_step else episode_start
            if starts.shape != (batch_size, sequence_length) or starts.dtype != torch.bool or starts.device != device:
                raise ValueError("episode_start must be bool and match the input's leading dimensions.")
        if valid_mask is None:
            valid = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
        else:
            valid = valid_mask.unsqueeze(1) if single_step else valid_mask
            if valid.shape != (batch_size, sequence_length) or valid.dtype != torch.bool or valid.device != device:
                raise ValueError("valid_mask must be bool and match the input's leading dimensions.")

        hidden_steps: list[torch.Tensor] = []
        encoded = torch.tanh(self.input_layer(sequence))
        for step in range(sequence_length):
            if step == burn_in and burn_in:
                # Burn-in reconstructs the recurrent state but is outside the
                # optimization graph, matching truncated BPTT semantics.
                hidden = hidden.detach()
            hidden = torch.where(starts[:, step].unsqueeze(-1), torch.zeros_like(hidden), hidden)
            candidate = self.recurrent(encoded[:, step], hidden)
            hidden = torch.where(valid[:, step].unsqueeze(-1), candidate, hidden)
            hidden_steps.append(hidden)
        features = torch.stack(hidden_steps, dim=1)
        actor_parameters = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)

        effective_mask = action_mask
        if effective_mask is not None:
            expected = (batch_size, sequence_length, self.action_dim)
            if single_step:
                effective_mask = effective_mask.unsqueeze(1)
            if effective_mask.shape != expected or effective_mask.dtype != torch.bool:
                raise ValueError(f"action_mask must be bool with shape {expected}.")
            # Padded rows carry dummy actions. Make their distribution well-defined;
            # the valid/loss masks guarantee they never affect optimization.
            effective_mask = torch.where(
                valid.unsqueeze(-1), effective_mask, torch.ones_like(effective_mask)
            )
        if self.action_kind == "categorical":
            distribution: ActionDistribution = MaskedCategorical(actor_parameters, effective_mask)
        elif self.action_kind == "normal":
            assert self.log_std is not None
            log_std = self.log_std.view(1, 1, -1).expand_as(actor_parameters)
            distribution = DiagonalNormal(
                actor_parameters,
                log_std,
                min_log_std=self.min_log_std,
                max_log_std=self.max_log_std,
            )
        else:
            concentration_parameters = (
                actor_parameters.float()
                if actor_parameters.dtype in (torch.float16, torch.bfloat16)
                else actor_parameters
            )
            concentration = (F.softplus(concentration_parameters) + self.min_concentration).clamp_max(
                self.max_concentration
            )
            distribution = MaskedDirichlet(concentration, effective_mask)
        if single_step:
            if self.action_kind == "categorical":
                assert isinstance(distribution, MaskedCategorical)
                distribution = MaskedCategorical(
                    distribution.logits.squeeze(1),
                    distribution.mask.squeeze(1),
                )
            elif self.action_kind == "normal":
                assert isinstance(distribution, DiagonalNormal)
                distribution = DiagonalNormal(
                    distribution.mean.squeeze(1),
                    distribution.log_std.squeeze(1),
                    min_log_std=self.min_log_std,
                    max_log_std=self.max_log_std,
                )
            else:
                assert isinstance(distribution, MaskedDirichlet)
                distribution = MaskedDirichlet(
                    distribution.concentration.squeeze(1),
                    distribution.mask.squeeze(1),
                )
            values = values.squeeze(1)
        return PPOModelOutput(distribution=distribution, value=values, recurrent_state={"hidden": hidden})


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-5
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_sequences: int | None = None
    target_kl: float | None = None
    advantage_epsilon: float = 1e-8
    max_log_ratio: float = 20.0
    seed: int = 0

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "clip_range": self.clip_range,
            "max_grad_norm": self.max_grad_norm,
            "advantage_epsilon": self.advantage_epsilon,
            "max_log_ratio": self.max_log_ratio,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "weight_decay": self.weight_decay,
            "value_coefficient": self.value_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.value_clip_range is not None and (
            not math.isfinite(self.value_clip_range) or self.value_clip_range <= 0
        ):
            raise ValueError("value_clip_range must be finite and positive or None.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.minibatch_sequences is not None and self.minibatch_sequences <= 0:
            raise ValueError("minibatch_sequences must be positive or None.")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be finite and positive or None.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")


def _index_mapping(values: Mapping[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in values.items()}


class RecurrentPPO(Algorithm):
    """Clipped PPO with recurrent sequence minibatches and checkpointable state."""

    def __init__(
        self,
        model: PPOActorCritic,
        config: PPOConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = PPOConfig() if config is None else config
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        self._minibatch_generator = torch.Generator(device="cpu")
        self._minibatch_generator.manual_seed(self.config.seed)
        self.update_count = 0

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("PPO model has no parameters.") from exc

    @torch.no_grad()
    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        if observation.device != self.device:
            raise ValueError(f"Observation is on {observation.device}, but PPO model is on {self.device}.")
        state = dict(self.model.initial_recurrent_state(observation))
        for name, tensor in state.items():
            if tensor.ndim == 0 or tensor.shape[0] != observation.batch_size:
                raise ValueError(
                    f"Initial recurrent state {name!r} needs leading batch size {observation.batch_size}."
                )
            if tensor.device != observation.device:
                raise ValueError(f"Initial recurrent state {name!r} is on {tensor.device}, expected {observation.device}.")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"Initial recurrent state {name!r} must be finite.")
        return state

    @torch.no_grad()
    def value(
        self,
        observation: ObservationBatch,
        recurrent_state: RecurrentState | None = None,
    ) -> torch.Tensor:
        """Evaluate ``V(observation)`` from the supplied input recurrent state."""

        if observation.device != self.device:
            raise ValueError(f"Observation is on {observation.device}, but PPO model is on {self.device}.")
        state = self.initial_recurrent_state(observation) if recurrent_state is None else recurrent_state
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(
                observation.tensors,
                action_mask=observation.action_mask,
                recurrent_state=state,
                episode_start=observation.episode_start,
                valid_mask=torch.ones(observation.batch_size, dtype=torch.bool, device=observation.device),
                burn_in=0,
            )
        finally:
            self.model.train(was_training)
        return output.value

    @torch.no_grad()
    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: RecurrentState | None = None,
    ) -> ActionBatch:
        if observation.device != self.device:
            raise ValueError(f"Observation is on {observation.device}, but PPO model is on {self.device}.")
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(
                observation.tensors,
                action_mask=observation.action_mask,
                recurrent_state=recurrent_state,
                episode_start=observation.episode_start,
                valid_mask=torch.ones(observation.batch_size, dtype=torch.bool, device=observation.device),
                burn_in=0,
            )
            action = output.distribution.mode() if deterministic else output.distribution.sample()
            log_prob = output.distribution.log_prob(action)
            entropy = output.distribution.entropy()
        finally:
            self.model.train(was_training)
        return ActionBatch(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            recurrent_state=output.recurrent_state,
            extras={"value": output.value},
        )

    def _validate_batch(self, batch: RecurrentSequenceBatch) -> None:
        if batch.rewards.device != self.device:
            raise ValueError(f"Trajectory is on {batch.rewards.device}, but PPO model is on {self.device}.")
        expected = batch.rewards.shape
        fields = {
            "old_log_probs": batch.old_log_probs,
            "old_values": batch.old_values,
            "advantages": batch.advantages,
            "returns": batch.returns,
            "valid_mask": batch.valid_mask,
            "loss_mask": batch.loss_mask,
        }
        for name, value in fields.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}; got {tuple(value.shape)}.")
        if batch.valid_mask.dtype != torch.bool or batch.loss_mask.dtype != torch.bool:
            raise ValueError("valid_mask and loss_mask must be bool.")
        if bool((batch.loss_mask & ~batch.valid_mask).any().item()):
            raise ValueError("loss_mask must be a subset of valid_mask.")
        if batch.burn_in < 0 or batch.burn_in >= batch.sequence_width:
            raise ValueError("batch.burn_in must lie within the sequence width.")
        if batch.burn_in and bool(batch.loss_mask[:, : batch.burn_in].any().item()):
            raise ValueError("Burn-in positions cannot contribute to PPO losses.")
        if not bool(batch.loss_mask.any().item()):
            raise ValueError("PPO batch has no learning positions.")
        for name in ("old_log_probs", "old_values", "advantages", "returns"):
            value = fields[name][batch.loss_mask]
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite on all learning positions.")

    @staticmethod
    def _prepared_action_mask(batch: RecurrentSequenceBatch) -> torch.Tensor | None:
        if batch.action_masks is None:
            return None
        return torch.where(
            batch.valid_mask.unsqueeze(-1),
            batch.action_masks,
            torch.ones_like(batch.action_masks),
        )

    def update(self, batch: RecurrentSequenceBatch) -> Mapping[str, MetricValue]:
        self._validate_batch(batch)
        self.model.train(True)
        loss_mask = batch.loss_mask
        selected_advantages = batch.advantages[loss_mask]
        advantage_mean = selected_advantages.mean()
        advantage_std = selected_advantages.std(unbiased=False)
        normalized_advantages = torch.where(
            loss_mask,
            (batch.advantages - advantage_mean) / advantage_std.clamp_min(self.config.advantage_epsilon),
            torch.zeros_like(batch.advantages),
        )
        action_masks = self._prepared_action_mask(batch)
        learning_sequences = torch.nonzero(loss_mask.any(dim=1), as_tuple=False).flatten()
        num_sequences = learning_sequences.shape[0]
        minibatch_size = self.config.minibatch_sequences or num_sequences
        metric_names = (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
        )
        # Accumulate diagnostics on-device so a GPU rollout does not incur one
        # host synchronization per metric and minibatch.
        metric_sums = torch.zeros(len(metric_names), dtype=torch.float64, device=batch.rewards.device)
        metric_weight = torch.zeros((), dtype=torch.float64, device=batch.rewards.device)
        minibatches = 0
        epochs_completed = 0
        stopped_for_kl = False

        for epoch in range(self.config.epochs):
            permutation = torch.randperm(
                num_sequences,
                generator=self._minibatch_generator,
                device="cpu",
            ).to(batch.rewards.device)
            for start in range(0, num_sequences, minibatch_size):
                indices = learning_sequences[permutation[start : start + minibatch_size]]
                mask = loss_mask[indices]
                # Recurrent sequence construction guarantees at least one
                # learning token per row, so keep the weight on-device.
                count = mask.sum().to(dtype=torch.float64)
                output = self.model(
                    _index_mapping(batch.observations, indices),
                    action_mask=None if action_masks is None else action_masks[indices],
                    recurrent_state=_index_mapping(batch.initial_recurrent_state, indices),
                    episode_start=batch.episode_start[indices],
                    valid_mask=batch.valid_mask[indices],
                    burn_in=batch.burn_in,
                )
                actions = batch.actions[indices]
                if actions.is_floating_point() and actions.ndim == mask.ndim + 1:
                    # Padded rows carry zero tensors, which are outside a
                    # simplex distribution's support. A uniform dummy is valid
                    # for Dirichlet and harmless for Normal; it never enters a
                    # loss because valid_mask is false there.
                    dummy = torch.full_like(actions, 1.0 / actions.shape[-1])
                    actions = torch.where(batch.valid_mask[indices].unsqueeze(-1), actions, dummy)
                new_log_prob = output.distribution.log_prob(actions)
                entropy = output.distribution.entropy()
                if new_log_prob.shape != mask.shape or entropy.shape != mask.shape or output.value.shape != mask.shape:
                    raise ValueError("PPO model outputs must match the sequence batch's [sequence, time] shape.")

                old_log_prob = batch.old_log_probs[indices]
                log_ratio = (new_log_prob - old_log_prob).clamp(
                    min=-self.config.max_log_ratio,
                    max=self.config.max_log_ratio,
                )
                ratio = log_ratio.exp()
                advantages = normalized_advantages[indices]
                unclipped_objective = ratio * advantages
                clipped_objective = ratio.clamp(
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages
                policy_loss = -torch.minimum(unclipped_objective, clipped_objective)[mask].mean()

                values = output.value
                old_values = batch.old_values[indices]
                returns = batch.returns[indices]
                value_error = (values - returns).square()
                if self.config.value_clip_range is not None:
                    clipped_values = old_values + (values - old_values).clamp(
                        min=-self.config.value_clip_range,
                        max=self.config.value_clip_range,
                    )
                    clipped_error = (clipped_values - returns).square()
                    value_error = torch.maximum(value_error, clipped_error)
                value_loss = 0.5 * value_error[mask].mean()
                entropy_mean = entropy[mask].mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # This also gates non-finite loss gradients without a separate
                # loss.item() synchronization in every minibatch.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio)[mask].mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range)[mask].float().mean()
                local_metrics = torch.stack(
                    (
                        loss.detach(),
                        policy_loss.detach(),
                        value_loss.detach(),
                        entropy_mean.detach(),
                        approx_kl.detach(),
                        clip_fraction.detach(),
                        grad_norm.detach(),
                    )
                ).to(dtype=torch.float64)
                metric_sums += local_metrics * count
                metric_weight += count
                minibatches += 1
                if self.config.target_kl is not None and float(approx_kl.item()) > self.config.target_kl:
                    stopped_for_kl = True
                    break
            epochs_completed = epoch + 1
            if stopped_for_kl:
                break
        if minibatches == 0:
            raise RuntimeError("PPO produced no non-empty minibatches.")

        with torch.no_grad():
            final_output = self.model(
                batch.observations,
                action_mask=action_masks,
                recurrent_state=batch.initial_recurrent_state,
                episode_start=batch.episode_start,
                valid_mask=batch.valid_mask,
                burn_in=batch.burn_in,
            )
            targets = batch.returns[loss_mask]
            predictions = final_output.value[loss_mask]
            target_variance = targets.var(unbiased=False)
            explained_variance = torch.where(
                target_variance > self.config.advantage_epsilon,
                1.0 - (targets - predictions).var(unbiased=False) / target_variance,
                torch.zeros_like(target_variance),
            )
        self.update_count += 1
        metric_values = (metric_sums / metric_weight).cpu().tolist()
        metrics: dict[str, MetricValue] = dict(zip(metric_names, metric_values, strict=True))
        summary_values = torch.stack(
            (advantage_mean.detach(), advantage_std.detach(), explained_variance.detach())
        ).to(dtype=torch.float64).cpu().tolist()
        metrics.update(
            {
                "advantage_mean": summary_values[0],
                "advantage_std": summary_values[1],
                "explained_variance": summary_values[2],
                "epochs_completed": epochs_completed,
                "minibatches": minibatches,
                "stopped_for_kl": int(stopped_for_kl),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "update_count": self.update_count,
            }
        )
        return metrics

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": asdict(self.config),
            "minibatch_rng_state": self._minibatch_generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing = {"model", "optimizer", "update_count", "minibatch_rng_state"} - set(state)
        if missing:
            raise ValueError(f"PPO checkpoint is missing fields: {sorted(missing)}.")
        checkpoint_config = state.get("config")
        if checkpoint_config is not None and checkpoint_config != asdict(self.config):
            raise ValueError(
                "PPO checkpoint configuration differs from the active configuration; "
                "construct RecurrentPPO with the saved config before restoring optimizer state."
            )
        update_count_value = state["update_count"]
        if (
            isinstance(update_count_value, bool)
            or not isinstance(update_count_value, int)
            or update_count_value < 0
        ):
            raise ValueError("PPO checkpoint update_count must be a nonnegative integer.")
        rng_state = state["minibatch_rng_state"]
        if (
            not torch.is_tensor(rng_state)
            or rng_state.device.type != "cpu"
            or rng_state.dtype != torch.uint8
            or rng_state.ndim != 1
        ):
            raise ValueError("PPO checkpoint minibatch_rng_state must be a CPU uint8 tensor.")
        # Validate all cheap continuation metadata before mutating model or
        # optimizer state so corrupt checkpoints fail closed.
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = update_count_value
        self._minibatch_generator.set_state(rng_state.detach().clone())

    def train(self, mode: bool = True) -> RecurrentPPO:
        self.model.train(mode)
        return self
