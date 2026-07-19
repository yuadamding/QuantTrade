"""Implicit Q-learning with conservative transformed-environment targets.

IQL is a suitable reference for partial-offline problems: it learns only from
logged actions, avoids querying an out-of-distribution action inside the Bellman
target, and trains its actor through advantage-weighted behavior cloning.  The
optional transition transforms provide a MetaTrader-style lower-envelope target
across plausible environments without coupling this module to financial data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass, replace
import copy
import json
import math
from typing import Any, Literal, Mapping, cast

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.rl.algorithm import Algorithm, MetricValue, RecurrentState
from rl_quant.rl.mixture import MixtureActionDistribution, RegimeRouter
from rl_quant.rl.ppo import ActionDistribution, DiagonalNormal, MaskedDirichlet
from rl_quant.rl.replay import ReplayBatch
from rl_quant.rl.robust import TransitionTransform
from rl_quant.rl.types import ActionBatch, ObservationBatch


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("MLP dimensions must be positive and include at least one hidden layer.")
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class OfflineActorCritic(nn.Module, ABC):
    """Model surface required by :class:`ImplicitQLearning`."""

    @abstractmethod
    def distribution(
        self,
        observations: Mapping[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> ActionDistribution: ...

    @abstractmethod
    def q_values(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def value(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def actor_parameters(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def critic_parameters(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def value_parameters(self) -> list[nn.Parameter]: ...


class VectorIQLActorCritic(OfflineActorCritic):
    """Reference twin-Q IQL network for one vector observation field.

    ``normal`` supports unconstrained continuous actions. ``dirichlet`` produces
    exact masked simplex actions and connects directly to portfolio allocation
    environments when logged active weights are strictly positive.
    """

    def __init__(
        self,
        *,
        observation_key: str,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        action_kind: str = "normal",
        min_log_std: float = -10.0,
        max_log_std: float = 2.0,
        min_concentration: float = 1e-4,
        max_concentration: float = 1e4,
    ) -> None:
        super().__init__()
        if not observation_key:
            raise ValueError("observation_key cannot be empty.")
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive.")
        if action_kind not in ("normal", "dirichlet"):
            raise ValueError("IQL action_kind must be 'normal' or 'dirichlet'.")
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
        self.action_dim = int(action_dim)
        self.action_kind = action_kind
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.actor: nn.Module = _mlp(observation_dim, action_dim, hidden_dims)
        self.q1 = _mlp(observation_dim + action_dim, 1, hidden_dims)
        self.q2 = _mlp(observation_dim + action_dim, 1, hidden_dims)
        self.value_network = _mlp(observation_dim, 1, hidden_dims)
        if action_kind == "normal":
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.register_parameter("log_std", None)

    def _observation(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.observation_key not in observations:
            raise ValueError(f"Missing observation field {self.observation_key!r}.")
        value = observations[self.observation_key]
        if not value.is_floating_point() or value.ndim != 2 or value.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Observation {self.observation_key!r} must be floating [batch, {self.observation_dim}]."
            )
        return value

    def distribution(
        self,
        observations: Mapping[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> ActionDistribution:
        features = self._observation(observations)
        parameters = self.actor(features)
        if self.action_kind == "normal":
            if action_mask is not None and not bool(action_mask.all().item()):
                raise ValueError("Masked continuous dimensions require the simplex/Dirichlet actor.")
            assert self.log_std is not None
            return DiagonalNormal(
                parameters,
                self.log_std.view(1, -1).expand_as(parameters),
                min_log_std=self.min_log_std,
                max_log_std=self.max_log_std,
            )
        concentration = F.softplus(parameters) + self.min_concentration
        concentration = concentration.clamp(max=self.max_concentration)
        return MaskedDirichlet(concentration, action_mask)

    def q_values(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features = self._observation(observations)
        if (
            not actions.is_floating_point()
            or actions.shape != (features.shape[0], self.action_dim)
            or actions.device != features.device
        ):
            raise ValueError(f"IQL actions must be floating [batch, {self.action_dim}] on the model device.")
        inputs = torch.cat((features, actions), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def value(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.value_network(self._observation(observations)).squeeze(-1)

    def actor_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.actor.parameters())
        if self.log_std is not None:
            parameters.append(self.log_std)
        return parameters

    def critic_parameters(self) -> list[nn.Parameter]:
        return [*self.q1.parameters(), *self.q2.parameters()]

    def value_parameters(self) -> list[nn.Parameter]:
        return list(self.value_network.parameters())


class RegimeMixtureIQLActorCritic(VectorIQLActorCritic):
    """IQL actor with same-support specialist policies and a learned router.

    Critics and the expectile value network are shared across experts, which
    keeps Q estimates calibrated on the same return scale while allowing the
    behavior policy to specialize by regime.  The router is initialized to a
    uniform mixture; optional balancing/entropy terms are configured on IQL.
    """

    def __init__(
        self,
        *,
        observation_key: str,
        observation_dim: int,
        action_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        action_kind: str = "normal",
        router_hidden_dim: int | None = None,
        router_temperature: float = 1.0,
        min_log_std: float = -10.0,
        max_log_std: float = 2.0,
        min_concentration: float = 1e-4,
        max_concentration: float = 1e4,
    ) -> None:
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be an integer of at least two.")
        super().__init__(
            observation_key=observation_key,
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_kind=action_kind,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )
        self.num_experts = int(num_experts)
        self.actor = nn.ModuleList(
            _mlp(observation_dim, action_dim, hidden_dims) for _ in range(self.num_experts)
        )
        self.router = RegimeRouter(
            observation_dim,
            self.num_experts,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
        )
        if self.action_kind == "normal":
            self.log_std = nn.Parameter(torch.zeros(self.num_experts, action_dim))

    def distribution(
        self,
        observations: Mapping[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> ActionDistribution:
        features = self._observation(observations)
        router = self.router(features)
        components: list[DiagonalNormal | MaskedDirichlet] = []
        experts = cast(nn.ModuleList, self.actor)
        for index, expert in enumerate(experts):
            parameters = expert(features)
            if self.action_kind == "normal":
                if action_mask is not None and not bool(action_mask.all().item()):
                    raise ValueError("Masked continuous dimensions require Dirichlet experts.")
                assert self.log_std is not None
                components.append(
                    DiagonalNormal(
                        parameters,
                        self.log_std[index].view(1, -1).expand_as(parameters),
                        min_log_std=self.min_log_std,
                        max_log_std=self.max_log_std,
                    )
                )
            else:
                concentration = (F.softplus(parameters) + self.min_concentration).clamp(
                    max=self.max_concentration
                )
                components.append(MaskedDirichlet(concentration, action_mask))
        return MixtureActionDistribution(components, router)

    def actor_parameters(self) -> list[nn.Parameter]:
        parameters = [*self.actor.parameters(), *self.router.parameters()]
        if self.log_std is not None:
            parameters.append(self.log_std)
        return parameters


@dataclass(frozen=True)
class IQLConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    weight_decay: float = 0.0
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0
    target_tau: float = 0.005
    critic_uncertainty_penalty: float = 0.0
    action_source: Literal["executed_if_available", "requested"] = "executed_if_available"
    simplex_behavior_smoothing: float = 1e-6
    router_balance_coefficient: float = 0.0
    router_entropy_coefficient: float = 0.0
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        positive = {
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "value_learning_rate": self.value_learning_rate,
            "advantage_temperature": self.advantage_temperature,
            "max_advantage_weight": self.max_advantage_weight,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not math.isfinite(self.expectile) or not 0 < self.expectile < 1:
            raise ValueError("expectile must lie strictly between zero and one.")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must lie in (0, 1].")
        if not math.isfinite(self.critic_uncertainty_penalty) or self.critic_uncertainty_penalty < 0:
            raise ValueError("critic_uncertainty_penalty must be finite and nonnegative.")
        if self.action_source not in ("executed_if_available", "requested"):
            raise ValueError("action_source must be 'executed_if_available' or 'requested'.")
        if (
            not math.isfinite(self.simplex_behavior_smoothing)
            or not 0 <= self.simplex_behavior_smoothing < 1
        ):
            raise ValueError("simplex_behavior_smoothing must lie in [0, 1).")
        for name, value in (
            ("router_balance_coefficient", self.router_balance_coefficient),
            ("router_entropy_coefficient", self.router_entropy_coefficient),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


class ImplicitQLearning(Algorithm):
    """Twin-critic IQL with optional worst-case transformed transition targets."""

    def __init__(
        self,
        model: OfflineActorCritic,
        config: IQLConfig | None = None,
        *,
        transforms: tuple[TransitionTransform, ...] = (),
    ) -> None:
        self.model = model
        self.config = IQLConfig() if config is None else config
        self.transforms = tuple(transforms)
        if any(not isinstance(transform, TransitionTransform) for transform in self.transforms):
            raise TypeError("Every robust target transform must implement TransitionTransform.")
        actor_parameters = model.actor_parameters()
        critic_parameters = model.critic_parameters()
        value_parameters = model.value_parameters()
        parameter_groups = (actor_parameters, critic_parameters, value_parameters)
        flattened_ids = [id(parameter) for group in parameter_groups for parameter in group]
        if len(flattened_ids) != len(set(flattened_ids)):
            raise ValueError("IQL actor, critic, and value optimizer parameter sets must be disjoint.")
        if not all(parameter_groups):
            raise ValueError("IQL model parameter groups cannot be empty.")
        self.actor_optimizer = torch.optim.AdamW(
            actor_parameters,
            lr=self.config.actor_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic_parameters,
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            value_parameters,
            lr=self.config.value_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.target_model = copy.deepcopy(model).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.update_count = 0

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("IQL model has no parameters.") from exc

    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        del observation
        return {}

    @torch.no_grad()
    def value(
        self,
        observation: ObservationBatch,
        recurrent_state: RecurrentState | None = None,
    ) -> torch.Tensor:
        if recurrent_state:
            raise ValueError("The reference IQL model is feed-forward and accepts no recurrent state.")
        if observation.device != self.device:
            raise ValueError(f"Observation is on {observation.device}, but IQL is on {self.device}.")
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model.value(observation.tensors)
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: RecurrentState | None = None,
    ) -> ActionBatch:
        if recurrent_state:
            raise ValueError("The reference IQL model is feed-forward and accepts no recurrent state.")
        if observation.device != self.device:
            raise ValueError(f"Observation is on {observation.device}, but IQL is on {self.device}.")
        was_training = self.model.training
        self.model.eval()
        try:
            distribution = self.model.distribution(observation.tensors, observation.action_mask)
            routed_expert: torch.Tensor | None = None
            if isinstance(distribution, MixtureActionDistribution) and not deterministic:
                routed = distribution.sample_with_expert()
                action, routed_expert = routed.action, routed.expert_index
            else:
                action = distribution.mode() if deterministic else distribution.sample()
            q_values = torch.stack(self.model.q_values(observation.tensors, action), dim=0)
            q_min = q_values.min(dim=0).values
            q_std = q_values.std(dim=0, unbiased=False)
            value = self.model.value(observation.tensors)
            extras = {"q_min": q_min, "critic_uncertainty": q_std, "value": value}
            if isinstance(distribution, MixtureActionDistribution):
                extras["router_probabilities"] = distribution.router_probabilities
                extras["router_entropy"] = distribution.router.entropy
                if routed_expert is not None:
                    extras["routed_expert"] = routed_expert
            return ActionBatch(
                action=action,
                log_prob=distribution.log_prob(action),
                entropy=distribution.entropy(),
                extras=extras,
            )
        finally:
            self.model.train(was_training)

    def _validate_batch(self, batch: ReplayBatch) -> None:
        if batch.device != self.device:
            raise ValueError(f"Replay batch is on {batch.device}, but IQL is on {self.device}.")
        if not batch.actions.is_floating_point():
            raise ValueError("IQL requires floating continuous actions.")

    def _training_actions(self, batch: ReplayBatch) -> tuple[torch.Tensor, bool]:
        """Return the action whose transition economics produced ``batch.rewards``.

        A constrained environment records both the policy request and the
        allocation/order that was actually executed.  Q(s, a) must use the
        latter or it learns the wrong action/reward identity.  Requested actions
        remain available for diagnostics and for explicitly unconstrained data.
        """

        if self.config.action_source == "executed_if_available" and batch.executed_actions is not None:
            return batch.executed_actions, True
        return batch.actions, False

    def _behavior_actions(
        self,
        distribution: ActionDistribution,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare logged actions for the actor likelihood without changing Q inputs.

        Dirichlet densities are defined on the open simplex, while real target-
        weight logs commonly contain exact zero weights.  Additive smoothing is
        therefore applied only to the behavior-cloning likelihood, only on the
        declared active support.  The returned distance makes this approximation
        observable in training metrics.
        """

        is_simplex = isinstance(distribution, MaskedDirichlet) or (
            isinstance(distribution, MixtureActionDistribution)
            and distribution.action_kind == "simplex"
        )
        if not is_simplex:
            return actions, torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
        if isinstance(distribution, MaskedDirichlet):
            mask = distribution.mask
        else:
            assert isinstance(distribution, MixtureActionDistribution)
            mixture_mask = distribution.action_mask
            assert mixture_mask is not None
            mask = mixture_mask
        if actions.shape != mask.shape:
            raise ValueError("Logged simplex actions do not match the Dirichlet action schema.")
        tolerance = 1e-6
        if bool((actions[~mask].abs() > tolerance).any().item()):
            raise ValueError("Logged simplex actions allocate weight to a masked dimension.")
        if bool((actions[mask] < -tolerance).any().item()):
            raise ValueError("Logged simplex actions contain a materially negative active weight.")
        active = torch.where(mask, actions.clamp_min(0.0), torch.zeros_like(actions))
        totals = active.sum(dim=-1, keepdim=True)
        if bool((totals <= 0).any().item()):
            raise ValueError("Every logged simplex action must have positive active mass.")
        if not bool(torch.allclose(totals, torch.ones_like(totals), atol=tolerance, rtol=0.0)):
            raise ValueError("Every logged simplex action must sum to one before likelihood smoothing.")
        active = active / totals
        epsilon = self.config.simplex_behavior_smoothing
        if epsilon > 0:
            active = torch.where(mask, active + epsilon, torch.zeros_like(active))
            active = active / active.sum(dim=-1, keepdim=True)
        distance = (active - actions).abs().sum(dim=-1)
        return active, distance

    @torch.no_grad()
    def _conservative_target(self, batch: ReplayBatch) -> tuple[torch.Tensor, torch.Tensor]:
        base_target = batch.rewards + batch.discounts * self.model.value(batch.next_observations)
        if not self.transforms:
            return base_target, torch.zeros_like(base_target)

        candidates: list[torch.Tensor] = [base_target]
        row_identity_key = "__rl_quant_iql_transform_row_identity__"
        while row_identity_key in batch.extras:
            row_identity_key += "_"
        row_identity = torch.arange(batch.batch_size, dtype=torch.long, device=batch.device)
        tagged = replace(
            batch,
            extras={**batch.extras, row_identity_key: row_identity},
            _validate_values=False,
        )
        for transform in self.transforms:
            transformed = transform(tagged)
            if transformed.batch_size != batch.batch_size or transformed.device != batch.device:
                raise ValueError("Robust transforms must preserve replay batch size and device.")
            transformed_identity = transformed.extras.get(row_identity_key)
            if transformed_identity is not row_identity:
                raise ValueError(
                    "Robust transforms must preserve exact replay row identity and order."
                )
            changes_target = (
                not torch.equal(transformed.rewards, batch.rewards)
                or not torch.equal(transformed.discounts, batch.discounts)
                or any(
                    not torch.equal(transformed.next_observations[name], batch.next_observations[name])
                    for name in batch.next_observations
                )
            )
            # A declared next-state transform may be numerically neutral on a
            # particular batch (for example, sign-reversing an all-zero
            # terminal feature). It is still a valid scenario. A transform
            # with no target-relevant declaration/change is configuration
            # drift and must not masquerade as robustness.
            if not changes_target and getattr(transform, "transform_next", None) is not True:
                raise ValueError(
                    "A robust IQL target transform must change reward, discount, or next observation; "
                    "current-observation-only transforms do not affect this Bellman target."
                )
            valid_target = (
                torch.isfinite(transformed.rewards).all()
                & torch.isfinite(transformed.discounts).all()
                & ((transformed.discounts >= 0) & (transformed.discounts <= 1)).all()
                & (transformed.discounts[transformed.terminated] == 0).all()
            )
            for value in transformed.next_observations.values():
                if value.is_floating_point():
                    valid_target = valid_target & torch.isfinite(value).all()
            if not bool(valid_target.item()):
                raise ValueError("Robust transform produced an invalid Bellman-target field.")
            next_value = self.model.value(transformed.next_observations)
            candidates.append(transformed.rewards + transformed.discounts * next_value)
        stacked = torch.stack(candidates, dim=0)
        return stacked.min(dim=0).values, stacked.max(dim=0).values - stacked.min(dim=0).values

    def _clip_grad(self, parameters: list[nn.Parameter]) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )

    @torch.no_grad()
    def _update_target(self) -> None:
        tau = self.config.target_tau
        for target, source in zip(self.target_model.parameters(), self.model.parameters(), strict=True):
            target.lerp_(source, tau)
        for target_buffer, source_buffer in zip(
            self.target_model.buffers(), self.model.buffers(), strict=True
        ):
            target_buffer.copy_(source_buffer)

    def update(self, batch: ReplayBatch) -> Mapping[str, MetricValue]:
        if not isinstance(batch, ReplayBatch):
            raise TypeError("ImplicitQLearning.update expects a ReplayBatch.")
        self._validate_batch(batch)
        self.model.train(True)
        training_actions, used_executed_actions = self._training_actions(batch)
        # Validate behavior support before any optimizer mutates state. This is
        # especially important for offline simplex logs: corrupt weights must not
        # update the critics and only fail later in the actor likelihood.
        distribution = self.model.distribution(batch.observations, batch.action_masks)
        behavior_actions, behavior_smoothing_l1 = self._behavior_actions(distribution, training_actions)
        # Validate likelihood support before critic/value state can change.
        # Sparse simplex logs with smoothing disabled must fail atomically.
        behavior_log_prob = distribution.log_prob(behavior_actions)
        if behavior_log_prob.shape != batch.rewards.shape or not bool(
            torch.isfinite(behavior_log_prob).all().item()
        ):
            raise ValueError("IQL behavior log probability must be a finite batch vector.")

        with torch.no_grad():
            q_target, transform_spread = self._conservative_target(batch)
        q_values = self.model.q_values(batch.observations, training_actions)
        if len(q_values) < 2 or any(value.shape != batch.rewards.shape for value in q_values):
            raise ValueError("IQL needs at least two critic batch vectors.")
        critic_loss = torch.stack([F.mse_loss(value, q_target) for value in q_values]).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._clip_grad(self.model.critic_parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            target_q_values = torch.stack(
                self.target_model.q_values(batch.observations, training_actions), dim=0
            )
            target_q_min = target_q_values.min(dim=0).values
            critic_uncertainty = target_q_values.std(dim=0, unbiased=False)
            conservative_q = target_q_min - self.config.critic_uncertainty_penalty * critic_uncertainty
        value = self.model.value(batch.observations)
        advantage = conservative_q - value
        expectile_weight = torch.where(
            advantage > 0,
            torch.full_like(advantage, self.config.expectile),
            torch.full_like(advantage, 1.0 - self.config.expectile),
        )
        value_loss = (expectile_weight * advantage.square()).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = self._clip_grad(self.model.value_parameters())
        self.value_optimizer.step()

        with torch.no_grad():
            refreshed_value = self.model.value(batch.observations)
            actor_advantage = conservative_q - refreshed_value
            advantage_weight = torch.exp(
                self.config.advantage_temperature * actor_advantage
            ).clamp(max=self.config.max_advantage_weight)
        actor_loss = -(advantage_weight * behavior_log_prob).mean()
        router_entropy = torch.zeros((), dtype=actor_loss.dtype, device=actor_loss.device)
        router_balance = torch.zeros_like(router_entropy)
        router_min_utilization = torch.zeros_like(router_entropy)
        if isinstance(distribution, MixtureActionDistribution):
            router_entropy = distribution.router.entropy.mean()
            utilization = distribution.router_probabilities.mean(dim=0)
            uniform = torch.full_like(utilization, 1.0 / utilization.numel())
            router_balance = (utilization - uniform).square().sum()
            router_min_utilization = utilization.min()
            actor_loss = (
                actor_loss
                + self.config.router_balance_coefficient * router_balance
                - self.config.router_entropy_coefficient * router_entropy
            )
        entropy = distribution.entropy().mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip_grad(self.model.actor_parameters())
        self.actor_optimizer.step()

        self._update_target()
        self.update_count += 1
        tensor_metrics = {
            "critic_loss": critic_loss.detach(),
            "value_loss": value_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "entropy": entropy.detach(),
            "q_target_mean": q_target.mean(),
            "q_min_mean": target_q_min.mean(),
            "value_mean": value.detach().mean(),
            "advantage_mean": actor_advantage.mean(),
            "advantage_weight_mean": advantage_weight.mean(),
            "critic_uncertainty_mean": critic_uncertainty.mean(),
            "transform_target_spread_mean": transform_spread.mean(),
            "behavior_smoothing_l1_mean": behavior_smoothing_l1.mean(),
            "router_entropy": router_entropy.detach(),
            "router_balance_loss": router_balance.detach(),
            "router_min_utilization": router_min_utilization.detach(),
            "action_projection_l1_mean": (
                torch.zeros((), dtype=batch.actions.dtype, device=batch.device)
                if batch.executed_actions is None
                else (batch.executed_actions - batch.actions).abs().reshape(batch.batch_size, -1).sum(dim=-1).mean()
            ),
            "critic_grad_norm": critic_grad_norm.detach(),
            "value_grad_norm": value_grad_norm.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
        }
        metric_names = tuple(tensor_metrics)
        # One device-to-host transfer avoids synchronizing once per diagnostic
        # in this update hot path.
        metric_values = torch.stack(tuple(tensor_metrics.values())).to(dtype=torch.float64).cpu()
        if not bool(torch.isfinite(metric_values).all().item()):
            raise FloatingPointError("IQL produced a non-finite metric.")
        return dict(zip(metric_names, metric_values.tolist(), strict=True)) | {
            "update_count": self.update_count,
            "robust_transform_count": len(self.transforms),
            "critic_uses_executed_actions": int(used_executed_actions),
        }

    def _transform_fingerprints(self) -> tuple[str, ...]:
        fingerprints: list[str] = []
        for transform in self.transforms:
            type_name = f"{type(transform).__module__}.{type(transform).__qualname__}"
            if is_dataclass(transform) and not isinstance(transform, type):
                payload: Any = asdict(transform)
            else:
                provider = getattr(transform, "checkpoint_fingerprint", None)
                if not callable(provider):
                    raise TypeError(
                        f"Robust transform {type_name} is not checkpoint-stable; use a dataclass or "
                        "implement checkpoint_fingerprint()."
                    )
                payload = provider()
            try:
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Robust transform {type_name} has a non-serializable fingerprint.") from exc
            fingerprints.append(f"{type_name}:{encoded}")
        return tuple(fingerprints)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "config": asdict(self.config),
            "transforms": self._transform_fingerprints(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "model", "target_model", "actor_optimizer", "critic_optimizer", "value_optimizer",
            "config", "transforms", "update_count",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"IQL checkpoint is missing fields: {sorted(missing)}.")
        if state["config"] != asdict(self.config):
            raise ValueError("IQL checkpoint config differs from the active configuration.")
        if tuple(state["transforms"]) != self._transform_fingerprints():
            raise ValueError("IQL checkpoint robust transforms differ from the active transforms.")
        update_count_value = state["update_count"]
        if (
            isinstance(update_count_value, bool)
            or not isinstance(update_count_value, int)
            or update_count_value < 0
        ):
            raise ValueError("IQL update_count must be a nonnegative integer.")
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.update_count = update_count_value

    def train(self, mode: bool = True) -> ImplicitQLearning:
        self.model.train(mode)
        self.target_model.eval()
        return self
