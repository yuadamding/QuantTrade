"""Validated tensor batches passed between RL environments and algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import torch


def _move_tensor(
    value: torch.Tensor,
    device: torch.device | str,
    *,
    non_blocking: bool,
) -> torch.Tensor:
    # Batch containers mix floating data with boolean masks and integer indices.
    # A device-only helper cannot accidentally cast a mask into an invalid dtype.
    return value.to(device=device, non_blocking=non_blocking)


def _detach_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in values.items()}


def _move_mapping(
    values: Mapping[str, torch.Tensor],
    device: torch.device | str,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    return {
        name: _move_tensor(value, device, non_blocking=non_blocking)
        for name, value in values.items()
    }


def _require_batch_vector(value: torch.Tensor, batch_size: int, *, name: str, dtype: torch.dtype | None = None) -> None:
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},); got {tuple(value.shape)}.")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}; got {value.dtype}.")


@dataclass(frozen=True)
class ObservationBatch:
    """Named observation tensors with an optional feasible-action mask.

    Every tensor has the same leading environment batch dimension.  The container
    is intentionally agnostic to market data, images, language, or other domains.
    """

    tensors: Mapping[str, torch.Tensor]
    action_mask: torch.Tensor | None = None
    episode_start: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("ObservationBatch.tensors cannot be empty.")
        batch_size: int | None = None
        device: torch.device | None = None
        for name, value in self.tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Observation tensor names must be non-empty strings.")
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                raise ValueError(f"Observation {name!r} needs a leading batch dimension.")
            if batch_size is None:
                batch_size = value.shape[0]
                device = value.device
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"Observation {name!r} has batch size {value.shape[0]}, expected {batch_size}."
                )
            if value.device != device:
                raise ValueError(f"Observation {name!r} is on {value.device}, expected {device}.")
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f"Observation {name!r} must be finite; encode missingness with explicit validity channels."
                )
        assert batch_size is not None
        if self.action_mask is not None:
            if self.action_mask.ndim < 2 or self.action_mask.shape[0] != batch_size:
                raise ValueError("action_mask needs shape [batch, ...].")
            if self.action_mask.dtype != torch.bool:
                raise ValueError(f"action_mask must be torch.bool; got {self.action_mask.dtype}.")
            if self.action_mask.device != device:
                raise ValueError(f"action_mask is on {self.action_mask.device}, expected {device}.")
        if self.episode_start is not None:
            _require_batch_vector(self.episode_start, batch_size, name="episode_start", dtype=torch.bool)
            if self.episode_start.device != device:
                raise ValueError(f"episode_start is on {self.episode_start.device}, expected {device}.")

    @property
    def batch_size(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    @property
    def device(self) -> torch.device:
        return next(iter(self.tensors.values())).device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> ObservationBatch:
        return ObservationBatch(
            tensors=_move_mapping(self.tensors, device, non_blocking=non_blocking),
            action_mask=None
            if self.action_mask is None
            else _move_tensor(self.action_mask, device, non_blocking=non_blocking),
            episode_start=None
            if self.episode_start is None
            else _move_tensor(self.episode_start, device, non_blocking=non_blocking),
        )

    def detach(self) -> ObservationBatch:
        return ObservationBatch(
            tensors=_detach_mapping(self.tensors),
            action_mask=None if self.action_mask is None else self.action_mask.detach(),
            episode_start=None if self.episode_start is None else self.episode_start.detach(),
        )


@dataclass(frozen=True)
class ActionBatch:
    """Policy output before an environment applies domain constraints."""

    action: torch.Tensor
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    recurrent_state: Mapping[str, torch.Tensor] = field(default_factory=dict)
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.ndim == 0:
            raise ValueError("ActionBatch.action needs a leading batch dimension.")
        batch_size = self.action.shape[0]
        device = self.action.device
        if self.action.is_floating_point() and not bool(torch.isfinite(self.action).all().item()):
            raise ValueError("ActionBatch.action must contain only finite values.")
        for name, value in (("log_prob", self.log_prob), ("entropy", self.entropy)):
            if value is not None:
                _require_batch_vector(value, batch_size, name=name)
                if value.device != device or not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{name} must be finite and on {device}.")
        for group_name, values in (("recurrent_state", self.recurrent_state), ("extras", self.extras)):
            for name, value in values.items():
                if value.ndim == 0 or value.shape[0] != batch_size:
                    raise ValueError(f"{group_name}[{name!r}] needs leading batch size {batch_size}.")
                if value.device != device:
                    raise ValueError(f"{group_name}[{name!r}] is on {value.device}, expected {device}.")
                if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{group_name}[{name!r}] must be finite.")

    @property
    def batch_size(self) -> int:
        return self.action.shape[0]

    @property
    def device(self) -> torch.device:
        return self.action.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> ActionBatch:
        return ActionBatch(
            action=_move_tensor(self.action, device, non_blocking=non_blocking),
            log_prob=None
            if self.log_prob is None
            else _move_tensor(self.log_prob, device, non_blocking=non_blocking),
            entropy=None
            if self.entropy is None
            else _move_tensor(self.entropy, device, non_blocking=non_blocking),
            recurrent_state=_move_mapping(self.recurrent_state, device, non_blocking=non_blocking),
            extras=_move_mapping(self.extras, device, non_blocking=non_blocking),
        )

    def detach(self) -> ActionBatch:
        return ActionBatch(
            action=self.action.detach(),
            log_prob=None if self.log_prob is None else self.log_prob.detach(),
            entropy=None if self.entropy is None else self.entropy.detach(),
            recurrent_state=_detach_mapping(self.recurrent_state),
            extras=_detach_mapping(self.extras),
        )


@dataclass(frozen=True)
class RewardComponents:
    """Additive return-unit reward ledger.

    Cost and penalty fields are nonnegative and subtracted from ``gross_return``.
    ``extras`` are signed domain-specific reward terms and are added to the total.
    """

    gross_return: torch.Tensor
    execution_cost: torch.Tensor
    impact_cost: torch.Tensor
    risk_penalty: torch.Tensor
    constraint_penalty: torch.Tensor
    liquidation_cost: torch.Tensor
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.gross_return.shape
        device = self.gross_return.device
        reserved = {
            "gross_return",
            "execution_cost",
            "impact_cost",
            "risk_penalty",
            "constraint_penalty",
            "liquidation_cost",
        }
        collision = reserved.intersection(self.extras)
        if collision:
            raise ValueError(f"Extra reward component names collide with standard fields: {sorted(collision)}.")
        if self.gross_return.ndim != 1:
            raise ValueError(f"Reward components must be batch vectors; got {shape}.")
        fields = {
            "gross_return": self.gross_return,
            "execution_cost": self.execution_cost,
            "impact_cost": self.impact_cost,
            "risk_penalty": self.risk_penalty,
            "constraint_penalty": self.constraint_penalty,
            "liquidation_cost": self.liquidation_cost,
            **self.extras,
        }
        for name, value in fields.items():
            if value.shape != shape or value.device != device:
                raise ValueError(f"Reward component {name!r} must have shape {shape} on {device}.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Reward component {name!r} must be finite.")
        for name in ("execution_cost", "impact_cost", "risk_penalty", "constraint_penalty", "liquidation_cost"):
            if bool((getattr(self, name) < 0).any().item()):
                raise ValueError(f"Reward cost component {name!r} must be nonnegative.")

    @property
    def total(self) -> torch.Tensor:
        reward = (
            self.gross_return
            - self.execution_cost
            - self.impact_cost
            - self.risk_penalty
            - self.constraint_penalty
            - self.liquidation_cost
        )
        for value in self.extras.values():
            reward = reward + value
        return reward

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "gross_return": self.gross_return,
            "execution_cost": self.execution_cost,
            "impact_cost": self.impact_cost,
            "risk_penalty": self.risk_penalty,
            "constraint_penalty": self.constraint_penalty,
            "liquidation_cost": self.liquidation_cost,
            **self.extras,
        }

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> RewardComponents:
        return replace(
            self,
            gross_return=_move_tensor(self.gross_return, device, non_blocking=non_blocking),
            execution_cost=_move_tensor(self.execution_cost, device, non_blocking=non_blocking),
            impact_cost=_move_tensor(self.impact_cost, device, non_blocking=non_blocking),
            risk_penalty=_move_tensor(self.risk_penalty, device, non_blocking=non_blocking),
            constraint_penalty=_move_tensor(self.constraint_penalty, device, non_blocking=non_blocking),
            liquidation_cost=_move_tensor(self.liquidation_cost, device, non_blocking=non_blocking),
            extras=_move_mapping(self.extras, device, non_blocking=non_blocking),
        )

    def detach(self) -> RewardComponents:
        return replace(
            self,
            gross_return=self.gross_return.detach(),
            execution_cost=self.execution_cost.detach(),
            impact_cost=self.impact_cost.detach(),
            risk_penalty=self.risk_penalty.detach(),
            constraint_penalty=self.constraint_penalty.detach(),
            liquidation_cost=self.liquidation_cost.detach(),
            extras=_detach_mapping(self.extras),
        )


@dataclass(frozen=True)
class TransitionBatch:
    """One vectorized environment transition with explicit terminal semantics."""

    observation: ObservationBatch
    action: ActionBatch
    executed_action: torch.Tensor
    rewards: RewardComponents
    next_observation: ObservationBatch
    terminated: torch.Tensor
    truncated: torch.Tensor
    discount: torch.Tensor
    info: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch_size = self.observation.batch_size
        device = self.observation.device
        if self.next_observation.batch_size != batch_size or self.action.batch_size != batch_size:
            raise ValueError("Observation, action, and next observation batch sizes must match.")
        if self.next_observation.device != device or self.action.device != device:
            raise ValueError("Transition tensors must share one device.")
        if self.executed_action.shape != self.action.action.shape or self.executed_action.device != device:
            raise ValueError("executed_action must match the requested action's shape and device.")
        if self.executed_action.is_floating_point() and not bool(torch.isfinite(self.executed_action).all().item()):
            raise ValueError("executed_action must be finite.")
        if self.rewards.total.shape != (batch_size,) or self.rewards.total.device != device:
            raise ValueError("Rewards must match the transition batch and device.")
        for name, value in (("terminated", self.terminated), ("truncated", self.truncated)):
            _require_batch_vector(value, batch_size, name=name, dtype=torch.bool)
            if value.device != device:
                raise ValueError(f"{name} is on {value.device}, expected {device}.")
        if bool((self.terminated & self.truncated).any().item()):
            raise ValueError("A transition cannot be both terminated and truncated.")
        _require_batch_vector(self.discount, batch_size, name="discount")
        if not self.discount.is_floating_point():
            raise ValueError("discount must use a floating-point dtype.")
        if self.discount.device != device or not bool(torch.isfinite(self.discount).all().item()):
            raise ValueError("discount must be finite and share the transition device.")
        if bool(((self.discount < 0) | (self.discount > 1)).any().item()):
            raise ValueError("discount must lie in [0, 1].")
        if bool((self.discount[self.terminated] != 0).any().item()):
            raise ValueError("Terminated transitions must have zero discount.")
        for name, value in self.info.items():
            if value.ndim == 0 or value.shape[0] != batch_size or value.device != device:
                raise ValueError(f"info[{name!r}] needs leading batch size {batch_size} on {device}.")
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"info[{name!r}] must be finite.")

    @property
    def reward(self) -> torch.Tensor:
        return self.rewards.total

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> TransitionBatch:
        return TransitionBatch(
            observation=self.observation.to(device, non_blocking=non_blocking),
            action=self.action.to(device, non_blocking=non_blocking),
            executed_action=_move_tensor(self.executed_action, device, non_blocking=non_blocking),
            rewards=self.rewards.to(device, non_blocking=non_blocking),
            next_observation=self.next_observation.to(device, non_blocking=non_blocking),
            terminated=_move_tensor(self.terminated, device, non_blocking=non_blocking),
            truncated=_move_tensor(self.truncated, device, non_blocking=non_blocking),
            discount=_move_tensor(self.discount, device, non_blocking=non_blocking),
            info=_move_mapping(self.info, device, non_blocking=non_blocking),
        )

    def detach(self) -> TransitionBatch:
        return TransitionBatch(
            observation=self.observation.detach(),
            action=self.action.detach(),
            executed_action=self.executed_action.detach(),
            rewards=self.rewards.detach(),
            next_observation=self.next_observation.detach(),
            terminated=self.terminated.detach(),
            truncated=self.truncated.detach(),
            discount=self.discount.detach(),
            info=_detach_mapping(self.info),
        )
