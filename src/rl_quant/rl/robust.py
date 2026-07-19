"""Distribution-shift transforms and uncertainty-aware deployment helpers.

These utilities encode the transferable part of regime-robust trading research
without importing market concepts into the RL core.  A domain adapter defines
economically plausible transforms; an offline algorithm evaluates conservative
targets across the original and transformed transition batches.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Protocol, runtime_checkable

import torch

from rl_quant.rl.replay import ReplayBatch
from rl_quant.rl.types import ActionBatch


@runtime_checkable
class TransitionTransform(Protocol):
    """Deterministically stress an offline transition batch.

    Implementations must preserve batch size, device, action schema, and point-
    in-time meaning.  A transformed target is a robustness stress, never an
    excuse to manufacture observations from the held-out test period.
    """

    def __call__(self, batch: ReplayBatch) -> ReplayBatch: ...


@dataclass(frozen=True)
class ObservationAffineTransform:
    """Apply a finite affine stress to selected current/next observation fields."""

    keys: tuple[str, ...]
    scale: float = 1.0
    shift: float = 0.0
    transform_current: bool = True
    transform_next: bool = True

    def __post_init__(self) -> None:
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("Observation transform keys must be unique and non-empty.")
        if not math.isfinite(self.scale) or not math.isfinite(self.shift):
            raise ValueError("Observation transform scale/shift must be finite.")
        if not self.transform_current and not self.transform_next:
            raise ValueError("ObservationAffineTransform must transform at least one side.")

    def _apply(self, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(self.keys) - set(values)
        if missing:
            raise ValueError(f"Observation transform keys are missing: {sorted(missing)}.")
        output = dict(values)
        for key in self.keys:
            value = output[key]
            if not value.is_floating_point():
                raise ValueError(f"Observation transform field {key!r} must be floating point.")
            transformed = value * self.scale + self.shift
            if not bool(torch.isfinite(transformed).all().item()):
                raise ValueError(f"Observation transform field {key!r} produced non-finite values.")
            output[key] = transformed
        return output

    def __call__(self, batch: ReplayBatch) -> ReplayBatch:
        return replace(
            batch,
            observations=self._apply(batch.observations) if self.transform_current else batch.observations,
            next_observations=self._apply(batch.next_observations) if self.transform_next else batch.next_observations,
            _validate_values=False,
        )


@dataclass(frozen=True)
class AdverseRewardTransform:
    """Subtract a fixed nonnegative return-unit stress from every transition."""

    penalty: float
    component_name: str = "robustness_stress"

    def __post_init__(self) -> None:
        if not math.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("Adverse reward penalty must be finite and nonnegative.")
        if not self.component_name:
            raise ValueError("component_name cannot be empty.")

    def __call__(self, batch: ReplayBatch) -> ReplayBatch:
        penalty = torch.full_like(batch.rewards, self.penalty)
        components = dict(batch.reward_components)
        if self.component_name in components:
            raise ValueError(f"Reward component {self.component_name!r} already exists.")
        components[self.component_name] = penalty
        stressed_reward = batch.rewards - penalty
        if not bool(torch.isfinite(stressed_reward).all().item()):
            raise ValueError("Adverse reward transform produced non-finite values.")
        return replace(
            batch,
            rewards=stressed_reward,
            reward_components=components,
            _validate_values=False,
        )


@dataclass(frozen=True)
class AbstentionResult:
    action: ActionBatch
    abstained: torch.Tensor


@dataclass(frozen=True)
class UncertaintyAbstention:
    """Replace uncertain actions with a caller-supplied safe action.

    This helper is for deterministic evaluation/shadow/deployment.  Replacing a
    stochastic sample changes its behavior density, so the returned ActionBatch
    deliberately clears ``log_prob`` and ``entropy``.  It must not be fed into
    an on-policy update as if it were sampled from the original policy.
    """

    threshold: float
    fallback_action: torch.Tensor

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Uncertainty threshold must be finite and nonnegative.")
        if self.fallback_action.ndim < 1:
            raise ValueError("fallback_action must describe at least one action dimension.")
        if self.fallback_action.is_floating_point() and not bool(torch.isfinite(self.fallback_action).all().item()):
            raise ValueError("fallback_action must be finite.")

    def apply(self, action: ActionBatch, uncertainty: torch.Tensor) -> AbstentionResult:
        if uncertainty.shape != (action.batch_size,) or uncertainty.device != action.device:
            raise ValueError("uncertainty must be a batch vector on the action device.")
        if not uncertainty.is_floating_point() or not bool(torch.isfinite(uncertainty).all().item()):
            raise ValueError("uncertainty must be finite floating point.")
        if self.fallback_action.shape == action.action.shape[1:]:
            fallback = self.fallback_action.to(action.device, dtype=action.action.dtype)
            fallback = fallback.expand_as(action.action)
        elif self.fallback_action.shape == action.action.shape:
            fallback = self.fallback_action.to(action.device, dtype=action.action.dtype)
        else:
            raise ValueError(
                "fallback_action must match the action event shape or the complete batched action shape."
            )
        abstained = uncertainty > self.threshold
        selector = abstained.view(action.batch_size, *([1] * (action.action.ndim - 1)))
        selected = torch.where(selector, fallback, action.action)
        extras = dict(action.extras)
        extras["uncertainty"] = uncertainty
        extras["abstained"] = abstained
        return AbstentionResult(
            action=ActionBatch(
                action=selected,
                recurrent_state=action.recurrent_state,
                extras=extras,
            ),
            abstained=abstained,
        )
