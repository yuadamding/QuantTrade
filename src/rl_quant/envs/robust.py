"""Market-specific transition stresses for conservative offline targets.

The generic RL layer defines :class:`TransitionTransform` but deliberately has
no opinion about market feature or reward-component names.  This module keeps
those names caller-declared and applies only deterministic, point-in-time
stresses to an already logged :class:`ReplayBatch`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch

from rl_quant.rl.replay import ReplayBatch
from rl_quant.rl.robust import TransitionTransform


_EXECUTION_COST_COMPONENTS = frozenset({"execution_cost", "impact_cost", "liquidation_cost"})


def _validate_names(names: tuple[str, ...], *, label: str) -> None:
    if not isinstance(names, tuple) or not names:
        raise ValueError(f"{label} must be a non-empty tuple of names.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicate names.")


def _validate_named_factors(
    values: tuple[tuple[str, float], ...],
    *,
    label: str,
    minimum: float,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} must be a non-empty tuple of (name, factor) pairs.")
    names: list[str] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{label} entries must be immutable (name, factor) pairs.")
        name, factor = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} names must be non-empty strings.")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ValueError(f"{label} factors must be real numbers.")
        numeric = float(factor)
        if not math.isfinite(numeric) or numeric < minimum:
            raise ValueError(f"{label} factors must be finite and >= {minimum}.")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicate names.")


def _require_observation_fields(
    values: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"{label} observation fields are missing: {sorted(missing)}.")
    for name in names:
        value = values[name]
        if not value.is_floating_point():
            raise ValueError(f"{label} observation field {name!r} must be floating point.")


def _check_finite(value: torch.Tensor, *, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} produced non-finite values.")


@dataclass(frozen=True)
class TrendReturnFeatureReversal:
    """Reverse the sign of explicitly named trend/return observation fields.

    No name is inferred from spelling or position.  Every requested field must
    exist and be floating point on each transformed side of the transition.
    """

    feature_names: tuple[str, ...]
    transform_current: bool = True
    transform_next: bool = True

    def __post_init__(self) -> None:
        _validate_names(self.feature_names, label="feature_names")
        if not self.transform_current and not self.transform_next:
            raise ValueError("TrendReturnFeatureReversal must transform at least one transition side.")

    def _apply(self, values: Mapping[str, torch.Tensor], *, label: str) -> dict[str, torch.Tensor]:
        _require_observation_fields(values, self.feature_names, label=label)
        output = dict(values)
        for name in self.feature_names:
            output[name] = -values[name]
        return output

    def __call__(self, batch: ReplayBatch) -> ReplayBatch:
        return replace(
            batch,
            observations=(
                self._apply(batch.observations, label="current")
                if self.transform_current
                else batch.observations
            ),
            next_observations=(
                self._apply(batch.next_observations, label="next")
                if self.transform_next
                else batch.next_observations
            ),
            _validate_values=False,
        )


@dataclass(frozen=True)
class LiquidityCostStress:
    """Scale named liquidity features and named nonnegative cost components.

    Feature factors are caller-defined because low volume, high spread, and a
    high illiquidity score have different adverse directions.  Cost factors
    must be at least one.  The total replay reward is reduced by exactly the
    increase in the declared cost components, leaving all other reward terms
    unchanged.
    """

    liquidity_feature_factors: tuple[tuple[str, float], ...]
    cost_component_factors: tuple[tuple[str, float], ...]
    transform_current: bool = True
    transform_next: bool = True

    def __post_init__(self) -> None:
        _validate_named_factors(
            self.liquidity_feature_factors,
            label="liquidity_feature_factors",
            minimum=0.0,
        )
        _validate_named_factors(
            self.cost_component_factors,
            label="cost_component_factors",
            minimum=1.0,
        )
        unsupported = set(self.cost_component_names) - _EXECUTION_COST_COMPONENTS
        if unsupported:
            raise ValueError(
                "cost_component_factors may name only execution_cost, impact_cost, or liquidation_cost; "
                f"got {sorted(unsupported)}."
            )
        if not self.transform_current and not self.transform_next:
            raise ValueError("LiquidityCostStress must transform at least one observation side.")
        if all(float(factor) == 1.0 for _, factor in self.liquidity_feature_factors) and all(
            float(factor) == 1.0 for _, factor in self.cost_component_factors
        ):
            raise ValueError("LiquidityCostStress cannot be a no-op.")

    @property
    def liquidity_feature_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.liquidity_feature_factors)

    @property
    def cost_component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.cost_component_factors)

    def _scale_features(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        label: str,
    ) -> dict[str, torch.Tensor]:
        _require_observation_fields(values, self.liquidity_feature_names, label=label)
        output = dict(values)
        for name, factor in self.liquidity_feature_factors:
            scaled = values[name] * float(factor)
            _check_finite(scaled, label=f"{label} liquidity field {name!r}")
            output[name] = scaled
        return output

    def _scale_costs(self, batch: ReplayBatch) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        missing = set(self.cost_component_names) - set(batch.reward_components)
        if missing:
            raise ValueError(f"Cost reward components are missing: {sorted(missing)}.")
        components = dict(batch.reward_components)
        extra_cost = torch.zeros_like(batch.rewards)
        for name, factor in self.cost_component_factors:
            original = batch.reward_components[name]
            if original.shape != batch.rewards.shape:
                raise ValueError(
                    f"Cost reward component {name!r} must have shape {tuple(batch.rewards.shape)}."
                )
            if not original.is_floating_point() or original.dtype != batch.rewards.dtype:
                raise ValueError(
                    f"Cost reward component {name!r} must use reward dtype {batch.rewards.dtype}."
                )
            if bool((original < 0).any().item()):
                raise ValueError(f"Cost reward component {name!r} must be nonnegative.")
            stressed = original * float(factor)
            _check_finite(stressed, label=f"cost reward component {name!r}")
            components[name] = stressed
            extra_cost = extra_cost + (stressed - original)
        return components, extra_cost

    def __call__(self, batch: ReplayBatch) -> ReplayBatch:
        components, extra_cost = self._scale_costs(batch)
        stressed_reward = batch.rewards - extra_cost
        _check_finite(stressed_reward, label="liquidity-stressed reward")
        return replace(
            batch,
            observations=(
                self._scale_features(batch.observations, label="current")
                if self.transform_current
                else batch.observations
            ),
            next_observations=(
                self._scale_features(batch.next_observations, label="next")
                if self.transform_next
                else batch.next_observations
            ),
            rewards=stressed_reward,
            reward_components=components,
            _validate_values=False,
        )


def _validate_transform_result(
    source: ReplayBatch,
    transformed: ReplayBatch,
    *,
    label: str,
) -> None:
    if not isinstance(transformed, ReplayBatch):
        raise TypeError(f"{label} did not return a ReplayBatch.")
    if transformed.batch_size != source.batch_size or transformed.device != source.device:
        raise ValueError(f"{label} changed replay batch size or device.")
    if transformed.actions.shape != source.actions.shape or transformed.actions.dtype != source.actions.dtype:
        raise ValueError(f"{label} changed the requested-action schema.")
    if set(transformed.observations) != set(source.observations):
        raise ValueError(f"{label} changed observation field names.")
    for name, original in source.observations.items():
        current = transformed.observations[name]
        next_value = transformed.next_observations[name]
        if current.shape != original.shape or current.dtype != original.dtype:
            raise ValueError(f"{label} changed observation schema for {name!r}.")
        if next_value.shape != original.shape or next_value.dtype != original.dtype:
            raise ValueError(f"{label} changed next-observation schema for {name!r}.")


@dataclass(frozen=True)
class SequentialTransitionTransform:
    """Apply transition transforms in order as one lower-envelope scenario."""

    transforms: tuple[TransitionTransform, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transforms, tuple) or not self.transforms:
            raise ValueError("SequentialTransitionTransform needs a non-empty transform tuple.")
        if any(not isinstance(transform, TransitionTransform) for transform in self.transforms):
            raise TypeError("Every sequential item must implement TransitionTransform.")

    def __call__(self, batch: ReplayBatch) -> ReplayBatch:
        transformed = batch
        for index, transform in enumerate(self.transforms):
            candidate = transform(transformed)
            _validate_transform_result(
                transformed,
                candidate,
                label=f"Sequential transform {index} ({type(transform).__name__})",
            )
            transformed = candidate
        return transformed


@dataclass(frozen=True)
class LowerEnvelopeScenario:
    """A stable scenario name paired with one transition transform."""

    name: str
    transform: TransitionTransform

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Lower-envelope scenario names must be non-empty strings.")
        if not isinstance(self.transform, TransitionTransform):
            raise TypeError("A lower-envelope scenario must implement TransitionTransform.")


@dataclass(frozen=True)
class LowerEnvelopeTransformSuite:
    """Named adverse scenarios whose transforms can be passed directly to IQL.

    IQL already includes the unmodified batch in its elementwise lower envelope;
    this suite therefore contains only stressed scenarios.
    """

    scenarios: tuple[LowerEnvelopeScenario, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValueError("LowerEnvelopeTransformSuite needs at least one scenario.")
        if any(not isinstance(scenario, LowerEnvelopeScenario) for scenario in self.scenarios):
            raise TypeError("Lower-envelope scenarios must be LowerEnvelopeScenario instances.")
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(set(names)) != len(names):
            raise ValueError("Lower-envelope scenario names must be unique.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(scenario.name for scenario in self.scenarios)

    @property
    def transforms(self) -> tuple[TransitionTransform, ...]:
        """Return the exact tuple accepted by ``ImplicitQLearning``."""

        return tuple(scenario.transform for scenario in self.scenarios)

    def apply(self, batch: ReplayBatch) -> dict[str, ReplayBatch]:
        """Materialize each named scenario without modifying ``batch``."""

        output: dict[str, ReplayBatch] = {}
        for scenario in self.scenarios:
            transformed = scenario.transform(batch)
            _validate_transform_result(batch, transformed, label=f"Scenario {scenario.name!r}")
            output[scenario.name] = transformed
        return output


def market_lower_envelope_suite(
    reversal: TrendReturnFeatureReversal,
    liquidity_cost: LiquidityCostStress,
    *,
    include_joint: bool = True,
) -> LowerEnvelopeTransformSuite:
    """Build reversal, liquidity/cost, and optionally joint market scenarios."""

    if not isinstance(reversal, TrendReturnFeatureReversal):
        raise TypeError("reversal must be a TrendReturnFeatureReversal.")
    if not isinstance(liquidity_cost, LiquidityCostStress):
        raise TypeError("liquidity_cost must be a LiquidityCostStress.")
    if not isinstance(include_joint, bool):
        raise TypeError("include_joint must be boolean.")
    scenarios = [
        LowerEnvelopeScenario("trend_return_reversal", reversal),
        LowerEnvelopeScenario("liquidity_cost_stress", liquidity_cost),
    ]
    if include_joint:
        scenarios.append(
            LowerEnvelopeScenario(
                "joint_reversal_liquidity_cost",
                SequentialTransitionTransform((reversal, liquidity_cost)),
            )
        )
    return LowerEnvelopeTransformSuite(tuple(scenarios))


__all__ = [
    "LiquidityCostStress",
    "LowerEnvelopeScenario",
    "LowerEnvelopeTransformSuite",
    "SequentialTransitionTransform",
    "TrendReturnFeatureReversal",
    "market_lower_envelope_suite",
]
