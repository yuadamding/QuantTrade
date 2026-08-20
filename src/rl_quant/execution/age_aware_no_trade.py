"""Soft age-aware replacement logic for deterministic forecast translation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


class AgeAwareNoTradeError(ValueError):
    """The replacement decision inputs are economically invalid."""


@dataclass(frozen=True, slots=True)
class AgeAwareNoTradeConfig:
    preferred_holding_sessions: int = 30
    young_position_penalty_return: float = 0.001
    downside_penalty: float = 1.0
    epistemic_penalty: float = 1.0

    def validate(self) -> None:
        if (
            isinstance(self.preferred_holding_sessions, bool)
            or not isinstance(self.preferred_holding_sessions, int)
            or self.preferred_holding_sessions <= 0
        ):
            raise AgeAwareNoTradeError("preferred holding period must be positive")
        for name in (
            "young_position_penalty_return",
            "downside_penalty",
            "epistemic_penalty",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise AgeAwareNoTradeError(f"{name} must be finite and nonnegative")

    def young_sale_penalty(self, age_sessions: int) -> float:
        self.validate()
        if isinstance(age_sessions, bool) or not isinstance(age_sessions, int) or age_sessions < 0:
            raise AgeAwareNoTradeError("position age must be nonnegative")
        remaining = max(self.preferred_holding_sessions - age_sessions, 0)
        fraction = remaining / self.preferred_holding_sessions
        return self.young_position_penalty_return * fraction * fraction


@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    expected_return: float
    downside_quantile: float
    epistemic_uncertainty: float

    def validate(self) -> None:
        for name in ("expected_return", "downside_quantile", "epistemic_uncertainty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AgeAwareNoTradeError(f"{name} must be finite")
        if self.epistemic_uncertainty < 0.0:
            raise AgeAwareNoTradeError("epistemic uncertainty cannot be negative")

    def conservative_score(self, config: AgeAwareNoTradeConfig) -> float:
        self.validate()
        config.validate()
        downside_width = max(self.expected_return - self.downside_quantile, 0.0)
        return (
            self.expected_return
            - config.downside_penalty * downside_width
            - config.epistemic_penalty * self.epistemic_uncertainty
        )


@dataclass(frozen=True, slots=True)
class ReplacementDecision:
    action: Literal["keep", "replace", "forced-exit"]
    replacement_advantage: float
    required_advantage: float
    young_sale_penalty: float


def evaluate_replacement(
    *,
    held: ForecastDistribution,
    candidate: ForecastDistribution,
    held_age_sessions: int,
    sell_cost_return: float,
    buy_cost_return: float,
    config: AgeAwareNoTradeConfig,
    forced_exit: bool = False,
) -> ReplacementDecision:
    """Replace only when forecast uplift clears cost, uncertainty, and age friction."""

    held.validate()
    candidate.validate()
    config.validate()
    for name, value in (
        ("sell cost", sell_cost_return),
        ("buy cost", buy_cost_return),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise AgeAwareNoTradeError(f"{name} must be finite and nonnegative")
    advantage = candidate.conservative_score(config) - held.conservative_score(config)
    if forced_exit:
        return ReplacementDecision(
            action="forced-exit",
            replacement_advantage=advantage,
            required_advantage=0.0,
            young_sale_penalty=0.0,
        )
    young_penalty = config.young_sale_penalty(held_age_sessions)
    required = sell_cost_return + buy_cost_return + young_penalty
    return ReplacementDecision(
        action="replace" if advantage > required else "keep",
        replacement_advantage=advantage,
        required_advantage=required,
        young_sale_penalty=young_penalty,
    )


__all__ = [
    "AgeAwareNoTradeConfig",
    "AgeAwareNoTradeError",
    "ForecastDistribution",
    "ReplacementDecision",
    "evaluate_replacement",
]
