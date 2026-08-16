"""Immutable soft holding-target specifications and release schedules.

The holding target is deliberately independent of prediction horizon, label
support, age-state capacity, and evaluation horizon.  New generic APIs use a
three-session neutral expected holding duration.  Historical Hold-30 APIs bind
the exact legacy age clock and therefore retain their previous behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import torch

DEFAULT_HOLD_TARGET_SESSIONS = 3
DEFAULT_HOLD_AGE_CAP_SESSIONS = 60
_GENERIC_HAZARD_RESIDUAL_MIN = -20.0
_GENERIC_HAZARD_RESIDUAL_MAX = 20.0
_LEGACY_HAZARD_RESIDUAL_MIN = -12.0
_LEGACY_HAZARD_RESIDUAL_MAX = 12.0
_CALIBRATION_ITERATIONS = 160

HoldPriorFamily = Literal["calibrated-logistic-v2", "legacy-hold30-v1"]
TerminalAgeMode = Literal["repeat-last-hazard"]


class HoldTargetProtocolError(ValueError):
    """A holding-target specification or schedule is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _neutral_hazard(
    age: int,
    *,
    location: float,
    width: float,
    residual_minimum: float,
) -> float:
    beta = -2.0 + (age - location) / width
    if residual_minimum == _LEGACY_HAZARD_RESIDUAL_MIN:
        # Match the historical tensor path's endpoint logit clamp exactly.
        p_min = _sigmoid(max(-20.0, min(20.0, beta + residual_minimum)))
        release = _sigmoid(max(-20.0, min(20.0, beta)))
        return (release - p_min) / (1.0 - p_min)
    # Stable form of the generic normalized sigmoid at residual zero:
    # (sigmoid(beta) - sigmoid(beta + minimum)) /
    # (1 - sigmoid(beta + minimum)).
    return (1.0 - math.exp(residual_minimum)) * _sigmoid(beta)


@lru_cache(maxsize=None)
def _survival_schedule_from_location(
    *,
    age_cap_sessions: int,
    location: float,
    width: float,
    residual_minimum: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    survival = 1.0
    hazards: list[float] = []
    weights: list[float] = []
    for age in range(1, age_cap_sessions + 1):
        weights.append(survival)
        hazard = _neutral_hazard(
            age,
            location=location,
            width=width,
            residual_minimum=residual_minimum,
        )
        hazards.append(hazard)
        survival *= 1.0 - hazard
    return tuple(hazards), tuple(weights)


def _runtime_expectation(
    hazards: tuple[float, ...],
    survival_weights: tuple[float, ...],
) -> tuple[float, float, float]:
    """Return terminal survival, terminal tail, and total runtime expectation."""

    if not hazards or len(hazards) != len(survival_weights):
        raise HoldTargetProtocolError("holding schedule is empty or misaligned")
    terminal_hazard = hazards[-1]
    survival_after_age_cap = survival_weights[-1] * (1.0 - terminal_hazard)
    if not 0.0 < terminal_hazard <= 1.0:
        raise HoldTargetProtocolError("terminal holding hazard is invalid")
    terminal_tail = survival_after_age_cap / terminal_hazard
    expectation = math.fsum(survival_weights) + terminal_tail
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (survival_after_age_cap, terminal_tail, expectation)
    ):
        raise HoldTargetProtocolError("holding runtime expectation is invalid")
    return survival_after_age_cap, terminal_tail, expectation


@lru_cache(maxsize=None)
def _calibrated_location(
    *,
    target_sessions: int,
    age_cap_sessions: int,
    width: float,
) -> float:
    lower = -float(age_cap_sessions) - 64.0 * width
    upper = 2.0 * float(age_cap_sessions) + 64.0 * width
    target = float(target_sessions)
    for _ in range(_CALIBRATION_ITERATIONS):
        midpoint = 0.5 * (lower + upper)
        hazards, survival = _survival_schedule_from_location(
            age_cap_sessions=age_cap_sessions,
            location=midpoint,
            width=width,
            residual_minimum=_GENERIC_HAZARD_RESIDUAL_MIN,
        )
        _, _, expectation = _runtime_expectation(hazards, survival)
        if expectation < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


@dataclass(frozen=True, slots=True)
class HoldTargetSpec:
    """One run-level soft holding target.

    ``target_sessions`` is the requested neutral expected duration in earned
    trading sessions.  It is not a minimum hold or a forced expiry.
    """

    target_sessions: int = DEFAULT_HOLD_TARGET_SESSIONS
    age_cap_sessions: int = DEFAULT_HOLD_AGE_CAP_SESSIONS
    prior_family: HoldPriorFamily = "calibrated-logistic-v2"
    release_transition_width_sessions: float = 4.0
    hard_minimum_hold: bool = False
    terminal_age_mode: TerminalAgeMode = "repeat-last-hazard"

    def validate(self) -> None:
        if (
            isinstance(self.target_sessions, bool)
            or not isinstance(self.target_sessions, int)
            or isinstance(self.age_cap_sessions, bool)
            or not isinstance(self.age_cap_sessions, int)
            or not 1 <= self.target_sessions <= self.age_cap_sessions
            or self.age_cap_sessions != DEFAULT_HOLD_AGE_CAP_SESSIONS
            or not math.isfinite(self.release_transition_width_sessions)
            or self.release_transition_width_sessions <= 0.0
            or self.prior_family not in {"calibrated-logistic-v2", "legacy-hold30-v1"}
            or self.hard_minimum_hold
            or self.terminal_age_mode != "repeat-last-hazard"
        ):
            raise HoldTargetProtocolError("holding-target specification is invalid")
        if self.prior_family == "legacy-hold30-v1" and (
            self.target_sessions != 30
            or self.age_cap_sessions != 60
            or self.release_transition_width_sessions != 4.0
        ):
            raise HoldTargetProtocolError("legacy Hold-30 specification drifted")

    @property
    def calibrated_release_location(self) -> float:
        self.validate()
        if self.prior_family == "legacy-hold30-v1":
            return 30.0
        return _calibrated_location(
            target_sessions=self.target_sessions,
            age_cap_sessions=self.age_cap_sessions,
            width=self.release_transition_width_sessions,
        )

    @property
    def neutral_release_hazards(self) -> tuple[float, ...]:
        self.validate()
        residual_minimum = (
            _LEGACY_HAZARD_RESIDUAL_MIN
            if self.prior_family == "legacy-hold30-v1"
            else _GENERIC_HAZARD_RESIDUAL_MIN
        )
        hazards, _ = _survival_schedule_from_location(
            age_cap_sessions=self.age_cap_sessions,
            location=self.calibrated_release_location,
            width=self.release_transition_width_sessions,
            residual_minimum=residual_minimum,
        )
        return hazards

    @property
    def neutral_survival_weights(self) -> tuple[float, ...]:
        self.validate()
        residual_minimum = (
            _LEGACY_HAZARD_RESIDUAL_MIN
            if self.prior_family == "legacy-hold30-v1"
            else _GENERIC_HAZARD_RESIDUAL_MIN
        )
        _, survival = _survival_schedule_from_location(
            age_cap_sessions=self.age_cap_sessions,
            location=self.calibrated_release_location,
            width=self.release_transition_width_sessions,
            residual_minimum=residual_minimum,
        )
        return survival

    @property
    def expected_neutral_hold_sessions(self) -> float:
        return _runtime_expectation(
            self.neutral_release_hazards,
            self.neutral_survival_weights,
        )[2]

    @property
    def finite_support_expected_hold_sessions(self) -> float:
        return math.fsum(self.neutral_survival_weights)

    @property
    def survival_after_age_cap(self) -> float:
        return _runtime_expectation(
            self.neutral_release_hazards,
            self.neutral_survival_weights,
        )[0]

    @property
    def terminal_hazard(self) -> float:
        return self.neutral_release_hazards[-1]

    @property
    def terminal_expected_tail_sessions(self) -> float:
        return _runtime_expectation(
            self.neutral_release_hazards,
            self.neutral_survival_weights,
        )[1]

    @property
    def hazard_schedule_sha256(self) -> str:
        return _sha256(self.neutral_release_hazards)

    @property
    def survival_schedule_sha256(self) -> str:
        return _sha256((*self.neutral_survival_weights, self.survival_after_age_cap))

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                **asdict(self),
                "calibrated_release_location": self.calibrated_release_location,
                "finite_support_expected_hold_sessions": (
                    self.finite_support_expected_hold_sessions
                ),
                "survival_after_age_cap": self.survival_after_age_cap,
                "terminal_hazard": self.terminal_hazard,
                "terminal_expected_tail_sessions": (
                    self.terminal_expected_tail_sessions
                ),
                "runtime_expected_hold_sessions": (self.expected_neutral_hold_sessions),
                "hazard_schedule_sha256": self.hazard_schedule_sha256,
                "survival_schedule_sha256": self.survival_schedule_sha256,
            }
        )


DEFAULT_HOLD_TARGET_SPEC = HoldTargetSpec()
LEGACY_HOLD30_TARGET_SPEC = HoldTargetSpec(
    target_sessions=30,
    prior_family="legacy-hold30-v1",
)


def _clip_with_zero_boundary_gradient(
    value: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    lo = value.new_tensor(lower)
    hi = value.new_tensor(upper)
    return torch.where(value <= lo, lo, torch.where(value >= hi, hi, value))


def hold_release_hazard(
    age: torch.Tensor,
    learned_residual: torch.Tensor,
    *,
    hold_spec: HoldTargetSpec = DEFAULT_HOLD_TARGET_SPEC,
    exact_hold_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the normalized soft release hazard for one immutable target."""

    hold_spec.validate()
    if not learned_residual.is_floating_point():
        raise TypeError("learned_residual must be a floating-point tensor")
    age_value = age.to(device=learned_residual.device, dtype=learned_residual.dtype)
    beta = (
        -2.0
        + (
            age_value.clamp(min=0.0, max=float(hold_spec.age_cap_sessions))
            - hold_spec.calibrated_release_location
        )
        / hold_spec.release_transition_width_sessions
    )
    if hold_spec.prior_family == "legacy-hold30-v1":
        residual_minimum = _LEGACY_HAZARD_RESIDUAL_MIN
        residual_maximum = _LEGACY_HAZARD_RESIDUAL_MAX
    else:
        residual_minimum = _GENERIC_HAZARD_RESIDUAL_MIN
        residual_maximum = _GENERIC_HAZARD_RESIDUAL_MAX
    bounded = _clip_with_zero_boundary_gradient(
        learned_residual,
        residual_minimum,
        residual_maximum,
    )
    if hold_spec.prior_family == "legacy-hold30-v1":
        # Frozen historical arithmetic, including its endpoint logit clamp.
        p_min = torch.sigmoid(
            _clip_with_zero_boundary_gradient(
                beta + residual_minimum,
                -20.0,
                20.0,
            )
        )
        release = torch.sigmoid(
            _clip_with_zero_boundary_gradient(beta + bounded, -20.0, 20.0)
        )
        normalized = (release - p_min) / (1.0 - p_min)
    else:
        # Stable exact form of
        # (sigmoid(beta+r) - sigmoid(beta+minimum)) /
        # (1 - sigmoid(beta+minimum)).  It remains finite when both ordinary
        # sigmoid terms round to one and matches the calibration schedule.
        normalized = (
            1.0 - torch.exp(bounded.new_tensor(residual_minimum) - bounded)
        ) * torch.sigmoid(beta + bounded)
    if exact_hold_probability is None:
        return cast(torch.Tensor, normalized)
    if (
        not exact_hold_probability.is_floating_point()
        or not bool(torch.isfinite(exact_hold_probability).all())
        or bool(((exact_hold_probability < 0) | (exact_hold_probability > 1)).any())
    ):
        raise HoldTargetProtocolError(
            "exact_hold_probability must be finite and lie in [0,1]"
        )
    try:
        return cast(
            torch.Tensor,
            normalized
            * (
                1.0
                - exact_hold_probability.to(
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
            ),
        )
    except RuntimeError as error:
        raise HoldTargetProtocolError(
            "exact_hold_probability must broadcast with the release hazard"
        ) from error


def hold_survival_weights(
    *,
    hold_spec: HoldTargetSpec = DEFAULT_HOLD_TARGET_SPEC,
    support_sessions: int | None = None,
) -> torch.Tensor:
    """Return CPU float64 neutral survival weights for a declared support."""

    hold_spec.validate()
    support = (
        hold_spec.age_cap_sessions if support_sessions is None else support_sessions
    )
    if (
        isinstance(support, bool)
        or not isinstance(support, int)
        or not 1 <= support <= hold_spec.age_cap_sessions
    ):
        raise HoldTargetProtocolError("holding survival support is invalid")
    return torch.tensor(
        hold_spec.neutral_survival_weights[:support],
        dtype=torch.float64,
    )


__all__ = [
    "DEFAULT_HOLD_AGE_CAP_SESSIONS",
    "DEFAULT_HOLD_TARGET_SESSIONS",
    "DEFAULT_HOLD_TARGET_SPEC",
    "LEGACY_HOLD30_TARGET_SPEC",
    "HoldPriorFamily",
    "TerminalAgeMode",
    "HoldTargetProtocolError",
    "HoldTargetSpec",
    "hold_release_hazard",
    "hold_survival_weights",
]
