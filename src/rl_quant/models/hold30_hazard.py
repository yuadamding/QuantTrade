"""Versioned hazard transforms and diagnostics for Hold-30 policies.

The v2/v3 contracts use the historical closed-interval hard clip.  Later
research generations can opt into the smooth ``12*tanh(raw/12)`` transform
without changing those frozen semantics.  Exact holding is represented by a
separate mixture probability, rather than requiring a learned continuous
hazard logit to saturate at an endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

HOLD30_HAZARD_MIN = -12.0
HOLD30_HAZARD_MAX = 12.0
HOLD30_HAZARD_BOUND_MODES = ("hard_clip", "smooth_tanh")
Hold30HazardBoundMode = Literal["hard_clip", "smooth_tanh"]


def straight_through_exact_hold_decision(
    hold_logit: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return a hard forward hold decision with a sigmoid gradient surrogate.

    A finite sigmoid probability is never exactly one, so multiplying release
    by ``1-p`` would only attenuate trading.  M03R instead executes the hard
    branch ``1[sigmoid(logit) >= threshold]``.  The straight-through term keeps
    the sigmoid derivative for optimization while the economic forward path is
    exactly zero release whenever the branch is active.
    """

    if not hold_logit.is_floating_point() or not bool(torch.isfinite(hold_logit).all()):
        raise ValueError("hold_logit must be finite and floating point")
    if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be finite and lie in (0,1)")
    probability = torch.sigmoid(hold_logit)
    hard = (probability >= float(threshold)).to(dtype=probability.dtype)
    return (probability - probability.detach()) + hard


def _clip_with_zero_boundary_gradient(
    value: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Closed-interval clip whose derivative is zero at and beyond endpoints."""

    lo = value.new_tensor(lower)
    hi = value.new_tensor(upper)
    return torch.where(value <= lo, lo, torch.where(value >= hi, hi, value))


def clip_hold30_hazard_residual(raw_hazard: torch.Tensor) -> torch.Tensor:
    """Apply the frozen v2/v3 hard bound in ``[-12, 12]``."""

    if not raw_hazard.is_floating_point():
        raise TypeError("raw_hazard must be a floating-point tensor")
    return _clip_with_zero_boundary_gradient(
        raw_hazard,
        HOLD30_HAZARD_MIN,
        HOLD30_HAZARD_MAX,
    )


def smooth_hold30_hazard_residual(raw_hazard: torch.Tensor) -> torch.Tensor:
    """Smoothly bound a learned residual while retaining finite output limits.

    Unlike the historical hard clip, this transform has nonzero mathematical
    derivative for every finite input.  Exact zero-release behavior remains a
    separate mixture action and is not approximated by driving ``raw_hazard``
    toward negative infinity.
    """

    if not raw_hazard.is_floating_point():
        raise TypeError("raw_hazard must be a floating-point tensor")
    scale = raw_hazard.new_tensor(HOLD30_HAZARD_MAX)
    return scale * torch.tanh(raw_hazard / scale)


def bound_hold30_hazard_residual(
    raw_hazard: torch.Tensor,
    *,
    mode: Hold30HazardBoundMode = "hard_clip",
) -> torch.Tensor:
    """Apply one explicitly named residual transform.

    The default is intentionally the frozen hard clip so existing checkpoints,
    state dictionaries, and v3 result identities remain unchanged.
    """

    if mode == "hard_clip":
        return clip_hold30_hazard_residual(raw_hazard)
    if mode == "smooth_tanh":
        return smooth_hold30_hazard_residual(raw_hazard)
    raise ValueError(
        f"hazard bound mode must be one of {HOLD30_HAZARD_BOUND_MODES}; got {mode!r}"
    )


@dataclass(frozen=True, slots=True)
class Hold30HazardTelemetry:
    """Detached, JSON-safe diagnostics for one raw hazard tensor."""

    bound_mode: str
    observation_count: int
    raw_min: float
    raw_q01: float
    raw_median: float
    raw_q99: float
    raw_max: float
    bounded_min: float
    bounded_max: float
    fraction_raw_at_or_below_min: float
    fraction_raw_at_or_above_max: float
    transform_nonzero_gradient_fraction: float
    exact_hold_probability_mean: float | None = None
    exact_hold_probability_near_zero_fraction: float | None = None
    exact_hold_probability_near_one_fraction: float | None = None

    def as_dict(self) -> dict[str, str | int | float | None]:
        """Return stable receipt-friendly field names with explicit meaning."""

        return {
            "hazard_bound_mode": self.bound_mode,
            "hazard_observation_count": self.observation_count,
            "raw_hazard_min": self.raw_min,
            "raw_hazard_q01": self.raw_q01,
            "raw_hazard_median": self.raw_median,
            "raw_hazard_q99": self.raw_q99,
            "raw_hazard_max": self.raw_max,
            "bounded_hazard_min": self.bounded_min,
            "bounded_hazard_max": self.bounded_max,
            "fraction_raw_hazard_at_or_below_minus_12": (
                self.fraction_raw_at_or_below_min
            ),
            "fraction_raw_hazard_at_or_above_plus_12": (
                self.fraction_raw_at_or_above_max
            ),
            "hazard_transform_nonzero_gradient_fraction": (
                self.transform_nonzero_gradient_fraction
            ),
            "exact_hold_probability_mean": self.exact_hold_probability_mean,
            "exact_hold_probability_near_zero_fraction": (
                self.exact_hold_probability_near_zero_fraction
            ),
            "exact_hold_probability_near_one_fraction": (
                self.exact_hold_probability_near_one_fraction
            ),
        }


def hold30_hazard_telemetry(
    raw_hazard: torch.Tensor,
    *,
    mode: Hold30HazardBoundMode,
    eligible: torch.Tensor | None = None,
    exact_hold_probability: torch.Tensor | None = None,
) -> Hold30HazardTelemetry:
    """Measure saturation and usable transform gradients on eligible assets.

    ``eligible`` should exclude CASH and unavailable assets so their deliberate
    exact-hold sentinels do not inflate saturation statistics.  This helper is
    diagnostic only: all values are detached before aggregation.
    """

    if not raw_hazard.is_floating_point() or not bool(torch.isfinite(raw_hazard).all()):
        raise ValueError("raw_hazard must be finite and floating point")
    if eligible is None:
        eligible = torch.ones_like(raw_hazard, dtype=torch.bool)
    if eligible.shape != raw_hazard.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be boolean with the raw_hazard shape")
    selected = raw_hazard.detach()[eligible]
    if selected.numel() == 0:
        raise ValueError("hazard telemetry requires at least one eligible value")

    bounded = bound_hold30_hazard_residual(selected, mode=mode)
    if mode == "hard_clip":
        nonzero = (selected > HOLD30_HAZARD_MIN) & (selected < HOLD30_HAZARD_MAX)
    elif mode == "smooth_tanh":
        derivative = 1.0 - torch.tanh(selected / HOLD30_HAZARD_MAX).square()
        nonzero = derivative > 0
    else:  # fail before producing partially meaningful diagnostics
        raise ValueError(
            f"hazard bound mode must be one of {HOLD30_HAZARD_BOUND_MODES}; got {mode!r}"
        )

    hold_mean: float | None = None
    hold_zero: float | None = None
    hold_one: float | None = None
    if exact_hold_probability is not None:
        if (
            exact_hold_probability.shape != raw_hazard.shape
            or not exact_hold_probability.is_floating_point()
            or not bool(torch.isfinite(exact_hold_probability).all())
            or bool(((exact_hold_probability < 0) | (exact_hold_probability > 1)).any())
        ):
            raise ValueError(
                "exact_hold_probability must be finite, floating point, in [0,1], "
                "and match raw_hazard"
            )
        selected_hold = exact_hold_probability.detach()[eligible]
        hold_mean = float(selected_hold.mean().item())
        hold_zero = float((selected_hold <= 1e-6).to(torch.float64).mean().item())
        hold_one = float((selected_hold >= 1.0 - 1e-6).to(torch.float64).mean().item())

    quantiles = torch.quantile(
        selected.to(dtype=torch.float64),
        torch.tensor([0.01, 0.5, 0.99], device=selected.device, dtype=torch.float64),
    )
    return Hold30HazardTelemetry(
        bound_mode=mode,
        observation_count=selected.numel(),
        raw_min=float(selected.min().item()),
        raw_q01=float(quantiles[0].item()),
        raw_median=float(quantiles[1].item()),
        raw_q99=float(quantiles[2].item()),
        raw_max=float(selected.max().item()),
        bounded_min=float(bounded.min().item()),
        bounded_max=float(bounded.max().item()),
        fraction_raw_at_or_below_min=float(
            (selected <= HOLD30_HAZARD_MIN).to(torch.float64).mean().item()
        ),
        fraction_raw_at_or_above_max=float(
            (selected >= HOLD30_HAZARD_MAX).to(torch.float64).mean().item()
        ),
        transform_nonzero_gradient_fraction=float(
            nonzero.to(torch.float64).mean().item()
        ),
        exact_hold_probability_mean=hold_mean,
        exact_hold_probability_near_zero_fraction=hold_zero,
        exact_hold_probability_near_one_fraction=hold_one,
    )


__all__ = [
    "HOLD30_HAZARD_BOUND_MODES",
    "HOLD30_HAZARD_MAX",
    "HOLD30_HAZARD_MIN",
    "Hold30HazardBoundMode",
    "Hold30HazardTelemetry",
    "bound_hold30_hazard_residual",
    "clip_hold30_hazard_residual",
    "hold30_hazard_telemetry",
    "smooth_hold30_hazard_residual",
    "straight_through_exact_hold_decision",
]
