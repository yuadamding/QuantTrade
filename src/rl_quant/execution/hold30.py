"""Deterministic Hold-30 portfolio construction.

The actor emits intent at the decision timestamp.  This module applies that
already-pending intent to the future, fill-time repaired book.  It owns no
neural state and receives only fill-time masks and risk ceilings, keeping the
observation/execution chronology explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.models.daily_policy import hold30_proposed_release, hold30_release_hazard
from rl_quant.protocol.hold30_alpha_v3 import HOLD30_ALPHA_TE_TARGET_ANNUAL

HOLD30_MAX_STOCK_WEIGHT = 0.01
HOLD30_MAX_DISCRETIONARY_TURNOVER = 0.10
HOLD30_EXPOSURE_BAND = 0.02
HOLD30_EXPOSURE_STEP = 0.10


def _require_matrix(name: str, value: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point [batch, asset] tensor")
    if reference is not None and value.shape != reference.shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} must match {tuple(reference.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _require_vector(name: str, value: torch.Tensor, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tuple(value.shape) != (batch_size,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite [batch] tensor")
    return value


def _zero_boundary_clip(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    lo = value.new_tensor(lower)
    hi = value.new_tensor(upper)
    return torch.where(value <= lo, lo, torch.where(value >= hi, hi, value))


def centered_benchmark_tilt(
    scores: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    *,
    cash_index: int = 0,
    score_clip: float = 2.0,
) -> torch.Tensor:
    """Return the RFC benchmark-relative entry direction over risky assets."""

    scores = _require_matrix("scores", scores)
    benchmark_weights = _require_matrix("benchmark_weights", benchmark_weights, scores)
    if trade_mask.shape != scores.shape:
        raise ValueError("trade_mask must match scores")
    mask = trade_mask.bool().clone()
    mask[:, cash_index] = False
    count = mask.sum(-1).clamp_min(1).to(scores.dtype)
    mean = torch.where(mask, scores, torch.zeros_like(scores)).sum(-1) / count
    centered = _zero_boundary_clip(scores - mean.unsqueeze(-1), -score_clip, score_clip)
    unnormalized = torch.where(
        mask,
        benchmark_weights.clamp_min(0.0) * torch.exp(centered),
        torch.zeros_like(scores),
    )
    total = unnormalized.sum(-1, keepdim=True)
    return torch.where(total > 0, unnormalized / total.clamp_min(1e-18), torch.zeros_like(unnormalized))


@dataclass(frozen=True)
class Hold30ExposureEnvelope:
    cap: torch.Tensor
    hard_max: torch.Tensor
    band_min: torch.Tensor
    band_max: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor


def hold30_exposure_envelope(
    repaired_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    *,
    cash_index: int = 0,
) -> Hold30ExposureEnvelope:
    """Fill-time cap and hold-preserving C1 exposure envelope."""

    repaired_weights = _require_matrix("repaired_weights", repaired_weights)
    benchmark_weights = _require_matrix("benchmark_weights", benchmark_weights, repaired_weights)
    risk_asset_caps = _require_matrix("risk_asset_caps", risk_asset_caps, repaired_weights)
    batch, assets = repaired_weights.shape
    risk_gross_max = _require_vector("risk_gross_max", risk_gross_max, batch, repaired_weights)
    if not 0 <= cash_index < assets:
        raise ValueError("cash_index is outside the asset axis")
    risky = torch.ones_like(repaired_weights, dtype=torch.bool)
    risky[:, cash_index] = False
    cap = torch.where(
        risky,
        torch.minimum(risk_asset_caps.clamp_min(0.0), repaired_weights.new_tensor(HOLD30_MAX_STOCK_WEIGHT)),
        torch.zeros_like(repaired_weights),
    )
    hard_max = torch.minimum(
        torch.ones_like(risk_gross_max),
        torch.minimum(risk_gross_max.clamp_min(0.0), cap.sum(-1)),
    )
    benchmark_gross = torch.where(risky, benchmark_weights, torch.zeros_like(benchmark_weights)).sum(-1)
    repaired_gross = torch.where(risky, repaired_weights, torch.zeros_like(repaired_weights)).sum(-1)
    band_min = torch.minimum((benchmark_gross - HOLD30_EXPOSURE_BAND).clamp_min(0.0), hard_max)
    band_max = torch.minimum(benchmark_gross + HOLD30_EXPOSURE_BAND, hard_max)
    minimum = torch.minimum(repaired_gross, band_min)
    maximum = torch.maximum(repaired_gross, band_max)
    return Hold30ExposureEnvelope(cap, hard_max, band_min, band_max, minimum, maximum)


def _waterfill_one(
    requested_mass: torch.Tensor,
    direction: torch.Tensor,
    capacity: torch.Tensor,
    *,
    iterations: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise differentiable capped proportional allocation for one row."""

    valid = (direction > 0) & (capacity > 0)
    available = torch.where(valid, capacity, torch.zeros_like(capacity)).sum()
    effective = torch.minimum(requested_mass.clamp_min(0.0), available)
    if float(effective.detach()) <= 1e-15:
        return torch.zeros_like(direction), effective
    if float((available - effective).detach().abs()) <= 1e-12:
        # The v1 custom contract has a zero backward on the full-cap branch.
        return torch.where(valid, capacity.detach(), torch.zeros_like(capacity)), effective

    safe_direction = torch.where(valid, direction, torch.ones_like(direction))
    ratios = torch.where(valid, capacity / safe_direction, torch.zeros_like(capacity))
    lo = torch.zeros((), dtype=direction.dtype, device=direction.device)
    hi = ratios.max()
    # The comparisons select the local active set and intentionally carry no
    # gradient.  The final value is recomputed analytically below.
    with torch.no_grad():
        lo_search = lo.detach()
        hi_search = hi.detach()
        for _ in range(iterations):
            mid = (lo_search + hi_search) * 0.5
            mass = torch.minimum(capacity.detach(), mid * direction.detach())
            mass = torch.where(valid, mass, torch.zeros_like(mass)).sum()
            if bool(mass >= effective.detach()):
                hi_search = mid
            else:
                lo_search = mid
        alpha_search = hi_search
        proposed = alpha_search * direction.detach()
        tie_tol = max(1e-12, float(effective.detach().abs()) * 1e-12)
        tie = valid & ((proposed - capacity.detach()).abs() <= tie_tol)
        saturated = valid & (proposed > capacity.detach() + tie_tol)
        unsaturated = valid & ~saturated & ~tie

    fixed = saturated | tie
    saturated_mass = torch.where(fixed, capacity, torch.zeros_like(capacity)).sum()
    unsaturated_direction = torch.where(unsaturated, direction, torch.zeros_like(direction)).sum()
    alpha = torch.where(
        unsaturated_direction > 0,
        (effective - saturated_mass) / unsaturated_direction.clamp_min(1e-18),
        torch.zeros_like(effective),
    )
    allocated = torch.where(saturated, capacity, torch.zeros_like(capacity))
    allocated = allocated + torch.where(unsaturated, alpha * direction, torch.zeros_like(direction))
    allocated = allocated + torch.where(tie, capacity.detach(), torch.zeros_like(capacity))
    return allocated, effective


def capped_waterfill(
    requested_mass: torch.Tensor,
    direction: torch.Tensor,
    capacity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch wrapper around the frozen 100-iteration water-fill contract."""

    direction = _require_matrix("direction", direction)
    capacity = _require_matrix("capacity", capacity, direction)
    requested_mass = _require_vector("requested_mass", requested_mass, direction.shape[0], direction)
    allocations = []
    effective = []
    for row in range(direction.shape[0]):
        allocation, mass = _waterfill_one(requested_mass[row], direction[row], capacity[row])
        allocations.append(allocation)
        effective.append(mass)
    return torch.stack(allocations), torch.stack(effective)


@dataclass(frozen=True)
class Hold30BuiltAction:
    target_weights: torch.Tensor
    requested_delta: torch.Tensor
    constructed_delta: torch.Tensor
    requested_turnover: torch.Tensor
    constructed_turnover: torch.Tensor
    desired_risky_exposure: torch.Tensor
    proposed_release_by_age: torch.Tensor
    proposed_release: torch.Tensor
    capacity_shortfall: torch.Tensor


def _turnover(delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * delta.abs().sum(-1)


def _turnover_limit(delta: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    requested = _turnover(delta)
    scale = torch.where(requested > limit, delta.new_tensor(limit) / requested.clamp_min(1e-18), 1.0)
    constructed = scale.unsqueeze(-1) * delta
    return constructed, requested, _turnover(constructed)


def build_h2_hold30_action(
    repaired_weights: torch.Tensor,
    age_notional: torch.Tensor,
    entry_scores: torch.Tensor,
    hazard_residual: torch.Tensor,
    exposure_residual: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    *,
    exact_hold_probability: torch.Tensor | None = None,
    cash_index: int = 0,
    max_turnover: float = HOLD30_MAX_DISCRETIONARY_TURNOVER,
    exposure_step: float = HOLD30_EXPOSURE_STEP,
) -> Hold30BuiltAction:
    """Apply one H2 intent to a fill-time repaired portfolio."""

    repaired_weights = _require_matrix("repaired_weights", repaired_weights)
    entry_scores = _require_matrix("entry_scores", entry_scores, repaired_weights)
    hazard_residual = _require_matrix("hazard_residual", hazard_residual, repaired_weights)
    if age_notional.shape != (*repaired_weights.shape, 61):
        raise ValueError("age_notional must be [batch, asset, 61]")
    if not age_notional.is_floating_point() or not bool(torch.isfinite(age_notional).all()):
        raise ValueError("age_notional must be finite and floating point")
    if bool((age_notional < 0).any()):
        raise ValueError("age_notional cannot be negative")
    batch, _assets = repaired_weights.shape
    exposure_residual = _require_vector("exposure_residual", exposure_residual, batch, repaired_weights)
    if trade_mask.shape != repaired_weights.shape:
        raise ValueError("trade_mask must match repaired_weights")
    envelope = hold30_exposure_envelope(
        repaired_weights,
        benchmark_weights,
        risk_asset_caps,
        risk_gross_max,
        cash_index=cash_index,
    )
    direction = centered_benchmark_tilt(
        entry_scores,
        benchmark_weights,
        trade_mask,
        cash_index=cash_index,
    )
    risky = torch.ones_like(repaired_weights, dtype=torch.bool)
    risky[:, cash_index] = False
    ages = torch.arange(61, device=age_notional.device, dtype=age_notional.dtype)
    if exact_hold_probability is not None:
        exact_hold_probability = _require_matrix(
            "exact_hold_probability",
            exact_hold_probability,
            repaired_weights,
        )
        if bool(
            ((exact_hold_probability < 0) | (exact_hold_probability > 1)).any()
        ):
            raise ValueError("exact_hold_probability must lie in [0,1]")
    hazards = hold30_release_hazard(
        ages,
        hazard_residual.unsqueeze(-1).to(age_notional.dtype),
        exact_hold_probability=(
            None
            if exact_hold_probability is None
            else exact_hold_probability.unsqueeze(-1).to(age_notional.dtype)
        ),
    )
    release_by_age = age_notional * hazards
    release_by_age = torch.where(risky.unsqueeze(-1), release_by_age, torch.zeros_like(release_by_age))
    proposed_release = hold30_proposed_release(
        age_notional,
        hazard_residual.to(age_notional.dtype),
        exact_hold_probability=(
            None
            if exact_hold_probability is None
            else exact_hold_probability.to(age_notional.dtype)
        ),
    )
    proposed_release = torch.where(risky, proposed_release, torch.zeros_like(proposed_release))
    proposed_release = torch.minimum(proposed_release, repaired_weights.clamp_min(0.0))
    retained = torch.where(risky, repaired_weights - proposed_release, torch.zeros_like(repaired_weights)).clamp_min(0.0)
    retained_gross = retained.sum(-1)
    repaired_gross = torch.where(risky, repaired_weights, torch.zeros_like(repaired_weights)).sum(-1)
    if (
        isinstance(exposure_step, bool)
        or not isinstance(exposure_step, (int, float))
        or not torch.isfinite(torch.tensor(float(exposure_step)))
        or float(exposure_step) < 0
    ):
        raise ValueError("exposure_step must be a finite non-negative scalar")
    desired_gross = repaired_gross + float(exposure_step) * torch.tanh(
        exposure_residual
    )
    desired_gross = torch.maximum(envelope.minimum, torch.minimum(desired_gross, envelope.maximum))

    buy_mass = (desired_gross - retained_gross).clamp_min(0.0)
    capacity = (envelope.cap - retained).clamp_min(0.0)
    buys, effective = capped_waterfill(buy_mass, direction, capacity)
    buy_case = desired_gross >= retained_gross
    sell_mass = (retained_gross - desired_gross).clamp_min(0.0)
    proportional = torch.where(
        retained_gross.unsqueeze(-1) > 0,
        retained * (1.0 - sell_mass / retained_gross.clamp_min(1e-18)).unsqueeze(-1),
        torch.zeros_like(retained),
    )
    desired_risky = torch.where(buy_case.unsqueeze(-1), retained + buys, proportional)
    desired_risky = torch.where(risky, desired_risky, torch.zeros_like(desired_risky))
    desired = desired_risky.clone()
    desired[:, cash_index] = 1.0 - desired_risky.sum(-1)
    requested_delta = desired - repaired_weights
    constructed_delta, requested_turnover, constructed_turnover = _turnover_limit(requested_delta, max_turnover)
    target = repaired_weights + constructed_delta
    shortfall = torch.where(buy_case, buy_mass - effective, torch.zeros_like(buy_mass)).clamp_min(0.0)
    return Hold30BuiltAction(
        target_weights=target,
        requested_delta=requested_delta,
        constructed_delta=constructed_delta,
        requested_turnover=requested_turnover,
        constructed_turnover=constructed_turnover,
        desired_risky_exposure=desired_gross,
        proposed_release_by_age=release_by_age,
        proposed_release=proposed_release,
        capacity_shortfall=shortfall,
    )


def build_alpha_hold30_action(
    repaired_weights: torch.Tensor,
    age_notional: torch.Tensor,
    risk_adjusted_score: torch.Tensor,
    hazard_residual: torch.Tensor,
    active_risk_scale: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    *,
    exact_hold_probability: torch.Tensor | None = None,
    total_risk_overlay: torch.Tensor | None = None,
    total_risk_step: float | None = None,
    te_target: float = HOLD30_ALPHA_TE_TARGET_ANNUAL,
    cash_index: int = 0,
    max_turnover: float = HOLD30_MAX_DISCRETIONARY_TURNOVER,
) -> Hold30BuiltAction:
    """Construct one v3 alpha action through the age-aware H2 ledger.

    The active-risk scale changes only the benchmark-relative cross-sectional
    score.  Canonical alpha variants pass no total-risk overlay and therefore
    cannot time gross market exposure.  a06 must supply both its raw overlay
    and an explicit manifest coefficient ``total_risk_step``.
    """

    repaired_weights = _require_matrix("repaired_weights", repaired_weights)
    risk_adjusted_score = _require_matrix(
        "risk_adjusted_score", risk_adjusted_score, repaired_weights
    )
    batch = repaired_weights.shape[0]
    active_risk_scale = _require_vector(
        "active_risk_scale", active_risk_scale, batch, repaired_weights
    )
    if bool((active_risk_scale < 0).any()):
        raise ValueError("active_risk_scale must be nonnegative")
    if not 0 < float(te_target) < 1:
        raise ValueError("te_target must lie in (0,1)")
    scaled_score = risk_adjusted_score * (
        active_risk_scale / float(te_target)
    ).unsqueeze(-1)
    if total_risk_overlay is None:
        if total_risk_step is not None:
            raise ValueError(
                "total_risk_step is forbidden without the registered a06 overlay"
            )
        exposure = repaired_weights.new_zeros((batch,))
        exposure_step = 0.0
    else:
        exposure = _require_vector(
            "total_risk_overlay", total_risk_overlay, batch, repaired_weights
        )
        if total_risk_step is None:
            raise ValueError(
                "a06 total-risk overlay requires an explicit manifest step"
            )
        exposure_step = float(total_risk_step)
    return build_h2_hold30_action(
        repaired_weights,
        age_notional,
        scaled_score,
        hazard_residual,
        exposure,
        benchmark_weights,
        trade_mask,
        risk_asset_caps,
        risk_gross_max,
        exact_hold_probability=exact_hold_probability,
        cash_index=cash_index,
        max_turnover=max_turnover,
        exposure_step=exposure_step,
    )


def build_scalar_gate_hold30_action(
    repaired_weights: torch.Tensor,
    target_logits: torch.Tensor,
    gate: torch.Tensor,
    benchmark_weights: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    *,
    cash_index: int = 0,
    temperature: float = 0.5,
    max_turnover: float = HOLD30_MAX_DISCRETIONARY_TURNOVER,
) -> Hold30BuiltAction:
    """Common corrected H0/H1 fill-time adapter."""

    repaired_weights = _require_matrix("repaired_weights", repaired_weights)
    target_logits = _require_matrix("target_logits", target_logits, repaired_weights)
    gate = _require_vector("gate", gate, repaired_weights.shape[0], repaired_weights).clamp(0.0, 1.0)
    envelope = hold30_exposure_envelope(
        repaired_weights, benchmark_weights, risk_asset_caps, risk_gross_max, cash_index=cash_index
    )
    mask = trade_mask.bool().clone()
    mask[:, cash_index] = True
    count = mask.sum(-1).clamp_min(1).to(target_logits.dtype)
    mean = torch.where(mask, target_logits, torch.zeros_like(target_logits)).sum(-1) / count
    centered = _zero_boundary_clip(target_logits - mean.unsqueeze(-1), -8.0, 8.0)
    logits = centered / temperature
    logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    absolute_direction = torch.softmax(logits, dim=-1)
    risky_direction = absolute_direction.clone()
    risky_direction[:, cash_index] = 0.0
    risky_total = risky_direction.sum(-1, keepdim=True)
    risky_direction = torch.where(
        risky_total > 0, risky_direction / risky_total.clamp_min(1e-18), torch.zeros_like(risky_direction)
    )
    risky_mask = torch.ones_like(absolute_direction, dtype=torch.bool)
    risky_mask[:, cash_index] = False
    requested_gross = torch.where(
        risky_mask, absolute_direction, torch.zeros_like(absolute_direction)
    ).sum(-1)
    desired_gross = torch.maximum(envelope.band_min, torch.minimum(requested_gross, envelope.band_max))
    buys, effective = capped_waterfill(desired_gross, risky_direction, envelope.cap)
    desired = buys.clone()
    desired[:, cash_index] = 1.0 - buys.sum(-1)
    interpolated = (1.0 - gate.unsqueeze(-1)) * repaired_weights + gate.unsqueeze(-1) * desired
    requested_delta = interpolated - repaired_weights
    constructed_delta, requested_turnover, constructed_turnover = _turnover_limit(requested_delta, max_turnover)
    target = repaired_weights + constructed_delta
    return Hold30BuiltAction(
        target_weights=target,
        requested_delta=requested_delta,
        constructed_delta=constructed_delta,
        requested_turnover=requested_turnover,
        constructed_turnover=constructed_turnover,
        desired_risky_exposure=desired_gross,
        proposed_release_by_age=repaired_weights.new_zeros((*repaired_weights.shape, 61)),
        proposed_release=repaired_weights.new_zeros(repaired_weights.shape),
        capacity_shortfall=(desired_gross - effective).clamp_min(0.0),
    )


__all__ = [
    "HOLD30_EXPOSURE_BAND",
    "HOLD30_EXPOSURE_STEP",
    "HOLD30_MAX_DISCRETIONARY_TURNOVER",
    "HOLD30_MAX_STOCK_WEIGHT",
    "Hold30BuiltAction",
    "Hold30ExposureEnvelope",
    "build_alpha_hold30_action",
    "build_h2_hold30_action",
    "build_scalar_gate_hold30_action",
    "capped_waterfill",
    "centered_benchmark_tilt",
    "hold30_exposure_envelope",
]
