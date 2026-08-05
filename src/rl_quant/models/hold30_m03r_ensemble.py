"""M03R-only output-space ensemble for five seeded alpha policies.

The frozen V2/V3 ensemble remains in :mod:`rl_quant.models.hold30_ensemble`.
This module accepts only exact M03R artifact identities and only settings that
actually emit residual-alpha heads.  Seed outputs are combined before any
portfolio constraint is applied; the execution layer consequently has one,
and only one, requested portfolio to project.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_hazard import bound_hold30_hazard_residual
from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    validate_m03r_artifact_identity,
)

M03R_ENSEMBLE_MEMBER_COUNT = M03R_DESIGN.ensemble_execution.ensemble_member_count
M03R_ENTRY_SCORE_CLIP = 2.0


class M03REnsembleError(ValueError):
    """An M03R seed ensemble is incomplete, ambiguous, or misidentified."""


@dataclass(frozen=True, slots=True)
class M03REnsembleMember:
    """One seeded member's raw decision-time intent."""

    protocol_generation: str
    design_id: str
    setting_id: str
    seed: int
    intent: Hold30Intent


@dataclass(frozen=True, slots=True)
class M03REnsembleIntent:
    """The single output-space intent passed to M03R execution."""

    protocol_generation: str
    design_id: str
    setting_id: str
    ordered_seeds: tuple[int, ...]
    intent: Hold30Intent


def _required_matrix(
    members: Sequence[M03REnsembleMember],
    name: str,
    shape: tuple[int, int],
) -> torch.Tensor:
    values = [getattr(member.intent, name) for member in members]
    if any(value is None or tuple(value.shape) != shape for value in values):
        raise M03REnsembleError(f"every M03R member {name} must have shape {shape}")
    tensors = [value for value in values if value is not None]
    reference = tensors[0]
    if not all(
        value.device == reference.device
        and value.dtype == reference.dtype
        and value.is_floating_point()
        and bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise M03REnsembleError(
            f"M03R member {name} values must share device/dtype and be finite"
        )
    return torch.stack(tensors)


def _required_vector(
    members: Sequence[M03REnsembleMember],
    name: str,
    batch: int,
) -> torch.Tensor:
    values = [getattr(member.intent, name) for member in members]
    if any(value is None or tuple(value.shape) != (batch,) for value in values):
        raise M03REnsembleError(f"every M03R member {name} must have shape {(batch,)}")
    tensors = [value for value in values if value is not None]
    reference = tensors[0]
    if not all(
        value.device == reference.device
        and value.dtype == reference.dtype
        and value.is_floating_point()
        and bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise M03REnsembleError(
            f"M03R member {name} values must share device/dtype and be finite"
        )
    return torch.stack(tensors)


def _optional_matrix(
    members: Sequence[M03REnsembleMember],
    name: str,
    shape: tuple[int, int],
) -> torch.Tensor | None:
    values = [getattr(member.intent, name) for member in members]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise M03REnsembleError(
            f"M03R member {name} must be populated by every seed or none"
        )
    return _required_matrix(members, name, shape)


def _center_and_clip_member_scores(
    values: torch.Tensor,
    risky_available: torch.Tensor,
) -> torch.Tensor:
    expanded = risky_available.unsqueeze(0)
    count = expanded.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
    means = torch.where(expanded, values, torch.zeros_like(values)).sum(dim=-1)
    means = means / count
    centered = (values - means.unsqueeze(-1)).clamp(
        -M03R_ENTRY_SCORE_CLIP,
        M03R_ENTRY_SCORE_CLIP,
    )
    return torch.where(expanded, centered, torch.zeros_like(centered))


def aggregate_m03r_alpha_intents(
    members: Sequence[M03REnsembleMember],
    decision_available: torch.Tensor,
    *,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
    cash_index: int = 0,
) -> M03REnsembleIntent:
    """Apply the immutable M03R five-seed output-space aggregation rule.

    The rule is intentionally explicit: clipped cross-sectionally centered
    entry scores and alpha means are arithmetic means; downside, raw exit
    hazard, exact-hold mass, active-risk scale, and calibrated signal
    confidence use the five-member median. Auxiliary horizon means are
    arithmetic means. A06's isolated total-risk overlay uses the median; it is
    forbidden for every other M03R setting.
    """

    setting = validate_m03r_artifact_identity(
        protocol_generation=protocol_generation,
        design_id=design_id,
        setting_id=setting_id,
    )
    if not setting.residual_alpha_heads:
        raise M03REnsembleError(
            f"{setting_id} has no residual-alpha heads and cannot use the M03R alpha ensemble"
        )
    if len(members) != M03R_ENSEMBLE_MEMBER_COUNT:
        raise M03REnsembleError("M03R requires exactly five seeded members")
    for member in members:
        member_setting = validate_m03r_artifact_identity(
            protocol_generation=member.protocol_generation,
            design_id=member.design_id,
            setting_id=member.setting_id,
        )
        if member_setting.setting_id != setting.setting_id:
            raise M03REnsembleError(
                "every M03R ensemble member must carry the exact requested setting identity"
            )
    if decision_available.ndim != 2 or decision_available.dtype != torch.bool:
        raise M03REnsembleError(
            "decision_available must be a boolean [batch, asset] tensor"
        )
    batch, assets = decision_available.shape
    if not 0 <= cash_index < assets:
        raise M03REnsembleError("cash_index is outside the asset axis")
    if not bool(decision_available[:, cash_index].all()):
        raise M03REnsembleError("CASH must be available for every M03R decision")
    if any(
        isinstance(member.seed, bool) or not isinstance(member.seed, int)
        for member in members
    ):
        raise M03REnsembleError("every ensemble seed must be an integer")
    if len({member.seed for member in members}) != M03R_ENSEMBLE_MEMBER_COUNT:
        raise M03REnsembleError("M03R ensemble seeds must be distinct")
    ordered = tuple(sorted(members, key=lambda member: member.seed))
    shape = (batch, assets)
    risky = decision_available.clone()
    risky[:, cash_index] = False

    entry_stack = _required_matrix(ordered, "entry_scores", shape)
    entry = _center_and_clip_member_scores(entry_stack, risky).mean(dim=0)
    alpha_mean = _required_matrix(ordered, "alpha_mean_30d", shape).mean(dim=0)
    downside_stack = _required_matrix(ordered, "alpha_downside_30d", shape)
    if bool((downside_stack < 0).any()):
        raise M03REnsembleError("alpha_downside_30d must be nonnegative")
    downside = downside_stack.median(dim=0).values
    raw_hazard_stack = _required_matrix(ordered, "raw_hazard_residual", shape)
    member_hazard_stack = _required_matrix(ordered, "hazard_residual", shape)
    expected_member_hazard = bound_hold30_hazard_residual(
        raw_hazard_stack, mode="smooth_tanh"
    )
    member_risky = risky.unsqueeze(0).expand_as(raw_hazard_stack)
    if not bool(
        torch.allclose(
            member_hazard_stack[member_risky],
            expected_member_hazard[member_risky],
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        raise M03REnsembleError(
            "member hazard_residual does not match the M03R smooth bound of raw hazard"
        )
    if bool((raw_hazard_stack[~member_risky] != 0).any()) or bool(
        (member_hazard_stack[~member_risky] != -12).any()
    ):
        raise M03REnsembleError(
            "CASH/unavailable hazard sentinels must be raw=0 and bounded=-12"
        )
    raw_hazard = raw_hazard_stack.median(dim=0).values
    hazard = bound_hold30_hazard_residual(raw_hazard, mode="smooth_tanh")
    exact_hold_stack = _optional_matrix(ordered, "exact_hold_probability", shape)
    learned_hazard = setting.exit_hazard_mode == "learned-age-aware"
    if not learned_hazard and (
        bool((raw_hazard_stack[member_risky] != 0).any())
        or bool((member_hazard_stack[member_risky] != 0).any())
    ):
        raise M03REnsembleError(
            "A08 fixed Hold-30 prior requires zero raw and bounded hazard residual"
        )
    if learned_hazard and exact_hold_stack is None:
        raise M03REnsembleError(
            "learned-hazard M03R members require hard exact-hold decisions"
        )
    if exact_hold_stack is not None:
        if bool(((exact_hold_stack != 0) & (exact_hold_stack != 1)).any()):
            raise M03REnsembleError(
                "every exact_hold_probability must be a hard binary decision"
            )
        if bool((exact_hold_stack[~member_risky] != 1).any()):
            raise M03REnsembleError(
                "CASH/unavailable exact-hold sentinels must equal one"
            )
    exact_hold = (
        None if exact_hold_stack is None else exact_hold_stack.median(dim=0).values
    )
    active_risk_stack = _required_vector(ordered, "active_risk_scale", batch)
    if bool((active_risk_stack < 0).any()):
        raise M03REnsembleError("active_risk_scale must be nonnegative")
    active_risk = active_risk_stack.median(dim=0).values
    signal_confidence_stack = _required_vector(ordered, "signal_confidence", batch)
    if bool(((signal_confidence_stack < 0) | (signal_confidence_stack > 1)).any()):
        raise M03REnsembleError("every signal_confidence must lie in [0,1]")
    expected_active_risk = (
        M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        * signal_confidence_stack
    )
    if not bool(
        torch.allclose(
            active_risk_stack,
            expected_active_risk,
            atol=1e-7,
            rtol=1e-6,
        )
    ):
        raise M03REnsembleError(
            "every active_risk_scale must equal 0.04 * signal_confidence"
        )
    signal_confidence = signal_confidence_stack.median(dim=0).values
    if not bool(
        torch.allclose(
            active_risk,
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
            * signal_confidence,
            atol=1e-7,
            rtol=1e-6,
        )
    ):
        raise M03REnsembleError(
            "aggregate active_risk_scale must equal 0.04 * aggregate confidence"
        )

    auxiliaries = [member.intent.auxiliary_alpha_mean for member in ordered]
    if any(value is None for value in auxiliaries):
        raise M03REnsembleError(
            "every M03R alpha member must populate auxiliary_alpha_mean"
        )
    auxiliary_tensors = [value for value in auxiliaries if value is not None]
    auxiliary_shape = tuple(auxiliary_tensors[0].shape)
    if auxiliary_shape[:2] != shape or len(auxiliary_shape) != 3:
        raise M03REnsembleError("auxiliary_alpha_mean must be [batch, asset, horizon]")
    if not all(
        tuple(value.shape) == auxiliary_shape
        and value.device == auxiliary_tensors[0].device
        and value.dtype == auxiliary_tensors[0].dtype
        and value.is_floating_point()
        and bool(torch.isfinite(value).all())
        for value in auxiliary_tensors
    ):
        raise M03REnsembleError("M03R auxiliary alpha tensors are inconsistent")
    auxiliary = torch.stack(auxiliary_tensors).mean(dim=0)

    exposure = _required_vector(ordered, "exposure_residual", batch)
    if bool((exposure != 0).any()):
        raise M03REnsembleError(
            "M03R alpha-core members must not encode gross exposure timing"
        )
    overlay_values = [member.intent.total_risk_overlay for member in ordered]
    if setting.sharpe_mode == "separate-total-risk-overlay":
        overlay = (
            _required_vector(ordered, "total_risk_overlay", batch).median(dim=0).values
        )
    else:
        if any(value is not None for value in overlay_values):
            raise M03REnsembleError(
                "total_risk_overlay is exclusive to exact A06 M03R identity"
            )
        overlay = None

    # Fields belonging to the legacy absolute-target mechanisms are forbidden.
    for name in ("target_logits", "gate"):
        if any(getattr(member.intent, name) is not None for member in ordered):
            raise M03REnsembleError(f"M03R alpha members must not populate {name}")

    zero = torch.zeros_like(alpha_mean)
    intent = Hold30Intent(
        entry_scores=entry,
        hazard_residual=torch.where(risky, hazard, torch.full_like(hazard, -12.0)),
        raw_hazard_residual=torch.where(risky, raw_hazard, zero),
        exact_hold_probability=(
            None
            if exact_hold is None
            else torch.where(risky, exact_hold, torch.ones_like(exact_hold))
        ),
        exposure_residual=torch.zeros_like(active_risk),
        alpha_mean_30d=torch.where(risky, alpha_mean, zero),
        alpha_downside_30d=torch.where(risky, downside, zero),
        active_risk_scale=active_risk,
        signal_confidence=signal_confidence,
        total_risk_overlay=overlay,
        auxiliary_alpha_mean=torch.where(
            risky.unsqueeze(-1), auxiliary, torch.zeros_like(auxiliary)
        ),
    )
    return M03REnsembleIntent(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=setting.setting_id,
        ordered_seeds=tuple(member.seed for member in ordered),
        intent=intent,
    )


__all__ = [
    "M03R_ENSEMBLE_MEMBER_COUNT",
    "M03R_ENTRY_SCORE_CLIP",
    "M03REnsembleError",
    "M03REnsembleIntent",
    "M03REnsembleMember",
    "aggregate_m03r_alpha_intents",
]
