"""Output-space ensemble contract for Hold-30 policies.

Member policies receive one shared economic book/age state but retain separate
causal model states.  Their raw outputs are aggregated once before the common
portfolio builder; member weights or member returns are never averaged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rl_quant.models.daily_policy import Hold30Intent


def _masked_center_clip(
    value: torch.Tensor,
    mask: torch.Tensor,
    bound: float,
) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("member outputs must have shape [member, batch, asset]")
    if mask.shape != value.shape[1:] or mask.dtype != torch.bool:
        raise ValueError("ensemble mask must be boolean [batch, asset]")
    expanded = mask.unsqueeze(0)
    count = expanded.sum(-1).clamp_min(1).to(value.dtype)
    mean = torch.where(expanded, value, torch.zeros_like(value)).sum(-1) / count
    centered = (value - mean.unsqueeze(-1)).clamp(-bound, bound)
    return torch.where(expanded, centered, torch.zeros_like(centered))


def _stack_required(
    intents: Sequence[Hold30Intent],
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    values = [getattr(intent, name) for intent in intents]
    if any(value is None or tuple(value.shape) != shape for value in values):
        raise ValueError(f"every member {name} must have shape {shape}")
    tensors = [value for value in values if value is not None]
    if not all(value.dtype == tensors[0].dtype and value.device == tensors[0].device for value in tensors):
        raise ValueError(f"member {name} tensors must share dtype and device")
    if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError(f"member {name} tensors must be finite and floating point")
    return torch.stack(tensors)


def _reject_extra_fields(intents: Sequence[Hold30Intent], allowed: set[str]) -> None:
    fields = {
        "entry_scores",
        "target_logits",
        "gate",
        "hazard_residual",
        "raw_hazard_residual",
        "exact_hold_probability",
        "exposure_residual",
        "alpha_mean_30d",
        "alpha_downside_30d",
        "active_risk_scale",
        "total_risk_overlay",
        "auxiliary_alpha_mean",
    }
    for field in fields - allowed:
        if any(getattr(intent, field) is not None for intent in intents):
            raise ValueError(f"member intents for this mechanism must not populate {field}")


def aggregate_hold30_intents(
    mechanism: str,
    intents: Sequence[Hold30Intent],
    decision_available: torch.Tensor,
    *,
    cash_index: int = 0,
) -> Hold30Intent:
    """Apply the frozen five-member output-space aggregation rule."""

    if mechanism not in {"H0", "H1", "H2", "H3"}:
        raise ValueError("mechanism must be H0, H1, H2, or H3")
    if len(intents) != 5:
        raise ValueError("the pre-lockbox ensemble requires exactly five members")
    if decision_available.ndim != 2 or decision_available.dtype != torch.bool:
        raise ValueError("decision_available must be boolean [batch, asset]")
    batch, assets = decision_available.shape
    if not 0 <= cash_index < assets:
        raise ValueError("cash_index is outside the asset axis")
    if not bool(decision_available[:, cash_index].all()):
        raise ValueError("CASH must be decision-available for every ensemble row")
    matrix_shape = (batch, assets)

    if mechanism in {"H0", "H1"}:
        _reject_extra_fields(intents, {"target_logits", "gate"})
        logits = _stack_required(intents, "target_logits", matrix_shape)
        eligible = decision_available.clone()
        eligible[:, cash_index] = True
        centered = _masked_center_clip(logits, eligible, 8.0)
        gates = _stack_required(intents, "gate", (batch,))
        return Hold30Intent(
            target_logits=centered.mean(dim=0),
            gate=gates.median(dim=0).values,
        )

    scores = _stack_required(intents, "entry_scores", matrix_shape)
    risky = decision_available.clone()
    risky[:, cash_index] = False
    entry = _masked_center_clip(scores, risky, 2.0).mean(dim=0)
    if mechanism == "H3":
        _reject_extra_fields(intents, {"entry_scores"})
        return Hold30Intent(entry_scores=entry)
    alpha_fields = (
        "alpha_mean_30d",
        "alpha_downside_30d",
        "active_risk_scale",
        "total_risk_overlay",
        "auxiliary_alpha_mean",
    )
    if any(getattr(intent, name) is not None for intent in intents for name in alpha_fields):
        raise ValueError(
            "v3 alpha output-space ensemble aggregation is not frozen; "
            "silently dropping alpha/risk outputs is forbidden"
        )
    _reject_extra_fields(
        intents,
        {
            "entry_scores",
            "hazard_residual",
            "raw_hazard_residual",
            "exact_hold_probability",
            "exposure_residual",
        },
    )
    hazards = _stack_required(intents, "hazard_residual", matrix_shape)
    exposures = _stack_required(intents, "exposure_residual", (batch,))
    raw_values = [intent.raw_hazard_residual for intent in intents]
    if any(value is not None for value in raw_values) and any(
        value is None for value in raw_values
    ):
        raise ValueError(
            "raw_hazard_residual must be populated by every member or none"
        )
    raw_hazard = (
        None
        if all(value is None for value in raw_values)
        else _stack_required(
            intents,
            "raw_hazard_residual",
            matrix_shape,
        ).median(dim=0).values
    )
    hold_values = [intent.exact_hold_probability for intent in intents]
    if any(value is not None for value in hold_values) and any(
        value is None for value in hold_values
    ):
        raise ValueError(
            "exact_hold_probability must be populated by every member or none"
        )
    exact_hold = (
        None
        if all(value is None for value in hold_values)
        else _stack_required(
            intents,
            "exact_hold_probability",
            matrix_shape,
        ).median(dim=0).values.clamp(0.0, 1.0)
    )
    return Hold30Intent(
        entry_scores=entry,
        hazard_residual=hazards.median(dim=0).values.clamp(-12.0, 12.0),
        raw_hazard_residual=raw_hazard,
        exact_hold_probability=exact_hold,
        exposure_residual=exposures.median(dim=0).values,
    )


@dataclass(frozen=True)
class Hold30EnsembleDecision:
    member_intents: tuple[Hold30Intent, ...]
    aggregate_intent: Hold30Intent


def decide_hold30_ensemble(
    mechanism: str,
    policies: Sequence[torch.nn.Module],
    member_states: torch.Tensor,
    decision_weights: torch.Tensor,
    decision_available: torch.Tensor,
    age_summaries: torch.Tensor,
    *,
    cash_index: int = 0,
) -> Hold30EnsembleDecision:
    """Evaluate every member once against one shared economic state."""

    if len(policies) != 5 or member_states.shape[0] != 5:
        raise ValueError("exactly five policies and five member states are required")
    intents = tuple(
        policy.hold30_intent(
            member_states[index],
            decision_weights,
            decision_available,
            age_summaries,
        )
        for index, policy in enumerate(policies)
    )
    aggregate = aggregate_hold30_intents(
        mechanism,
        intents,
        decision_available,
        cash_index=cash_index,
    )
    return Hold30EnsembleDecision(intents, aggregate)


__all__ = [
    "Hold30EnsembleDecision",
    "aggregate_hold30_intents",
    "decide_hold30_ensemble",
]
