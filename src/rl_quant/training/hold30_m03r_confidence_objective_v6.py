"""Isolated raw-confidence supervision for M03R v6 training.

Deployment calibration is deliberately unavailable while policy parameters are
trainable.  The raw confidence head is instead trained as a binary classifier
on the frozen v6 standardized-unit-risk 30-session outcome definition.  The
model supplies detached market features to this head, so this objective can
update only the confidence head through its separate optimizer route.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_DESIGN

M03R_V6_RAW_CONFIDENCE_OBJECTIVE_SCHEMA = (
    "rl-quant.m03r-v6-raw-confidence-binary-objective-v1"
)


class M03RV6ConfidenceObjectiveError(ValueError):
    """Raw confidence supervision is malformed or violates v6 semantics."""


@dataclass(frozen=True, slots=True)
class M03RV6ConfidenceObjectiveDiagnostics:
    """Detached evidence for one confidence-head-only training step."""

    valid_observation_count: int
    positive_target_count: int
    observed_positive_rate: float
    binary_log_loss: torch.Tensor
    target_definition: str
    market_features_detached_from_confidence_objective: bool = True


def m03r_v6_raw_confidence_objective(
    raw_confidence_logits: torch.Tensor,
    standardized_unit_risk_30_session_active_log_returns: torch.Tensor,
    valid_observations: torch.Tensor,
) -> tuple[torch.Tensor, M03RV6ConfidenceObjectiveDiagnostics]:
    """Return BCE on internally derived v6 unit-risk outcome signs.

    Callers provide continuous detached outcomes, never binary labels.  This
    keeps target construction package-owned and aligned with post-freeze
    calibration evidence.
    """

    logits = raw_confidence_logits
    outcomes = standardized_unit_risk_30_session_active_log_returns
    valid = valid_observations
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim < 1
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise M03RV6ConfidenceObjectiveError(
            "raw_confidence_logits must be finite floating observations"
        )
    if (
        not isinstance(outcomes, torch.Tensor)
        or tuple(outcomes.shape) != tuple(logits.shape)
        or not outcomes.is_floating_point()
        or outcomes.requires_grad
        or outcomes.device != logits.device
        or not bool(torch.isfinite(outcomes).all())
    ):
        raise M03RV6ConfidenceObjectiveError(
            "standardized unit-risk outcomes must be detached finite floating "
            "and align exactly with logits"
        )
    if (
        not isinstance(valid, torch.Tensor)
        or tuple(valid.shape) != tuple(logits.shape)
        or valid.dtype != torch.bool
        or valid.device != logits.device
        or not bool(valid.any())
    ):
        raise M03RV6ConfidenceObjectiveError(
            "valid_observations must be a nonempty aligned boolean mask"
        )

    selected_logits = logits[valid]
    selected_targets = (outcomes[valid] > 0.0).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        reduction="mean",
    )
    count = int(selected_targets.numel())
    positives = int(selected_targets.detach().sum().item())
    diagnostics = M03RV6ConfidenceObjectiveDiagnostics(
        valid_observation_count=count,
        positive_target_count=positives,
        observed_positive_rate=positives / count,
        binary_log_loss=loss.detach().clone(),
        target_definition=M03R_DESIGN.model.confidence_target_definition,
    )
    return loss, diagnostics


__all__ = [
    "M03R_V6_RAW_CONFIDENCE_OBJECTIVE_SCHEMA",
    "M03RV6ConfidenceObjectiveDiagnostics",
    "M03RV6ConfidenceObjectiveError",
    "m03r_v6_raw_confidence_objective",
]
