"""Raw-confidence target and gradient isolation for M03R v6 training."""

from __future__ import annotations

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_DESIGN
from rl_quant.training.hold30_m03r_confidence_objective_v6 import (
    M03RV6ConfidenceObjectiveError,
    m03r_v6_raw_confidence_objective,
)


def test_v6_confidence_targets_are_derived_from_continuous_unit_risk_outcomes() -> None:
    logits = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    outcomes = torch.tensor([0.02, -0.01, 0.0, 0.03], dtype=torch.float64)
    valid = torch.tensor([True, True, True, False])

    loss, diagnostics = m03r_v6_raw_confidence_objective(logits, outcomes, valid)
    loss.backward()

    assert diagnostics.valid_observation_count == 3
    assert diagnostics.positive_target_count == 1
    assert diagnostics.observed_positive_rate == pytest.approx(1.0 / 3.0)
    assert (
        diagnostics.target_definition == M03R_DESIGN.model.confidence_target_definition
    )
    assert diagnostics.market_features_detached_from_confidence_objective
    assert logits.grad is not None
    assert float(logits.grad[0]) < 0.0
    assert float(logits.grad[1]) > 0.0
    assert float(logits.grad[2]) > 0.0
    assert float(logits.grad[3]) == 0.0


def test_v6_confidence_objective_rejects_attached_or_misaligned_evidence() -> None:
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    outcomes = torch.tensor([0.01, -0.01], dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, dtype=torch.bool)
    with pytest.raises(M03RV6ConfidenceObjectiveError, match="detached"):
        m03r_v6_raw_confidence_objective(logits, outcomes, valid)
    with pytest.raises(M03RV6ConfidenceObjectiveError, match="nonempty"):
        m03r_v6_raw_confidence_objective(
            logits,
            outcomes.detach(),
            torch.zeros(2, dtype=torch.bool),
        )


def test_v6_confidence_objective_has_no_caller_binary_target_api() -> None:
    with pytest.raises(TypeError):
        m03r_v6_raw_confidence_objective(  # type: ignore[call-arg]
            raw_confidence_logits=torch.zeros(2),
            standardized_unit_risk_30_session_active_log_returns=torch.ones(2),
            valid_observations=torch.ones(2, dtype=torch.bool),
            binary_targets=torch.ones(2),
        )
