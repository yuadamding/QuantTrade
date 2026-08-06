"""Focused qualification for the standalone M02 v6 confidence path."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch
from torch import nn

from rl_quant.models.hold30_confidence_v6 import (
    M03RV6StandaloneConfidenceConfig,
    M03RV6StandaloneConfidenceHead,
    bind_m03r_v6_frozen_policy_confidence,
    m03r_v6_policy_state_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
)
from rl_quant.training.hold30_m03r_confidence_fit import (
    M03RConfidenceCalibrationFitEvidence,
    build_m03r_v6_confidence_outcome_evidence,
    fit_and_bind_m03r_confidence_calibration,
)

M02 = "M02-active-risk-no-alpha-heads-v6"
SEED = 43


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fit_evidence(
    *,
    checkpoint_sha256: str,
    model_state_sha256: str,
) -> M03RConfidenceCalibrationFitEvidence:
    raw_logits = torch.linspace(-2.5, 3.0, 12, dtype=torch.float64)
    outcomes = torch.tensor(
        [-0.04, -0.03, 0.01, -0.02, 0.02, -0.01, 0.04, 0.01, 0.03, 0.05, 0.02, 0.06],
        dtype=torch.float64,
    )
    c1 = torch.full((12, 30), 0.0002, dtype=torch.float64)
    policy = torch.expm1(torch.log1p(c1) + outcomes.unsqueeze(1) / 30.0)
    folds = tuple("inner-00" if index < 6 else "inner-01" for index in range(12))
    dates = tuple(f"2024-02-{index + 1:02d}" for index in range(12))
    outcome_evidence = build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=policy,
        c1_net_simple_returns=c1,
        fold_ids=folds,
        trading_sessions=dates,
        proposal_path_manifest_sha256=_digest("m02-proposal-path"),
    )
    return fit_and_bind_m03r_confidence_calibration(
        setting_id=M02,
        seed=SEED,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        raw_logits=raw_logits,
        binary_targets=None,
        fold_ids=folds,
        trading_sessions=dates,
        checkpoint_frozen_before_calibration=True,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        v6_outcome_evidence=outcome_evidence,
    )


def _post_freeze_config(
    *,
    checkpoint_sha256: str,
    model_state_sha256: str,
) -> M03RV6StandaloneConfidenceConfig:
    evidence = _fit_evidence(
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
    )
    manifest = evidence.calibration_manifest
    return M03RV6StandaloneConfidenceConfig(
        setting_id=M02,
        hidden_dim=8,
        lifecycle_stage="v6-post-freeze-calibrated",
        calibration_manifest_sha256=manifest.manifest_sha256,
        calibration_manifest=manifest,
        calibration_seed=manifest.seed,
        calibration_checkpoint_sha256=manifest.checkpoint_sha256,
        calibration_model_state_sha256=manifest.model_state_sha256,
        calibration_source_score_array_sha256=manifest.source_score_array_sha256,
        calibration_source_target_array_sha256=manifest.source_target_array_sha256,
        calibration_fit_evidence=evidence,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(43)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return hidden, available


def test_standalone_confidence_is_exclusive_to_m02_and_requires_lifecycle() -> None:
    with pytest.raises(ValueError, match="exclusive"):
        M03RV6StandaloneConfidenceConfig(
            setting_id=M03R_CANONICAL_SETTING_ID,
            hidden_dim=8,
            lifecycle_stage="v6-training-uncalibrated",
        )
    training = M03RV6StandaloneConfidenceConfig(
        setting_id=M02,
        hidden_dim=8,
        lifecycle_stage="v6-training-uncalibrated",
    )
    with pytest.raises(ValueError, match="uncalibrated training forbids"):
        replace(training, calibration_manifest_sha256=_digest("placeholder"))


def test_training_stage_emits_detached_logit_and_constant_unit_risk_budget() -> None:
    head = M03RV6StandaloneConfidenceHead(
        M03RV6StandaloneConfidenceConfig(
            setting_id=M02,
            hidden_dim=8,
            lifecycle_stage="v6-training-uncalibrated",
        )
    )
    hidden, available = _inputs()
    output = head(hidden, available)
    assert output.signal_confidence is None
    torch.testing.assert_close(
        output.active_risk_scale,
        torch.full_like(
            output.active_risk_scale,
            float(
                M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
            ),
        ),
    )
    torch.testing.assert_close(
        output.benchmark_derisk_request,
        torch.zeros_like(output.benchmark_derisk_request),
    )
    output.uncalibrated_logit.sum().backward()
    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_post_freeze_stage_requires_evidence_and_applies_bound_calibrator() -> None:
    training_head = M03RV6StandaloneConfidenceHead(
        M03RV6StandaloneConfidenceConfig(
            setting_id=M02,
            hidden_dim=8,
            lifecycle_stage="v6-training-uncalibrated",
        )
    )
    training_policy = nn.Sequential(training_head)
    loaded_state = {
        name: value.detach().clone()
        for name, value in training_policy.state_dict().items()
    }
    checkpoint_sha256 = _digest("m02-checkpoint")
    model_state_sha256 = m03r_v6_policy_state_sha256(training_policy)
    config = _post_freeze_config(
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
    )
    with pytest.raises(ValueError, match="fit evidence"):
        replace(config, calibration_fit_evidence=None)
    head = M03RV6StandaloneConfidenceHead(config)
    policy = nn.Sequential(head)
    policy.load_state_dict(loaded_state)
    hidden, available = _inputs()
    with pytest.raises(ValueError, match="full-policy state binding"):
        head(hidden, available)

    with pytest.raises(ValueError, match="loaded checkpoint"):
        bind_m03r_v6_frozen_policy_confidence(
            policy,
            loaded_checkpoint_sha256=_digest("different-checkpoint"),
        )
    assert any(parameter.requires_grad for parameter in policy.parameters())

    with torch.no_grad():
        next(head.confidence_head.parameters()).add_(1e-4)
    with pytest.raises(ValueError, match="loaded policy state"):
        bind_m03r_v6_frozen_policy_confidence(
            policy,
            loaded_checkpoint_sha256=checkpoint_sha256,
        )
    policy.load_state_dict(loaded_state)
    binding = bind_m03r_v6_frozen_policy_confidence(
        policy,
        loaded_checkpoint_sha256=checkpoint_sha256,
    )
    assert binding.loaded_policy_state_sha256 == model_state_sha256
    assert binding.calibration_fit_evidence_sha256s == (
        config.calibration_fit_evidence.evidence_sha256,
    )
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert not head.confidence_head.training
    policy.train()
    assert not head.confidence_head.training
    output = head(hidden, available)
    assert output.signal_confidence is not None
    assert not output.signal_confidence.requires_grad
    assert not output.active_risk_scale.requires_grad
    assert hidden.grad is None
    assert all(parameter.grad is None for parameter in head.parameters())
    torch.testing.assert_close(
        output.active_risk_scale,
        float(
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        )
        * output.signal_confidence,
    )
    assert bool(
        ((output.signal_confidence > 0.0) & (output.signal_confidence < 1.0)).all()
    )
