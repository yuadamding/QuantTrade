"""Generation-qualified model integration for the M03R v6 exit action."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal

import pytest
import torch
from torch import nn

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.execution.hold30_exit_v6 import build_m03r_v6_exit_release
from rl_quant.models.daily_policy import (
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    resolve_hold30_m03r_v6_model_switches,
)
from rl_quant.models.hold30_alpha import Hold30AlphaHead, Hold30AlphaHeadConfig
from rl_quant.models.hold30_confidence_v6 import (
    bind_m03r_v6_frozen_policy_confidence,
    m03r_v6_policy_state_sha256,
)
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_EXIT_ACTION_INDEX,
    M03R_V6_HOLD_ACTION_INDEX,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_AGE_LEDGER_BIN_COUNT,
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationManifest,
)
from rl_quant.training.hold30_alpha_m03r_v6 import (
    M03RV6ExitNotionalByAge,
    M03RV6TrainingPlan,
    M03RV6TrainingProgress,
    m03r_v6_soft_persistence_objective,
)
from rl_quant.training.hold30_m03r_confidence_fit import (
    M03RConfidenceCalibrationFitEvidence,
    build_m03r_v6_confidence_outcome_evidence,
    fit_and_bind_m03r_confidence_calibration,
)
from rl_quant.training.hold30_runtime import (
    Hold30ChronologicalRuntime,
    Hold30Sequence,
)

A11 = "A11-no-exact-hold-atom"
A08 = "A08-fixed-exit-hazard-v6"
M00 = "M00-absolute-return-v6"
M01 = "M01-benchmark-subtraction-v6"
M02 = "M02-active-risk-no-alpha-heads-v6"
SEED = 29


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _v6_fit_evidence(
    setting_id: str,
    *,
    checkpoint_sha256: str | None = None,
    model_state_sha256: str | None = None,
) -> M03RConfidenceCalibrationFitEvidence:
    checkpoint = (
        _digest(f"{setting_id}-checkpoint")
        if checkpoint_sha256 is None
        else checkpoint_sha256
    )
    model_state = (
        _digest(f"{setting_id}-model")
        if model_state_sha256 is None
        else model_state_sha256
    )
    raw_logits = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        dtype=torch.float64,
    )
    outcomes = torch.tensor(
        [-0.06, -0.02, 0.01, -0.04, 0.03, -0.01, 0.08, 0.02, -0.03, 0.04, 0.07, 0.05],
        dtype=torch.float64,
    )
    c1_returns = torch.full((outcomes.numel(), 30), 0.0002, dtype=torch.float64)
    policy_returns = torch.expm1(torch.log1p(c1_returns) + outcomes.unsqueeze(1) / 30.0)
    folds = tuple("inner-00" if index < 6 else "inner-01" for index in range(12))
    dates = tuple(f"2024-01-{index + 2:02d}" for index in range(12))
    outcome_evidence = build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=policy_returns,
        c1_net_simple_returns=c1_returns,
        fold_ids=folds,
        trading_sessions=dates,
        proposal_path_manifest_sha256=_digest(f"{setting_id}-proposal-path"),
    )
    return fit_and_bind_m03r_confidence_calibration(
        setting_id=setting_id,
        seed=SEED,
        checkpoint_sha256=checkpoint,
        model_state_sha256=model_state,
        raw_logits=raw_logits,
        binary_targets=None,
        fold_ids=folds,
        trading_sessions=dates,
        checkpoint_frozen_before_calibration=True,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        v6_outcome_evidence=outcome_evidence,
    )


def _v6_alpha_config(
    setting_id: str,
    *,
    stage: Literal[
        "v6-training-uncalibrated", "v6-post-freeze-calibrated"
    ] = "v6-post-freeze-calibrated",
    fit_evidence: M03RConfidenceCalibrationFitEvidence | None = None,
) -> Hold30AlphaHeadConfig:
    evidence: M03RConfidenceCalibrationFitEvidence | None = None
    manifest: M03RConfidenceCalibrationManifest | None = None
    if stage == "v6-post-freeze-calibrated":
        evidence = (
            _v6_fit_evidence(setting_id) if fit_evidence is None else fit_evidence
        )
        manifest = evidence.calibration_manifest
    return Hold30AlphaHeadConfig(
        setting_id=setting_id,
        hidden_dim=8,
        downside_penalty_kappa=0.75,
        uncertainty_log_scale_bounds=(-4.0, 2.0),
        mechanism_generation="m03r-v3",
        hazard_bound_mode="smooth_tanh",
        exact_hold_mixture=False,
        fixed_hazard_residual=0.0 if setting_id == A08 else None,
        confidence_calibration_manifest_sha256=(
            None if manifest is None else manifest.manifest_sha256
        ),
        confidence_calibration_manifest=manifest,
        confidence_calibration_seed=None if manifest is None else manifest.seed,
        confidence_calibration_checkpoint_sha256=(
            None if manifest is None else manifest.checkpoint_sha256
        ),
        confidence_calibration_model_state_sha256=(
            None if manifest is None else manifest.model_state_sha256
        ),
        confidence_calibration_source_score_array_sha256=(
            None if manifest is None else manifest.source_score_array_sha256
        ),
        confidence_calibration_source_target_array_sha256=(
            None if manifest is None else manifest.source_target_array_sha256
        ),
        m03r_v6_confidence_stage=stage,
        confidence_calibration_fit_evidence=evidence,
    )


def _head_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(109)
    market = torch.randn(2, 5, 8)
    weights = torch.tensor(
        [[0.90, 0.03, 0.03, 0.02, 0.02], [0.91, 0.03, 0.03, 0.03, 0.00]]
    )
    ages = torch.rand(2, 5, 5)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return market, weights, ages, available


def _v6_m02_policy_config(
    *,
    stage: Literal[
        "v6-training-uncalibrated", "v6-post-freeze-calibrated"
    ] = "v6-training-uncalibrated",
    fit_evidence: M03RConfidenceCalibrationFitEvidence | None = None,
) -> DailyCrossSectionConfig:
    evidence: M03RConfidenceCalibrationFitEvidence | None = None
    manifest: M03RConfidenceCalibrationManifest | None = None
    if stage == "v6-post-freeze-calibrated":
        evidence = _v6_fit_evidence(M02) if fit_evidence is None else fit_evidence
        manifest = evidence.calibration_manifest
    return DailyCrossSectionConfig(
        context_dim=4,
        bar_feature_dim=5,
        raw_policy_dim=4,
        raw_policy_layers=1,
        raw_policy_heads=1,
        raw_block_seconds=2,
        session_seconds=4,
        news_raw_dim=1,
        max_news=2,
        news_embed_dim=4,
        token_dim=8,
        temporal_layers=1,
        temporal_heads=1,
        daily_lookback=252,
        max_days=252,
        alloc_layers=1,
        alloc_heads=1,
        feedforward_dim=16,
        dropout=0.0,
        raw_recent_days=42,
        hold30_setting=M02,
        hold30_mechanism_generation="m03r-v3",
        hold30_fast_raw_context_sessions=42,
        hold30_slow_context_sessions=252,
        hold30_hazard_bound_mode="smooth_tanh",
        alpha_confidence_calibration_manifest_sha256=(
            None if manifest is None else manifest.manifest_sha256
        ),
        alpha_confidence_calibration_manifest=manifest,
        alpha_confidence_calibration_seed=(None if manifest is None else manifest.seed),
        alpha_confidence_calibration_checkpoint_sha256=(
            None if manifest is None else manifest.checkpoint_sha256
        ),
        alpha_confidence_calibration_model_state_sha256=(
            None if manifest is None else manifest.model_state_sha256
        ),
        alpha_confidence_calibration_source_score_array_sha256=(
            None if manifest is None else manifest.source_score_array_sha256
        ),
        alpha_confidence_calibration_source_target_array_sha256=(
            None if manifest is None else manifest.source_target_array_sha256
        ),
        alpha_m03r_v6_confidence_stage=stage,
        alpha_confidence_calibration_fit_evidence=evidence,
    )


def _bound_v6_alpha_head(
    setting_id: str,
) -> tuple[Hold30AlphaHead, nn.Sequential, str]:
    training_head = Hold30AlphaHead(
        _v6_alpha_config(setting_id, stage="v6-training-uncalibrated")
    )
    training_policy = nn.Sequential(training_head)
    loaded_state = {
        name: value.detach().clone()
        for name, value in training_policy.state_dict().items()
    }
    model_state_sha256 = m03r_v6_policy_state_sha256(training_policy)
    checkpoint_sha256 = _digest(f"{setting_id}-checkpoint")
    evidence = _v6_fit_evidence(
        setting_id,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
    )
    post_head = Hold30AlphaHead(
        _v6_alpha_config(setting_id, fit_evidence=evidence)
    )
    post_policy = nn.Sequential(post_head)
    post_policy.load_state_dict(loaded_state)
    bind_m03r_v6_frozen_policy_confidence(
        post_policy,
        loaded_checkpoint_sha256=checkpoint_sha256,
    )
    return post_head, post_policy, checkpoint_sha256


def _bound_v6_m02_policy() -> tuple[DailyCrossSectionPolicy, str]:
    training_policy = DailyCrossSectionPolicy(_v6_m02_policy_config())
    loaded_state = {
        name: value.detach().clone()
        for name, value in training_policy.state_dict().items()
    }
    model_state_sha256 = m03r_v6_policy_state_sha256(training_policy)
    checkpoint_sha256 = _digest(f"{M02}-checkpoint")
    evidence = _v6_fit_evidence(
        M02,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
    )
    post_policy = DailyCrossSectionPolicy(
        _v6_m02_policy_config(
            stage="v6-post-freeze-calibrated",
            fit_evidence=evidence,
        )
    )
    post_policy.load_state_dict(loaded_state)
    bind_m03r_v6_frozen_policy_confidence(
        post_policy,
        loaded_checkpoint_sha256=checkpoint_sha256,
    )
    return post_policy, checkpoint_sha256


def test_v6_model_resolver_is_exact_and_a11_removes_only_hold() -> None:
    assert (
        tuple(
            resolve_hold30_m03r_v6_model_switches(setting_id).setting_id
            for setting_id in M03R_SETTING_IDS
        )
        == M03R_SETTING_IDS
    )
    canonical = resolve_hold30_m03r_v6_model_switches(M03R_CANONICAL_SETTING_ID)
    assert canonical.use_three_way_exit_action
    assert canonical.allow_exact_hold_atom

    a11 = resolve_hold30_m03r_v6_model_switches(A11)
    assert a11.use_three_way_exit_action
    assert not a11.allow_exact_hold_atom

    fixed = resolve_hold30_m03r_v6_model_switches(A08)
    assert not fixed.use_three_way_exit_action
    assert not fixed.allow_exact_hold_atom

    with pytest.raises(ValueError, match="unknown M03R v6 setting"):
        resolve_hold30_m03r_v6_model_switches("M03R-active-alpha-hold30")


def test_v6_fixed_hazard_ablation_has_no_three_way_action_head() -> None:
    valid = _v6_alpha_config(A08)
    with pytest.raises(ValueError, match="requires fixed residual 0.0"):
        replace(valid, fixed_hazard_residual=None)
    head, policy, _checkpoint = _bound_v6_alpha_head(A08)
    assert head.exit_action_head_v6 is None
    assert head(*_head_inputs()).exit_action_v6 is None
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())


def test_v6_confidence_lifecycle_separates_training_from_calibrated_execution() -> None:
    training = _v6_alpha_config(
        M03R_CANONICAL_SETTING_ID,
        stage="v6-training-uncalibrated",
    )
    head = Hold30AlphaHead(training)
    market, weights, ages, available = _head_inputs()
    market.requires_grad_(True)
    output = head(market, weights, ages, available)
    assert output.signal_confidence is None
    assert output.uncalibrated_signal_confidence_logit is not None
    expected_risk = torch.full_like(
        output.active_risk_scale,
        float(
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        ),
    )
    torch.testing.assert_close(output.active_risk_scale, expected_risk)
    assert output.benchmark_derisk_request is not None
    torch.testing.assert_close(
        output.benchmark_derisk_request,
        torch.zeros_like(output.benchmark_derisk_request),
    )

    output.uncalibrated_signal_confidence_logit.sum().backward()
    assert market.grad is None
    assert head.confidence_head is not None
    confidence_gradients = tuple(
        parameter.grad for parameter in head.confidence_head.parameters()
    )
    assert all(gradient is not None for gradient in confidence_gradients)
    assert any(
        bool((gradient != 0.0).any())
        for gradient in confidence_gradients
        if gradient is not None
    )

    post_freeze = _v6_alpha_config(M03R_CANONICAL_SETTING_ID)
    with pytest.raises(ValueError, match="fit evidence"):
        replace(post_freeze, confidence_calibration_fit_evidence=None)
    with pytest.raises(ValueError, match="uncalibrated training forbids"):
        replace(
            post_freeze,
            m03r_v6_confidence_stage="v6-training-uncalibrated",
        )
    assert post_freeze.confidence_calibration_fit_evidence is not None
    with pytest.raises(ValueError, match="confidence-fit evidence"):
        replace(
            post_freeze,
            confidence_calibration_fit_evidence=replace(
                post_freeze.confidence_calibration_fit_evidence,
                post_calibration_policy_updates_permitted=True,
            ),
        )

    policy = DailyCrossSectionPolicy(
        replace(
            _v6_m02_policy_config(),
            hold30_setting=M03R_CANONICAL_SETTING_ID,
            alpha_downside_penalty_kappa=0.75,
            alpha_uncertainty_log_scale_bounds=(-4.0, 2.0),
            alpha_m03r_v6_confidence_stage="v6-training-uncalibrated",
        )
    )
    assert policy.alpha_head is not None
    assert policy.exit_action_head_v6 is None
    assert (
        policy.alpha_head.config.m03r_v6_confidence_stage == "v6-training-uncalibrated"
    )


def test_m02_keeps_active_risk_confidence_without_residual_alpha_heads() -> None:
    switches = resolve_hold30_m03r_v6_model_switches(M02)
    assert not switches.use_alpha_head
    assert switches.use_confidence_scaled_active_risk

    with pytest.raises(ValueError, match="requires an explicit"):
        DailyCrossSectionPolicy(
            replace(
                _v6_m02_policy_config(),
                alpha_m03r_v6_confidence_stage=None,
            )
        )

    training_policy = DailyCrossSectionPolicy(_v6_m02_policy_config())
    assert training_policy.alpha_head is None
    confidence_head = training_policy.standalone_confidence_head_v6
    assert confidence_head is not None
    state = torch.randn(2, 5, 8, requires_grad=True)
    weights = torch.tensor(
        [[0.90, 0.03, 0.03, 0.02, 0.02], [0.91, 0.03, 0.03, 0.03, 0.00]]
    )
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    ages = torch.rand(2, 5, 5)
    training_intent = training_policy.hold30_intent(
        state,
        weights,
        available,
        ages,
    )
    assert training_intent.alpha_mean_30d is None
    assert training_intent.alpha_downside_30d is None
    assert training_intent.auxiliary_alpha_mean is None
    assert training_intent.signal_confidence is None
    assert training_intent.uncalibrated_signal_confidence_logit is not None
    assert training_intent.active_risk_scale is not None
    torch.testing.assert_close(
        training_intent.active_risk_scale,
        torch.full_like(
            training_intent.active_risk_scale,
            float(
                M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
            ),
        ),
    )
    training_intent.uncalibrated_signal_confidence_logit.sum().backward()
    assert state.grad is None
    confidence_gradients = tuple(
        parameter.grad for parameter in confidence_head.parameters()
    )
    assert all(gradient is not None for gradient in confidence_gradients)
    assert any(
        bool((gradient != 0.0).any())
        for gradient in confidence_gradients
        if gradient is not None
    )

    post_config = _v6_m02_policy_config(stage="v6-post-freeze-calibrated")
    with pytest.raises(ValueError, match="fit evidence"):
        DailyCrossSectionPolicy(
            replace(post_config, alpha_confidence_calibration_fit_evidence=None)
        )
    post_policy, _checkpoint = _bound_v6_m02_policy()
    assert post_policy.alpha_head is None
    assert post_policy.standalone_confidence_head_v6 is not None
    assert not post_policy.training
    assert all(not parameter.requires_grad for parameter in post_policy.parameters())
    assert all(parameter.grad is None for parameter in post_policy.parameters())
    post_intent = post_policy.hold30_intent(state.detach(), weights, available, ages)
    assert post_intent.signal_confidence is not None
    assert post_intent.active_risk_scale is not None
    torch.testing.assert_close(
        post_intent.active_risk_scale,
        float(
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        )
        * post_intent.signal_confidence,
    )
    assert post_intent.benchmark_derisk_request is not None
    torch.testing.assert_close(
        post_intent.benchmark_derisk_request,
        torch.zeros_like(post_intent.benchmark_derisk_request),
    )
    assert not post_intent.signal_confidence.requires_grad
    assert not post_intent.active_risk_scale.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        post_intent.active_risk_scale.sum().backward()
    assert all(parameter.grad is None for parameter in post_policy.parameters())


def test_post_freeze_binding_rejects_a_different_loaded_full_policy_state() -> None:
    training_policy = DailyCrossSectionPolicy(_v6_m02_policy_config())
    loaded_state = {
        name: value.detach().clone()
        for name, value in training_policy.state_dict().items()
    }
    checkpoint_sha256 = _digest("m02-state-mismatch-checkpoint")
    evidence = _v6_fit_evidence(
        M02,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=m03r_v6_policy_state_sha256(training_policy),
    )
    post_policy = DailyCrossSectionPolicy(
        _v6_m02_policy_config(
            stage="v6-post-freeze-calibrated",
            fit_evidence=evidence,
        )
    )
    post_policy.load_state_dict(loaded_state)
    with torch.no_grad():
        next(post_policy.raw_encoder.parameters()).add_(1e-4)

    with pytest.raises(ValueError, match="loaded policy state"):
        bind_m03r_v6_frozen_policy_confidence(
            post_policy,
            loaded_checkpoint_sha256=checkpoint_sha256,
        )
    assert post_policy.training
    assert any(parameter.requires_grad for parameter in post_policy.parameters())


@pytest.mark.parametrize(
    "stage",
    ("v6-training-uncalibrated", "v6-post-freeze-calibrated"),
)
def test_m02_confidence_only_intent_runs_through_chronological_runtime(
    stage: Literal["v6-training-uncalibrated", "v6-post-freeze-calibrated"],
) -> None:
    policy = (
        DailyCrossSectionPolicy(_v6_m02_policy_config(stage=stage))
        if stage == "v6-training-uncalibrated"
        else _bound_v6_m02_policy()[0]
    )
    positions, batch, assets = 2, 1, 4
    initial_weights = torch.tensor([[0.90, 0.04, 0.03, 0.03]])
    decision_state = torch.randn(positions, batch, assets, 8)
    available = torch.ones((positions, batch, assets), dtype=torch.bool)
    benchmark = initial_weights.unsqueeze(0).expand(positions, -1, -1).clone()
    sequence = Hold30Sequence(
        decision_state=decision_state,
        asset_returns=torch.zeros((positions - 1, batch, assets)),
        decision_available=available.clone(),
        fill_membership=available.clone(),
        fill_availability=available.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.ones_like(benchmark),
        risk_gross_max=torch.ones((positions, batch)),
        benchmark_net_returns=torch.zeros((positions - 1, batch)),
        initial_ledger=CohortLedger.from_weights(
            initial_weights,
            cash_index=0,
            track_initial_units=True,
        ),
        cost_rate=0.0,
        axis_id="m02-confidence-only-runtime-v1",
    )

    terminal, transitions = Hold30ChronologicalRuntime("H2").run_to_terminal(
        policy,
        sequence,
    )

    assert terminal.position_index == positions - 1
    assert len(transitions) == 1
    intent = transitions[0].raw_intent
    assert intent.alpha_mean_30d is None
    assert intent.alpha_downside_30d is None
    assert intent.auxiliary_alpha_mean is None
    assert intent.active_risk_scale is not None
    assert intent.uncalibrated_signal_confidence_logit is not None
    assert (intent.signal_confidence is None) is (stage == "v6-training-uncalibrated")


@pytest.mark.parametrize("setting_id", (M00, M01))
def test_pre_m02_controls_do_not_gain_the_confidence_mechanism(
    setting_id: str,
) -> None:
    config = replace(
        _v6_m02_policy_config(),
        hold30_setting=setting_id,
        alpha_m03r_v6_confidence_stage=None,
    )
    policy = DailyCrossSectionPolicy(config)
    assert policy.alpha_head is None
    assert policy.standalone_confidence_head_v6 is None
    with pytest.raises(ValueError, match="without confidence-scaled active risk"):
        DailyCrossSectionPolicy(
            replace(config, alpha_m03r_v6_confidence_stage="v6-training-uncalibrated")
        )


def test_actual_v6_alpha_head_learns_an_exact_age_two_exit() -> None:
    head = Hold30AlphaHead(
        _v6_alpha_config(
            M03R_CANONICAL_SETTING_ID,
            stage="v6-training-uncalibrated",
        )
    )
    assert head.exit_action_head_v6 is not None
    optimizer = torch.optim.SGD(head.exit_action_head_v6.parameters(), lr=5.0)
    market = torch.zeros((1, 3, 8), dtype=torch.float32)
    weights = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    ages = torch.zeros((1, 3, 5), dtype=torch.float32)
    available = torch.tensor([[True, True, False]])
    ledger = torch.zeros((1, 3, M03R_AGE_LEDGER_BIN_COUNT), dtype=torch.float32)
    ledger[0, 1, 2] = 1.0
    zeros = torch.zeros(M03R_AGE_LEDGER_BIN_COUNT, dtype=torch.float32)
    progress = M03RV6TrainingProgress(
        completed_optimizer_steps=100,
        training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
    )

    initial = head(market, weights, ages, available).exit_action_v6
    assert initial is not None
    assert initial.continuous_decision_st[0, 1].item() == 1.0
    for _step in range(6):
        output = head(market, weights, ages, available)
        assert output.exit_action_v6 is not None
        release = build_m03r_v6_exit_release(
            ledger,
            output.hazard_residual,
            output.exit_action_v6,
        )
        persistence, _ = m03r_v6_soft_persistence_objective(
            M03RV6ExitNotionalByAge(
                discretionary_policy=release.discretionary_release_by_age.sum(
                    dim=(0, 1)
                ),
                other_forced=zeros,
                unavailable=zeros,
                risk_repair=zeros,
                corporate_action=zeros,
                terminal=zeros,
                valid_decision_session_count=1,
            ),
            progress,
        )
        # The severe adverse outcome dominates the 5-bp soft preference.
        loss = 0.05 * (1.0 - release.discretionary_release[0, 1]) + persistence
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    learned = head(market, weights, ages, available)
    assert learned.exit_action_v6 is not None
    assert learned.exit_action_v6.exit_decision_st[0, 1].item() == 1.0
    final_release = build_m03r_v6_exit_release(
        ledger,
        learned.hazard_residual,
        learned.exit_action_v6,
    )
    assert final_release.discretionary_release[0, 1].item() == 1.0


@pytest.mark.parametrize(
    ("setting_id", "hold_atom_enabled"),
    ((M03R_CANONICAL_SETTING_ID, True), (A11, False)),
)
def test_v6_alpha_head_exposes_three_way_action(
    setting_id: str,
    hold_atom_enabled: bool,
) -> None:
    head, policy, _checkpoint = _bound_v6_alpha_head(setting_id)
    assert head.exit_action_head_v6 is not None
    output = head(*_head_inputs())
    action = output.exit_action_v6
    assert action is not None
    assert output.signal_confidence is not None
    assert not output.signal_confidence.requires_grad
    assert not output.active_risk_scale.requires_grad
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())
    torch.testing.assert_close(
        output.active_risk_scale,
        float(
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        )
        * output.signal_confidence,
    )
    assert action.exact_hold_atom_enabled is hold_atom_enabled
    risky = action.risky_available
    if hold_atom_enabled:
        torch.testing.assert_close(
            action.soft_probabilities[..., M03R_V6_HOLD_ACTION_INDEX][risky],
            torch.full_like(
                action.soft_probabilities[..., M03R_V6_HOLD_ACTION_INDEX][risky],
                1.0 / 3.0,
            ),
        )
    else:
        assert bool(
            (
                action.soft_probabilities[..., M03R_V6_HOLD_ACTION_INDEX][risky] == 0.0
            ).all()
        )

    with torch.no_grad():
        head.exit_action_head_v6.action_logits.bias.fill_(-8.0)
        head.exit_action_head_v6.action_logits.bias[M03R_V6_EXIT_ACTION_INDEX] = 8.0
    exit_action = head(*_head_inputs()).exit_action_v6
    assert exit_action is not None
    assert bool(
        (exit_action.exit_decision_st[exit_action.risky_available] == 1.0).all()
    )


def test_daily_policy_accepts_only_exact_v6_identity_and_exposes_action() -> None:
    policy = DailyCrossSectionPolicy(_v6_m02_policy_config())
    assert policy.exit_action_head_v6 is not None
    state = torch.randn(1, 4, 8)
    weights = torch.tensor([[0.90, 0.04, 0.03, 0.03]])
    available = torch.tensor([[True, True, True, True]])
    ages = torch.rand(1, 4, 5)
    intent = policy.hold30_intent(state, weights, available, ages)
    assert intent.exit_action_v6 is not None
    assert intent.exit_action_v6.exact_hold_atom_enabled

    with pytest.raises(ValueError, match="unknown M03R setting"):
        DailyCrossSectionPolicy(
            replace(
                _v6_m02_policy_config(),
                hold30_mechanism_generation="m03r-v2",
                alpha_m03r_v6_confidence_stage=None,
            )
        )
