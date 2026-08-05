"""Focused v5 calibration, action naming, and gradient-null tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from rl_quant.execution.hold30_m03r_projection_v5 import (
    M03R_ACTIVE_BETA_EXPOSURE_NAME,
    M03R_RISK_MANIFEST_SCHEMA,
    M03RObjectiveRiskContract,
    M03RQualifiedRiskManifest,
    bind_m03r_objective_risk_contract,
    bind_m03r_risk_manifest,
    qualify_m03r_risk_manifest,
)
from rl_quant.models.daily_policy import resolve_hold30_m03r_v5_model_switches
from rl_quant.models.hold30_alpha import (
    Hold30AlphaHead,
    Hold30AlphaHeadConfig,
    Hold30AlphaModelError,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import resolve_m03r_v5_setting
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationError,
    apply_m03r_confidence_calibration,
    bind_m03r_confidence_calibration,
    compute_m03r_confidence_calibration_sha256,
    validate_m03r_confidence_calibration_manifest,
)
from rl_quant.training.hold30_alpha_m03r_v5 import (
    M03RObjectiveConfig,
    M03RObjectiveError,
    M03RObjectiveInputs,
    m03r_active_objective,
)

M00 = "M00-absolute-return"
M01 = "M01-benchmark-subtraction"
M02 = "M02-active-risk-no-alpha-heads"
M03R = "M03R-active-alpha-hold30"
A04 = "A04-no-downside-score-adjustment"
SEED = 19


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


CHECKPOINT_SHA256 = _digest("m03r-v5-checkpoint-seed-19")
MODEL_STATE_SHA256 = _digest("m03r-v5-model-state-seed-19")
SOURCE_SCORE_ARRAY_SHA256 = _digest("inner-validation-confidence-scores-seed-19")
SOURCE_TARGET_ARRAY_SHA256 = _digest("inner-validation-confidence-targets-seed-19")


def _manifest(setting_id: str = M03R):
    return bind_m03r_confidence_calibration(
        setting_id=setting_id,
        seed=SEED,
        checkpoint_sha256=CHECKPOINT_SHA256,
        model_state_sha256=MODEL_STATE_SHA256,
        source_score_array_sha256=SOURCE_SCORE_ARRAY_SHA256,
        source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
        fit_fold_ids=("inner-00", "inner-01"),
        fit_start_trading_session="2021-01-04",
        fit_end_trading_session="2024-12-31",
        temperature=2.0,
        intercept=-0.25,
        fit_observation_count=756,
        brier_score=0.21,
        expected_calibration_error=0.03,
        observed_target_rate=0.54,
    )


def _head_config(setting_id: str = M03R) -> Hold30AlphaHeadConfig:
    manifest = _manifest(setting_id)
    use_downside = resolve_m03r_v5_setting(
        setting_id
    ).use_downside_adjusted_stock_score
    return Hold30AlphaHeadConfig(
        setting_id=setting_id,
        hidden_dim=8,
        downside_penalty_kappa=0.75 if use_downside else None,
        uncertainty_log_scale_bounds=(-4.0, 2.0) if use_downside else None,
        mechanism_generation="m03r-v2",
        hazard_bound_mode="smooth_tanh",
        exact_hold_mixture=True,
        exact_hold_logit_bias=0.0,
        confidence_calibration_manifest_sha256=manifest.manifest_sha256,
        confidence_calibration_manifest=manifest,
        confidence_calibration_seed=SEED,
        confidence_calibration_checkpoint_sha256=CHECKPOINT_SHA256,
        confidence_calibration_model_state_sha256=MODEL_STATE_SHA256,
        confidence_calibration_source_score_array_sha256=(SOURCE_SCORE_ARRAY_SHA256),
        confidence_calibration_source_target_array_sha256=(SOURCE_TARGET_ARRAY_SHA256),
    )


def _head_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(87)
    market = torch.randn(2, 5, 8)
    weights = torch.tensor(
        [[0.90, 0.03, 0.03, 0.02, 0.02], [0.91, 0.03, 0.03, 0.03, 0.00]]
    )
    age = torch.rand(2, 5, 5)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return market, weights, age, available


def test_calibration_manifest_is_applied_and_content_verified() -> None:
    manifest = _manifest()
    assert compute_m03r_confidence_calibration_sha256(manifest) == (
        manifest.manifest_sha256
    )
    raw = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    calibrated = apply_m03r_confidence_calibration(
        raw,
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_setting_id=M03R,
        expected_seed=SEED,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_model_state_sha256=MODEL_STATE_SHA256,
        expected_source_score_array_sha256=SOURCE_SCORE_ARRAY_SHA256,
        expected_source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
    )
    torch.testing.assert_close(calibrated, torch.sigmoid(raw / 2.0 - 0.25))
    calibrated.sum().backward()
    assert raw.grad is not None and bool((raw.grad > 0).all())

    tampered = replace(manifest, temperature=1.0)
    with pytest.raises(M03RConfidenceCalibrationError, match="payload"):
        validate_m03r_confidence_calibration_manifest(
            tampered,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=M03R,
            expected_seed=SEED,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_model_state_sha256=MODEL_STATE_SHA256,
            expected_source_score_array_sha256=SOURCE_SCORE_ARRAY_SHA256,
            expected_source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
        )
    with pytest.raises(M03RConfidenceCalibrationError, match="outer data"):
        validate_m03r_confidence_calibration_manifest(
            replace(manifest, outer_data_used=True),
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=M03R,
            expected_seed=SEED,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_model_state_sha256=MODEL_STATE_SHA256,
            expected_source_score_array_sha256=SOURCE_SCORE_ARRAY_SHA256,
            expected_source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
        )

    with pytest.raises(M03RConfidenceCalibrationError, match="checkpoint_sha256"):
        validate_m03r_confidence_calibration_manifest(
            manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=M03R,
            expected_seed=SEED,
            expected_checkpoint_sha256=_digest("wrong-checkpoint"),
            expected_model_state_sha256=MODEL_STATE_SHA256,
            expected_source_score_array_sha256=SOURCE_SCORE_ARRAY_SHA256,
            expected_source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
        )
    with pytest.raises(M03RConfidenceCalibrationError, match="source_score"):
        validate_m03r_confidence_calibration_manifest(
            manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=M03R,
            expected_seed=SEED,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_model_state_sha256=MODEL_STATE_SHA256,
            expected_source_score_array_sha256=_digest("wrong-score-array"),
            expected_source_target_array_sha256=SOURCE_TARGET_ARRAY_SHA256,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", 23),
        ("checkpoint_sha256", _digest("changed-checkpoint")),
        ("model_state_sha256", _digest("changed-model-state")),
        ("source_score_array_sha256", _digest("changed-score-array")),
        ("source_target_array_sha256", _digest("changed-target-array")),
    ),
)
def test_calibration_digest_binds_checkpoint_and_fit_arrays(
    field: str, replacement: object
) -> None:
    manifest = _manifest()
    mutated = replace(manifest, **{field: replacement})
    assert compute_m03r_confidence_calibration_sha256(mutated) != (
        manifest.manifest_sha256
    )


def test_v5_head_fails_closed_without_the_typed_calibrator() -> None:
    config = _head_config()
    with pytest.raises(Hold30AlphaModelError, match="typed, content-bound"):
        Hold30AlphaHead(
            replace(
                config,
                confidence_calibration_manifest=None,
                confidence_calibration_manifest_sha256="a" * 64,
            )
        )

    with pytest.raises(Hold30AlphaModelError, match="checkpoint_sha256"):
        Hold30AlphaHead(
            replace(
                config,
                confidence_calibration_checkpoint_sha256=_digest(
                    "unrelated-checkpoint"
                ),
            )
        )


def test_v5_setting_and_model_resolvers_are_generation_qualified() -> None:
    assert resolve_m03r_v5_setting(A04).setting_id == A04
    switches = resolve_hold30_m03r_v5_model_switches(A04)
    assert switches.setting_id == A04
    assert not switches.use_uncertainty


def test_v5_head_applies_calibration_once_and_names_exact_hold_outputs() -> None:
    head = Hold30AlphaHead(_head_config())
    assert head.confidence_head is not None
    with torch.no_grad():
        head.confidence_head[-1].weight.zero_()
        head.confidence_head[-1].bias.fill_(1.0)
    output = head(*_head_inputs())

    expected = torch.full_like(
        output.signal_confidence, torch.sigmoid(torch.tensor(0.25))
    )
    torch.testing.assert_close(output.signal_confidence, expected)
    torch.testing.assert_close(output.active_risk_scale, 0.04 * expected)
    assert output.uncalibrated_signal_confidence_logit is not None
    torch.testing.assert_close(
        output.uncalibrated_signal_confidence_logit,
        torch.ones_like(output.uncalibrated_signal_confidence_logit),
    )
    assert output.benchmark_derisk_request is not None
    torch.testing.assert_close(
        output.benchmark_derisk_request,
        torch.zeros_like(output.benchmark_derisk_request),
    )
    assert output.exact_hold_probability is None
    assert output.exact_hold_logit is not None
    assert output.exact_hold_soft_probability is not None
    assert output.exact_hold_decision_st is not None
    assert set(output.exact_hold_decision_st.unique().tolist()) <= {0.0, 1.0}


def test_a04_disables_only_downside_score_adjustment_not_confidence_budget() -> None:
    head = Hold30AlphaHead(_head_config(A04))
    assert head.downside_head is None
    assert head.confidence_head is not None
    output = head(*_head_inputs())
    assert output.downside_30d is None
    assert output.signal_confidence is not None
    torch.testing.assert_close(
        output.risk_adjusted_score,
        output.mean_30d,
    )
    torch.testing.assert_close(
        output.active_risk_scale,
        0.04 * output.signal_confidence,
    )


def _objective_risk_evidence(
    observation_count: int = 4,
) -> tuple[
    tuple[str, ...],
    tuple[M03RQualifiedRiskManifest, ...],
    M03RObjectiveRiskContract,
]:
    asset_ids = ("CASH", *(f"S{index}" for index in range(1, 8)))
    exposure_names = (
        M03R_ACTIVE_BETA_EXPOSURE_NAME,
        "sector:test",
        "factor:size",
        "factor:momentum",
        "factor:value",
        "factor:volatility",
        "factor:liquidity",
    )
    lower = torch.full((7,), -0.01, dtype=torch.float64)
    upper = torch.full((7,), 0.01, dtype=torch.float64)
    lower[0], upper[0] = -0.10, 0.10
    covariance = torch.eye(8, dtype=torch.float64) * 0.001
    covariance[0, 0] = 0.0
    common = {
        "schema": M03R_RISK_MANIFEST_SCHEMA,
        "asset_ids": asset_ids,
        "exposure_names": exposure_names,
        "exposure_families": (
            "market",
            "sector",
            "size",
            "momentum",
            "value",
            "volatility",
            "liquidity",
        ),
        "exposure_units": (
            "unit-beta",
            *("normalized-loading" for _ in range(6)),
        ),
        "exposure_normalization_ids": tuple("pit-zscore-v1" for _ in range(7)),
        "exposure_estimation_window_trading_sessions": 252,
        "missing_value_policy": "fail-closed",
        "covariance_estimator_id": "sample-covariance-v1",
        "covariance_shrinkage_id": "none",
        "covariance_return_convention": "daily-simple-return",
        "stale_loading_policy": "same-session-required",
        "infeasibility_policy": "fail-closed-no-artifact",
        "exposure_loadings": torch.cat(
            (torch.zeros((7, 1), dtype=torch.float64), torch.eye(7, dtype=torch.float64)),
            dim=1,
        ),
        "exposure_lower_bounds": lower,
        "exposure_upper_bounds": upper,
        "daily_return_covariance": covariance,
        "annual_tracking_error_ceiling": 0.06,
        "maximum_risky_asset_weight": 0.01,
    }
    qualified = []
    for index in range(observation_count):
        manifest = bind_m03r_risk_manifest(
            **common,
            as_of_trading_session=f"2025-12-{index + 1:02d}",
        )
        qualified.append(
            qualify_m03r_risk_manifest(
                manifest,
                expected_manifest_sha256=manifest.manifest_sha256,
            )
        )
    qualified_tuple = tuple(qualified)
    return (
        asset_ids,
        qualified_tuple,
        bind_m03r_objective_risk_contract(qualified_tuple),
    )


def _objective_risk_contract(observation_count: int = 4) -> M03RObjectiveRiskContract:
    return _objective_risk_evidence(observation_count)[2]


def _objective_books(
    observation_count: int,
    asset_count: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    benchmark = torch.full(
        (observation_count, asset_count),
        1.0 / asset_count,
        dtype=dtype,
    )
    return benchmark.clone(), benchmark


def _objective_config(
    setting_id: str,
    *,
    risk_contract: M03RObjectiveRiskContract | None = None,
) -> M03RObjectiveConfig:
    bound_risk = _objective_risk_contract() if risk_contract is None else risk_contract
    return M03RObjectiveConfig(
        setting_id=setting_id,
        risk_contract=bound_risk,
        lambda_turnover=0.0,
        lambda_early_exit=0.0,
        lambda_forced_turnover=0.0,
    )


def _objective_loss(setting_id: str, policy: torch.Tensor) -> torch.Tensor:
    benchmark = torch.tensor([0.003, -0.002, 0.001, 0.004], dtype=policy.dtype)
    asset_ids, qualified, risk_contract = _objective_risk_evidence()
    policy_weights, benchmark_weights = _objective_books(
        4, len(asset_ids), dtype=policy.dtype
    )
    inputs = M03RObjectiveInputs(
        policy_net_return=policy,
        benchmark_net_return=benchmark,
        market_excess_return=torch.tensor(
            [-0.02, -0.01, 0.01, 0.02], dtype=policy.dtype
        ),
        discretionary_one_way_turnover=torch.zeros(4, dtype=policy.dtype),
        early_exit_notional=torch.zeros(4, dtype=policy.dtype),
        forced_one_way_turnover=torch.zeros(4, dtype=policy.dtype),
        asset_ids=asset_ids,
        policy_weights=policy_weights,
        benchmark_weights=benchmark_weights,
        qualified_risk_manifests=qualified,
        risk_manifest_sha256s=risk_contract.ordered_risk_manifest_sha256s,
    )
    loss, _metrics = m03r_active_objective(
        inputs,
        _objective_config(setting_id, risk_contract=risk_contract),
    )
    return loss


def test_m01_and_m00_have_identical_gradients_and_optimizer_updates() -> None:
    initial = torch.tensor([0.002, -0.001, 0.004, 0.003], dtype=torch.float64)
    p00 = torch.nn.Parameter(initial.clone())
    p01 = torch.nn.Parameter(initial.clone())
    loss00 = _objective_loss(M00, p00)
    loss01 = _objective_loss(M01, p01)
    loss00.backward()
    loss01.backward()
    assert p00.grad is not None and p01.grad is not None
    torch.testing.assert_close(p00.grad, p01.grad, rtol=0.0, atol=0.0)

    optimizer00 = torch.optim.AdamW([p00], lr=1e-3, weight_decay=0.01)
    optimizer01 = torch.optim.AdamW([p01], lr=1e-3, weight_decay=0.01)
    optimizer00.step()
    optimizer01.step()
    torch.testing.assert_close(p00, p01, rtol=0.0, atol=0.0)


def test_objective_risk_contract_binds_manifest_order_units_and_bounds() -> None:
    contract = _objective_risk_contract()
    assert len(set(contract.ordered_risk_manifest_sha256s)) == 4
    assert contract.exposure_names[0] == M03R_ACTIVE_BETA_EXPOSURE_NAME
    assert contract.exposure_units[0] == "unit-beta"
    assert contract.exposure_lower_bounds[0] == -0.10
    assert contract.exposure_upper_bounds[0] == 0.10

    with pytest.raises(M03RObjectiveError, match="content hash"):
        _objective_config(
            M00,
            risk_contract=replace(
                contract,
                exposure_units=("wrong-unit", *contract.exposure_units[1:]),
            ),
        )


def test_objective_rejects_factor_rows_from_a_different_manifest_order() -> None:
    asset_ids, qualified, contract = _objective_risk_evidence()
    policy_weights, benchmark_weights = _objective_books(
        4, len(asset_ids), dtype=torch.float64
    )
    policy = torch.tensor(
        [0.002, -0.001, 0.004, 0.003],
        dtype=torch.float64,
        requires_grad=True,
    )
    inputs = M03RObjectiveInputs(
        policy_net_return=policy,
        benchmark_net_return=torch.tensor(
            [0.003, -0.002, 0.001, 0.004], dtype=torch.float64
        ),
        market_excess_return=torch.tensor(
            [-0.02, -0.01, 0.01, 0.02], dtype=torch.float64
        ),
        discretionary_one_way_turnover=torch.zeros(4, dtype=torch.float64),
        early_exit_notional=torch.zeros(4, dtype=torch.float64),
        forced_one_way_turnover=torch.zeros(4, dtype=torch.float64),
        asset_ids=asset_ids,
        policy_weights=policy_weights,
        benchmark_weights=benchmark_weights,
        qualified_risk_manifests=tuple(reversed(qualified)),
        risk_manifest_sha256s=tuple(
            reversed(contract.ordered_risk_manifest_sha256s)
        ),
    )
    with pytest.raises(M03RObjectiveError, match="ordered risk-manifest contract"):
        m03r_active_objective(
            inputs,
            _objective_config(M00, risk_contract=contract),
        )


def test_objective_derives_factor_exposure_from_bound_weights_and_manifests() -> None:
    asset_ids, qualified, contract = _objective_risk_evidence()
    policy_weights, benchmark_weights = _objective_books(
        4, len(asset_ids), dtype=torch.float64
    )
    policy_weights[:, 1] += 0.02
    policy_weights[:, 2] -= 0.02
    policy_parameter = torch.nn.Parameter(policy_weights)
    zero_return = torch.zeros(4, dtype=torch.float64)
    inputs = M03RObjectiveInputs(
        policy_net_return=zero_return,
        benchmark_net_return=zero_return.clone(),
        market_excess_return=torch.tensor(
            [-0.02, -0.01, 0.01, 0.02], dtype=torch.float64
        ),
        discretionary_one_way_turnover=zero_return.clone(),
        early_exit_notional=zero_return.clone(),
        forced_one_way_turnover=zero_return.clone(),
        asset_ids=asset_ids,
        policy_weights=policy_parameter,
        benchmark_weights=benchmark_weights,
        qualified_risk_manifests=qualified,
        risk_manifest_sha256s=contract.ordered_risk_manifest_sha256s,
    )
    config = M03RObjectiveConfig(
        setting_id=M02,
        risk_contract=contract,
        lambda_tracking_error_ceiling=0.0,
        lambda_active_beta=0.0,
        lambda_factor_exposure=1.0,
        lambda_turnover=0.0,
        lambda_early_exit=0.0,
        lambda_forced_turnover=0.0,
    )
    loss, metrics = m03r_active_objective(inputs, config)
    assert metrics.factor_exposure_penalty > 0.0
    loss.backward()
    assert policy_parameter.grad is not None
    assert float(policy_parameter.grad.abs().sum()) > 0.0

    with pytest.raises(M03RObjectiveError, match="asset axis"):
        M03RObjectiveInputs(
            policy_net_return=zero_return,
            benchmark_net_return=zero_return.clone(),
            market_excess_return=torch.tensor(
                [-0.02, -0.01, 0.01, 0.02], dtype=torch.float64
            ),
            discretionary_one_way_turnover=zero_return.clone(),
            early_exit_notional=zero_return.clone(),
            forced_one_way_turnover=zero_return.clone(),
            asset_ids=tuple(reversed(asset_ids)),
            policy_weights=policy_weights.detach(),
            benchmark_weights=benchmark_weights,
            qualified_risk_manifests=qualified,
            risk_manifest_sha256s=contract.ordered_risk_manifest_sha256s,
        )
