"""Objective tests for the setting-bound M03R active-alpha contract."""

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r import resolve_m03r_setting
from rl_quant.training.hold30_alpha_m03r import (
    M03R_DIRECT_SHARPE_CONSTRUCTION_ID,
    M03RDirectSharpeSurrogate,
    M03RObjectiveConfig,
    M03RObjectiveError,
    M03RObjectiveInputs,
    confidence_scaled_preferred_tracking_error,
    m03r_active_objective,
)

M00 = "M00-absolute-return"
M01 = "M01-benchmark-subtraction"
M02 = "M02-active-risk-no-alpha-heads"
M03R = "M03R-active-alpha-hold30"
A04 = "A04-no-uncertainty-scaling"
A05 = "A05-fixed-te-floor"
A06 = "A06-sharpe-overlay"
A07 = "A07-direct-sharpe"
A08 = "A08-fixed-exit-hazard"
A09 = "A09-no-long-context"
A10 = "A10-no-factor-neutral-projection"


def _config(setting_id: str, **changes: object) -> M03RObjectiveConfig:
    setting = resolve_m03r_setting(setting_id)
    config = M03RObjectiveConfig(
        setting_id=setting_id,
        factor_names=("sector", "size", "momentum"),
        absolute_factor_exposure_limits=(0.02, 0.02, 0.02),
        lambda_tracking_error_ceiling=(
            2.0 if setting.annual_tracking_error_ceiling is not None else None
        ),
        lambda_tracking_error_floor=1.5 if setting_id == A05 else None,
        lambda_active_beta=3.0 if setting.active_beta_neutrality else None,
        lambda_factor_exposure=4.0 if setting.factor_sector_projection else None,
        lambda_turnover=5.0,
        lambda_early_exit=6.0,
        lambda_forced_turnover=7.0,
        lambda_auxiliary=8.0 if setting.residual_alpha_heads else None,
        lambda_direct_sharpe=2.5 if setting_id == A07 else None,
    )
    return replace(config, **changes)


def _inputs(
    *,
    policy: torch.Tensor | None = None,
    factors: torch.Tensor | None = None,
) -> M03RObjectiveInputs:
    dtype = torch.float64
    benchmark = torch.tensor([0.001, -0.001, 0.002, 0.0], dtype=dtype)
    if policy is None:
        policy = benchmark.clone().requires_grad_(True)
    if factors is None:
        factors = torch.zeros(4, 3, dtype=dtype)
    return M03RObjectiveInputs(
        policy_net_return=policy,
        benchmark_net_return=benchmark,
        market_excess_return=torch.tensor([-0.02, -0.01, 0.01, 0.02], dtype=dtype),
        discretionary_one_way_turnover=torch.zeros(4, dtype=dtype),
        early_exit_notional=torch.zeros(4, dtype=dtype),
        forced_one_way_turnover=torch.zeros(4, dtype=dtype),
        active_factor_exposure=factors,
    )


def _auxiliary(setting_id: str, policy: torch.Tensor) -> torch.Tensor | None:
    if not resolve_m03r_setting(setting_id).residual_alpha_heads:
        return None
    return policy.sum() * 0.0


def _objective(
    setting_id: str,
    inputs: M03RObjectiveInputs,
    *,
    config: M03RObjectiveConfig | None = None,
    direct_sharpe_surrogate: M03RDirectSharpeSurrogate | None = None,
):
    return m03r_active_objective(
        inputs,
        _config(setting_id) if config is None else config,
        auxiliary_loss=_auxiliary(setting_id, inputs.policy_net_return),
        direct_sharpe_surrogate=direct_sharpe_surrogate,
    )


def test_exact_setting_identity_rejects_aliases_and_cross_generation_ids() -> None:
    with pytest.raises(M03RObjectiveError, match="V3 setting"):
        M03RObjectiveConfig(
            setting_id="hold30a-m03-alpha-core",
            factor_names=("market",),
            absolute_factor_exposure_limits=(0.1,),
        )
    with pytest.raises(M03RObjectiveError, match="unknown M03R setting"):
        M03RObjectiveConfig(
            setting_id="m03r",
            factor_names=("market",),
            absolute_factor_exposure_limits=(0.1,),
        )
    with pytest.raises(M03RObjectiveError, match="exact non-empty"):
        M03RObjectiveConfig(  # type: ignore[arg-type]
            setting_id=None,
            factor_names=("market",),
            absolute_factor_exposure_limits=(0.1,),
        )


def test_m00_uses_absolute_log_return_and_m01_only_subtracts_c1() -> None:
    inputs = _inputs()
    m00_loss, m00_metrics = _objective(M00, inputs)
    m01_loss, m01_metrics = _objective(M01, inputs)

    expected_absolute = -torch.log1p(inputs.policy_net_return).mean()
    torch.testing.assert_close(m00_loss, expected_absolute)
    assert m01_loss.item() == pytest.approx(0.0, abs=1e-15)
    assert m00_metrics.mean_net_active_log_return == pytest.approx(0.0)
    assert m01_metrics.mean_net_active_log_return == pytest.approx(0.0)


def test_m01_has_no_active_risk_or_factor_penalty() -> None:
    benchmark = _inputs().benchmark_net_return
    market = torch.tensor([-0.02, -0.01, 0.01, 0.02], dtype=torch.float64)
    policy = (benchmark + 0.5 * market).requires_grad_(True)
    factors = torch.full((4, 3), 50.0, dtype=torch.float64)
    inputs = _inputs(policy=policy, factors=factors)
    loss, metrics = _objective(M01, inputs)

    expected = -(torch.log1p(policy) - torch.log1p(inputs.benchmark_net_return)).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics.active_market_beta == pytest.approx(0.5)
    assert metrics.factor_exposure_penalty > 0.0
    with pytest.raises(M03RObjectiveError, match="irrelevant for exact setting"):
        _objective(M01, inputs, config=_config(M01, lambda_active_beta=1.0))


@pytest.mark.parametrize("setting_id", [M02, M03R])
def test_m02_and_canonical_apply_active_risk_controls(setting_id: str) -> None:
    benchmark = _inputs().benchmark_net_return
    market = torch.tensor([-0.02, -0.01, 0.01, 0.02], dtype=torch.float64)
    policy = (benchmark + 0.5 * market).requires_grad_(True)
    factors = torch.full((4, 3), 0.04, dtype=torch.float64)
    loss, metrics = _objective(setting_id, _inputs(policy=policy, factors=factors))

    assert loss.item() > 0.0
    assert metrics.annual_tracking_error_penalty > 0.0
    assert metrics.active_market_beta_penalty == pytest.approx(0.25)
    assert metrics.factor_exposure_penalty > 0.0
    loss.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()


def test_canonical_zero_tracking_error_has_finite_autograd_without_a_floor() -> None:
    inputs = _inputs()
    loss, metrics = _objective(M03R, inputs)

    assert metrics.annual_tracking_error == pytest.approx(0.0)
    assert metrics.annual_tracking_error_penalty == pytest.approx(0.0)
    assert metrics.annual_tracking_error_floor_penalty == pytest.approx(0.0)
    loss.backward()
    assert inputs.policy_net_return.grad is not None
    assert torch.isfinite(inputs.policy_net_return.grad).all()


def test_only_a05_requires_and_applies_the_frozen_two_percent_te_floor() -> None:
    inputs = _inputs()
    loss, metrics = _objective(A05, inputs)

    assert metrics.annual_tracking_error == pytest.approx(0.0)
    assert metrics.annual_tracking_error_floor_penalty == pytest.approx(0.02**2)
    assert loss.item() == pytest.approx(1.5 * 0.02**2)
    loss.backward()
    assert inputs.policy_net_return.grad is not None
    assert torch.isfinite(inputs.policy_net_return.grad).all()

    with pytest.raises(M03RObjectiveError, match="lambda_tracking_error_floor"):
        _objective(
            A05, _inputs(), config=_config(A05, lambda_tracking_error_floor=None)
        )
    with pytest.raises(M03RObjectiveError, match="irrelevant for exact setting"):
        _objective(
            M03R,
            _inputs(),
            config=_config(M03R, lambda_tracking_error_floor=1.0),
        )


def test_a07_requires_full_batch_two_pass_surrogate_and_coefficient() -> None:
    inputs = _inputs()
    with pytest.raises(M03RObjectiveError, match="explicit precomputed two-pass"):
        _objective(A07, inputs)
    with pytest.raises(M03RObjectiveError, match="lambda_direct_sharpe"):
        _objective(
            A07,
            _inputs(),
            config=_config(A07, lambda_direct_sharpe=None),
        )

    surrogate_loss = inputs.policy_net_return.sum() * 0.0 + 0.25
    with pytest.raises(M03RObjectiveError, match="complete effective batch"):
        _objective(
            A07,
            inputs,
            direct_sharpe_surrogate=M03RDirectSharpeSurrogate(
                loss_term=surrogate_loss,
                observation_count=3,
            ),
        )

    loss, metrics = _objective(
        A07,
        inputs,
        direct_sharpe_surrogate=M03RDirectSharpeSurrogate(
            loss_term=surrogate_loss,
            observation_count=4,
            construction_id=M03R_DIRECT_SHARPE_CONSTRUCTION_ID,
        ),
    )
    assert loss.item() == pytest.approx(2.5 * 0.25)
    assert metrics.direct_sharpe_surrogate_loss == pytest.approx(0.25)
    loss.backward()
    assert inputs.policy_net_return.grad is not None
    assert torch.isfinite(inputs.policy_net_return.grad).all()

    with pytest.raises(M03RObjectiveError, match="only to A07"):
        _objective(
            M03R,
            _inputs(),
            direct_sharpe_surrogate=M03RDirectSharpeSurrogate(
                loss_term=_inputs().policy_net_return.sum() * 0.0,
                observation_count=4,
            ),
        )


def test_a10_disables_factor_projection_penalty_only() -> None:
    baseline_inputs = _inputs()
    exposed_inputs = replace(
        baseline_inputs,
        active_factor_exposure=torch.full((4, 3), 50.0, dtype=torch.float64),
    )
    baseline_loss, _ = _objective(A10, baseline_inputs)
    exposed_loss, exposed_metrics = _objective(A10, exposed_inputs)

    torch.testing.assert_close(exposed_loss, baseline_loss)
    assert exposed_metrics.factor_exposure_penalty > 0.0
    with pytest.raises(M03RObjectiveError, match="irrelevant for exact setting"):
        _objective(
            A10,
            exposed_inputs,
            config=_config(A10, lambda_factor_exposure=1.0),
        )


@pytest.mark.parametrize("setting_id", [A04, A06, A08, A09])
def test_other_ablation_rows_retain_the_canonical_core_objective(
    setting_id: str,
) -> None:
    canonical_inputs = _inputs()
    ablation_inputs = replace(
        canonical_inputs,
        policy_net_return=canonical_inputs.policy_net_return.detach()
        .clone()
        .requires_grad_(True),
    )
    canonical_loss, _ = _objective(M03R, canonical_inputs)
    ablation_loss, _ = _objective(setting_id, ablation_inputs)
    torch.testing.assert_close(ablation_loss, canonical_loss)


def test_auxiliary_and_result_moving_coefficients_fail_closed() -> None:
    with pytest.raises(M03RObjectiveError, match="lambda_auxiliary"):
        _objective(M03R, _inputs(), config=_config(M03R, lambda_auxiliary=None))
    with pytest.raises(M03RObjectiveError, match="auxiliary_loss is irrelevant"):
        m03r_active_objective(
            _inputs(),
            _config(M02),
            auxiliary_loss=torch.zeros((), dtype=torch.float64),
        )
    with pytest.raises(M03RObjectiveError, match="irrelevant for exact setting"):
        _objective(
            A06,
            _inputs(),
            config=_config(A06, lambda_direct_sharpe=1.0),
        )


def test_confidence_controls_preferred_risk_without_forcing_a_floor() -> None:
    confidence = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
    preferred = confidence_scaled_preferred_tracking_error(confidence)
    torch.testing.assert_close(
        preferred,
        torch.tensor([0.0, 0.01, 0.04], dtype=torch.float64),
    )
