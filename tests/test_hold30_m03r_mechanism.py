"""Focused qualification for opt-in post-v3 Hold-30 mechanism primitives."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution.hold30 import (
    build_alpha_hold30_action,
    build_h2_hold30_action,
    capped_waterfill,
)
from rl_quant.models.daily_policy import (
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    hold30_proposed_release,
)
from rl_quant.models.hold30_alpha import (
    Hold30AlphaHead,
    Hold30AlphaHeadConfig,
    Hold30AlphaModelError,
)
from rl_quant.models.hold30_hazard import (
    bound_hold30_hazard_residual,
    clip_hold30_hazard_residual,
    hold30_hazard_telemetry,
)

M03R = "M03R-active-alpha-hold30"
A08 = "A08-fixed-exit-hazard"


def _m03r_policy_config(**updates: object) -> DailyCrossSectionConfig:
    values: dict[str, object] = {
        "context_dim": 4,
        "bar_feature_dim": 5,
        "raw_policy_dim": 4,
        "raw_policy_layers": 1,
        "raw_policy_heads": 1,
        "raw_block_seconds": 2,
        "session_seconds": 4,
        "news_raw_dim": 1,
        "max_news": 2,
        "news_embed_dim": 4,
        "token_dim": 8,
        "temporal_layers": 1,
        "temporal_heads": 1,
        "daily_lookback": 252,
        "max_days": 252,
        "alloc_layers": 1,
        "alloc_heads": 1,
        "feedforward_dim": 16,
        "dropout": 0.0,
        "raw_recent_days": 42,
        "hold30_setting": "M02-active-risk-no-alpha-heads",
        "hold30_mechanism_generation": "m03r-v1",
        "hold30_fast_raw_context_sessions": 42,
        "hold30_slow_context_sessions": 252,
        "hold30_hazard_bound_mode": "smooth_tanh",
        "hold30_exact_hold_mixture": True,
        "hold30_exact_hold_logit_bias": 0.0,
        "alpha_confidence_calibration_manifest_sha256": "a" * 64,
    }
    values.update(updates)
    return DailyCrossSectionConfig(**values)


def _alpha_config(**updates: object) -> Hold30AlphaHeadConfig:
    values: dict[str, object] = {
        "setting_id": M03R,
        "hidden_dim": 8,
        "downside_penalty_kappa": 0.75,
        "active_log_scale_bounds": None,
        "uncertainty_log_scale_bounds": (-4.0, 2.0),
        "mechanism_generation": "m03r-v1",
        "hazard_bound_mode": "smooth_tanh",
        "exact_hold_mixture": True,
        "exact_hold_logit_bias": 0.0,
        "confidence_calibration_manifest_sha256": "a" * 64,
    }
    values.update(updates)
    return Hold30AlphaHeadConfig(**values)


def _head_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(41)
    market = torch.randn(2, 5, 8)
    weights = torch.tensor(
        [[0.90, 0.03, 0.03, 0.02, 0.02], [0.91, 0.03, 0.03, 0.03, 0.00]]
    )
    age = torch.rand(2, 5, 5)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return market, weights, age, available


def _ledger_from_weights(weights: torch.Tensor, age: int = 20) -> torch.Tensor:
    ledger = weights.new_zeros((*weights.shape, 61))
    ledger[..., age] = weights
    return ledger


def test_frozen_generation_rejects_post_v3_hazard_options() -> None:
    with pytest.raises(ValueError, match="explicit m03r-v1"):
        DailyCrossSectionPolicy(
            replace(
                _m03r_policy_config(),
                hold30_mechanism_generation="v2-v3-frozen",
            )
        )
    frozen_geometry = replace(
        _m03r_policy_config(),
        hold30_mechanism_generation="v2-v3-frozen",
        hold30_fast_raw_context_sessions=None,
        hold30_slow_context_sessions=None,
        hold30_hazard_bound_mode="hard_clip",
        hold30_exact_hold_mixture=False,
        hold30_exact_hold_logit_bias=None,
        raw_recent_days=0,
        daily_lookback=63,
        max_days=63,
    )
    with pytest.raises(ValueError, match="M03R setting identity requires"):
        DailyCrossSectionPolicy(frozen_geometry)
    for legacy_id in ("hold30-m02-age-hazard", "hold30a-m03-alpha-core"):
        with pytest.raises(ValueError, match="exact M03R setting identity"):
            DailyCrossSectionPolicy(
                replace(_m03r_policy_config(), hold30_setting=legacy_id)
            )
    with pytest.raises(ValueError, match="m03r-v1 requires explicit"):
        DailyCrossSectionPolicy(
            replace(
                _m03r_policy_config(),
                hold30_fast_raw_context_sessions=None,
                hold30_slow_context_sessions=None,
            )
        )


def test_explicit_two_speed_contract_binds_fast42_and_slow252() -> None:
    policy = DailyCrossSectionPolicy(_m03r_policy_config()).eval()
    assert policy.hold30_context_contract is not None
    assert policy.hold30_context_contract.fast_raw_context_sessions == 42
    assert policy.hold30_context_contract.slow_context_sessions == 252
    mask = policy._raw_day_mask(252)
    assert mask[:210] == [False] * 210
    assert mask[210:] == [True] * 42

    with pytest.raises(ValueError, match="daily_lookback must equal"):
        DailyCrossSectionPolicy(replace(_m03r_policy_config(), daily_lookback=63))
    with pytest.raises(ValueError, match="max_days must cover"):
        DailyCrossSectionPolicy(replace(_m03r_policy_config(), max_days=251))

    short = DailyCrossSectionPolicy(
        replace(
            _m03r_policy_config(),
            hold30_setting="A09-no-long-context",
            daily_lookback=63,
            max_days=63,
            hold30_slow_context_sessions=63,
            alpha_downside_penalty_kappa=0.75,
            alpha_uncertainty_log_scale_bounds=(-4.0, 2.0),
        )
    )
    assert short.hold30_context_contract is not None
    assert short.hold30_context_contract.slow_context_sessions == 63


def test_smooth_hazard_has_nonzero_gradient_beyond_old_clip_boundaries() -> None:
    raw = torch.tensor(
        [-100.0, -13.0, -12.0, 0.0, 12.0, 13.0, 100.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    bounded = bound_hold30_hazard_residual(raw, mode="smooth_tanh")
    assert bool((bounded > -12.0).all())
    assert bool((bounded < 12.0).all())
    bounded.sum().backward()
    assert raw.grad is not None
    assert bool((raw.grad > 0).all())

    legacy = raw.detach().clone().requires_grad_(True)
    clip_hold30_hazard_residual(legacy).sum().backward()
    assert legacy.grad is not None
    assert legacy.grad[0].item() == 0.0
    assert legacy.grad[-1].item() == 0.0


def test_hazard_telemetry_excludes_ineligible_sentinels_and_reports_gradient() -> None:
    raw = torch.tensor([[-99.0, -12.0, 0.0, 12.0, 99.0]], dtype=torch.float64)
    eligible = torch.tensor([[False, True, True, True, False]])
    hold = torch.tensor([[1.0, 0.0, 0.5, 1.0, 1.0]], dtype=torch.float64)
    hard = hold30_hazard_telemetry(
        raw,
        mode="hard_clip",
        eligible=eligible,
        exact_hold_probability=hold,
    )
    assert hard.observation_count == 3
    assert hard.fraction_raw_at_or_below_min == pytest.approx(1.0 / 3.0)
    assert hard.fraction_raw_at_or_above_max == pytest.approx(1.0 / 3.0)
    assert hard.transform_nonzero_gradient_fraction == pytest.approx(1.0 / 3.0)
    assert hard.exact_hold_probability_mean == pytest.approx(0.5)
    assert hard.exact_hold_probability_near_zero_fraction == pytest.approx(1.0 / 3.0)
    assert hard.exact_hold_probability_near_one_fraction == pytest.approx(1.0 / 3.0)

    smooth = hold30_hazard_telemetry(raw, mode="smooth_tanh", eligible=eligible)
    assert smooth.transform_nonzero_gradient_fraction == 1.0
    assert smooth.bounded_min > -12.0
    assert smooth.bounded_max < 12.0


def test_exact_hold_mixture_has_an_exact_atom_and_differentiable_release() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    common = {
        "benchmark_weights": weights.clone(),
        "trade_mask": torch.ones_like(weights, dtype=torch.bool),
        "risk_asset_caps": torch.ones_like(weights),
        "risk_gross_max": torch.ones(1, dtype=torch.float64),
    }
    built = build_h2_hold30_action(
        weights,
        _ledger_from_weights(weights),
        entry_scores=torch.zeros_like(weights),
        hazard_residual=torch.zeros_like(weights),
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        exact_hold_probability=torch.ones_like(weights),
        **common,
    )
    assert torch.equal(built.proposed_release, torch.zeros_like(weights))
    assert torch.equal(built.target_weights, weights)

    probability = torch.full_like(weights, 0.5, requires_grad=True)
    released = hold30_proposed_release(
        _ledger_from_weights(weights),
        torch.zeros_like(weights),
        exact_hold_probability=probability,
    )
    released[:, 1:].sum().backward()
    assert probability.grad is not None
    assert bool((probability.grad[:, 1:] < 0).all())


def test_alpha_head_supports_smooth_mixture_and_fixed_hazard_comparator() -> None:
    market, weights, age, available = _head_inputs()
    mixture = Hold30AlphaHead(
        _alpha_config(exact_hold_mixture=True, exact_hold_logit_bias=1.25)
    )
    output = mixture(market, weights, age, available)
    assert output.raw_hazard_residual.shape == weights.shape
    assert output.exact_hold_probability is not None
    assert set(output.exact_hold_probability.unique().tolist()) <= {0.0, 1.0}
    assert output.signal_confidence is not None
    torch.testing.assert_close(
        output.active_risk_scale,
        0.04 * output.signal_confidence,
    )
    replace(output, active_risk_scale=torch.zeros_like(output.active_risk_scale)).validate()
    risky = available.clone()
    risky[:, 0] = False
    assert bool((output.hazard_residual[risky] > -12.0).all())
    assert bool((output.hazard_residual[risky] < 12.0).all())
    loss = (
        output.hazard_residual[available].sum()
        + output.exact_hold_probability[available].sum()
    )
    loss.backward()
    assert mixture.hazard_head.weight.grad is not None
    assert mixture.exact_hold_head is not None
    assert mixture.exact_hold_head.weight.grad is not None

    fixed = Hold30AlphaHead(
        _alpha_config(
            setting_id=A08,
            fixed_hazard_residual=0.0,
            exact_hold_mixture=False,
            exact_hold_logit_bias=None,
        )
    ).eval()
    first = fixed(market, weights, age, available)
    second = fixed(market * -3.0, weights.flip(-1), 1.0 - age, available)
    assert torch.equal(
        first.hazard_residual[risky], torch.zeros_like(first.hazard_residual[risky])
    )
    torch.testing.assert_close(first.hazard_residual, second.hazard_residual)
    assert all(
        not parameter.requires_grad for parameter in fixed.hazard_head.parameters()
    )

    with pytest.raises(Hold30AlphaModelError, match="confidence calibration"):
        Hold30AlphaHead(_alpha_config(confidence_calibration_manifest_sha256=None))


def test_zero_active_risk_scale_is_a_valid_null_copy_action() -> None:
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64)
    built = build_h2_hold30_action(
        weights,
        _ledger_from_weights(weights),
        entry_scores=torch.zeros_like(weights),
        hazard_residual=torch.zeros_like(weights),
        exposure_residual=torch.zeros(1, dtype=torch.float64),
        exact_hold_probability=torch.ones_like(weights),
        benchmark_weights=weights.clone(),
        trade_mask=torch.ones_like(weights, dtype=torch.bool),
        risk_asset_caps=torch.ones_like(weights),
        risk_gross_max=torch.ones(1, dtype=torch.float64),
    )
    assert torch.equal(built.target_weights, weights)

    # The alpha builder must also accept zero confidence/active risk. With an
    # exact-hold decision, this is the protocol's literal null-copy action.
    alpha = build_alpha_hold30_action(
        weights,
        _ledger_from_weights(weights),
        risk_adjusted_score=torch.ones_like(weights),
        hazard_residual=torch.zeros_like(weights),
        active_risk_scale=torch.zeros(1, dtype=torch.float64),
        benchmark_weights=weights.clone(),
        trade_mask=torch.ones_like(weights, dtype=torch.bool),
        risk_asset_caps=torch.ones_like(weights),
        risk_gross_max=torch.ones(1, dtype=torch.float64),
        exact_hold_probability=torch.ones_like(weights),
    )
    assert torch.equal(alpha.target_weights, weights)


def _reference_waterfill(
    requested_mass: torch.Tensor,
    direction: torch.Tensor,
    capacity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent sorted-breakpoint solution for small test rows."""

    result = torch.zeros_like(direction)
    effective_rows = []
    for row in range(direction.shape[0]):
        valid = (direction[row] > 0) & (capacity[row] > 0)
        available = capacity[row][valid].sum()
        effective = torch.minimum(requested_mass[row].clamp_min(0), available)
        effective_rows.append(effective)
        if not bool(valid.any()) or effective.item() == 0:
            continue
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        ratios = capacity[row, indices] / direction[row, indices]
        order = indices[torch.argsort(ratios)]
        fixed_mass = direction.new_zeros(())
        active_direction = direction[row, valid].sum()
        alpha = None
        for index in order:
            breakpoint = capacity[row, index] / direction[row, index]
            mass_at_breakpoint = fixed_mass + breakpoint * active_direction
            if bool(effective <= mass_at_breakpoint + 1e-14):
                alpha = (effective - fixed_mass) / active_direction
                break
            fixed_mass = fixed_mass + capacity[row, index]
            active_direction = active_direction - direction[row, index]
        if alpha is None:
            result[row, valid] = capacity[row, valid]
        else:
            result[row] = torch.where(
                valid,
                torch.minimum(capacity[row], alpha * direction[row]),
                torch.zeros_like(direction[row]),
            )
    return result, torch.stack(effective_rows)


def test_random_waterfill_matches_reference_and_is_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(20260805)
    for _ in range(32):
        direction = torch.rand(3, 11, generator=generator, dtype=torch.float64)
        direction[direction < 0.15] = 0.0
        capacity = 0.03 * torch.rand(3, 11, generator=generator, dtype=torch.float64)
        capacity[capacity < 0.004] = 0.0
        requested = 0.20 * torch.rand(3, generator=generator, dtype=torch.float64)
        actual, effective = capped_waterfill(requested, direction, capacity)
        expected, expected_effective = _reference_waterfill(
            requested,
            direction,
            capacity,
        )
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
        torch.testing.assert_close(effective, expected_effective, atol=1e-13, rtol=0)
        assert bool((actual >= 0).all())
        assert bool((actual <= capacity + 2e-12).all())
        torch.testing.assert_close(actual.sum(-1), effective, atol=2e-11, rtol=0)

        permutation = torch.randperm(direction.shape[1], generator=generator)
        permuted, _ = capped_waterfill(
            requested,
            direction[:, permutation],
            capacity[:, permutation],
        )
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(permuted[:, inverse], actual, atol=2e-11, rtol=2e-10)
