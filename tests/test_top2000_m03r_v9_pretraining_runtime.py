from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03RV9HorizonBinding,
    resolve_m03r_v9_setting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingError,
    m03r_v9_alpha_pretraining_loss,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    _factor_residualize,
    build_m03r_v9_alpha_batch_from_origin_states,
    qualify_m03r_v9_origin_risk_exposures,
)


def _sequence() -> Hold30Sequence:
    positions = 70
    assets = 4
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    ledger = CohortLedger.from_weights(
        weights, cash_index=0, initial_age=0, track_initial_units=True
    )
    availability = torch.ones((positions, 1, assets), dtype=torch.bool)
    returns = torch.zeros((positions - 1, 1, assets))
    returns[..., 1] = -0.001
    returns[..., 2] = 0.0
    returns[..., 3] = 0.001
    return Hold30Sequence(
        decision_state=torch.zeros((positions, 1, assets, 1)),
        asset_returns=returns,
        decision_available=availability,
        fill_membership=availability.clone(),
        fill_availability=availability.clone(),
        benchmark_weights=weights.unsqueeze(0).expand(positions, -1, -1).clone(),
        risk_asset_caps=torch.ones((positions, 1, assets)),
        risk_gross_max=torch.ones((positions, 1)),
        benchmark_net_returns=torch.zeros((positions - 1, 1)),
        initial_ledger=ledger,
        cost_rate=0.002,
        axis_id="v9-target-test",
    )


def _policy(setting: int) -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        setting,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _exposures() -> object:
    loadings = torch.zeros((70, 4, 6), dtype=torch.float64)
    loadings[:, 1:, 0] = 1.0
    loadings[:, 1:3, 1] = 1.0
    loadings[:, 3, 2] = 1.0
    loadings[:, 1:, 3] = torch.tensor([-1.0, 0.0, 1.0])
    loadings[:, 1:, 4] = torch.tensor([-1.0, 0.0, 1.0])
    loadings[:, 1:, 5] = torch.tensor([1.0, -2.0, 1.0])
    weights = torch.ones((70, 4), dtype=torch.float64)
    weights[:, 0] = 0.0
    decision_time = torch.arange(70, dtype=torch.int64) * 86_400_000
    available_time = decision_time[:, None, None].expand(70, 4, 3).clone()
    return qualify_m03r_v9_origin_risk_exposures(
        state_start_index=0,
        cash_index=0,
        projector_exposure_names=(
            "sector-a",
            "sector-b",
            "active-beta",
            "style-return",
            "style-volatility",
        ),
        projector_exposure_families=(
            "sector",
            "sector",
            "active-beta",
            "style-risk",
            "style-risk",
        ),
        asset_axis_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        exposure_loadings=loadings,
        regression_weights=weights,
        decision_timestamp_ms=decision_time,
        exposure_available_timestamp_ms=available_time,
    )


def test_factor_residual_target_removes_origin_available_style_return() -> None:
    sequence = _sequence()
    states = torch.randn((1, 1, 4, 16), requires_grad=True)
    residual = build_m03r_v9_alpha_batch_from_origin_states(
        _policy(0),
        resolve_m03r_v9_setting(0),
        states,
        sequence,
        torch.tensor([0]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=70,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=_exposures(),  # type: ignore[arg-type]
    )
    benchmark = build_m03r_v9_alpha_batch_from_origin_states(
        _policy(2),
        resolve_m03r_v9_setting(2),
        states.detach().clone().requires_grad_(True),
        sequence,
        torch.tensor([0]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=70,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=None,
    )

    expected_h5 = 5.0 * math.log1p(0.001)
    assert benchmark.target_log_return[0, 3, 0].item() == pytest.approx(expected_h5)
    # The frozen ridge makes the residual intentionally nonzero but removes
    # more than 99.9% of this exactly style-spanned cross section.
    assert residual.target_log_return[0, 1:, 0].abs().max().item() < 2.0e-6
    m03r_v9_alpha_pretraining_loss(residual, ranking_enabled=True).total.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_factor_residualization_accepts_cpu_exposures_for_cuda_targets() -> None:
    target_cpu = torch.tensor([0.0, -0.01, 0.0, 0.01], dtype=torch.float32)
    valid_cpu = torch.tensor([False, True, True, True])
    loadings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64)

    expected = _factor_residualize(target_cpu, valid_cpu, loadings, weights)
    observed = _factor_residualize(
        target_cpu.cuda(),
        valid_cpu.cuda(),
        loadings,
        weights,
    )

    assert observed.device.type == "cuda"
    torch.testing.assert_close(observed.cpu(), expected)


def test_asset_axis_and_target_mode_mismatch_fail_before_training() -> None:
    sequence = _sequence()
    states = torch.randn((1, 1, 4, 16))
    with pytest.raises(M03RV9AlphaPretrainingError, match="asset axes"):
        build_m03r_v9_alpha_batch_from_origin_states(
            _policy(0),
            resolve_m03r_v9_setting(0),
            states,
            sequence,
            torch.tensor([0]),
            sequence_global_state_start=0,
            split="training",
            split_start_inclusive=0,
            split_stop_exclusive=70,
            fold_index=0,
            source_array_sha256="c" * 64,
            asset_axis_sha256="d" * 64,
            origin_risk_exposures=_exposures(),  # type: ignore[arg-type]
        )
    with pytest.raises(M03RV9AlphaPretrainingError, match="must not residualize"):
        build_m03r_v9_alpha_batch_from_origin_states(
            _policy(2),
            resolve_m03r_v9_setting(2),
            states,
            sequence,
            torch.tensor([0]),
            sequence_global_state_start=0,
            split="training",
            split_start_inclusive=0,
            split_stop_exclusive=70,
            fold_index=0,
            source_array_sha256="c" * 64,
            asset_axis_sha256="a" * 64,
            origin_risk_exposures=_exposures(),  # type: ignore[arg-type]
        )


def test_future_available_or_incomplete_exposure_family_fails_qualification() -> None:
    exposure = _exposures()
    future = exposure.exposure_available_timestamp_ms.clone()  # type: ignore[attr-defined]
    future[4, 2, 0] += 1
    with pytest.raises(M03RV9AlphaPretrainingError, match="malformed"):
        replace(  # type: ignore[arg-type]
            exposure,
            exposure_available_timestamp_ms=future,
        ).validate()
    with pytest.raises(M03RV9AlphaPretrainingError, match="malformed"):
        replace(  # type: ignore[arg-type]
            exposure,
            projector_exposure_families=(
                "style-risk",
                "style-risk",
                "active-beta",
                "style-risk",
                "style-risk",
            ),
        ).validate()
    with pytest.raises(M03RV9AlphaPretrainingError, match="malformed"):
        replace(  # type: ignore[arg-type]
            exposure,
            availability_family_names=("sector", "style-risk", "active-beta"),
        ).validate()
