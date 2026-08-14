from __future__ import annotations

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import M03R_V12_SETTINGS
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v12_pretraining_runtime import (
    build_m03r_v12_batch_from_origin_states,
)


def test_v12_batch_uses_one_corrected_mask_and_separate_rank_scores() -> None:
    states = 70
    assets = 12
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.89
    initial[0, 1:] = 0.01
    availability = torch.ones((states, 1, assets), dtype=torch.bool)
    returns = torch.zeros((states - 1, 1, assets), dtype=torch.float32)
    returns[:, 0, 1:] = torch.linspace(-0.001, 0.001, assets - 1)
    sequence = Hold30Sequence(
        decision_state=torch.zeros((states, 1, assets, 1)),
        asset_returns=returns,
        decision_available=availability,
        fill_membership=availability.clone(),
        fill_availability=availability.clone(),
        benchmark_weights=initial.unsqueeze(0).expand(states, -1, -1).clone(),
        risk_asset_caps=torch.ones((states, 1, assets)),
        risk_gross_max=torch.ones((states, 1)),
        benchmark_net_returns=torch.zeros((states - 1, 1)),
        initial_ledger=CohortLedger.from_weights(
            initial, cash_index=0, initial_age=0, track_initial_units=True
        ),
        cost_rate=0.002,
        axis_id="v12-runtime-test",
    )
    loadings = torch.zeros((states, assets, 6), dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, assets - 1, dtype=torch.float64)
    loadings[:, 1:, 0] = 1.0
    loadings[:, 1:7, 1] = 1.0
    loadings[:, 7:, 2] = 1.0
    loadings[:, 1:, 3] = x
    loadings[:, 1:, 4] = x.square()
    loadings[:, 1:, 5] = x.pow(3)
    weights = torch.ones((states, assets), dtype=torch.float64)
    weights[:, 0] = 0.0
    weights[:, -1] = 0.0
    decision_time = torch.arange(states, dtype=torch.int64) * 86_400_000
    exposure = qualify_m03r_v9_origin_risk_exposures(
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
        exposure_available_timestamp_ms=decision_time[:, None, None]
        .expand(states, assets, 3)
        .clone(),
    )
    policy = Top2000M03RV12PredictivePolicy(
        1,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    batch = build_m03r_v12_batch_from_origin_states(
        policy,
        M03R_V12_SETTINGS[1],
        torch.randn((1, 1, assets, 16)),
        sequence,
        torch.tensor([0]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=70,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,
    )
    assert batch.predicted_rank_score.shape == (1, assets, 5)
    assert batch.predicted_rank_score.data_ptr() != (batch.predicted_mean.data_ptr())
    assert not bool(batch.valid[0, -1].any())
    assert batch.valid.shape[-1] == 5
    assert batch.receipt_sha256
