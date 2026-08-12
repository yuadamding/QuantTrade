from __future__ import annotations

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import M03R_V11_SETTINGS
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    build_m03r_v11_alpha_batch_from_origin_states,
)


def test_regression_ineligible_available_asset_never_enters_v11_batch() -> None:
    states = 70
    assets = 12
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.89
    initial[0, 1:] = 0.01
    ledger = CohortLedger.from_weights(
        initial, cash_index=0, initial_age=0, track_initial_units=True
    )
    availability = torch.ones((states, 1, assets), dtype=torch.bool)
    asset_returns = torch.zeros((states - 1, 1, assets), dtype=torch.float32)
    asset_returns[:, 0, 1:] = torch.linspace(-0.001, 0.001, assets - 1)
    sequence = Hold30Sequence(
        decision_state=torch.zeros((states, 1, assets, 1)),
        asset_returns=asset_returns,
        decision_available=availability,
        fill_membership=availability.clone(),
        fill_availability=availability.clone(),
        benchmark_weights=initial.unsqueeze(0).expand(states, -1, -1).clone(),
        risk_asset_caps=torch.ones((states, 1, assets)),
        risk_gross_max=torch.ones((states, 1)),
        benchmark_net_returns=torch.zeros((states - 1, 1)),
        initial_ledger=ledger,
        cost_rate=0.002,
        axis_id="v11-qualified-target-test",
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
    policy = Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    batch = build_m03r_v11_alpha_batch_from_origin_states(
        policy,
        M03R_V11_SETTINGS[1],
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
    assert not bool(batch.corrected_batch.valid[0, -1].any())
    assert bool((batch.corrected_batch.target_log_return[0, -1] == 0.0).all())
    assert batch.available_risky_asset_count == (11, 11, 11, 11)
    assert batch.factor_qualified_risky_asset_count == (10, 10, 10, 10)
