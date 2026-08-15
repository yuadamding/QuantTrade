from __future__ import annotations

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import M03R_V14_SETTINGS
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    Top2000M03RV14PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v14_fold import (
    M03RV14PanelEpisodeSchedule,
    build_m03r_v14_paired_input_binding,
    render_m03r_v14_fold_geometries,
    render_m03r_v14_training_update_plan,
)
from rl_quant.training.top2000_m03r_v14_pretraining_optimizer import (
    build_m03r_v14_optimizer,
)
from rl_quant.training.top2000_m03r_v14_pretraining_runtime import (
    M03RV14PretrainingRuntimeError,
    build_m03r_v14_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v14_pretraining_step import (
    _state_is_finite,
    train_m03r_v14_predictive_batch_update,
)


def _surfaces() -> tuple[Hold30Sequence, object]:
    states = 378
    assets = 14
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.87
    initial[0, 1:] = 0.01
    availability = torch.ones((states, 1, assets), dtype=torch.bool)
    # Asset 12 is eligible at the decision origin but unavailable in the
    # future label path.  It must remain in the action operator only.
    availability[252:256, 0, 12] = False
    # Asset 11 has a complete future outcome but was unavailable at the
    # decision origin.  It must enter neither operator nor any loss reduction.
    availability[251, 0, 11] = False
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
        axis_id="v14-runtime-test",
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
    return sequence, exposure


def test_v14_batch_is_h3_only_full_context_and_separates_action_mask() -> None:
    sequence, exposure = _surfaces()
    policy = Top2000M03RV14PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    batch = build_m03r_v14_batch_from_origin_states(
        policy,
        M03R_V14_SETTINGS[0],
        torch.randn((1, 1, sequence.num_assets, 16)),
        sequence,
        torch.tensor([251]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=378,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    assert batch.objective.predicted_mean.shape == (1, sequence.num_assets)
    assert batch.objective.target_log_return.shape == (1, sequence.num_assets)
    assert batch.objective.setting.selected_horizon_sessions == 3
    assert not bool(batch.target_residual_operators[0].qualified_asset_mask[12])
    assert bool(batch.action_residual_operators[0].qualified_asset_mask[12])
    assert not bool(batch.target_residual_operators[0].qualified_asset_mask[11])
    assert not bool(batch.action_residual_operators[0].qualified_asset_mask[11])
    assert not bool(batch.objective.valid[0, 11])
    assert batch.objective.predicted_mean.data_ptr() != batch.raw_predicted_mean.data_ptr()
    assert batch.objective.predicted_mean.grad_fn is not None
    assert (
        batch.target_residual_operators[0].receipt_sha256
        != batch.action_residual_operators[0].receipt_sha256
    )
    assert batch.receipt_sha256


def test_v14_batch_rejects_less_than_full_local_context() -> None:
    sequence, exposure = _surfaces()
    policy = Top2000M03RV14PredictivePolicy(
        1,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    with pytest.raises(M03RV14PretrainingRuntimeError, match="full-context"):
        build_m03r_v14_batch_from_origin_states(
            policy,
            M03R_V14_SETTINGS[1],
            torch.randn((1, 1, sequence.num_assets, 16)),
            sequence,
            torch.tensor([250]),
            sequence_global_state_start=0,
            split="training",
            split_start_inclusive=0,
            split_stop_exclusive=378,
            fold_index=0,
            source_array_sha256="c" * 64,
            asset_axis_sha256="a" * 64,
            origin_risk_exposures=exposure,  # type: ignore[arg-type]
        )


def test_v14_optimizer_has_no_disconnected_rank_group_and_mutates_once() -> None:
    sequence, exposure = _surfaces()
    policy = Top2000M03RV14PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    optimizer, partition = build_m03r_v14_optimizer(policy)
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "encoder",
        "direct-h3-heads",
    ]
    assert all("rank" not in name for name in partition.head_parameter_names)
    geometry = render_m03r_v14_fold_geometries(1001)[0]
    schedule = M03RV14PanelEpisodeSchedule(
        protocol_common_data_sha256="1" * 64,
        cache_sha256="2" * 64,
        asset_axis_sha256="a" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v14_fold_geometries(1001)
        ),
    )
    candidates = tuple(
        render_m03r_v14_training_update_plan(
            schedule,
            geometry,
            setting_index=0,
            completed_update=update,
        )
        for update in range(geometry.training_block_count)
    )
    plan = next(row for row in candidates if row.epoch_block_index == 0)
    assert plan.episode_start == 0
    rank_origins = torch.tensor(plan.rank_origins[0], dtype=torch.long)
    batch = build_m03r_v14_batch_from_origin_states(
        policy,
        M03R_V14_SETTINGS[0],
        torch.randn((rank_origins.numel(), 1, sequence.num_assets, 16)),
        sequence,
        rank_origins,
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=geometry.training_target_stop_exclusive,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    receipt = train_m03r_v14_predictive_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        plan,
        build_m03r_v14_paired_input_binding(
            plan,
            cache_sha256="2" * 64,
            source_array_sha256="c" * 64,
            asset_axis_sha256="a" * 64,
        ),
        completed_updates=plan.completed_update,
        distributed_rank=0,
        distributed_world_size=2,
        gradient_synchronizer=lambda _parameters: None,
    )
    assert receipt.completed_updates_after == plan.completed_update + 1
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256
    assert receipt.optimizer_state_before_sha256 != receipt.optimizer_state_after_sha256
    assert receipt.economic_optimizer_updates == 0
    assert receipt.outer_2026_accessed is False
    assert receipt.distributed_gradient_synchronized is True
    assert receipt.local_origin_count == len(plan.rank_origins[0])
    assert receipt.global_origin_count == len(plan.global_origins)


def test_v14_post_step_finite_guard_covers_parameters_and_adam_state() -> None:
    layer = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1.0e-3)
    layer(torch.ones((2, 3))).sum().backward()
    optimizer.step()
    parameters = tuple(layer.parameters())
    assert _state_is_finite(parameters, optimizer)
    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].view(-1)[0] = float("inf")
    assert not _state_is_finite(parameters, optimizer)
    first_state["exp_avg"].view(-1)[0] = 0.0
    parameters[0].data.view(-1)[0] = float("nan")
    assert not _state_is_finite(parameters, optimizer)
