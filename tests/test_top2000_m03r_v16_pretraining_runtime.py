from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SETTINGS,
    M03R_V16_SURVIVAL_WEIGHTS,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
    render_m03r_v16_training_update_plan,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    build_m03r_v16_optimizer,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16PretrainingRuntimeError,
    build_m03r_v16_batch_from_origin_states,
    m03r_v16_selection_weights,
)
from rl_quant.training.top2000_m03r_v16_pretraining_step import (
    train_m03r_v16_score_batch_update,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16ValidationError,
    evaluate_m03r_v16_inner_validation_batch,
)


def _surfaces() -> tuple[Hold30Sequence, object]:
    states = 345
    assets = 14
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.87
    initial[0, 1:] = 0.01
    availability = torch.ones((states, 1, assets), dtype=torch.bool)
    availability[252:283, 0, 12] = False
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
        axis_id="v16-runtime-test",
    )
    exposure_states = 1001
    loadings = torch.zeros((exposure_states, assets, 6), dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, assets - 1, dtype=torch.float64)
    loadings[:, 1:, 0] = 1.0
    loadings[:, 1:7, 1] = 1.0
    loadings[:, 7:, 2] = 1.0
    loadings[:, 1:, 3] = x
    loadings[:, 1:, 4] = x.square()
    loadings[:, 1:, 5] = x.pow(3)
    weights = torch.ones((exposure_states, assets), dtype=torch.float64)
    weights[:, 0] = 0.0
    decision_time = torch.arange(exposure_states, dtype=torch.int64) * 86_400_000
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
        .expand(exposure_states, assets, 3)
        .clone(),
    )
    return sequence, exposure


def test_v16_target_weights_preserve_declared_economic_units() -> None:
    h21 = m03r_v16_selection_weights(M03R_V16_SETTINGS[0])
    h30 = m03r_v16_selection_weights(M03R_V16_SETTINGS[1])
    survival = m03r_v16_selection_weights(M03R_V16_SETTINGS[2])
    assert torch.equal(h21, torch.ones(21, dtype=torch.float64))
    assert torch.equal(h30, torch.ones(30, dtype=torch.float64))
    assert tuple(float(value) for value in survival) == pytest.approx(
        M03R_V16_SURVIVAL_WEIGHTS
    )
    assert float(survival.sum()) > 20.0


def test_v16_survival_weighting_is_not_a_mandatory_expiry() -> None:
    survival = m03r_v16_selection_weights(M03R_V16_SETTINGS[2])
    assert bool((survival > 0.0).all())
    assert float(survival[-1]) > 0.0
    assert float(survival[0]) > float(survival[-1])


def test_v16_settings_use_identical_common30_masks_and_operators() -> None:
    sequence, exposure = _surfaces()
    state = torch.randn((1, 1, sequence.num_assets, 16))
    batches = []
    for setting_index in range(3):
        torch.manual_seed(17)
        policy = Top2000M03RV16PredictivePolicy(
            setting_index,
            token_dim=16,
            raw_stock_chunk=8,
            activation_checkpointing=False,
        )
        batches.append(
            build_m03r_v16_batch_from_origin_states(
                policy,
                M03R_V16_SETTINGS[setting_index],
                state,
                sequence,
                torch.tensor([251]),
                sequence_global_state_start=0,
                split="training",
                split_start_inclusive=0,
                split_stop_exclusive=345,
                fold_index=0,
                source_array_sha256="c" * 64,
                asset_axis_sha256="a" * 64,
                origin_risk_exposures=exposure,  # type: ignore[arg-type]
            )
        )
    assert all(
        batch.objective.executable_selection_score_z.shape == (1, 14)
        for batch in batches
    )
    assert (
        len({batch.selection_target_operators[0].receipt_sha256 for batch in batches})
        == 1
    )
    assert len({batch.action_operators[0].receipt_sha256 for batch in batches}) == 1
    assert all(
        torch.equal(
            batches[0].objective.selection_valid, batch.objective.selection_valid
        )
        for batch in batches[1:]
    )
    assert bool(batches[0].action_operators[0].qualified_asset_mask[12])
    assert not bool(batches[0].selection_target_operators[0].qualified_asset_mask[12])
    assert not bool(batches[0].action_operators[0].qualified_asset_mask[11])
    assert batches[0].objective.executable_selection_score_z.grad_fn is not None
    assert max(batches[0].action_returned_dtype_exposure_errors) <= 1.0e-5


def test_v16_score_step_mutates_encoder_and_selection_head_only() -> None:
    sequence, exposure = _surfaces()
    geometries = render_m03r_v16_fold_geometries(1001)
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256="d" * 64,
        cache_sha256="e" * 64,
        asset_axis_sha256="a" * 64,
        fold_geometry_sha256=tuple(value.receipt_sha256 for value in geometries),
    )
    geometry = geometries[0]
    plans = tuple(
        render_m03r_v16_training_update_plan(
            schedule, geometry, setting_index=0, completed_update=cursor
        )
        for cursor in range(geometry.training_block_count)
    )
    plan = next(
        value
        for value in plans
        if value.episode_start == 0 and len(value.global_origins) == 63
    )
    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    optimizer, partition = build_m03r_v16_optimizer(policy, "score")
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        torch.randn((63, 1, sequence.num_assets, 16)),
        sequence,
        torch.tensor(plan.global_origins),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=geometry.training_origin_start_inclusive,
        split_stop_exclusive=geometry.inner_validation_origin_start_inclusive,
        fold_index=0,
        source_array_sha256="f" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    receipt = train_m03r_v16_score_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        plan,
        completed_updates=plan.completed_update,
        distributed_rank=0,
        distributed_world_size=1,
    )
    receipt.validate()
    assert receipt.encoder_version_root_before != receipt.encoder_version_root_after
    assert (
        receipt.selection_head_version_root_before
        != receipt.selection_head_version_root_after
    )
    assert receipt.timing_optimizer_updates == 0
    assert receipt.uncertainty_calibration_updates == 0


def test_v16_inner_validation_scores_the_executable_selection_tensor() -> None:
    sequence, exposure = _surfaces()
    geometry = render_m03r_v16_fold_geometries(1001)[0]
    policy = Top2000M03RV16PredictivePolicy(
        1,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    local_origins = torch.arange(251, 314, dtype=torch.long)
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[1],
        torch.randn((63, 1, sequence.num_assets, 16)),
        sequence,
        local_origins,
        sequence_global_state_start=160,
        split="inner_validation",
        split_start_inclusive=geometry.inner_validation_origin_start_inclusive,
        split_stop_exclusive=geometry.training_target_stop_exclusive,
        fold_index=0,
        source_array_sha256="f" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    receipt = evaluate_m03r_v16_inner_validation_batch(
        batch,
        geometry,
        epoch_index=0,
        completed_score_updates=geometry.training_block_count,
        model_state_sha256=batch.policy_state_binding_sha256,
        epoch_checkpoint_file_sha256="2" * 64,
    )
    receipt.validate()
    assert receipt.origin_count == 63
    assert receipt.setting_index == 1
    assert receipt.batch_receipt_sha256 == batch.receipt_sha256
    with pytest.raises(M03RV16ValidationError, match="batch or epoch"):
        evaluate_m03r_v16_inner_validation_batch(
            batch,
            geometry,
            epoch_index=0,
            completed_score_updates=geometry.training_block_count,
            model_state_sha256="9" * 64,
            epoch_checkpoint_file_sha256="2" * 64,
        )


def test_v16_validation_batch_rejects_training_state_binding() -> None:
    sequence, exposure = _surfaces()
    policy = Top2000M03RV16PredictivePolicy(
        0, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False
    )
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        torch.randn((1, 1, sequence.num_assets, 16)),
        sequence,
        torch.tensor([251]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=345,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    with pytest.raises(M03RV16PretrainingRuntimeError, match="built batch drifted"):
        replace(
            batch,
            split="inner_validation",
            policy_state_binding_kind="parameter-version-root",
        ).validate()
