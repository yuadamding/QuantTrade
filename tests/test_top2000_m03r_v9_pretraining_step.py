from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerError,
    build_m03r_v9_alpha_pretraining_optimizer,
    validate_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    M03RV9AlphaStepError,
    train_m03r_v9_alpha_pretraining_update,
)


def _policy(setting: int = 0) -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        setting,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(policy: Top2000M03RV9PredictivePolicy) -> M03RV9AlphaPretrainingBatch:
    dates, assets = 3, 5
    mean = torch.randn((dates, assets, 4), dtype=torch.float32)
    mean[:, 0] = 0.0
    # Use actual policy heads so every expected predictive parameter receives
    # a gradient through one reviewed hidden-state surface.
    hidden = torch.randn((dates, assets, 16), dtype=torch.float32)
    alpha_head = policy.source_policy.core.alpha_head
    assert alpha_head is not None
    predicted_mean = alpha_head.auxiliary_head(hidden)
    predicted_scale = policy.alpha_scale_head(hidden)
    target = mean.detach()
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV9AlphaPretrainingBatch(
        predicted_mean=predicted_mean,
        predicted_log_scale=predicted_scale,
        target_log_return=target,
        valid=valid,
        origin_indices=torch.tensor((0, 1, 2), dtype=torch.long),
        split="training",
        target_mode=policy.setting.target_mode,
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256=(
            "c" * 64 if policy.setting.target_mode == "factor-residual" else None
        ),
    )


def test_optimizer_contains_only_encoder_and_two_alpha_heads() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)
    assert any(
        "alpha_head.auxiliary_head" in name
        for name in partition.prediction_head_parameter_names
    )
    assert any(
        "alpha_scale_head" in name for name in partition.prediction_head_parameter_names
    )
    assert not any(
        "hazard" in name
        for name in (
            *partition.encoder_parameter_names,
            *partition.prediction_head_parameter_names,
        )
    )
    optimizer.param_groups[0]["lr"] = 9.0
    with pytest.raises(M03RV9AlphaOptimizerError, match="group drifted"):
        validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)


def test_single_step_binds_normalized_setting_loss_and_no_qualification() -> None:
    policy = _policy(1)
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    receipt = train_m03r_v9_alpha_pretraining_update(
        policy,
        _batch(policy),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.component_weights == (0.0, 0.60, 0.40)
    assert not receipt.early_stopping_enabled
    assert not receipt.qualification_evaluated_during_update
    assert receipt.completed_updates_after == 1
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256


def test_qualification_batch_and_update_64_resume_are_rejected() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    with pytest.raises(M03RV9AlphaStepError, match="training batch"):
        train_m03r_v9_alpha_pretraining_update(
            policy,
            replace(_batch(policy), split="qualification"),
            optimizer,
            partition,
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
    with pytest.raises(M03RV9AlphaStepError, match="outside 0..63"):
        train_m03r_v9_alpha_pretraining_update(
            policy,
            _batch(policy),
            optimizer,
            partition,
            completed_updates=64,
            distributed_rank=0,
            distributed_world_size=1,
        )
