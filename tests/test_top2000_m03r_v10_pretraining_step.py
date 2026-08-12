from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTINGS
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03RV10AlphaPretrainingBatch,
    M03RV10AlphaStepError,
    train_m03r_v10_alpha_pretraining_update,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    build_m03r_v9_alpha_pretraining_optimizer,
)


def _policy(setting: int = 0) -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        setting,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(
    policy: Top2000M03RV9PredictivePolicy,
    *,
    v10_setting: int = 1,
) -> M03RV10AlphaPretrainingBatch:
    hidden = torch.randn((3, 5, 16), dtype=torch.float32)
    alpha_head = policy.source_policy.core.alpha_head
    assert alpha_head is not None
    mean = alpha_head.auxiliary_head(hidden)
    log_scale = policy.alpha_scale_head(hidden)
    target = torch.randn_like(mean)
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    base = M03RV9AlphaPretrainingBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        target_log_return=target,
        valid=valid,
        origin_indices=torch.tensor((0, 1, 2), dtype=torch.long),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
    )
    return M03RV10AlphaPretrainingBatch(base, M03R_V10_SETTINGS[v10_setting])


def test_v10_step_binds_fresh_setting_and_imported_architecture() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    batch = _batch(policy)
    receipt = train_m03r_v10_alpha_pretraining_update(
        policy,
        batch,
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.setting_id == M03R_V10_SETTINGS[1].setting_id
    assert receipt.batch_receipt_sha256 == batch.receipt_sha256
    assert receipt.horizon_loss_weights == (0.10, 0.35, 0.40, 0.15)
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256
    assert not receipt.qualification_evaluated_during_update


def test_v10_21_30_step_binds_focused_horizon_weights() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    receipt = train_m03r_v10_alpha_pretraining_update(
        policy,
        _batch(policy, v10_setting=2),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.horizon_loss_weights == (0.0, 7.0 / 15.0, 8.0 / 15.0, 0.0)


def test_v10_rejects_qualification_batch_v9_state_route_and_cursor_64() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    batch = _batch(policy)
    with pytest.raises(M03RV10AlphaStepError, match="training batch"):
        train_m03r_v10_alpha_pretraining_update(
            policy,
            replace(
                batch,
                imported_v9_batch=replace(
                    batch.imported_v9_batch,
                    split="qualification",
                ),
            ),
            optimizer,
            partition,
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
    with pytest.raises(M03RV10AlphaStepError, match="outside 0..63"):
        train_m03r_v10_alpha_pretraining_update(
            policy,
            batch,
            optimizer,
            partition,
            completed_updates=64,
            distributed_rank=0,
            distributed_world_size=1,
        )
    wrong_policy = _policy(1)
    wrong_optimizer, wrong_partition = build_m03r_v9_alpha_pretraining_optimizer(
        wrong_policy
    )
    with pytest.raises(M03RV10AlphaStepError, match="architecture identity"):
        train_m03r_v10_alpha_pretraining_update(
            wrong_policy,
            _batch(wrong_policy),
            wrong_optimizer,
            wrong_partition,
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
