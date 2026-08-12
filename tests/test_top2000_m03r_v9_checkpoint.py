from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_checkpoint import (
    M03RV9AlphaCheckpointError,
    load_m03r_v9_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v9_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
    build_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    M03RV9AlphaStepReceipt,
    train_m03r_v9_alpha_pretraining_update,
)


def _policy() -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(policy: Top2000M03RV9PredictivePolicy) -> M03RV9AlphaPretrainingBatch:
    hidden = torch.randn((3, 5, 16))
    alpha_head = policy.source_policy.core.alpha_head
    assert alpha_head is not None
    mean = alpha_head.auxiliary_head(hidden)
    log_scale = policy.alpha_scale_head(hidden)
    target = torch.randn_like(mean)
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV9AlphaPretrainingBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        target_log_return=target,
        valid=valid,
        origin_indices=torch.tensor((0, 1, 2)),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
    )


def _trained() -> tuple[
    Top2000M03RV9PredictivePolicy,
    torch.optim.Optimizer,
    M03RV9AlphaOptimizerPartition,
    M03RV9AlphaStepReceipt,
]:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    step = train_m03r_v9_alpha_pretraining_update(
        policy,
        _batch(policy),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    final_step = replace(
        step,
        completed_updates_before=63,
        completed_updates_after=64,
    )
    final_step.validate()
    return policy, optimizer, partition, final_step


def test_update64_checkpoint_binds_both_heads_horizon_and_risk(
    tmp_path: Path,
) -> None:
    policy, optimizer, partition, final_step = _trained()
    path = tmp_path / "rank-0-update-64.pt"
    file_sha = write_immutable_m03r_v9_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        final_step,
        setting_index=0,
        fold_index=0,
        rank=0,
        world_size=1,
        plan_sha256="d" * 64,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_binding_sha256="e" * 64,
    )
    loaded_policy = _policy()
    loaded = load_m03r_v9_alpha_checkpoint_for_evaluation(
        path,
        expected_file_sha256=file_sha,
        expected_plan_sha256="d" * 64,
        expected_setting_index=0,
        expected_fold_index=0,
        expected_rank=0,
        expected_world_size=1,
        expected_source_array_sha256="a" * 64,
        expected_asset_axis_sha256="b" * 64,
        expected_risk_binding_sha256="e" * 64,
        policy=loaded_policy,
    )
    assert loaded.completed_updates == 64
    assert loaded.alpha_head_identity.selected_alpha_horizon == 30
    assert loaded.alpha_head_identity == policy.alpha_head_identity()
    assert loaded_policy.alpha_head_identity() == policy.alpha_head_identity()
    with pytest.raises(M03RV9AlphaCheckpointError, match="already exists"):
        write_immutable_m03r_v9_alpha_checkpoint(
            path,
            policy,
            optimizer,
            partition,
            final_step,
            setting_index=0,
            fold_index=0,
            rank=0,
            world_size=1,
            plan_sha256="d" * 64,
            source_array_sha256="a" * 64,
            asset_axis_sha256="b" * 64,
            risk_binding_sha256="e" * 64,
        )


def test_checkpoint_rejects_wrong_horizon_axis_or_external_hash(tmp_path: Path) -> None:
    policy, optimizer, partition, final_step = _trained()
    path = tmp_path / "rank-0-update-64.pt"
    file_sha = write_immutable_m03r_v9_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        final_step,
        setting_index=0,
        fold_index=0,
        rank=0,
        world_size=1,
        plan_sha256="d" * 64,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_binding_sha256="e" * 64,
    )
    with pytest.raises(M03RV9AlphaCheckpointError, match="file hash"):
        load_m03r_v9_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256="f" * 64,
            expected_plan_sha256="d" * 64,
            expected_setting_index=0,
            expected_fold_index=0,
            expected_rank=0,
            expected_world_size=1,
            expected_source_array_sha256="a" * 64,
            expected_asset_axis_sha256="b" * 64,
            expected_risk_binding_sha256="e" * 64,
            policy=_policy(),
        )
    with pytest.raises(M03RV9AlphaCheckpointError, match="identity drifted"):
        load_m03r_v9_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256=file_sha,
            expected_plan_sha256="d" * 64,
            expected_setting_index=0,
            expected_fold_index=0,
            expected_rank=0,
            expected_world_size=1,
            expected_source_array_sha256="a" * 64,
            expected_asset_axis_sha256="9" * 64,
            expected_risk_binding_sha256="e" * 64,
            policy=_policy(),
        )
    with pytest.raises(M03RV9AlphaCheckpointError, match="identity drifted"):
        load_m03r_v9_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256=file_sha,
            expected_plan_sha256="d" * 64,
            expected_setting_index=0,
            expected_fold_index=0,
            expected_rank=0,
            expected_world_size=1,
            expected_source_array_sha256="a" * 64,
            expected_asset_axis_sha256="b" * 64,
            expected_risk_binding_sha256="e" * 64,
            policy=Top2000M03RV9PredictivePolicy(
                0,
                M03RV9HorizonBinding(21, 21, 21),
                token_dim=16,
                raw_stock_chunk=8,
                activation_checkpointing=False,
            ),
        )
