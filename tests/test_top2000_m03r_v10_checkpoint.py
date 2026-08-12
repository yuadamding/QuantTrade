from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTINGS
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v10_checkpoint import (
    M03RV10AlphaCheckpointError,
    load_m03r_v10_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v10_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03RV10AlphaPretrainingBatch,
    train_m03r_v10_alpha_pretraining_update,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    build_m03r_v9_alpha_pretraining_optimizer,
)


def _policy(horizon: int = 30) -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(horizon, horizon, horizon),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _trained():  # type: ignore[no-untyped-def]
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    hidden = torch.randn((3, 5, 16))
    alpha_head = policy.source_policy.core.alpha_head
    assert alpha_head is not None
    mean = alpha_head.auxiliary_head(hidden)
    scale = policy.alpha_scale_head(hidden)
    base = M03RV9AlphaPretrainingBatch(
        predicted_mean=mean,
        predicted_log_scale=scale,
        target_log_return=torch.randn_like(mean),
        valid=torch.ones_like(mean, dtype=torch.bool),
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
    step = train_m03r_v10_alpha_pretraining_update(
        policy,
        M03RV10AlphaPretrainingBatch(base, M03R_V10_SETTINGS[1]),
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


def test_v10_checkpoint_is_fresh_evaluation_only_and_no_clobber(
    tmp_path: Path,
) -> None:
    policy, optimizer, partition, final_step = _trained()
    path = tmp_path / "v10-setting-01-fold-00-horizon-30.pt"
    file_sha = write_immutable_m03r_v10_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        final_step,
        setting_index=1,
        fold_index=0,
        plan_sha256="d" * 64,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_binding_sha256="e" * 64,
    )
    loaded = load_m03r_v10_alpha_checkpoint_for_evaluation(
        path,
        expected_file_sha256=file_sha,
        expected_plan_sha256="d" * 64,
        expected_setting_index=1,
        expected_fold_index=0,
        expected_source_array_sha256="a" * 64,
        expected_asset_axis_sha256="b" * 64,
        expected_risk_binding_sha256="e" * 64,
        policy=_policy(),
    )
    assert loaded.completed_updates == 64
    assert loaded.setting_index == 1
    assert loaded.selected_horizon == 30
    with pytest.raises(M03RV10AlphaCheckpointError, match="already exists"):
        write_immutable_m03r_v10_alpha_checkpoint(
            path,
            policy,
            optimizer,
            partition,
            final_step,
            setting_index=1,
            fold_index=0,
            plan_sha256="d" * 64,
            source_array_sha256="a" * 64,
            asset_axis_sha256="b" * 64,
            risk_binding_sha256="e" * 64,
        )


def test_v10_checkpoint_rejects_wrong_setting_horizon_or_hash(tmp_path: Path) -> None:
    policy, optimizer, partition, final_step = _trained()
    path = tmp_path / "v10.pt"
    file_sha = write_immutable_m03r_v10_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        final_step,
        setting_index=1,
        fold_index=0,
        plan_sha256="d" * 64,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_binding_sha256="e" * 64,
    )
    with pytest.raises(M03RV10AlphaCheckpointError, match="file hash"):
        load_m03r_v10_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256="f" * 64,
            expected_plan_sha256="d" * 64,
            expected_setting_index=1,
            expected_fold_index=0,
            expected_source_array_sha256="a" * 64,
            expected_asset_axis_sha256="b" * 64,
            expected_risk_binding_sha256="e" * 64,
            policy=_policy(),
        )
    with pytest.raises(M03RV10AlphaCheckpointError, match="identity drifted"):
        load_m03r_v10_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256=file_sha,
            expected_plan_sha256="d" * 64,
            expected_setting_index=2,
            expected_fold_index=0,
            expected_source_array_sha256="a" * 64,
            expected_asset_axis_sha256="b" * 64,
            expected_risk_binding_sha256="e" * 64,
            policy=_policy(21),
        )
