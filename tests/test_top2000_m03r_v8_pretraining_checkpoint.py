from __future__ import annotations

import pytest
import torch

from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_checkpoint import (
    M03RV8AlphaCheckpointError,
    load_m03r_v8_alpha_checkpoint,
    write_immutable_m03r_v8_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    build_m03r_v8_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    M03RV8AlphaEarlyStoppingState,
    model_state_sha256,
    optimizer_state_sha256,
    train_m03r_v8_alpha_pretraining_update,
)


def _policy() -> Top2000M03RV8DevelopmentPolicy:
    torch.manual_seed(3)
    return Top2000M03RV8DevelopmentPolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(policy: Top2000M03RV8DevelopmentPolicy) -> M03RV8AlphaPretrainingBatch:
    state = torch.randn(2, 1, 5, 16)
    available = torch.ones(1, 5, dtype=torch.bool)
    outputs = [policy.alpha_pretraining_distribution(row, available) for row in state]
    valid = torch.ones(2, 5, 4, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV8AlphaPretrainingBatch(
        predicted_mean=torch.stack([row.predicted_mean.squeeze(0) for row in outputs]),
        predicted_log_scale=torch.stack(
            [row.predicted_log_scale.squeeze(0) for row in outputs]
        ),
        target_residual_log_return=torch.randn(2, 5, 4) * 0.01,
        valid=valid,
        origin_indices=torch.tensor([0, 1]),
        split="training",
        fold_index=2,
        split_start_inclusive=0,
        split_stop_exclusive=70,
        source_array_sha256="a" * 64,
    )


def test_checkpoint_round_trip_and_no_clobber(tmp_path) -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    step = train_m03r_v8_alpha_pretraining_update(
        policy,
        _batch(policy),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    early = M03RV8AlphaEarlyStoppingState()
    path = tmp_path / "checkpoint.pt"
    file_hash = write_immutable_m03r_v8_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        early,
        step,
        setting_index=0,
        fold_index=2,
        rank=0,
        world_size=1,
        completed_updates=1,
        plan_sha256="b" * 64,
        source_array_sha256="a" * 64,
    )
    with pytest.raises(M03RV8AlphaCheckpointError, match="already exists"):
        write_immutable_m03r_v8_alpha_checkpoint(
            path,
            policy,
            optimizer,
            partition,
            early,
            step,
            setting_index=0,
            fold_index=2,
            rank=0,
            world_size=1,
            completed_updates=1,
            plan_sha256="b" * 64,
            source_array_sha256="a" * 64,
        )

    restored = _policy()
    restored_optimizer, restored_partition = (
        build_m03r_v8_alpha_pretraining_optimizer(restored)
    )
    completed, restored_early, restored_step = load_m03r_v8_alpha_checkpoint(
        path,
        expected_file_sha256=file_hash,
        expected_plan_sha256="b" * 64,
        expected_setting_index=0,
        expected_fold_index=2,
        expected_rank=0,
        expected_world_size=1,
        expected_source_array_sha256="a" * 64,
        policy=restored,
        optimizer=restored_optimizer,
        partition=restored_partition,
    )
    assert completed == 1
    assert restored_early == early
    assert restored_step == step
    assert model_state_sha256(restored) == model_state_sha256(policy)
    assert optimizer_state_sha256(restored_optimizer) == optimizer_state_sha256(
        optimizer
    )


def test_checkpoint_tamper_or_identity_mismatch_fails(tmp_path) -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    step = train_m03r_v8_alpha_pretraining_update(
        policy,
        _batch(policy),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    path = tmp_path / "checkpoint.pt"
    file_hash = write_immutable_m03r_v8_alpha_checkpoint(
        path,
        policy,
        optimizer,
        partition,
        M03RV8AlphaEarlyStoppingState(),
        step,
        setting_index=0,
        fold_index=2,
        rank=0,
        world_size=1,
        completed_updates=1,
        plan_sha256="b" * 64,
        source_array_sha256="a" * 64,
    )
    restored = _policy()
    restored_optimizer, restored_partition = (
        build_m03r_v8_alpha_pretraining_optimizer(restored)
    )
    with pytest.raises(M03RV8AlphaCheckpointError, match="identity drifted"):
        load_m03r_v8_alpha_checkpoint(
            path,
            expected_file_sha256=file_hash,
            expected_plan_sha256="c" * 64,
            expected_setting_index=0,
            expected_fold_index=2,
            expected_rank=0,
            expected_world_size=1,
            expected_source_array_sha256="a" * 64,
            policy=restored,
            optimizer=restored_optimizer,
            partition=restored_partition,
        )
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(M03RV8AlphaCheckpointError, match="hash drifted"):
        load_m03r_v8_alpha_checkpoint(
            path,
            expected_file_sha256=file_hash,
            expected_plan_sha256="b" * 64,
            expected_setting_index=0,
            expected_fold_index=2,
            expected_rank=0,
            expected_world_size=1,
            expected_source_array_sha256="a" * 64,
            policy=restored,
            optimizer=restored_optimizer,
            partition=restored_partition,
        )
