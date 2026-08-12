from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import M03R_V11_SETTINGS
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    build_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v11_fold import M03RV11TrainingShardPlan
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    M03RV11AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v11_pretraining_step import (
    M03RV11AlphaStepError,
    train_m03r_v11_alpha_pretraining_update,
)
from rl_quant.training.top2000_m03r_v11_schedule import (
    M03RV11PairedInputReceipt,
    build_m03r_v11_paired_input_receipt,
)


def _policy() -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(
    policy: Top2000M03RV9PredictivePolicy,
    *,
    setting: int = 1,
    origins: tuple[int, ...] = (1, 3, 5),
) -> M03RV11AlphaPretrainingBatch:
    hidden = torch.randn((len(origins), 10, 16), dtype=torch.float32)
    alpha_head = policy.source_policy.core.alpha_head
    assert alpha_head is not None
    mean = alpha_head.auxiliary_head(hidden)
    log_scale = policy.alpha_scale_head(hidden)
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    base = M03RV9AlphaPretrainingBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        target_log_return=torch.randn_like(mean),
        valid=valid,
        origin_indices=torch.tensor(origins, dtype=torch.long),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
    )
    evidence_count = len(origins) * 4
    return M03RV11AlphaPretrainingBatch(
        corrected_batch=base,
        setting=M03R_V11_SETTINGS[setting],
        residual_operator_receipt_sha256=tuple(
            f"{index:064x}" for index in range(1, evidence_count + 1)
        ),
        available_risky_asset_count=tuple(9 for _ in range(evidence_count)),
        factor_qualified_risky_asset_count=tuple(9 for _ in range(evidence_count)),
        effective_design_rank=tuple(5 for _ in range(evidence_count)),
        weighted_residual_degrees_of_freedom=tuple(4 for _ in range(evidence_count)),
    )


def _shard(*, setting: int = 1) -> M03RV11TrainingShardPlan:
    return M03RV11TrainingShardPlan(
        setting_index=setting,
        fold_index=0,
        completed_update=0,
        episode_start=1,
        global_origins=(1, 2, 3, 4, 5, 6),
        rank_origins=((1, 3, 5), (2, 4, 6)),
        panel_episode_schedule_sha256="d" * 64,
        fold_geometry_sha256="e" * 64,
    )


def _paired(shard: M03RV11TrainingShardPlan) -> M03RV11PairedInputReceipt:
    return build_m03r_v11_paired_input_receipt(
        shard,
        (torch.arange(24, dtype=torch.float32).reshape(6, 4),),
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
    )


def test_v11_step_binds_exact_paired_rank_shard_and_mutates_once() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    batch = _batch(policy)
    receipt = train_m03r_v11_alpha_pretraining_update(
        policy,
        batch,
        optimizer,
        partition,
        _shard(),
        _paired(_shard()),
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.setting_id == M03R_V11_SETTINGS[1].setting_id
    assert receipt.training_shard_receipt_sha256 == _shard().receipt_sha256
    assert receipt.paired_input_receipt_sha256 == _paired(_shard()).receipt_sha256
    assert receipt.panel_episode_schedule_sha256 == "d" * 64
    assert receipt.rank_local_origin_count == 3
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256
    assert receipt.optimizer_state_before_sha256 != receipt.optimizer_state_after_sha256
    assert not receipt.qualification_evaluated_during_update


def test_v11_step_rejects_wrong_rank_origins_setting_and_qualification() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    batch = _batch(policy)
    with pytest.raises(M03RV11AlphaStepError, match="paired rank shard"):
        train_m03r_v11_alpha_pretraining_update(
            policy,
            _batch(policy, origins=(2, 4, 6)),
            optimizer,
            partition,
            _shard(),
            _paired(_shard()),
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
    with pytest.raises(M03RV11AlphaStepError, match="paired rank shard"):
        train_m03r_v11_alpha_pretraining_update(
            policy,
            batch,
            optimizer,
            partition,
            _shard(setting=2),
            _paired(_shard(setting=2)),
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
    with pytest.raises(M03RV11AlphaStepError, match="training batch"):
        train_m03r_v11_alpha_pretraining_update(
            policy,
            replace(
                batch,
                corrected_batch=replace(batch.corrected_batch, split="qualification"),
            ),
            optimizer,
            partition,
            _shard(),
            _paired(_shard()),
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )
    with pytest.raises(M03RV11AlphaStepError, match="paired rank shard"):
        train_m03r_v11_alpha_pretraining_update(
            policy,
            batch,
            optimizer,
            partition,
            _shard(),
            replace(_paired(_shard()), source_array_sha256="f" * 64),
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )


def test_v11_nonfinite_gradient_is_rejected_without_state_mutation() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(policy)
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    parameter = next(policy.alpha_scale_head.parameters())
    handle = parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(M03RV11AlphaStepError, match="non-finite gradient"):
            train_m03r_v11_alpha_pretraining_update(
                policy,
                _batch(policy),
                optimizer,
                partition,
                _shard(),
                _paired(_shard()),
                completed_updates=0,
                distributed_rank=0,
                distributed_world_size=1,
            )
    finally:
        handle.remove()
    assert model_state_sha256(policy) == before_model
    assert optimizer_state_sha256(optimizer) == before_optimizer
