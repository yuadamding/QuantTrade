from __future__ import annotations

import torch

from rl_quant.training.top2000_m03r_v12_gradient_balance import (
    install_m03r_v12_balanced_gradients,
)
from rl_quant.training.top2000_m03r_v12_fold import M03RV12TrainingShardPlan
from rl_quant.training.top2000_m03r_v12_objective import (
    M03RV12PredictiveBatch,
    m03r_v12_predictive_loss,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v12_pretraining_optimizer import (
    build_m03r_v12_optimizer,
)
from rl_quant.training.top2000_m03r_v12_pretraining_step import (
    train_m03r_v12_predictive_batch_update,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    build_m03r_v12_paired_input_receipt,
)


def _case(
    setting: int,
) -> tuple[
    Top2000M03RV12PredictivePolicy,
    torch.optim.AdamW,
    M03RV12PredictiveBatch,
]:
    torch.manual_seed(11)
    policy = Top2000M03RV12PredictivePolicy(
        setting,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    state = torch.randn((3, 12, 16))
    available = torch.ones((3, 12), dtype=torch.bool)
    output = policy.predictive_output(state, available)
    economic = output.economic_distribution
    valid = torch.ones_like(economic.mean_by_horizon, dtype=torch.bool)
    valid[:, 0] = False
    count = 15
    batch = M03RV12PredictiveBatch(
        predicted_mean=economic.mean_by_horizon,
        predicted_log_scale=economic.log_scale_by_horizon,
        predicted_rank_score=output.rank_score_by_horizon,
        target_log_return=torch.randn_like(economic.mean_by_horizon),
        valid=valid,
        origin_indices=torch.tensor([1, 2, 3]),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
        setting=policy.v12_setting,
        residual_operator_receipt_sha256=tuple("d" * 64 for _ in range(count)),
        available_risky_asset_count=tuple(11 for _ in range(count)),
        factor_qualified_risky_asset_count=tuple(11 for _ in range(count)),
        effective_design_rank=tuple(5 for _ in range(count)),
        weighted_residual_degrees_of_freedom=tuple(6 for _ in range(count)),
    )
    optimizer, _partition = build_m03r_v12_optimizer(policy)
    return (
        policy,
        optimizer,
        batch,
    )


def test_v12_rank_encoder_gradient_is_bounded_without_scaling_economic_heads() -> None:
    policy, optimizer, batch = _case(1)
    _unused, partition = build_m03r_v12_optimizer(policy)
    receipt = install_m03r_v12_balanced_gradients(
        policy,
        optimizer,
        partition,
        m03r_v12_predictive_loss(batch),
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.rank_encoder_multiplier <= 1.0
    assert receipt.effective_rank_encoder_gradient_norm <= (
        0.25 * receipt.economic_encoder_gradient_norm + 1.0e-7
    )
    named = dict(policy.named_parameters())
    assert any(
        named[name].grad is not None for name in partition.economic_head_parameter_names
    )
    assert any(
        named[name].grad is not None for name in partition.rank_head_parameter_names
    )


def test_v12_no_rank_control_does_not_mutate_rank_head() -> None:
    policy, optimizer, batch = _case(2)
    _unused, partition = build_m03r_v12_optimizer(policy)
    before = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
        if name in partition.rank_head_parameter_names
    }
    receipt = install_m03r_v12_balanced_gradients(
        policy,
        optimizer,
        partition,
        m03r_v12_predictive_loss(batch),
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.raw_rank_encoder_gradient_norm == 0.0
    assert receipt.rank_head_gradient_norm_before_clip == 0.0
    optimizer.step()
    named = dict(policy.named_parameters())
    assert all(torch.equal(named[name], value) for name, value in before.items())


def test_v12_step_mutates_once_and_binds_balance_receipt() -> None:
    policy, optimizer, batch = _case(1)
    _unused, partition = build_m03r_v12_optimizer(policy)
    shard = M03RV12TrainingShardPlan(
        setting_index=1,
        fold_index=0,
        completed_update=0,
        episode_start=1,
        global_origins=(1, 2, 3, 4, 5, 6),
        rank_origins=((1, 3, 5), (2, 4, 6)),
        panel_episode_schedule_sha256="d" * 64,
        fold_geometry_sha256="e" * 64,
    )
    paired = build_m03r_v12_paired_input_receipt(
        shard,
        (torch.arange(6, dtype=torch.float32),),
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
    )
    receipt = train_m03r_v12_predictive_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        shard,
        paired,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )
    assert receipt.completed_updates_after == 1
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256
    assert receipt.optimizer_state_before_sha256 != receipt.optimizer_state_after_sha256
    assert receipt.gradient_balance.rank_encoder_multiplier <= 1.0
    assert receipt.training_shard_receipt_sha256 == shard.receipt_sha256
    assert receipt.paired_input_receipt_sha256 == paired.receipt_sha256
    assert not receipt.qualification_evaluated_during_update
    assert not receipt.outer_2026_accessed
