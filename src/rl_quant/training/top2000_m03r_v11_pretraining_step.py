"""Sole optimizer-mutation boundary for corrected M03R-v11 training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03R_V9_PROTOCOL_SHA256
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PREDICTIVE_SPEC,
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
    validate_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v11_fold import M03RV11TrainingShardPlan
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID,
    M03RV11AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v11_rank_objective import (
    m03r_v11_predictive_loss,
)
from rl_quant.training.top2000_m03r_v11_schedule import M03RV11PairedInputReceipt

M03R_V11_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v11-alpha-step-v1"


class M03RV11AlphaStepError(ValueError):
    """The v11 paired update violated its immutable mutation boundary."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11AlphaStepError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV11AlphaStepReceipt:
    setting_index: int
    setting_id: str
    fold_index: int
    completed_updates_before: int
    completed_updates_after: int
    distributed_rank: int
    distributed_world_size: int
    rank_local_origin_count: int
    rank_origin_sha256: str
    training_shard_receipt_sha256: str
    paired_input_receipt_sha256: str
    panel_episode_schedule_sha256: str
    batch_receipt_sha256: str
    residual_operator_root_sha256: str
    source_array_sha256: str
    optimizer_partition_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    optimizer_state_before_sha256: str
    optimizer_state_after_sha256: str
    total_loss: float
    ranking_loss: float
    robust_regression_loss: float
    distributional_loss: float
    component_weights: tuple[float, float, float]
    horizon_loss_weights: tuple[float, float, float, float]
    gradient_norm_before_clip: float
    gradient_reduction: str
    imported_architecture_protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    imported_architecture_setting_id: str = M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID
    nonfinite_rejected_before_step: bool = False
    early_stopping_enabled: bool = False
    qualification_evaluated_during_update: bool = False
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V11_ALPHA_STEP_SCHEMA

    def validate(self) -> None:
        digests = (
            self.rank_origin_sha256,
            self.training_shard_receipt_sha256,
            self.paired_input_receipt_sha256,
            self.panel_episode_schedule_sha256,
            self.batch_receipt_sha256,
            self.residual_operator_root_sha256,
            self.source_array_sha256,
            self.optimizer_partition_sha256,
            self.model_state_before_sha256,
            self.model_state_after_sha256,
            self.optimizer_state_before_sha256,
            self.optimizer_state_after_sha256,
        )
        metrics = (
            self.total_loss,
            self.ranking_loss,
            self.robust_regression_loss,
            self.distributional_loss,
            self.gradient_norm_before_clip,
        )
        expected_reduction = (
            "two-rank-sum-then-divide-world-size"
            if self.distributed_world_size == 2
            else "single-rank"
        )
        if (
            isinstance(self.setting_index, bool)
            or self.setting_index not in range(3)
            or self.setting_id != M03R_V11_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.completed_updates_after != self.completed_updates_before + 1
            or self.completed_updates_before
            not in range(M03R_V11_PREDICTIVE_SPEC.optimizer_updates)
            or self.distributed_world_size not in {1, 2}
            or self.distributed_rank not in range(self.distributed_world_size)
            or self.rank_local_origin_count <= 0
            or any(_digest("receipt field", value) != value for value in digests)
            or not all(math.isfinite(value) for value in metrics)
            or self.gradient_norm_before_clip < 0.0
            or self.component_weights != (0.50, 0.30, 0.20)
            or abs(sum(self.horizon_loss_weights) - 1.0) > 1.0e-12
            or self.gradient_reduction != expected_reduction
            or self.imported_architecture_protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.imported_architecture_setting_id
            != M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.nonfinite_rejected_before_step
            or self.early_stopping_enabled
            or self.qualification_evaluated_during_update
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V11_ALPHA_STEP_SCHEMA
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
        ):
            raise M03RV11AlphaStepError("v11 alpha step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, torch.Tensor)
    )


def _distributed_active(rank: int, world_size: int) -> bool:
    if world_size not in {1, 2} or rank not in range(world_size):
        raise M03RV11AlphaStepError("v11 distributed geometry drifted")
    if world_size == 1:
        return False
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != world_size
        or dist.get_rank() != rank
    ):
        raise M03RV11AlphaStepError("two-rank v11 step lacks its process group")
    return True


def _average_gradients(
    parameters: tuple[torch.Tensor, ...],
    *,
    world_size: int,
) -> None:
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        used = torch.tensor(
            0 if parameter.grad is None else 1,
            dtype=torch.int64,
            device=parameter.device,
        )
        dist.all_reduce(used, op=dist.ReduceOp.SUM)
        if int(used.item()) == 0:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def train_m03r_v11_alpha_pretraining_update(
    policy: Top2000M03RV9PredictivePolicy,
    batch: M03RV11AlphaPretrainingBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    training_shard: M03RV11TrainingShardPlan,
    paired_input: M03RV11PairedInputReceipt,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV11AlphaStepReceipt:
    """Execute one paired v11 update and fail before mutation on drift."""

    batch.validate()
    training_shard.validate()
    paired_input.validate()
    base = batch.corrected_batch
    distributed = _distributed_active(distributed_rank, distributed_world_size)
    expected_origins = training_shard.rank_origins[distributed_rank]
    observed_origins = tuple(int(value) for value in base.origin_indices.tolist())
    if base.split != "training":
        raise M03RV11AlphaStepError("v11 optimizer updates require a training batch")
    if (
        batch.setting.setting_index != training_shard.setting_index
        or base.fold_index != training_shard.fold_index
        or completed_updates != training_shard.completed_update
        or observed_origins != expected_origins
        or paired_input.schedule_sha256 != training_shard.panel_episode_schedule_sha256
        or paired_input.fold_index != training_shard.fold_index
        or paired_input.completed_update != training_shard.completed_update
        or paired_input.episode_start != training_shard.episode_start
        or paired_input.global_origins != training_shard.global_origins
        or paired_input.rank_origin_sha256[distributed_rank]
        != _sha256(observed_origins)
        or paired_input.source_array_sha256 != base.source_array_sha256
        or paired_input.asset_axis_sha256 != base.asset_axis_sha256
    ):
        raise M03RV11AlphaStepError("v11 batch is not the bound paired rank shard")
    if (
        not isinstance(policy, Top2000M03RV9PredictivePolicy)
        or policy.setting.setting_index != 0
        or policy.setting.setting_id != M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID
    ):
        raise M03RV11AlphaStepError("v11 imported architecture identity drifted")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates not in range(M03R_V11_PREDICTIVE_SPEC.optimizer_updates)
    ):
        raise M03RV11AlphaStepError("v11 update cursor is outside 0..63")
    validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)
    parameters = _optimizer_parameters(optimizer)
    selected_ids = {id(parameter) for parameter in parameters}
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v11_predictive_loss(batch)
    loss.total.backward()  # type: ignore[no-untyped-call]
    leaked = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None and id(parameter) not in selected_ids
    ]
    if leaked:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV11AlphaStepError(
            "v11 predictive loss leaked outside the optimizer partition: "
            + ",".join(leaked[:3])
        )
    if distributed:
        _average_gradients(parameters, world_size=distributed_world_size)
    populated = tuple(
        parameter for parameter in parameters if parameter.grad is not None
    )
    if not populated:
        raise M03RV11AlphaStepError("v11 predictive loss produced no gradients")
    try:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                populated,
                partition.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        )
    except RuntimeError as exc:
        optimizer.zero_grad(set_to_none=True)
        if (
            model_state_sha256(policy) != before_model
            or optimizer_state_sha256(optimizer) != before_optimizer
        ):
            raise M03RV11AlphaStepError(
                "v11 state changed before non-finite rejection"
            ) from exc
        raise M03RV11AlphaStepError(
            "v11 non-finite gradient rejected before optimizer step"
        ) from exc
    optimizer.step()
    metrics = torch.tensor(
        (
            float(loss.total.detach()),
            float(loss.ranking.detach()),
            float(loss.robust_regression.detach()),
            float(loss.distributional.detach()),
        ),
        dtype=torch.float64,
        device=base.predicted_mean.device,
    )
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics.div_(float(distributed_world_size))
    residual_root = _sha256(batch.residual_operator_receipt_sha256)
    receipt = M03RV11AlphaStepReceipt(
        setting_index=batch.setting.setting_index,
        setting_id=batch.setting.setting_id,
        fold_index=base.fold_index,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        rank_local_origin_count=len(observed_origins),
        rank_origin_sha256=_sha256(observed_origins),
        training_shard_receipt_sha256=training_shard.receipt_sha256,
        paired_input_receipt_sha256=paired_input.receipt_sha256,
        panel_episode_schedule_sha256=training_shard.panel_episode_schedule_sha256,
        batch_receipt_sha256=batch.receipt_sha256,
        residual_operator_root_sha256=residual_root,
        source_array_sha256=base.source_array_sha256,
        optimizer_partition_sha256=partition.receipt_sha256,
        model_state_before_sha256=before_model,
        model_state_after_sha256=model_state_sha256(policy),
        optimizer_state_before_sha256=before_optimizer,
        optimizer_state_after_sha256=optimizer_state_sha256(optimizer),
        total_loss=float(metrics[0]),
        ranking_loss=float(metrics[1]),
        robust_regression_loss=float(metrics[2]),
        distributional_loss=float(metrics[3]),
        component_weights=batch.setting.component_weights,
        horizon_loss_weights=batch.setting.horizon_loss_weights,
        gradient_norm_before_clip=gradient_norm,
        gradient_reduction=(
            "two-rank-sum-then-divide-world-size" if distributed else "single-rank"
        ),
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03R_V11_ALPHA_STEP_SCHEMA",
    "M03RV11AlphaStepError",
    "M03RV11AlphaStepReceipt",
    "train_m03r_v11_alpha_pretraining_update",
]
