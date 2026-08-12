"""Typed batch and sole predictive mutation boundary for M03R-v10."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_PREDICTIVE_SPEC,
    M03R_V10_PROTOCOL_SHA256,
    M03R_V10_SETTING_IDS,
    M03RV10PredictiveSetting,
)
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v10_rank_objective import (
    m03r_v10_predictive_loss,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
    validate_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)

M03R_V10_ALPHA_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v10-alpha-batch-v1"
M03R_V10_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v10-alpha-step-v1"
M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID = "V9-P0-factor-residual-ranked"


class M03RV10AlphaStepError(ValueError):
    """The v10 batch, optimizer, distributed state, or receipt drifted."""


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


def _digest(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV10AlphaStepError("v10 batch identity is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV10AlphaPretrainingBatch:
    imported_v9_batch: M03RV9AlphaPretrainingBatch
    setting: M03RV10PredictiveSetting
    imported_architecture_protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    imported_architecture_setting_id: str = M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_ALPHA_BATCH_SCHEMA

    def validate(self) -> None:
        self.imported_v9_batch.validate()
        self.setting.__post_init__()
        if (
            self.imported_v9_batch.target_mode != "factor-residual"
            or self.imported_v9_batch.exposure_receipt_sha256 is None
            or self.imported_architecture_protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.imported_architecture_setting_id
            != M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_ALPHA_BATCH_SCHEMA
        ):
            raise M03RV10AlphaStepError("v10 batch identity drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        base = self.imported_v9_batch
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "setting_sha256": self.setting.receipt_sha256,
                "imported_architecture_protocol_sha256": (
                    self.imported_architecture_protocol_sha256
                ),
                "imported_architecture_setting_id": (
                    self.imported_architecture_setting_id
                ),
                "base_batch_schema": base.schema,
                "split": base.split,
                "fold_index": base.fold_index,
                "origin_count": int(base.origin_indices.numel()),
                "origin_start": int(base.origin_indices[0]),
                "origin_stop_inclusive": int(base.origin_indices[-1]),
                "source_array_sha256": _digest(base.source_array_sha256),
                "asset_axis_sha256": _digest(base.asset_axis_sha256),
                "exposure_receipt_sha256": _digest(base.exposure_receipt_sha256 or ""),
                "outer_score_accessed": base.outer_score_accessed,
                "lockbox_accessed": base.lockbox_accessed,
            }
        )


@dataclass(frozen=True, slots=True)
class M03RV10AlphaStepReceipt:
    setting_index: int
    setting_id: str
    fold_index: int
    completed_updates_before: int
    completed_updates_after: int
    distributed_world_size: int
    rank_local_origin_count: int
    batch_receipt_sha256: str
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
    imported_architecture_setting_id: str = M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
    early_stopping_enabled: bool = False
    qualification_evaluated_during_update: bool = False
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V10_ALPHA_STEP_SCHEMA

    def validate(self) -> None:
        digests = (
            self.batch_receipt_sha256,
            self.source_array_sha256,
            self.optimizer_partition_sha256,
            self.model_state_before_sha256,
            self.model_state_after_sha256,
            self.optimizer_state_before_sha256,
            self.optimizer_state_after_sha256,
        )
        if (
            isinstance(self.setting_index, bool)
            or not 0 <= self.setting_index < len(M03R_V10_SETTING_IDS)
            or self.setting_id != M03R_V10_SETTING_IDS[self.setting_index]
            or not 0 <= self.fold_index < 6
            or self.completed_updates_after != self.completed_updates_before + 1
            or not 0
            <= self.completed_updates_before
            < M03R_V10_PREDICTIVE_SPEC.optimizer_updates
            or self.distributed_world_size not in {1, 2}
            or self.rank_local_origin_count <= 0
            or any(_digest(value) != value for value in digests)
            or not all(
                math.isfinite(value)
                for value in (
                    self.total_loss,
                    self.ranking_loss,
                    self.robust_regression_loss,
                    self.distributional_loss,
                    self.gradient_norm_before_clip,
                )
            )
            or self.gradient_norm_before_clip < 0.0
            or self.component_weights != (0.50, 0.30, 0.20)
            or abs(sum(self.horizon_loss_weights) - 1.0) > 1.0e-12
            or self.gradient_reduction
            != (
                "two-rank-sum-then-divide-world-size"
                if self.distributed_world_size == 2
                else "single-rank"
            )
            or self.imported_architecture_protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.imported_architecture_setting_id
            != M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.early_stopping_enabled
            or self.qualification_evaluated_during_update
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V10_ALPHA_STEP_SCHEMA
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
        ):
            raise M03RV10AlphaStepError("v10 alpha step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, torch.Tensor)
    )


def _distributed_active(rank: int, world_size: int) -> bool:
    if world_size not in {1, 2} or rank not in range(world_size):
        raise M03RV10AlphaStepError("v10 distributed geometry drifted")
    if world_size == 1:
        return False
    if not dist.is_available() or not dist.is_initialized():
        raise M03RV10AlphaStepError("two-rank v10 step lacks process group")
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise M03RV10AlphaStepError("v10 process-group identity drifted")
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


def train_m03r_v10_alpha_pretraining_update(
    policy: Top2000M03RV9PredictivePolicy,
    batch: M03RV10AlphaPretrainingBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV10AlphaStepReceipt:
    """Execute one v10 update using fresh state and the imported architecture."""

    batch.validate()
    base = batch.imported_v9_batch
    if base.split != "training":
        raise M03RV10AlphaStepError("v10 optimizer updates require a training batch")
    if (
        not isinstance(policy, Top2000M03RV9PredictivePolicy)
        or policy.setting.setting_index != 0
        or policy.setting.setting_id != M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
    ):
        raise M03RV10AlphaStepError("v10 imported architecture identity drifted")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0 <= completed_updates < M03R_V10_PREDICTIVE_SPEC.optimizer_updates
    ):
        raise M03RV10AlphaStepError("v10 update cursor is outside 0..63")
    distributed = _distributed_active(distributed_rank, distributed_world_size)
    validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)
    parameters = _optimizer_parameters(optimizer)
    selected_ids = {id(parameter) for parameter in parameters}
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v10_predictive_loss(
        base.predicted_mean,
        base.predicted_log_scale,
        base.target_log_return,
        base.valid,
        batch.setting,
    )
    loss.total.backward()  # type: ignore[no-untyped-call]
    leaked = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None and id(parameter) not in selected_ids
    ]
    if leaked:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV10AlphaStepError(
            "v10 predictive loss leaked outside the optimizer partition: "
            + ",".join(leaked[:3])
        )
    if distributed:
        _average_gradients(parameters, world_size=distributed_world_size)
    populated = tuple(
        parameter for parameter in parameters if parameter.grad is not None
    )
    if not populated:
        raise M03RV10AlphaStepError("v10 predictive loss produced no gradients")
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            populated,
            partition.gradient_clip_norm,
            error_if_nonfinite=True,
        )
    )
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
    receipt = M03RV10AlphaStepReceipt(
        setting_index=batch.setting.setting_index,
        setting_id=batch.setting.setting_id,
        fold_index=base.fold_index,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_world_size=distributed_world_size,
        rank_local_origin_count=int(base.origin_indices.numel()),
        batch_receipt_sha256=batch.receipt_sha256,
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
    "M03R_V10_ALPHA_BATCH_SCHEMA",
    "M03R_V10_ALPHA_STEP_SCHEMA",
    "M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID",
    "M03RV10AlphaPretrainingBatch",
    "M03RV10AlphaStepError",
    "M03RV10AlphaStepReceipt",
    "train_m03r_v10_alpha_pretraining_update",
]
