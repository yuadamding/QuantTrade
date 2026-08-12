"""Sole distributed mutation boundary for M03R-v9 predictive training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
    m03r_v9_alpha_pretraining_loss,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
    validate_m03r_v9_alpha_pretraining_optimizer,
)

M03R_V9_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v9-alpha-step-v1"


class M03RV9AlphaStepError(ValueError):
    """A v9 predictive step violated its exact mutation boundary."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return _canonical_sha256(
        [(name, _tensor_sha256(value)) for name, value in sorted(state.items())]
    )


def model_state_sha256(model: nn.Module) -> str:
    return state_dict_sha256(model.state_dict())


def _optimizer_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "sha256": _tensor_sha256(value)}
    if isinstance(value, Mapping):
        rows = [
            (f"{type(key).__name__}:{key}", _optimizer_value(item))
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: row[0])
        return {"type": "mapping", "rows": rows}
    if isinstance(value, tuple):
        return {"type": "tuple", "values": [_optimizer_value(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "values": [_optimizer_value(item) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    raise M03RV9AlphaStepError(
        f"optimizer state contains unsupported {type(value).__name__}"
    )


def optimizer_state_dict_sha256(state: Mapping[str, Any]) -> str:
    return _canonical_sha256(_optimizer_value(state))


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    return optimizer_state_dict_sha256(optimizer.state_dict())


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )


def _validate_distributed(rank: int, world_size: int) -> bool:
    if world_size not in {1, 2} or rank not in range(world_size):
        raise M03RV9AlphaStepError("v9 predictive training supports one or two ranks")
    distributed = world_size == 2
    if distributed and (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != rank
    ):
        raise M03RV9AlphaStepError("two-rank v9 step lacks its exact process group")
    return distributed


def _average_gradients(
    parameters: Iterable[nn.Parameter],
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


@dataclass(frozen=True, slots=True)
class M03RV9AlphaStepReceipt:
    setting_id: str
    fold_index: int
    completed_updates_before: int
    completed_updates_after: int
    distributed_world_size: int
    rank_local_origin_count: int
    source_array_sha256: str
    optimizer_partition_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    optimizer_state_before_sha256: str
    optimizer_state_after_sha256: str
    total_loss: float
    listwise_ranking_loss: float
    robust_regression_loss: float
    distributional_loss: float
    component_weights: tuple[float, float, float]
    valid_date_horizon_count: int
    gradient_norm_before_clip: float
    gradient_reduction: str
    nonfinite_rejected_before_step: bool
    early_stopping_enabled: bool = False
    qualification_evaluated_during_update: bool = False
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    development_only: bool = True
    promotion_eligible: bool = False
    schema: str = M03R_V9_ALPHA_STEP_SCHEMA

    def validate(self) -> None:
        digests = (
            self.source_array_sha256,
            self.optimizer_partition_sha256,
            self.model_state_before_sha256,
            self.model_state_after_sha256,
            self.optimizer_state_before_sha256,
            self.optimizer_state_after_sha256,
        )
        if (
            self.setting_id not in M03R_V9_SETTING_IDS
            or not 0 <= self.fold_index < 6
            or self.completed_updates_after != self.completed_updates_before + 1
            or not 0
            <= self.completed_updates_before
            < M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates
            or self.distributed_world_size not in {1, 2}
            or self.rank_local_origin_count <= 0
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or not all(
                math.isfinite(value)
                for value in (
                    self.total_loss,
                    self.listwise_ranking_loss,
                    self.robust_regression_loss,
                    self.distributional_loss,
                    self.gradient_norm_before_clip,
                )
            )
            or self.gradient_norm_before_clip < 0.0
            or self.gradient_reduction
            != (
                "two-rank-sum-then-divide-world-size"
                if self.distributed_world_size == 2
                else "single-rank"
            )
            or self.nonfinite_rejected_before_step
            or self.early_stopping_enabled
            or self.qualification_evaluated_during_update
            or self.component_weights
            not in {
                M03R_V9_PREDICTIVE_SPEC.ranked_component_weights,
                M03R_V9_PREDICTIVE_SPEC.no_ranking_component_weights,
            }
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or not self.development_only
            or self.promotion_eligible
            or self.schema != M03R_V9_ALPHA_STEP_SCHEMA
        ):
            raise M03RV9AlphaStepError("v9 alpha step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))


def train_m03r_v9_alpha_pretraining_update(
    policy: Top2000M03RV9PredictivePolicy,
    batch: M03RV9AlphaPretrainingBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV9AlphaStepReceipt:
    """Execute exactly one predictive update without qualification access."""

    batch.validate()
    if batch.split != "training":
        raise M03RV9AlphaStepError("optimizer updates require a training batch")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0
        <= completed_updates
        < M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates
    ):
        raise M03RV9AlphaStepError("v9 update cursor is outside 0..63")
    distributed = _validate_distributed(distributed_rank, distributed_world_size)
    validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)
    parameters = _optimizer_parameters(optimizer)
    selected_ids = {id(parameter) for parameter in parameters}
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v9_alpha_pretraining_loss(
        batch,
        ranking_enabled=policy.setting.ranking_enabled,
    )
    global_valid_count = loss.valid_date_horizon_count
    loss_scale = 1.0
    if distributed:
        count = torch.tensor(
            float(loss.valid_date_horizon_count),
            dtype=torch.float64,
            device=batch.predicted_mean.device,
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_valid_count = int(count.item())
        if global_valid_count <= 0:
            raise M03RV9AlphaStepError("distributed predictive support is empty")
        loss_scale = (
            float(distributed_world_size)
            * float(loss.valid_date_horizon_count)
            / float(global_valid_count)
        )
    (loss.total * loss_scale).backward()  # type: ignore[no-untyped-call]
    leaked = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None and id(parameter) not in selected_ids
    ]
    if leaked:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV9AlphaStepError(
            "predictive loss leaked outside the optimizer partition: "
            + ",".join(leaked[:3])
        )
    if distributed:
        _average_gradients(parameters, world_size=distributed_world_size)
    populated = tuple(
        parameter for parameter in parameters if parameter.grad is not None
    )
    if not populated:
        raise M03RV9AlphaStepError("predictive loss produced no gradients")
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
            float(loss.listwise_ranking.detach()),
            float(loss.robust_regression.detach()),
            float(loss.distributional.detach()),
            float(loss.valid_date_horizon_count),
        ),
        dtype=torch.float64,
        device=batch.predicted_mean.device,
    )
    if distributed:
        metrics[:4].mul_(metrics[4])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics[:4].div_(metrics[4])
    receipt = M03RV9AlphaStepReceipt(
        setting_id=policy.setting.setting_id,
        fold_index=batch.fold_index,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_world_size=distributed_world_size,
        rank_local_origin_count=int(batch.origin_indices.numel()),
        source_array_sha256=batch.source_array_sha256,
        optimizer_partition_sha256=partition.receipt_sha256,
        model_state_before_sha256=before_model,
        model_state_after_sha256=model_state_sha256(policy),
        optimizer_state_before_sha256=before_optimizer,
        optimizer_state_after_sha256=optimizer_state_sha256(optimizer),
        total_loss=float(metrics[0]),
        listwise_ranking_loss=float(metrics[1]),
        robust_regression_loss=float(metrics[2]),
        distributional_loss=float(metrics[3]),
        component_weights=loss.component_weights,
        valid_date_horizon_count=global_valid_count,
        gradient_norm_before_clip=gradient_norm,
        gradient_reduction=(
            "two-rank-sum-then-divide-world-size" if distributed else "single-rank"
        ),
        nonfinite_rejected_before_step=False,
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03R_V9_ALPHA_STEP_SCHEMA",
    "M03RV9AlphaStepError",
    "M03RV9AlphaStepReceipt",
    "model_state_sha256",
    "optimizer_state_dict_sha256",
    "optimizer_state_sha256",
    "state_dict_sha256",
    "train_m03r_v9_alpha_pretraining_update",
]
