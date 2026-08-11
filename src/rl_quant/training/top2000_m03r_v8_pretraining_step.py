"""Distributed optimizer step and early stopping for M03R-v8 pretraining.

The predictive stage is deliberately separate from economic fine tuning.  A
caller supplies one rank-local, causally bounded batch.  This module owns the
only permitted mutation boundary: backward, exact two-rank gradient averaging,
non-finite rejection, clipping, and one AdamW step.
"""

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

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaFoldEvidence,
    M03RV8AlphaPretrainingBatch,
    m03r_v8_alpha_pretraining_loss,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    M03RV8AlphaOptimizerPartition,
    validate_m03r_v8_alpha_pretraining_optimizer,
)

M03R_V8_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v8-alpha-step-v1"
M03R_V8_ALPHA_EARLY_STOP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-alpha-early-stop-v1"
)


class M03RV8AlphaStepError(ValueError):
    """The predictive optimizer step or selection state is invalid."""


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
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state mapping independently of serialization metadata."""

    return _canonical_sha256(
        [(name, _tensor_sha256(value)) for name, value in sorted(state.items())]
    )


def model_state_sha256(model: nn.Module) -> str:
    """Hash model state independently of torch serialization metadata."""

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
    raise M03RV8AlphaStepError(
        f"optimizer state contains unsupported {type(value).__name__}"
    )


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    """Hash an optimizer state independently of pickle metadata."""

    return optimizer_state_dict_sha256(optimizer.state_dict())


def optimizer_state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash a detached optimizer state mapping."""

    return _canonical_sha256(_optimizer_value(state))


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _validate_distributed(rank: int, world_size: int) -> bool:
    if world_size not in {1, 2} or rank not in range(world_size):
        raise M03RV8AlphaStepError("v8 pretraining supports world size one or two")
    distributed = world_size == 2
    if distributed and (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != rank
    ):
        raise M03RV8AlphaStepError(
            "two-rank v8 pretraining requires the matching process group"
        )
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
class M03RV8AlphaStepReceipt:
    """Content-bound result of exactly one predictive optimizer step."""

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
    valid_date_horizon_count: int
    gradient_norm_before_clip: float
    gradient_reduction: str
    nonfinite_rejected_before_step: bool
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    development_only: bool = True
    promotion_eligible: bool = False
    schema: str = M03R_V8_ALPHA_STEP_SCHEMA

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
            not self.setting_id.startswith("V8-")
            or not 0 <= self.fold_index < 6
            or self.completed_updates_after != self.completed_updates_before + 1
            or not 0 <= self.completed_updates_before
            < M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
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
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256
            == self.optimizer_state_before_sha256
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or not self.development_only
            or self.promotion_eligible
            or self.schema != M03R_V8_ALPHA_STEP_SCHEMA
        ):
            raise M03RV8AlphaStepError("v8 alpha step receipt is invalid")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))


def train_m03r_v8_alpha_pretraining_update(
    policy: Top2000M03RV8DevelopmentPolicy,
    batch: M03RV8AlphaPretrainingBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV8AlphaStepReceipt:
    """Execute the sole permitted predictive-stage parameter mutation."""

    batch.validate()
    if batch.split != "training":
        raise M03RV8AlphaStepError("optimizer updates require a training-split batch")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0 <= completed_updates
        < M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
    ):
        raise M03RV8AlphaStepError("completed update count is out of range")
    distributed = _validate_distributed(distributed_rank, distributed_world_size)
    validate_m03r_v8_alpha_pretraining_optimizer(policy, optimizer, partition)
    parameters = _optimizer_parameters(optimizer)
    selected_ids = {id(parameter) for parameter in parameters}
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v8_alpha_pretraining_loss(
        batch,
        ranking_loss_weight=policy.setting.ranking_loss_weight,
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
            raise M03RV8AlphaStepError("distributed predictive support is empty")
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
        raise M03RV8AlphaStepError(
            "predictive loss leaked gradients outside optimizer partition: "
            + ",".join(leaked[:3])
        )
    if distributed:
        _average_gradients(parameters, world_size=distributed_world_size)
    populated = tuple(parameter for parameter in parameters if parameter.grad is not None)
    if not populated:
        raise M03RV8AlphaStepError("predictive loss produced no optimizer gradients")
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            populated,
            partition.gradient_clip_norm,
            error_if_nonfinite=True,
        )
    )
    optimizer.step()

    metrics = torch.tensor(
        [
            float(loss.total.detach()),
            float(loss.listwise_ranking.detach()),
            float(loss.robust_regression.detach()),
            float(loss.distributional.detach()),
            float(loss.valid_date_horizon_count),
        ],
        dtype=torch.float64,
        device=batch.predicted_mean.device,
    )
    if distributed:
        metrics[:4].mul_(metrics[4])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics[:4].div_(metrics[4])
    receipt = M03RV8AlphaStepReceipt(
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
        valid_date_horizon_count=global_valid_count,
        gradient_norm_before_clip=gradient_norm,
        gradient_reduction=(
            "two-rank-sum-then-divide-world-size" if distributed else "single-rank"
        ),
        nonfinite_rejected_before_step=False,
    )
    receipt.validate()
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV8AlphaEarlyStoppingState:
    """Deterministic inner-validation checkpoint selector."""

    best_update: int | None = None
    best_metric: float | None = None
    best_model_state_sha256: str | None = None
    best_fold_evidence_sha256: str | None = None
    evaluation_count: int = 0
    consecutive_non_improving_evaluations: int = 0
    stopped: bool = False
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_ALPHA_EARLY_STOP_SCHEMA

    def validate(self) -> None:
        digests = (self.best_model_state_sha256, self.best_fold_evidence_sha256)
        empty = self.best_update is None
        if (
            self.evaluation_count < 0
            or self.consecutive_non_improving_evaluations < 0
            or (empty and any(value is not None for value in (self.best_metric, *digests)))
            or (
                not empty
                and (
                    self.best_update is None
                    or self.best_update <= 0
                    or self.best_update
                    % M03R_V8_ALPHA_PRETRAINING.inner_validation_interval_updates
                    != 0
                    or self.best_metric is None
                    or not math.isfinite(self.best_metric)
                    or any(
                        value is None
                        or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)
                        for value in digests
                    )
                )
            )
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != M03R_V8_ALPHA_EARLY_STOP_SCHEMA
        ):
            raise M03RV8AlphaStepError("v8 early-stopping state is invalid")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))


def advance_m03r_v8_alpha_early_stopping(
    state: M03RV8AlphaEarlyStoppingState,
    *,
    completed_updates: int,
    evidence: M03RV8AlphaFoldEvidence,
    model_state_sha256_value: str,
) -> M03RV8AlphaEarlyStoppingState:
    """Advance the frozen 21/30-day selector, breaking ties to earliest."""

    state.validate()
    evidence.__post_init__()
    interval = M03R_V8_ALPHA_PRETRAINING.inner_validation_interval_updates
    if (
        state.stopped
        or completed_updates <= 0
        or completed_updates > M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
        or completed_updates % interval != 0
        or len(model_state_sha256_value) != 64
        or any(c not in "0123456789abcdef" for c in model_state_sha256_value)
    ):
        raise M03RV8AlphaStepError("early-stopping update boundary is invalid")
    metric = max(evidence.mean_spearman_rank_ic[1:3])
    improved = state.best_metric is None or metric > state.best_metric
    non_improving = 0 if improved else state.consecutive_non_improving_evaluations + 1
    evaluations = state.evaluation_count + 1
    stopped = (
        completed_updates
        >= M03R_V8_ALPHA_PRETRAINING.minimum_optimizer_updates_before_early_stop
        and non_improving
        >= M03R_V8_ALPHA_PRETRAINING.early_stopping_patience_evaluations
    )
    result = M03RV8AlphaEarlyStoppingState(
        best_update=completed_updates if improved else state.best_update,
        best_metric=metric if improved else state.best_metric,
        best_model_state_sha256=(
            model_state_sha256_value if improved else state.best_model_state_sha256
        ),
        best_fold_evidence_sha256=(
            evidence.receipt_sha256 if improved else state.best_fold_evidence_sha256
        ),
        evaluation_count=evaluations,
        consecutive_non_improving_evaluations=non_improving,
        stopped=stopped,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V8_ALPHA_EARLY_STOP_SCHEMA",
    "M03R_V8_ALPHA_STEP_SCHEMA",
    "M03RV8AlphaEarlyStoppingState",
    "M03RV8AlphaStepError",
    "M03RV8AlphaStepReceipt",
    "advance_m03r_v8_alpha_early_stopping",
    "model_state_sha256",
    "optimizer_state_dict_sha256",
    "optimizer_state_sha256",
    "state_dict_sha256",
    "train_m03r_v8_alpha_pretraining_update",
]
