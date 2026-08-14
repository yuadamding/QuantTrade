"""Single fail-closed optimizer mutation boundary for M03R-v13."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v13_fold import (
    M03RV13PairedInputBinding,
    M03RV13TrainingUpdatePlan,
)
from rl_quant.training.top2000_m03r_v13_objective import m03r_v13_predictive_loss
from rl_quant.training.top2000_m03r_v13_policy import (
    Top2000M03RV13PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v13_pretraining_optimizer import (
    M03RV13OptimizerPartition,
    validate_m03r_v13_optimizer,
)
from rl_quant.training.top2000_m03r_v13_pretraining_runtime import (
    M03RV13BuiltPredictiveBatch,
)

M03R_V13_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v13-alpha-step-v1"


class M03RV13AlphaStepError(ValueError):
    """The v13 alpha step failed before or during mutation."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    gradients = tuple(
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    )
    if not gradients:
        return 0.0
    return float(
        torch.sqrt(
            torch.stack(tuple(gradient.square().sum() for gradient in gradients)).sum()
        ).detach()
    )


@dataclass(frozen=True, slots=True)
class M03RV13AlphaStepReceipt:
    setting_index: int
    setting_id: str
    fold_index: int
    completed_updates_before: int
    completed_updates_after: int
    distributed_rank: int
    distributed_world_size: int
    local_origin_count: int
    global_origin_count: int
    distributed_gradient_synchronized: bool
    local_gradient_weight: float
    batch_receipt_sha256: str
    training_update_plan_sha256: str
    paired_input_binding_sha256: str
    optimizer_partition_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    optimizer_state_before_sha256: str
    optimizer_state_after_sha256: str
    target_residual_operator_root_sha256: str
    action_residual_operator_root_sha256: str
    total_loss: float
    ranking_loss: float
    robust_regression_loss: float
    distributional_loss: float
    encoder_gradient_norm_before_clip: float
    head_gradient_norm_before_clip: float
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256
    training_performed: bool = True
    qualification_evaluated_during_update: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V13_ALPHA_STEP_SCHEMA

    def validate(self) -> None:
        digests = (
            self.batch_receipt_sha256,
            self.training_update_plan_sha256,
            self.paired_input_binding_sha256,
            self.optimizer_partition_sha256,
            self.model_state_before_sha256,
            self.model_state_after_sha256,
            self.optimizer_state_before_sha256,
            self.optimizer_state_after_sha256,
            self.target_residual_operator_root_sha256,
            self.action_residual_operator_root_sha256,
        )
        metrics = (
            self.total_loss,
            self.ranking_loss,
            self.robust_regression_loss,
            self.distributional_loss,
            self.encoder_gradient_norm_before_clip,
            self.head_gradient_norm_before_clip,
        )
        if (
            self.setting_index not in range(len(M03R_V13_SETTING_IDS))
            or self.setting_id != M03R_V13_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.completed_updates_before < 0
            or self.completed_updates_after != self.completed_updates_before + 1
            or self.distributed_world_size not in {1, 2}
            or self.distributed_rank not in range(self.distributed_world_size)
            or self.local_origin_count <= 0
            or self.global_origin_count < self.local_origin_count
            or self.distributed_gradient_synchronized
            != (self.distributed_world_size == 2)
            or not math.isclose(
                self.local_gradient_weight,
                self.local_origin_count / self.global_origin_count,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or not all(math.isfinite(value) and value >= 0.0 for value in metrics[4:])
            or not all(math.isfinite(value) for value in metrics[:4])
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
            or not self.training_performed
            or self.qualification_evaluated_during_update
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V13_ALPHA_STEP_SCHEMA
        ):
            raise M03RV13AlphaStepError("v13 alpha-step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def train_m03r_v13_predictive_batch_update(
    policy: Top2000M03RV13PredictivePolicy,
    batch: M03RV13BuiltPredictiveBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV13OptimizerPartition,
    update_plan: M03RV13TrainingUpdatePlan,
    paired_input: M03RV13PairedInputBinding,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    gradient_synchronizer: Callable[[tuple[torch.nn.Parameter, ...]], None]
    | None = None,
) -> M03RV13AlphaStepReceipt:
    batch.validate()
    update_plan.validate()
    paired_input.validate()
    if (
        policy.v13_setting != batch.objective.setting
        or update_plan.setting_index != batch.objective.setting.setting_index
        or completed_updates != update_plan.completed_update
        or paired_input.fold_index != update_plan.fold_index
        or paired_input.completed_update != update_plan.completed_update
        or paired_input.episode_start != update_plan.episode_start
        or paired_input.episode_stop_exclusive != update_plan.episode_stop_exclusive
        or paired_input.global_origins != update_plan.global_origins
        or paired_input.rank_origins != update_plan.rank_origins
        or paired_input.panel_episode_schedule_sha256
        != update_plan.panel_episode_schedule_sha256
        or paired_input.fold_geometry_sha256 != update_plan.fold_geometry_sha256
        or distributed_world_size not in {1, 2}
        or distributed_rank not in range(distributed_world_size)
        or tuple(int(value) for value in batch.origin_indices)
        != (
            update_plan.global_origins
            if distributed_world_size == 1
            else update_plan.rank_origins[distributed_rank]
        )
    ):
        raise M03RV13AlphaStepError("v13 policy, batch, rank, or update drifted")
    validate_m03r_v13_optimizer(policy, optimizer, partition)
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v13_predictive_loss(batch.objective)
    named = dict(policy.named_parameters())
    encoder = tuple(named[name] for name in partition.encoder_parameter_names)
    heads = tuple(named[name] for name in partition.head_parameter_names)
    trainable = (*encoder, *heads)
    local_origin_count = int(batch.origin_indices.numel())
    global_origin_count = len(update_plan.global_origins)
    local_gradient_weight = local_origin_count / global_origin_count
    (loss.total * local_gradient_weight).backward()  # type: ignore[no-untyped-call]
    if distributed_world_size == 2:
        synchronizer = gradient_synchronizer or _synchronize_distributed_gradients
        synchronizer(trainable)
    elif gradient_synchronizer is not None:
        raise M03RV13AlphaStepError(
            "v13 single-rank update must not invoke a distributed synchronizer"
        )
    encoder_norm = _gradient_norm(encoder)
    head_norm = _gradient_norm(heads)
    gradients = tuple(
        parameter.grad
        for parameter in (*encoder, *heads)
        if parameter.grad is not None
    )
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        optimizer.zero_grad(set_to_none=True)
        if (
            model_state_sha256(policy) != before_model
            or optimizer_state_sha256(optimizer) != before_optimizer
        ):
            raise M03RV13AlphaStepError(
                "v13 state changed before non-finite-gradient rejection"
            )
        raise M03RV13AlphaStepError("v13 gradients are absent or non-finite")
    torch.nn.utils.clip_grad_norm_(encoder, partition.encoder_gradient_clip_norm)
    torch.nn.utils.clip_grad_norm_(heads, partition.head_gradient_clip_norm)
    optimizer.step()
    after_model = model_state_sha256(policy)
    after_optimizer = optimizer_state_sha256(optimizer)
    receipt = M03RV13AlphaStepReceipt(
        setting_index=batch.objective.setting.setting_index,
        setting_id=batch.objective.setting.setting_id,
        fold_index=update_plan.fold_index,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        local_origin_count=local_origin_count,
        global_origin_count=global_origin_count,
        distributed_gradient_synchronized=distributed_world_size == 2,
        local_gradient_weight=local_gradient_weight,
        batch_receipt_sha256=batch.receipt_sha256,
        training_update_plan_sha256=update_plan.receipt_sha256,
        paired_input_binding_sha256=paired_input.receipt_sha256,
        optimizer_partition_sha256=partition.receipt_sha256,
        model_state_before_sha256=before_model,
        model_state_after_sha256=after_model,
        optimizer_state_before_sha256=before_optimizer,
        optimizer_state_after_sha256=after_optimizer,
        target_residual_operator_root_sha256=_sha256(
            tuple(
                operator.receipt_sha256
                for operator in batch.target_residual_operators
            )
        ),
        action_residual_operator_root_sha256=_sha256(
            tuple(
                operator.receipt_sha256
                for operator in batch.action_residual_operators
            )
        ),
        total_loss=float(loss.total.detach()),
        ranking_loss=float(loss.ranking.detach()),
        robust_regression_loss=float(loss.robust_regression.detach()),
        distributional_loss=float(loss.distributional.detach()),
        encoder_gradient_norm_before_clip=encoder_norm,
        head_gradient_norm_before_clip=head_norm,
    )
    receipt.validate()
    return receipt


def _synchronize_distributed_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    """Sum origin-weighted gradients without assuming equal rank shard sizes."""

    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise M03RV13AlphaStepError(
            "v13 two-rank update requires an initialized two-rank process group"
        )
    if not parameters:
        raise M03RV13AlphaStepError("v13 distributed gradient set is empty")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise M03RV13AlphaStepError("v13 trainable parameters span devices")
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    for index, parameter in enumerate(parameters):
        if int(presence[index]) == 0:
            parameter.grad = None
            continue
        gradient = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad
        )
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        parameter.grad = gradient


__all__ = [
    "M03R_V13_ALPHA_STEP_SCHEMA",
    "M03RV13AlphaStepError",
    "M03RV13AlphaStepReceipt",
    "train_m03r_v13_predictive_batch_update",
]
