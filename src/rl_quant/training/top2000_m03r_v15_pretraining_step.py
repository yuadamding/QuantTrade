"""Single fail-closed optimizer mutation boundary for M03R-v15."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_SHA256,
    M03R_V15_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v15_fold import (
    M03RV15PairedInputBinding,
    M03RV15TrainingUpdatePlan,
    render_m03r_v15_fold_geometries,
)
from rl_quant.training.top2000_m03r_v15_objective import m03r_v15_predictive_loss
from rl_quant.training.top2000_m03r_v15_policy import (
    Top2000M03RV15PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v15_pretraining_optimizer import (
    M03R_V15_ENCODER_LEARNING_RATE,
    M03R_V15_ENCODER_GRADIENT_CLIP_NORM,
    M03R_V15_MEAN_HEAD_LEARNING_RATE,
    M03R_V15_MEAN_HEAD_GRADIENT_CLIP_NORM,
    M03R_V15_SCALE_HEAD_LEARNING_RATE,
    M03R_V15_SCALE_HEAD_GRADIENT_CLIP_NORM,
    M03RV15OptimizerPartition,
    validate_m03r_v15_optimizer,
)
from rl_quant.training.top2000_m03r_v15_pretraining_runtime import (
    M03RV15BuiltPredictiveBatch,
)

M03R_V15_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v15-alpha-step-v1"


class M03RV15AlphaStepError(ValueError):
    """The v15 alpha step failed before or during mutation."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    gradients = tuple(
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    )
    if not gradients:
        return 0.0
    return float(
        torch.sqrt(
            torch.stack(
                tuple(
                    gradient.to(torch.float64).square().sum()
                    for gradient in gradients
                )
            ).sum()
        ).detach()
    )


def _optimizer_tensors(optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []

    def collect(value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(optimizer.state)
    return tuple(tensors)


def _state_is_finite(
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.Optimizer,
) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters) and all(
        bool(torch.isfinite(value).all()) for value in _optimizer_tensors(optimizer)
    )


def _learning_rate_multiplier_for_cursor(
    fold_index: int,
    completed_update: int,
) -> float:
    total = render_m03r_v15_fold_geometries(1001)[
        fold_index
    ].optimizer_updates
    if completed_update not in range(total):
        raise M03RV15AlphaStepError("v15 learning-rate cursor drifted")
    step = completed_update + 1
    warmup = max(
        1,
        math.ceil(total * M03R_V15_PREDICTIVE_SPEC.learning_rate_warmup_fraction),
    )
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    minimum = M03R_V15_PREDICTIVE_SPEC.minimum_learning_rate_multiplier
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def m03r_v15_learning_rate_multiplier(
    update_plan: M03RV15TrainingUpdatePlan,
) -> float:
    """Return the deterministic warm-up/cosine multiplier for one mutation."""

    update_plan.validate()
    return _learning_rate_multiplier_for_cursor(
        update_plan.fold_index,
        update_plan.completed_update,
    )


def _apply_learning_rate_multiplier(
    optimizer: torch.optim.Optimizer,
    multiplier: float,
) -> tuple[float, float, float]:
    observed: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", ""))
        logical_name = group_name.removesuffix("-no-decay").removesuffix("-decay")
        base_rate = float(group.get("base_lr", -1.0))
        if logical_name not in {"encoder", "mean-head", "scale-head"} or base_rate <= 0:
            raise M03RV15AlphaStepError("v15 optimizer learning-rate group drifted")
        rate = base_rate * multiplier
        previous = observed.setdefault(logical_name, rate)
        if not math.isclose(previous, rate, rel_tol=0.0, abs_tol=1.0e-18):
            raise M03RV15AlphaStepError("v15 decay pair learning rates diverged")
        group["lr"] = rate
    return observed["encoder"], observed["mean-head"], observed["scale-head"]


@dataclass(frozen=True, slots=True)
class M03RV15AlphaStepReceipt:
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
    valid_asset_observation_count: int
    raw_score_rms: float
    executable_score_rms: float
    target_rms: float
    raw_to_executable_score_retention: float
    encoder_gradient_norm_before_clip: float
    mean_head_gradient_norm_before_clip: float
    scale_head_gradient_norm_before_clip: float
    encoder_gradient_clipped: bool
    mean_head_gradient_clipped: bool
    scale_head_gradient_clipped: bool
    learning_rate_multiplier: float
    encoder_learning_rate: float
    mean_head_learning_rate: float
    scale_head_learning_rate: float
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    training_performed: bool = True
    qualification_evaluated_during_update: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V15_ALPHA_STEP_SCHEMA

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
            self.raw_score_rms,
            self.executable_score_rms,
            self.target_rms,
            self.raw_to_executable_score_retention,
            self.encoder_gradient_norm_before_clip,
            self.mean_head_gradient_norm_before_clip,
            self.scale_head_gradient_norm_before_clip,
            self.learning_rate_multiplier,
            self.encoder_learning_rate,
            self.mean_head_learning_rate,
            self.scale_head_learning_rate,
        )
        if (
            self.setting_index not in range(len(M03R_V15_SETTING_IDS))
            or self.setting_id != M03R_V15_SETTING_IDS[self.setting_index]
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
            or min(
                self.encoder_gradient_norm_before_clip,
                self.mean_head_gradient_norm_before_clip,
                self.scale_head_gradient_norm_before_clip,
            )
            <= 0.0
            or self.valid_asset_observation_count <= 0
            or self.encoder_gradient_clipped
            != (
                self.encoder_gradient_norm_before_clip
                > M03R_V15_ENCODER_GRADIENT_CLIP_NORM
            )
            or self.mean_head_gradient_clipped
            != (
                self.mean_head_gradient_norm_before_clip
                > M03R_V15_MEAN_HEAD_GRADIENT_CLIP_NORM
            )
            or self.scale_head_gradient_clipped
            != (
                self.scale_head_gradient_norm_before_clip
                > M03R_V15_SCALE_HEAD_GRADIENT_CLIP_NORM
            )
            or not math.isclose(
                self.learning_rate_multiplier,
                _learning_rate_multiplier_for_cursor(
                    self.fold_index,
                    self.completed_updates_before,
                ),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                self.encoder_learning_rate,
                M03R_V15_ENCODER_LEARNING_RATE * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                self.mean_head_learning_rate,
                M03R_V15_MEAN_HEAD_LEARNING_RATE * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                self.scale_head_learning_rate,
                M03R_V15_SCALE_HEAD_LEARNING_RATE * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or not self.training_performed
            or self.qualification_evaluated_during_update
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V15_ALPHA_STEP_SCHEMA
        ):
            raise M03RV15AlphaStepError("v15 alpha-step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def train_m03r_v15_predictive_batch_update(
    policy: Top2000M03RV15PredictivePolicy,
    batch: M03RV15BuiltPredictiveBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV15OptimizerPartition,
    update_plan: M03RV15TrainingUpdatePlan,
    paired_input: M03RV15PairedInputBinding,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    gradient_synchronizer: Callable[[tuple[torch.nn.Parameter, ...]], None]
    | None = None,
) -> M03RV15AlphaStepReceipt:
    batch.validate()
    update_plan.validate()
    paired_input.validate()
    if (
        policy.v15_setting != batch.objective.setting
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
        raise M03RV15AlphaStepError("v15 policy, batch, rank, or update drifted")
    validate_m03r_v15_optimizer(policy, optimizer, partition)
    learning_rate_multiplier = m03r_v15_learning_rate_multiplier(update_plan)
    encoder_learning_rate, mean_head_learning_rate, scale_head_learning_rate = (
        _apply_learning_rate_multiplier(optimizer, learning_rate_multiplier)
    )
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v15_predictive_loss(batch.objective)
    valid_values = batch.objective.valid
    raw_values = batch.raw_predicted_mean[valid_values].to(torch.float64)
    executable_values = batch.objective.predicted_mean[valid_values].to(torch.float64)
    target_values = batch.objective.target_log_return[valid_values].to(torch.float64)
    raw_norm = torch.linalg.vector_norm(raw_values)
    executable_norm = torch.linalg.vector_norm(executable_values)
    raw_to_executable = (
        torch.ones((), dtype=torch.float64, device=raw_norm.device)
        if float(raw_norm) <= 1.0e-14 and float(executable_norm) <= 1.0e-14
        else executable_norm / raw_norm.clamp_min(1.0e-14)
    )
    named = dict(policy.named_parameters())
    encoder = tuple(named[name] for name in partition.encoder_parameter_names)
    mean_head = tuple(
        named[name] for name in partition.mean_head_parameter_names
    )
    scale_head = tuple(
        named[name] for name in partition.scale_head_parameter_names
    )
    trainable = (*encoder, *mean_head, *scale_head)
    local_origin_count = int(batch.origin_indices.numel())
    global_origin_count = len(update_plan.global_origins)
    local_gradient_weight = local_origin_count / global_origin_count
    (loss.total * local_gradient_weight).backward()  # type: ignore[no-untyped-call]
    if distributed_world_size == 2:
        synchronizer = gradient_synchronizer or _synchronize_distributed_gradients
        synchronizer(trainable)
    elif gradient_synchronizer is not None:
        raise M03RV15AlphaStepError(
            "v15 single-rank update must not invoke a distributed synchronizer"
        )
    gradients = tuple(
        parameter.grad
        for parameter in trainable
        if parameter.grad is not None
    )
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        optimizer.zero_grad(set_to_none=True)
        if (
            model_state_sha256(policy) != before_model
            or optimizer_state_sha256(optimizer) != before_optimizer
        ):
            raise M03RV15AlphaStepError(
                "v15 state changed before non-finite-gradient rejection"
            )
        raise M03RV15AlphaStepError("v15 gradients are absent or non-finite")
    encoder_norm = _gradient_norm(encoder)
    mean_head_norm = _gradient_norm(mean_head)
    scale_head_norm = _gradient_norm(scale_head)
    if min(encoder_norm, mean_head_norm, scale_head_norm) <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV15AlphaStepError(
            "v15 encoder, mean, and scale groups must all receive gradients"
        )
    torch.nn.utils.clip_grad_norm_(
        encoder,
        partition.encoder_gradient_clip_norm,
        error_if_nonfinite=True,
    )
    torch.nn.utils.clip_grad_norm_(
        mean_head,
        partition.mean_head_gradient_clip_norm,
        error_if_nonfinite=True,
    )
    torch.nn.utils.clip_grad_norm_(
        scale_head,
        partition.scale_head_gradient_clip_norm,
        error_if_nonfinite=True,
    )
    clipped_gradients = tuple(
        parameter.grad
        for parameter in trainable
        if parameter.grad is not None
    )
    if not clipped_gradients or any(
        not bool(torch.isfinite(value).all()) for value in clipped_gradients
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV15AlphaStepError("v15 clipped gradients are absent or non-finite")
    optimizer.step()
    if not _state_is_finite(trainable, optimizer):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV15AlphaStepError(
            "v15 optimizer mutation produced non-finite model or optimizer state"
        )
    after_model = model_state_sha256(policy)
    after_optimizer = optimizer_state_sha256(optimizer)
    receipt = M03RV15AlphaStepReceipt(
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
        valid_asset_observation_count=int(valid_values.sum()),
        raw_score_rms=float(torch.sqrt(raw_values.square().mean())),
        executable_score_rms=float(torch.sqrt(executable_values.square().mean())),
        target_rms=float(torch.sqrt(target_values.square().mean())),
        raw_to_executable_score_retention=float(raw_to_executable),
        encoder_gradient_norm_before_clip=encoder_norm,
        mean_head_gradient_norm_before_clip=mean_head_norm,
        scale_head_gradient_norm_before_clip=scale_head_norm,
        encoder_gradient_clipped=(
            encoder_norm > partition.encoder_gradient_clip_norm
        ),
        mean_head_gradient_clipped=(
            mean_head_norm > partition.mean_head_gradient_clip_norm
        ),
        scale_head_gradient_clipped=(
            scale_head_norm > partition.scale_head_gradient_clip_norm
        ),
        learning_rate_multiplier=learning_rate_multiplier,
        encoder_learning_rate=encoder_learning_rate,
        mean_head_learning_rate=mean_head_learning_rate,
        scale_head_learning_rate=scale_head_learning_rate,
    )
    receipt.validate()
    return receipt


def _synchronize_distributed_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    """Sum origin-weighted gradients without assuming equal rank shard sizes."""

    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise M03RV15AlphaStepError(
            "v15 two-rank update requires an initialized two-rank process group"
        )
    if not parameters:
        raise M03RV15AlphaStepError("v15 distributed gradient set is empty")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise M03RV15AlphaStepError("v15 trainable parameters span devices")
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
    "M03R_V15_ALPHA_STEP_SCHEMA",
    "M03RV15AlphaStepError",
    "M03RV15AlphaStepReceipt",
    "m03r_v15_learning_rate_multiplier",
    "train_m03r_v15_predictive_batch_update",
]
