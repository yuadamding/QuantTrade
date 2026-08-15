"""Fail-closed score-stage optimizer mutation for M03R-v16."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16FoldGeometry,
    M03RV16TrainingUpdatePlan,
)
from rl_quant.training.top2000_m03r_v16_objective import (
    m03r_v16_scale_calibration_loss,
    m03r_v16_score_loss,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    M03RV16OptimizerPartition,
    validate_m03r_v16_optimizer,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
)

M03R_V16_SCORE_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v16-score-step-v1"
M03R_V16_SCALE_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v16-scale-step-v1"


class M03RV16ScoreStepError(ValueError):
    """The V16 score mutation failed before a checkpoint boundary."""


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
                tuple(value.to(torch.float64).square().sum() for value in gradients)
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


def _version_root(
    names: tuple[str, ...], named: dict[str, torch.nn.Parameter]
) -> str:
    return _sha256(tuple((name, named[name]._version) for name in names))


def _learning_rate_multiplier(update_plan: M03RV16TrainingUpdatePlan) -> float:
    update_plan.validate()
    spec = M03R_V16_PREDICTIVE_SPEC
    # The fold-specific maximum is encoded in the validated plan's epoch cursor.
    from rl_quant.training.top2000_m03r_v16_fold import (
        render_m03r_v16_fold_geometries,
    )

    total = render_m03r_v16_fold_geometries(1001)[
        update_plan.fold_index
    ].maximum_optimizer_updates
    step = update_plan.completed_update + 1
    warmup = max(1, math.ceil(total * spec.learning_rate_warmup_fraction))
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    minimum = spec.minimum_learning_rate_multiplier
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _apply_learning_rates(
    optimizer: torch.optim.Optimizer, multiplier: float
) -> tuple[float, float]:
    observed: dict[str, float] = {}
    for group in optimizer.param_groups:
        logical = str(group.get("group_name", "")).removesuffix(
            "-no-decay"
        ).removesuffix("-decay")
        if logical not in {"encoder", "mean"}:
            raise M03RV16ScoreStepError("V16 score optimizer group drifted")
        base = float(group.get("base_lr", -1.0))
        rate = base * multiplier
        previous = observed.setdefault(logical, rate)
        if base <= 0.0 or not math.isclose(
            previous, rate, rel_tol=0.0, abs_tol=1.0e-18
        ):
            raise M03RV16ScoreStepError("V16 score learning-rate pair diverged")
        group["lr"] = rate
    return observed["encoder"], observed["mean"]


def _synchronize_distributed_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise M03RV16ScoreStepError(
            "V16 two-rank score step requires an initialized two-rank group"
        )
    if not parameters:
        raise M03RV16ScoreStepError("V16 distributed parameter set is empty")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise M03RV16ScoreStepError("V16 trainable parameters span devices")
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
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        )
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        parameter.grad = gradient


@dataclass(frozen=True, slots=True)
class M03RV16ScoreStepReceipt:
    setting_index: int
    fold_index: int
    completed_updates_before: int
    completed_updates_after: int
    distributed_rank: int
    distributed_world_size: int
    local_origin_count: int
    global_origin_count: int
    local_gradient_weight: float
    distributed_gradient_synchronized: bool
    batch_receipt_sha256: str
    update_plan_sha256: str
    optimizer_partition_sha256: str
    encoder_version_root_before: str
    encoder_version_root_after: str
    mean_version_root_before: str
    mean_version_root_after: str
    selection_target_operator_root_sha256: str
    timing_target_operator_root_sha256: str
    action_operator_root_sha256: str
    total_loss: float
    selection_robust_loss: float
    timing_robust_loss: float
    selection_valid_observations: int
    timing_valid_observations: int
    encoder_gradient_norm_before_clip: float
    mean_gradient_norm_before_clip: float
    encoder_gradient_clipped: bool
    mean_gradient_clipped: bool
    learning_rate_multiplier: float
    encoder_learning_rate: float
    mean_learning_rate: float
    full_state_hashed_in_hot_path: bool = False
    qualification_tail_accessed: bool = False
    outer_2026_accessed: bool = False
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_SCORE_STEP_SCHEMA

    def validate(self) -> None:
        digests = (
            self.batch_receipt_sha256,
            self.update_plan_sha256,
            self.optimizer_partition_sha256,
            self.encoder_version_root_before,
            self.encoder_version_root_after,
            self.mean_version_root_before,
            self.mean_version_root_after,
            self.selection_target_operator_root_sha256,
            self.timing_target_operator_root_sha256,
            self.action_operator_root_sha256,
        )
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(6)
            or self.completed_updates_before < 0
            or self.completed_updates_after != self.completed_updates_before + 1
            or self.distributed_world_size not in {1, 2}
            or self.distributed_rank not in range(self.distributed_world_size)
            or self.local_origin_count <= 0
            or self.global_origin_count < self.local_origin_count
            or not math.isclose(
                self.local_gradient_weight,
                self.local_origin_count / self.global_origin_count,
                abs_tol=1.0e-15,
            )
            or self.distributed_gradient_synchronized
            != (self.distributed_world_size == 2)
            or not all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in digests
            )
            or self.encoder_version_root_before == self.encoder_version_root_after
            or self.mean_version_root_before == self.mean_version_root_after
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.total_loss,
                    self.selection_robust_loss,
                    self.timing_robust_loss,
                    self.encoder_gradient_norm_before_clip,
                    self.mean_gradient_norm_before_clip,
                    self.learning_rate_multiplier,
                    self.encoder_learning_rate,
                    self.mean_learning_rate,
                )
            )
            or min(
                self.encoder_gradient_norm_before_clip,
                self.mean_gradient_norm_before_clip,
                self.encoder_learning_rate,
                self.mean_learning_rate,
            )
            <= 0.0
            or not math.isclose(
                self.encoder_learning_rate,
                M03R_V16_PREDICTIVE_SPEC.score_learning_rates[0]
                * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                self.mean_learning_rate,
                M03R_V16_PREDICTIVE_SPEC.score_learning_rates[1]
                * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or self.selection_valid_observations <= 0
            or self.timing_valid_observations <= 0
            or self.encoder_gradient_clipped
            != (
                self.encoder_gradient_norm_before_clip
                > M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
            )
            or self.mean_gradient_clipped
            != (
                self.mean_gradient_norm_before_clip
                > M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
            )
            or self.full_state_hashed_in_hot_path
            or self.qualification_tail_accessed
            or self.outer_2026_accessed
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_SCORE_STEP_SCHEMA
        ):
            raise M03RV16ScoreStepError("V16 score-step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16ScaleStepReceipt:
    setting_index: int
    fold_index: int
    calibration_epoch_index: int
    distributed_rank: int
    distributed_world_size: int
    local_origin_count: int
    global_origin_count: int
    local_gradient_weight: float
    distributed_gradient_synchronized: bool
    batch_receipt_sha256: str
    optimizer_partition_sha256: str
    selected_score_model_state_sha256: str
    selected_score_component_state_sha256: str
    selected_score_checkpoint_file_sha256: str
    checkpoint_selection_receipt_sha256: str
    scale_version_root_before: str
    scale_version_root_after: str
    total_loss: float
    selection_distributional_loss: float
    timing_distributional_loss: float
    scale_gradient_norm_before_clip: float
    scale_gradient_clipped: bool
    scale_learning_rate: float
    encoder_and_mean_frozen: bool = True
    qualification_tail_accessed: bool = False
    outer_2026_accessed: bool = False
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_SCALE_STEP_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(6)
            or self.calibration_epoch_index
            not in range(M03R_V16_PREDICTIVE_SPEC.scale_calibration_epochs)
            or self.distributed_world_size not in {1, 2}
            or self.distributed_rank not in range(self.distributed_world_size)
            or self.local_origin_count <= 0
            or self.global_origin_count
            != M03R_V16_PREDICTIVE_SPEC.inner_validation_origins_per_fold
            or not math.isclose(
                self.local_gradient_weight,
                self.local_origin_count / self.global_origin_count,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or self.distributed_gradient_synchronized
            != (self.distributed_world_size == 2)
            or not all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (
                    self.batch_receipt_sha256,
                    self.optimizer_partition_sha256,
                    self.selected_score_model_state_sha256,
                    self.selected_score_component_state_sha256,
                    self.selected_score_checkpoint_file_sha256,
                    self.checkpoint_selection_receipt_sha256,
                    self.scale_version_root_before,
                    self.scale_version_root_after,
                )
            )
            or self.scale_version_root_before == self.scale_version_root_after
            or not all(
                math.isfinite(value)
                for value in (
                    self.total_loss,
                    self.selection_distributional_loss,
                    self.timing_distributional_loss,
                )
            )
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.scale_gradient_norm_before_clip,
                    self.scale_learning_rate,
                )
            )
            or min(self.scale_gradient_norm_before_clip, self.scale_learning_rate)
            <= 0.0
            or self.scale_gradient_clipped
            != (
                self.scale_gradient_norm_before_clip
                > M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
            )
            or not math.isclose(
                self.scale_learning_rate,
                M03R_V16_PREDICTIVE_SPEC.scale_calibration_learning_rate,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not self.encoder_and_mean_frozen
            or self.qualification_tail_accessed
            or self.outer_2026_accessed
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_SCALE_STEP_SCHEMA
        ):
            raise M03RV16ScoreStepError("V16 scale-step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def train_m03r_v16_score_batch_update(
    policy: Top2000M03RV16PredictivePolicy,
    batch: M03RV16BuiltPredictiveBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
    update_plan: M03RV16TrainingUpdatePlan,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    gradient_synchronizer: Callable[[tuple[torch.nn.Parameter, ...]], None]
    | None = None,
) -> M03RV16ScoreStepReceipt:
    batch.validate()
    update_plan.validate()
    if (
        partition.stage != "score"
        or policy.v16_setting != batch.objective.setting
        or update_plan.setting_index != policy.v16_setting.setting_index
        or update_plan.fold_index != batch.fold_index
        or completed_updates != update_plan.completed_update
        or distributed_world_size not in {1, 2}
        or distributed_rank not in range(distributed_world_size)
        or tuple(int(value) for value in batch.origin_indices)
        != (
            update_plan.global_origins
            if distributed_world_size == 1
            else update_plan.rank_origins[distributed_rank]
        )
    ):
        raise M03RV16ScoreStepError("V16 score batch, cursor, or rank drifted")
    validate_m03r_v16_optimizer(policy, optimizer, partition)
    multiplier = _learning_rate_multiplier(update_plan)
    encoder_rate, mean_rate = _apply_learning_rates(optimizer, multiplier)
    named = dict(policy.named_parameters())
    encoder = tuple(named[name] for name in partition.encoder_parameter_names)
    means = tuple(named[name] for name in partition.mean_parameter_names)
    trainable = (*encoder, *means)
    before_encoder = _version_root(partition.encoder_parameter_names, named)
    before_mean = _version_root(partition.mean_parameter_names, named)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v16_score_loss(batch.objective)
    local_count = int(batch.origin_indices.numel())
    global_count = len(update_plan.global_origins)
    local_weight = local_count / global_count
    (loss.total * local_weight).backward()  # type: ignore[no-untyped-call]
    if distributed_world_size == 2:
        (gradient_synchronizer or _synchronize_distributed_gradients)(trainable)
    elif gradient_synchronizer is not None:
        raise M03RV16ScoreStepError(
            "V16 single-rank score step must not synchronize gradients"
        )
    gradients = tuple(
        parameter.grad for parameter in trainable if parameter.grad is not None
    )
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 score gradients are absent or non-finite")
    encoder_norm = _gradient_norm(encoder)
    mean_norm = _gradient_norm(means)
    if min(encoder_norm, mean_norm) <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 encoder and mean groups require gradients")
    clip = M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
    torch.nn.utils.clip_grad_norm_(encoder, clip, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(means, clip, error_if_nonfinite=True)
    clipped_gradients = tuple(
        parameter.grad for parameter in trainable if parameter.grad is not None
    )
    if not clipped_gradients or any(
        not bool(torch.isfinite(value).all()) for value in clipped_gradients
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 clipped score gradients are invalid")
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable) or any(
        not bool(torch.isfinite(value).all()) for value in _optimizer_tensors(optimizer)
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError(
            "V16 score mutation produced non-finite model or optimizer state"
        )
    after_encoder = _version_root(partition.encoder_parameter_names, named)
    after_mean = _version_root(partition.mean_parameter_names, named)
    receipt = M03RV16ScoreStepReceipt(
        setting_index=policy.v16_setting.setting_index,
        fold_index=update_plan.fold_index,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        local_origin_count=local_count,
        global_origin_count=global_count,
        local_gradient_weight=local_weight,
        distributed_gradient_synchronized=distributed_world_size == 2,
        batch_receipt_sha256=batch.receipt_sha256,
        update_plan_sha256=update_plan.receipt_sha256,
        optimizer_partition_sha256=partition.receipt_sha256,
        encoder_version_root_before=before_encoder,
        encoder_version_root_after=after_encoder,
        mean_version_root_before=before_mean,
        mean_version_root_after=after_mean,
        selection_target_operator_root_sha256=_sha256(
            tuple(value.receipt_sha256 for value in batch.selection_target_operators)
        ),
        timing_target_operator_root_sha256=_sha256(
            tuple(value.receipt_sha256 for value in batch.timing_target_operators)
        ),
        action_operator_root_sha256=_sha256(
            tuple(value.receipt_sha256 for value in batch.action_operators)
        ),
        total_loss=float(loss.total.detach()),
        selection_robust_loss=float(loss.selection_robust.detach()),
        timing_robust_loss=float(loss.timing_robust.detach()),
        selection_valid_observations=int(batch.objective.selection_valid.sum()),
        timing_valid_observations=int(batch.objective.timing_valid.sum()),
        encoder_gradient_norm_before_clip=encoder_norm,
        mean_gradient_norm_before_clip=mean_norm,
        encoder_gradient_clipped=encoder_norm > clip,
        mean_gradient_clipped=mean_norm > clip,
        learning_rate_multiplier=multiplier,
        encoder_learning_rate=encoder_rate,
        mean_learning_rate=mean_rate,
    )
    receipt.validate()
    return receipt


def train_m03r_v16_scale_calibration_update(
    policy: Top2000M03RV16PredictivePolicy,
    batch: M03RV16BuiltPredictiveBatch,
    geometry: M03RV16FoldGeometry,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
    *,
    calibration_epoch_index: int,
    distributed_rank: int,
    distributed_world_size: int,
    gradient_synchronizer: Callable[[tuple[torch.nn.Parameter, ...]], None]
    | None = None,
    selected_score_model_state_sha256: str,
    selected_score_component_state_sha256: str,
    selected_score_checkpoint_file_sha256: str,
    checkpoint_selection_receipt_sha256: str,
) -> M03RV16ScaleStepReceipt:
    """Calibrate scale on training-only validation after the mean is frozen."""

    batch.validate()
    geometry.validate()
    expected_origins = tuple(
        range(
            geometry.inner_validation_origin_start_inclusive,
            geometry.inner_validation_origin_stop_exclusive,
        )
    )
    if (
        partition.stage != "scale_calibration"
        or policy.v16_setting != batch.objective.setting
        or batch.split != "inner_validation"
        or batch.fold_index != geometry.fold_index
        or calibration_epoch_index
        not in range(M03R_V16_PREDICTIVE_SPEC.scale_calibration_epochs)
        or batch.policy_state_binding_kind != "model-state-sha256"
        or batch.policy_score_component_state_sha256
        != selected_score_component_state_sha256
        or not all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (
                selected_score_model_state_sha256,
                selected_score_component_state_sha256,
                selected_score_checkpoint_file_sha256,
                checkpoint_selection_receipt_sha256,
            )
        )
        or distributed_world_size not in {1, 2}
        or distributed_rank not in range(distributed_world_size)
        or tuple(int(value) for value in batch.origin_indices)
        != (
            expected_origins
            if distributed_world_size == 1
            else expected_origins[distributed_rank::2]
        )
    ):
        raise M03RV16ScoreStepError("V16 calibration batch, cursor, or rank drifted")
    validate_m03r_v16_optimizer(policy, optimizer, partition)
    named = dict(policy.named_parameters())
    scales = tuple(named[name] for name in partition.scale_parameter_names)
    before = _version_root(partition.scale_parameter_names, named)
    optimizer.zero_grad(set_to_none=True)
    loss = m03r_v16_scale_calibration_loss(batch.objective)
    local_count = int(batch.origin_indices.numel())
    global_count = len(expected_origins)
    local_weight = local_count / global_count
    (loss.total * local_weight).backward()  # type: ignore[no-untyped-call]
    if distributed_world_size == 2:
        (gradient_synchronizer or _synchronize_distributed_gradients)(scales)
    elif gradient_synchronizer is not None:
        raise M03RV16ScoreStepError(
            "V16 single-rank scale step must not synchronize gradients"
        )
    gradients = tuple(
        parameter.grad for parameter in scales if parameter.grad is not None
    )
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 scale gradients are absent or non-finite")
    scale_norm = _gradient_norm(scales)
    if scale_norm <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 scale group requires a gradient")
    clip = M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
    torch.nn.utils.clip_grad_norm_(scales, clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in scales) or any(
        not bool(torch.isfinite(value).all()) for value in _optimizer_tensors(optimizer)
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError(
            "V16 scale mutation produced non-finite model or optimizer state"
        )
    after = _version_root(partition.scale_parameter_names, named)
    receipt = M03RV16ScaleStepReceipt(
        setting_index=policy.v16_setting.setting_index,
        fold_index=geometry.fold_index,
        calibration_epoch_index=calibration_epoch_index,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        local_origin_count=local_count,
        global_origin_count=global_count,
        local_gradient_weight=local_weight,
        distributed_gradient_synchronized=distributed_world_size == 2,
        batch_receipt_sha256=batch.receipt_sha256,
        optimizer_partition_sha256=partition.receipt_sha256,
        selected_score_model_state_sha256=selected_score_model_state_sha256,
        selected_score_component_state_sha256=(
            selected_score_component_state_sha256
        ),
        selected_score_checkpoint_file_sha256=(
            selected_score_checkpoint_file_sha256
        ),
        checkpoint_selection_receipt_sha256=(
            checkpoint_selection_receipt_sha256
        ),
        scale_version_root_before=before,
        scale_version_root_after=after,
        total_loss=float(loss.total.detach()),
        selection_distributional_loss=float(loss.selection_distributional.detach()),
        timing_distributional_loss=float(loss.timing_distributional.detach()),
        scale_gradient_norm_before_clip=scale_norm,
        scale_gradient_clipped=scale_norm > clip,
        scale_learning_rate=M03R_V16_PREDICTIVE_SPEC.scale_calibration_learning_rate,
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03R_V16_SCORE_STEP_SCHEMA",
    "M03R_V16_SCALE_STEP_SCHEMA",
    "M03RV16ScaleStepReceipt",
    "M03RV16ScoreStepError",
    "M03RV16ScoreStepReceipt",
    "train_m03r_v16_score_batch_update",
    "train_m03r_v16_scale_calibration_update",
]
