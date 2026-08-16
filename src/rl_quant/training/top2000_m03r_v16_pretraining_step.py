"""Fail-closed selection-only optimizer mutation for M03R-v16."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from rl_quant.protocol.canonical_artifact import semantic_sha256 as _sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_fold import M03RV16TrainingUpdatePlan
from rl_quant.training.top2000_m03r_v16_numerical import (
    M03RV16NumericalTrainingError,
)
from rl_quant.training.top2000_m03r_v16_objective import m03r_v16_score_loss
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

M03R_V16_SCORE_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v16-score-step-v3"


class M03RV16ScoreStepError(ValueError):
    """The V16 score mutation failed before a checkpoint boundary."""


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


def _version_root(names: tuple[str, ...], named: dict[str, torch.nn.Parameter]) -> str:
    return _sha256(tuple((name, named[name]._version) for name in names))


def _learning_rate_multiplier(update_plan: M03RV16TrainingUpdatePlan) -> float:
    update_plan.validate()
    from rl_quant.training.top2000_m03r_v16_fold import (
        render_m03r_v16_fold_geometries,
    )

    spec = M03R_V16_PREDICTIVE_SPEC
    total = render_m03r_v16_fold_geometries(1001)[
        update_plan.fold_index
    ].maximum_optimizer_updates
    step = update_plan.completed_update + 1
    warmup = max(1, math.ceil(total * spec.learning_rate_warmup_fraction))
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    minimum = spec.minimum_learning_rate_multiplier
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _apply_learning_rates(
    optimizer: torch.optim.Optimizer, multiplier: float
) -> tuple[float, float]:
    observed: dict[str, float] = {}
    for group in optimizer.param_groups:
        logical = str(group.get("group_name", ""))
        for suffix in ("-no-decay", "-decay"):
            if logical.endswith(suffix):
                logical = logical.removesuffix(suffix)
        if logical not in {"encoder", "selection-head"}:
            raise M03RV16ScoreStepError("V16 score optimizer group drifted")
        base = float(group.get("base_lr", -1.0))
        rate = base * multiplier
        previous = observed.setdefault(logical, rate)
        if base <= 0.0 or not math.isclose(
            previous, rate, rel_tol=0.0, abs_tol=1.0e-18
        ):
            raise M03RV16ScoreStepError("V16 score learning-rate pair diverged")
        group["lr"] = rate
    return observed["encoder"], observed["selection-head"]


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
    selection_head_version_root_before: str
    selection_head_version_root_after: str
    selection_target_operator_root_sha256: str
    action_operator_root_sha256: str
    total_loss: float
    selection_robust_loss: float
    selection_valid_observations: int
    encoder_gradient_norm_before_clip: float
    selection_head_gradient_norm_before_clip: float
    encoder_gradient_clipped: bool
    selection_head_gradient_clipped: bool
    learning_rate_multiplier: float
    encoder_learning_rate: float
    selection_head_learning_rate: float
    full_state_hashed_in_hot_path: bool = False
    timing_optimizer_updates: int = 0
    uncertainty_calibration_updates: int = 0
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
            self.selection_head_version_root_before,
            self.selection_head_version_root_after,
            self.selection_target_operator_root_sha256,
            self.action_operator_root_sha256,
        )
        spec = M03R_V16_PREDICTIVE_SPEC
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(spec.chronological_fold_count)
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
            or self.selection_head_version_root_before
            == self.selection_head_version_root_after
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.total_loss,
                    self.selection_robust_loss,
                    self.encoder_gradient_norm_before_clip,
                    self.selection_head_gradient_norm_before_clip,
                    self.learning_rate_multiplier,
                    self.encoder_learning_rate,
                    self.selection_head_learning_rate,
                )
            )
            or min(
                self.encoder_gradient_norm_before_clip,
                self.selection_head_gradient_norm_before_clip,
                self.encoder_learning_rate,
                self.selection_head_learning_rate,
            )
            <= 0.0
            or not math.isclose(
                self.encoder_learning_rate,
                spec.score_learning_rates[0] * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                self.selection_head_learning_rate,
                spec.score_learning_rates[1] * self.learning_rate_multiplier,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or self.selection_valid_observations <= 0
            or self.encoder_gradient_clipped
            != (self.encoder_gradient_norm_before_clip > spec.gradient_clip_norm)
            or self.selection_head_gradient_clipped
            != (self.selection_head_gradient_norm_before_clip > spec.gradient_clip_norm)
            or self.full_state_hashed_in_hot_path
            or self.timing_optimizer_updates != 0
            or self.uncertainty_calibration_updates != 0
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
    encoder_rate, head_rate = _apply_learning_rates(optimizer, multiplier)
    named = dict(policy.named_parameters())
    encoder = tuple(named[name] for name in partition.encoder_parameter_names)
    selection_head = tuple(
        named[name] for name in partition.selection_head_parameter_names
    )
    trainable = (*encoder, *selection_head)
    before_encoder = _version_root(partition.encoder_parameter_names, named)
    before_head = _version_root(partition.selection_head_parameter_names, named)
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
    if not gradients or any(
        not bool(torch.isfinite(value).all()) for value in gradients
    ):
        optimizer.zero_grad(set_to_none=True)
        if gradients:
            raise M03RV16NumericalTrainingError(
                "V16 score gradients are non-finite"
            )
        raise M03RV16ScoreStepError("V16 score gradients are absent")
    encoder_norm = _gradient_norm(encoder)
    head_norm = _gradient_norm(selection_head)
    if min(encoder_norm, head_norm) <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16ScoreStepError("V16 encoder and selection head require gradients")
    clip = M03R_V16_PREDICTIVE_SPEC.gradient_clip_norm
    try:
        torch.nn.utils.clip_grad_norm_(encoder, clip, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(
            selection_head, clip, error_if_nonfinite=True
        )
    except RuntimeError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16NumericalTrainingError(
            "V16 score gradient norm is non-finite"
        ) from exc
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16NumericalTrainingError(
            "V16 clipped score gradients are non-finite"
        )
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable) or any(
        not bool(torch.isfinite(value).all()) for value in _optimizer_tensors(optimizer)
    ):
        optimizer.zero_grad(set_to_none=True)
        raise M03RV16NumericalTrainingError(
            "V16 score mutation produced non-finite model or optimizer state"
        )
    after_encoder = _version_root(partition.encoder_parameter_names, named)
    after_head = _version_root(partition.selection_head_parameter_names, named)
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
        selection_head_version_root_before=before_head,
        selection_head_version_root_after=after_head,
        selection_target_operator_root_sha256=_sha256(
            tuple(value.receipt_sha256 for value in batch.selection_target_operators)
        ),
        action_operator_root_sha256=_sha256(
            tuple(value.receipt_sha256 for value in batch.action_operators)
        ),
        total_loss=float(loss.total.detach()),
        selection_robust_loss=float(loss.selection_robust.detach()),
        selection_valid_observations=int(batch.objective.selection_valid.sum()),
        encoder_gradient_norm_before_clip=encoder_norm,
        selection_head_gradient_norm_before_clip=head_norm,
        encoder_gradient_clipped=encoder_norm > clip,
        selection_head_gradient_clipped=head_norm > clip,
        learning_rate_multiplier=multiplier,
        encoder_learning_rate=encoder_rate,
        selection_head_learning_rate=head_rate,
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03R_V16_SCORE_STEP_SCHEMA",
    "M03RV16ScoreStepError",
    "M03RV16ScoreStepReceipt",
    "train_m03r_v16_score_batch_update",
]
