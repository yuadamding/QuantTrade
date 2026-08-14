"""Component-aware gradient boundary for M03R-v12 predictive training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX,
)
from rl_quant.training.top2000_m03r_v12_objective import M03RV12PredictiveLoss
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v12_pretraining_optimizer import (
    M03RV12OptimizerPartition,
    validate_m03r_v12_optimizer,
)


class M03RV12GradientBalanceError(ValueError):
    """The v12 component-gradient boundary failed closed."""


@dataclass(frozen=True, slots=True)
class M03RV12GradientBalanceReceipt:
    economic_encoder_gradient_norm: float
    raw_rank_encoder_gradient_norm: float
    rank_encoder_multiplier: float
    effective_rank_encoder_gradient_norm: float
    combined_encoder_gradient_norm_before_clip: float
    economic_head_gradient_norm_before_clip: float
    rank_head_gradient_norm_before_clip: float
    maximum_rank_to_economic_encoder_gradient_ratio: float
    distributed_world_size: int

    def validate(self) -> None:
        values = (
            self.economic_encoder_gradient_norm,
            self.raw_rank_encoder_gradient_norm,
            self.rank_encoder_multiplier,
            self.effective_rank_encoder_gradient_norm,
            self.combined_encoder_gradient_norm_before_clip,
            self.economic_head_gradient_norm_before_clip,
            self.rank_head_gradient_norm_before_clip,
            self.maximum_rank_to_economic_encoder_gradient_ratio,
        )
        bound = (
            self.maximum_rank_to_economic_encoder_gradient_ratio
            * self.economic_encoder_gradient_norm
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in values)
            or self.rank_encoder_multiplier > 1.0
            or self.effective_rank_encoder_gradient_norm > bound + 1.0e-7
            or self.maximum_rank_to_economic_encoder_gradient_ratio
            != M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX
            or self.distributed_world_size not in {1, 2}
        ):
            raise M03RV12GradientBalanceError("v12 gradient-balance receipt drifted")


def _parameters(
    policy: Top2000M03RV12PredictivePolicy,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    named = dict(policy.named_parameters())
    return tuple(named[name] for name in names)


def _norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    total = sum(
        float(value.detach().double().square().sum())
        for value in gradients
        if value is not None
    )
    return math.sqrt(total)


def _distributed_average(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.Tensor, ...],
    *,
    world_size: int,
) -> tuple[torch.Tensor | None, ...]:
    if world_size == 1:
        return gradients
    rows: list[torch.Tensor | None] = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        used = torch.tensor(
            0 if gradient is None else 1,
            dtype=torch.int64,
            device=parameter.device,
        )
        dist.all_reduce(used, op=dist.ReduceOp.SUM)
        if int(used.item()) == 0:
            rows.append(None)
            continue
        value = torch.zeros_like(parameter) if gradient is None else gradient.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(float(world_size))
        rows.append(value)
    return tuple(rows)


def _require_distributed(rank: int, world_size: int) -> None:
    if world_size not in {1, 2} or rank not in range(world_size):
        raise M03RV12GradientBalanceError("v12 distributed geometry drifted")
    if world_size == 2 and (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != world_size
        or dist.get_rank() != rank
    ):
        raise M03RV12GradientBalanceError(
            "v12 two-rank gradient boundary lacks its process group"
        )


def install_m03r_v12_balanced_gradients(
    policy: Top2000M03RV12PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV12OptimizerPartition,
    loss: M03RV12PredictiveLoss,
    *,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV12GradientBalanceReceipt:
    """Install bounded component gradients without mutating parameters."""

    validate_m03r_v12_optimizer(policy, optimizer, partition)
    _require_distributed(distributed_rank, distributed_world_size)
    encoder = _parameters(policy, partition.encoder_parameter_names)
    economic = _parameters(policy, partition.economic_head_parameter_names)
    rank = _parameters(policy, partition.rank_head_parameter_names)
    optimizer.zero_grad(set_to_none=True)
    economic_raw = torch.autograd.grad(
        loss.economic_total,
        encoder + economic,
        retain_graph=loss.component_weights[0] > 0.0,
        allow_unused=True,
    )
    economic_gradients: tuple[torch.Tensor | None, ...] = _distributed_average(
        economic_raw,
        encoder + economic,
        world_size=distributed_world_size,
    )
    economic_encoder = economic_gradients[: len(encoder)]
    economic_heads = economic_gradients[len(encoder) :]
    rank_encoder: tuple[torch.Tensor | None, ...]
    rank_heads: tuple[torch.Tensor | None, ...]
    if loss.component_weights[0] > 0.0:
        rank_raw = torch.autograd.grad(
            loss.component_weights[0] * loss.ranking,
            encoder + rank,
            allow_unused=True,
        )
        rank_gradients: tuple[torch.Tensor | None, ...] = _distributed_average(
            rank_raw,
            encoder + rank,
            world_size=distributed_world_size,
        )
        rank_encoder = rank_gradients[: len(encoder)]
        rank_heads = rank_gradients[len(encoder) :]
    else:
        rank_encoder = tuple(None for _ in encoder)
        rank_heads = tuple(None for _ in rank)
    economic_encoder_norm = _norm(economic_encoder)
    raw_rank_encoder_norm = _norm(rank_encoder)
    if raw_rank_encoder_norm == 0.0 or economic_encoder_norm == 0.0:
        multiplier = 0.0
    else:
        multiplier = min(
            1.0,
            M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX
            * economic_encoder_norm
            / raw_rank_encoder_norm,
        )

    combined_encoder: list[torch.Tensor | None] = []
    for economic_gradient, rank_gradient in zip(
        economic_encoder, rank_encoder, strict=True
    ):
        if economic_gradient is None and rank_gradient is None:
            combined_encoder.append(None)
        elif economic_gradient is None:
            assert rank_gradient is not None
            combined_encoder.append(rank_gradient * multiplier)
        elif rank_gradient is None:
            combined_encoder.append(economic_gradient)
        else:
            combined_encoder.append(economic_gradient + rank_gradient * multiplier)

    for parameter, gradient in zip(encoder, combined_encoder, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    for parameter, gradient in zip(economic, economic_heads, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    for parameter, gradient in zip(rank, rank_heads, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()

    populated_encoder = tuple(value for value in encoder if value.grad is not None)
    populated_economic = tuple(value for value in economic if value.grad is not None)
    populated_rank = tuple(value for value in rank if value.grad is not None)
    if not populated_encoder or not populated_economic:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV12GradientBalanceError(
            "v12 economic objective produced no trainable gradients"
        )
    try:
        encoder_before_clip = float(
            torch.nn.utils.clip_grad_norm_(
                populated_encoder,
                partition.encoder_gradient_clip_norm,
                error_if_nonfinite=True,
            )
        )
        economic_before_clip = float(
            torch.nn.utils.clip_grad_norm_(
                populated_economic,
                partition.economic_head_gradient_clip_norm,
                error_if_nonfinite=True,
            )
        )
        rank_before_clip = (
            0.0
            if not populated_rank
            else float(
                torch.nn.utils.clip_grad_norm_(
                    populated_rank,
                    partition.rank_head_gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            )
        )
    except RuntimeError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise M03RV12GradientBalanceError(
            "v12 non-finite component gradient rejected before mutation"
        ) from exc
    receipt = M03RV12GradientBalanceReceipt(
        economic_encoder_gradient_norm=economic_encoder_norm,
        raw_rank_encoder_gradient_norm=raw_rank_encoder_norm,
        rank_encoder_multiplier=multiplier,
        effective_rank_encoder_gradient_norm=raw_rank_encoder_norm * multiplier,
        combined_encoder_gradient_norm_before_clip=encoder_before_clip,
        economic_head_gradient_norm_before_clip=economic_before_clip,
        rank_head_gradient_norm_before_clip=rank_before_clip,
        maximum_rank_to_economic_encoder_gradient_ratio=(
            M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX
        ),
        distributed_world_size=distributed_world_size,
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03RV12GradientBalanceError",
    "M03RV12GradientBalanceReceipt",
    "install_m03r_v12_balanced_gradients",
]
