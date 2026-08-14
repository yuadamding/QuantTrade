"""Three-way optimizer partition for M03R-v12 rank/scale separation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PREDICTIVE_SPEC,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)

M03R_V12_OPTIMIZER_SCHEMA = "rl-quant.top2000-dev.m03r-v12-optimizer-v1"
M03R_V12_ENCODER_LEARNING_RATE = 2.0e-5
M03R_V12_HEAD_LEARNING_RATE = 1.0e-4
M03R_V12_WEIGHT_DECAY = 1.0e-4


class M03RV12OptimizerError(ValueError):
    """The v12 optimizer partition drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12OptimizerPartition:
    setting_id: str
    encoder_parameter_names: tuple[str, ...]
    economic_head_parameter_names: tuple[str, ...]
    rank_head_parameter_names: tuple[str, ...]
    encoder_learning_rate: float = M03R_V12_ENCODER_LEARNING_RATE
    economic_head_learning_rate: float = M03R_V12_HEAD_LEARNING_RATE
    rank_head_learning_rate: float = M03R_V12_HEAD_LEARNING_RATE
    weight_decay: float = M03R_V12_WEIGHT_DECAY
    encoder_gradient_clip_norm: float = (
        M03R_V12_PREDICTIVE_SPEC.encoder_gradient_clip_norm
    )
    economic_head_gradient_clip_norm: float = (
        M03R_V12_PREDICTIVE_SPEC.economic_head_gradient_clip_norm
    )
    rank_head_gradient_clip_norm: float = (
        M03R_V12_PREDICTIVE_SPEC.rank_head_gradient_clip_norm
    )
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        groups = (
            self.encoder_parameter_names,
            self.economic_head_parameter_names,
            self.rank_head_parameter_names,
        )
        if (
            self.setting_id not in M03R_V12_SETTING_IDS
            or any(not group for group in groups)
            or any(tuple(sorted(group)) != group for group in groups)
            or any(len(set(group)) != len(group) for group in groups)
            or set(groups[0]) & set(groups[1])
            or set(groups[0]) & set(groups[2])
            or set(groups[1]) & set(groups[2])
            or self.encoder_learning_rate != M03R_V12_ENCODER_LEARNING_RATE
            or self.economic_head_learning_rate != M03R_V12_HEAD_LEARNING_RATE
            or self.rank_head_learning_rate != M03R_V12_HEAD_LEARNING_RATE
            or self.weight_decay != M03R_V12_WEIGHT_DECAY
            or self.encoder_gradient_clip_norm != 1.0
            or self.economic_head_gradient_clip_norm != 1.0
            or self.rank_head_gradient_clip_norm != 0.25
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_OPTIMIZER_SCHEMA
            or not all(name.startswith("source_policy.core.") for name in groups[0])
            or not all(
                name.startswith(
                    (
                        "economic_mean_head.",
                        "economic_scale_head.",
                    )
                )
                for name in groups[1]
            )
            or not all(name.startswith("rank_score_head.") for name in groups[2])
        ):
            raise M03RV12OptimizerError("v12 optimizer partition drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _parameter_names(
    policy: Top2000M03RV12PredictivePolicy,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    encoder_prefixes = (
        "source_policy.core.raw_encoder.",
        "source_policy.core.news_agg.",
        "source_policy.core.token_proj.",
        "source_policy.core.temporal.",
        "source_policy.core.alloc_in.",
        "source_policy.core.attn.",
    )
    economic_prefixes = (
        "economic_mean_head.",
        "economic_scale_head.",
    )
    encoder: list[str] = []
    economic: list[str] = []
    rank: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "source_policy.core.cash_bias" or name.startswith(encoder_prefixes):
            encoder.append(name)
        elif name.startswith(economic_prefixes):
            economic.append(name)
        elif name.startswith("rank_score_head."):
            rank.append(name)
    return tuple(sorted(encoder)), tuple(sorted(economic)), tuple(sorted(rank))


def build_m03r_v12_optimizer(
    policy: Top2000M03RV12PredictivePolicy,
) -> tuple[torch.optim.AdamW, M03RV12OptimizerPartition]:
    if not isinstance(policy, Top2000M03RV12PredictivePolicy):
        raise M03RV12OptimizerError("v12 optimizer requires its exact policy")
    encoder, economic, rank = _parameter_names(policy)
    partition = M03RV12OptimizerPartition(
        setting_id=policy.v12_setting.setting_id,
        encoder_parameter_names=encoder,
        economic_head_parameter_names=economic,
        rank_head_parameter_names=rank,
    )
    partition.validate()
    named = dict(policy.named_parameters())
    optimizer = torch.optim.AdamW(
        (
            {
                "params": [named[name] for name in encoder],
                "lr": partition.encoder_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "encoder",
            },
            {
                "params": [named[name] for name in economic],
                "lr": partition.economic_head_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "economic-heads",
            },
            {
                "params": [named[name] for name in rank],
                "lr": partition.rank_head_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "rank-head",
            },
        )
    )
    return optimizer, partition


def validate_m03r_v12_optimizer(
    policy: Top2000M03RV12PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV12OptimizerPartition,
) -> None:
    partition.validate()
    groups = _parameter_names(policy)
    expected_names = (
        partition.encoder_parameter_names,
        partition.economic_head_parameter_names,
        partition.rank_head_parameter_names,
    )
    if (
        partition.setting_id != policy.v12_setting.setting_id
        or groups != expected_names
        or len(optimizer.param_groups) != 3
    ):
        raise M03RV12OptimizerError("v12 optimizer inventory drifted")
    named = dict(policy.named_parameters())
    expected = (
        ("encoder", partition.encoder_learning_rate, expected_names[0]),
        ("economic-heads", partition.economic_head_learning_rate, expected_names[1]),
        ("rank-head", partition.rank_head_learning_rate, expected_names[2]),
    )
    for group, (group_name, rate, names) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        if (
            group.get("group_name") != group_name
            or float(group["lr"]) != rate
            or float(group["weight_decay"]) != partition.weight_decay
            or tuple(id(value) for value in group["params"])
            != tuple(id(named[name]) for name in names)
        ):
            raise M03RV12OptimizerError("v12 optimizer group drifted")


__all__ = [
    "M03R_V12_ENCODER_LEARNING_RATE",
    "M03R_V12_HEAD_LEARNING_RATE",
    "M03R_V12_OPTIMIZER_SCHEMA",
    "M03R_V12_WEIGHT_DECAY",
    "M03RV12OptimizerError",
    "M03RV12OptimizerPartition",
    "build_m03r_v12_optimizer",
    "validate_m03r_v12_optimizer",
]
