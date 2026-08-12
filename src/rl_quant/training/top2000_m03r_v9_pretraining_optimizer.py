"""Exact two-rate optimizer partition for M03R-v9 predictive training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)

M03R_V9_ALPHA_OPTIMIZER_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-alpha-optimizer-partition-v1"
)
M03R_V9_ENCODER_LEARNING_RATE = 2.0e-5
M03R_V9_PREDICTION_HEAD_LEARNING_RATE = 1.0e-4
M03R_V9_ADAMW_WEIGHT_DECAY = 1.0e-4
M03R_V9_GRADIENT_CLIP_NORM = 1.0


class M03RV9AlphaOptimizerError(ValueError):
    """The v9 predictive optimizer contains an unauthorized parameter."""


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


@dataclass(frozen=True, slots=True)
class M03RV9AlphaOptimizerPartition:
    setting_id: str
    encoder_parameter_names: tuple[str, ...]
    prediction_head_parameter_names: tuple[str, ...]
    encoder_learning_rate: float = M03R_V9_ENCODER_LEARNING_RATE
    prediction_head_learning_rate: float = M03R_V9_PREDICTION_HEAD_LEARNING_RATE
    weight_decay: float = M03R_V9_ADAMW_WEIGHT_DECAY
    gradient_clip_norm: float = M03R_V9_GRADIENT_CLIP_NORM
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_ALPHA_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        encoder = self.encoder_parameter_names
        heads = self.prediction_head_parameter_names
        if (
            self.setting_id not in M03R_V9_SETTING_IDS
            or not encoder
            or not heads
            or len(set(encoder)) != len(encoder)
            or len(set(heads)) != len(heads)
            or set(encoder) & set(heads)
            or tuple(sorted(encoder)) != encoder
            or tuple(sorted(heads)) != heads
            or self.encoder_learning_rate != M03R_V9_ENCODER_LEARNING_RATE
            or self.prediction_head_learning_rate
            != M03R_V9_PREDICTION_HEAD_LEARNING_RATE
            or not self.encoder_learning_rate < self.prediction_head_learning_rate
            or self.weight_decay != M03R_V9_ADAMW_WEIGHT_DECAY
            or self.gradient_clip_norm != M03R_V9_GRADIENT_CLIP_NORM
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.schema != M03R_V9_ALPHA_OPTIMIZER_SCHEMA
            or not all(name.startswith("source_policy.core.") for name in encoder)
            or not all(
                name.startswith(
                    (
                        "source_policy.core.alpha_head.auxiliary_head.",
                        "alpha_scale_head.",
                    )
                )
                for name in heads
            )
        ):
            raise M03RV9AlphaOptimizerError("v9 optimizer partition drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _partition_parameter_names(
    policy: Top2000M03RV9PredictivePolicy,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    encoder_prefixes = (
        "source_policy.core.raw_encoder.",
        "source_policy.core.news_agg.",
        "source_policy.core.token_proj.",
        "source_policy.core.temporal.",
        "source_policy.core.alloc_in.",
        "source_policy.core.attn.",
    )
    head_prefixes = (
        "source_policy.core.alpha_head.auxiliary_head.",
        "alpha_scale_head.",
    )
    encoder: list[str] = []
    heads: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "source_policy.core.cash_bias" or name.startswith(encoder_prefixes):
            encoder.append(name)
        elif name.startswith(head_prefixes):
            heads.append(name)
    return tuple(sorted(encoder)), tuple(sorted(heads))


def build_m03r_v9_alpha_pretraining_optimizer(
    policy: Top2000M03RV9PredictivePolicy,
) -> tuple[torch.optim.AdamW, M03RV9AlphaOptimizerPartition]:
    if not isinstance(policy, Top2000M03RV9PredictivePolicy):
        raise M03RV9AlphaOptimizerError(
            "v9 optimizer requires the generation-qualified policy"
        )
    encoder_names, head_names = _partition_parameter_names(policy)
    partition = M03RV9AlphaOptimizerPartition(
        setting_id=policy.setting.setting_id,
        encoder_parameter_names=encoder_names,
        prediction_head_parameter_names=head_names,
    )
    partition.validate()
    named = dict(policy.named_parameters())
    optimizer = torch.optim.AdamW(
        (
            {
                "params": [named[name] for name in encoder_names],
                "lr": partition.encoder_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "encoder",
            },
            {
                "params": [named[name] for name in head_names],
                "lr": partition.prediction_head_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "prediction-heads",
            },
        )
    )
    return optimizer, partition


def validate_m03r_v9_alpha_pretraining_optimizer(
    policy: Top2000M03RV9PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
) -> None:
    partition.validate()
    expected_encoder, expected_heads = _partition_parameter_names(policy)
    if (
        partition.setting_id != policy.setting.setting_id
        or partition.encoder_parameter_names != expected_encoder
        or partition.prediction_head_parameter_names != expected_heads
        or len(optimizer.param_groups) != 2
    ):
        raise M03RV9AlphaOptimizerError("v9 optimizer inventory drifted")
    named = dict(policy.named_parameters())
    expected = (
        (
            "encoder",
            partition.encoder_learning_rate,
            tuple(id(named[name]) for name in expected_encoder),
        ),
        (
            "prediction-heads",
            partition.prediction_head_learning_rate,
            tuple(id(named[name]) for name in expected_heads),
        ),
    )
    for group, (group_name, learning_rate, parameter_ids) in zip(
        optimizer.param_groups,
        expected,
        strict=True,
    ):
        if (
            group.get("group_name") != group_name
            or float(group["lr"]) != learning_rate
            or float(group["weight_decay"]) != partition.weight_decay
            or tuple(id(parameter) for parameter in group["params"]) != parameter_ids
        ):
            raise M03RV9AlphaOptimizerError("v9 optimizer group drifted")


__all__ = [
    "M03R_V9_ADAMW_WEIGHT_DECAY",
    "M03R_V9_ALPHA_OPTIMIZER_SCHEMA",
    "M03R_V9_ENCODER_LEARNING_RATE",
    "M03R_V9_GRADIENT_CLIP_NORM",
    "M03R_V9_PREDICTION_HEAD_LEARNING_RATE",
    "M03RV9AlphaOptimizerError",
    "M03RV9AlphaOptimizerPartition",
    "build_m03r_v9_alpha_pretraining_optimizer",
    "validate_m03r_v9_alpha_pretraining_optimizer",
]
