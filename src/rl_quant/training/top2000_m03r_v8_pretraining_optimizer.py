"""Exact optimizer partition for M03R-v8 alpha pretraining.

Only the raw/temporal/cross-sectional encoder and the four-horizon prediction
heads are optimized. Economic score, gate, hazard, exact-action, confidence,
and risk heads are excluded until policy fine-tuning, preventing pretraining
from silently changing the action mechanism through an unrelated parameter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)

M03R_V8_ALPHA_OPTIMIZER_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-alpha-optimizer-partition-v1"
)


class M03RV8AlphaOptimizerError(ValueError):
    """The v8 alpha-pretraining optimizer partition is invalid."""


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
class M03RV8AlphaOptimizerPartition:
    """Content-addressed names and hyperparameters for one optimizer."""

    setting_id: str
    encoder_parameter_names: tuple[str, ...]
    prediction_head_parameter_names: tuple[str, ...]
    encoder_learning_rate: float
    prediction_head_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_ALPHA_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        encoder = self.encoder_parameter_names
        heads = self.prediction_head_parameter_names
        if (
            not self.setting_id.startswith("V8-")
            or not encoder
            or not heads
            or len(set(encoder)) != len(encoder)
            or len(set(heads)) != len(heads)
            or set(encoder) & set(heads)
            or tuple(sorted(encoder)) != encoder
            or tuple(sorted(heads)) != heads
            or self.encoder_learning_rate
            != M03R_V8_ALPHA_PRETRAINING.encoder_learning_rate
            or self.prediction_head_learning_rate
            != M03R_V8_ALPHA_PRETRAINING.prediction_head_learning_rate
            or not self.encoder_learning_rate < self.prediction_head_learning_rate
            or self.weight_decay != M03R_V8_ALPHA_PRETRAINING.adamw_weight_decay
            or self.gradient_clip_norm != M03R_V8_ALPHA_PRETRAINING.gradient_clip_norm
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != M03R_V8_ALPHA_OPTIMIZER_SCHEMA
        ):
            raise M03RV8AlphaOptimizerError(
                "v8 alpha optimizer partition or hyperparameters drifted"
            )
        if not all(
            name.startswith("source_policy.core.") for name in encoder
        ) or not all(
            name.startswith(
                (
                    "source_policy.core.alpha_head.auxiliary_head.",
                    "alpha_log_scale_head.",
                )
            )
            for name in heads
        ):
            raise M03RV8AlphaOptimizerError(
                "v8 optimizer contains a parameter outside the predictive route"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _partition_parameter_names(
    policy: Top2000M03RV8DevelopmentPolicy,
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
        "alpha_log_scale_head.",
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


def build_m03r_v8_alpha_pretraining_optimizer(
    policy: Top2000M03RV8DevelopmentPolicy,
) -> tuple[torch.optim.AdamW, M03RV8AlphaOptimizerPartition]:
    """Create the two-rate AdamW partition for pretrained-alpha rows."""

    if not isinstance(policy, Top2000M03RV8DevelopmentPolicy):
        raise M03RV8AlphaOptimizerError(
            "alpha optimizer requires the generation-qualified v8 policy"
        )
    if policy.setting.alpha_pretraining_mode != "training-fold-pretrained":
        raise M03RV8AlphaOptimizerError(
            "the no-alpha-pretraining row cannot create a pretraining optimizer"
        )
    encoder_names, head_names = _partition_parameter_names(policy)
    partition = M03RV8AlphaOptimizerPartition(
        setting_id=policy.setting.setting_id,
        encoder_parameter_names=encoder_names,
        prediction_head_parameter_names=head_names,
        encoder_learning_rate=M03R_V8_ALPHA_PRETRAINING.encoder_learning_rate,
        prediction_head_learning_rate=(
            M03R_V8_ALPHA_PRETRAINING.prediction_head_learning_rate
        ),
        weight_decay=M03R_V8_ALPHA_PRETRAINING.adamw_weight_decay,
        gradient_clip_norm=M03R_V8_ALPHA_PRETRAINING.gradient_clip_norm,
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


def validate_m03r_v8_alpha_pretraining_optimizer(
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
) -> None:
    """Reject a mutated optimizer or a partition from another policy row."""

    partition.validate()
    if partition.setting_id != policy.setting.setting_id:
        raise M03RV8AlphaOptimizerError("optimizer setting identity drifted")
    expected_encoder, expected_heads = _partition_parameter_names(policy)
    if (
        partition.encoder_parameter_names != expected_encoder
        or partition.prediction_head_parameter_names != expected_heads
        or len(optimizer.param_groups) != 2
    ):
        raise M03RV8AlphaOptimizerError("optimizer parameter inventory drifted")
    named = dict(policy.named_parameters())
    expected_groups = (
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
        optimizer.param_groups, expected_groups, strict=True
    ):
        observed_ids = tuple(id(parameter) for parameter in group["params"])
        if (
            group.get("group_name") != group_name
            or float(group["lr"]) != learning_rate
            or float(group["weight_decay"]) != partition.weight_decay
            or observed_ids != parameter_ids
        ):
            raise M03RV8AlphaOptimizerError("optimizer group binding drifted")


__all__ = [
    "M03R_V8_ALPHA_OPTIMIZER_SCHEMA",
    "M03RV8AlphaOptimizerError",
    "M03RV8AlphaOptimizerPartition",
    "build_m03r_v8_alpha_pretraining_optimizer",
    "validate_m03r_v8_alpha_pretraining_optimizer",
]
