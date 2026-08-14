"""Two-way optimizer partition for the direct M03R-v13 predictor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v13_policy import (
    M03R_V13_ENCODER_PARAMETER_PREFIXES,
    Top2000M03RV13PredictivePolicy,
)

M03R_V13_OPTIMIZER_SCHEMA = "rl-quant.top2000-dev.m03r-v13-optimizer-v1"
M03R_V13_ENCODER_LEARNING_RATE = 2.0e-5
M03R_V13_HEAD_LEARNING_RATE = 1.0e-4
M03R_V13_WEIGHT_DECAY = 1.0e-4
M03R_V13_ENCODER_GRADIENT_CLIP_NORM = 1.0
M03R_V13_HEAD_GRADIENT_CLIP_NORM = 1.0


class M03RV13OptimizerError(ValueError):
    """The v13 optimizer partition drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV13OptimizerPartition:
    setting_id: str
    encoder_parameter_names: tuple[str, ...]
    head_parameter_names: tuple[str, ...]
    encoder_learning_rate: float = M03R_V13_ENCODER_LEARNING_RATE
    head_learning_rate: float = M03R_V13_HEAD_LEARNING_RATE
    weight_decay: float = M03R_V13_WEIGHT_DECAY
    encoder_gradient_clip_norm: float = M03R_V13_ENCODER_GRADIENT_CLIP_NORM
    head_gradient_clip_norm: float = M03R_V13_HEAD_GRADIENT_CLIP_NORM
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256
    schema: str = M03R_V13_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_id not in M03R_V13_SETTING_IDS
            or not self.encoder_parameter_names
            or not self.head_parameter_names
            or tuple(sorted(self.encoder_parameter_names))
            != self.encoder_parameter_names
            or tuple(sorted(self.head_parameter_names)) != self.head_parameter_names
            or len(set(self.encoder_parameter_names))
            != len(self.encoder_parameter_names)
            or len(set(self.head_parameter_names)) != len(self.head_parameter_names)
            or set(self.encoder_parameter_names).intersection(self.head_parameter_names)
            or self.encoder_learning_rate != M03R_V13_ENCODER_LEARNING_RATE
            or self.head_learning_rate != M03R_V13_HEAD_LEARNING_RATE
            or self.weight_decay != M03R_V13_WEIGHT_DECAY
            or self.encoder_gradient_clip_norm
            != M03R_V13_ENCODER_GRADIENT_CLIP_NORM
            or self.head_gradient_clip_norm != M03R_V13_HEAD_GRADIENT_CLIP_NORM
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
            or self.schema != M03R_V13_OPTIMIZER_SCHEMA
            or not all(
                name == "source_policy.core.cash_bias"
                or name.startswith(M03R_V13_ENCODER_PARAMETER_PREFIXES)
                for name in self.encoder_parameter_names
            )
            or not all(
                name.startswith(("economic_mean_head.", "economic_scale_head."))
                for name in self.head_parameter_names
            )
        ):
            raise M03RV13OptimizerError("v13 optimizer partition drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _parameter_names(
    policy: Top2000M03RV13PredictivePolicy,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    encoder: list[str] = []
    heads: list[str] = []
    unassigned: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "source_policy.core.cash_bias" or name.startswith(
            M03R_V13_ENCODER_PARAMETER_PREFIXES
        ):
            encoder.append(name)
        elif name.startswith(("economic_mean_head.", "economic_scale_head.")):
            heads.append(name)
        else:
            unassigned.append(name)
    if unassigned:
        raise M03RV13OptimizerError(
            f"v13 has unassigned trainable parameters: {tuple(sorted(unassigned))!r}"
        )
    return tuple(sorted(encoder)), tuple(sorted(heads))


def build_m03r_v13_optimizer(
    policy: Top2000M03RV13PredictivePolicy,
) -> tuple[torch.optim.AdamW, M03RV13OptimizerPartition]:
    if not isinstance(policy, Top2000M03RV13PredictivePolicy):
        raise M03RV13OptimizerError("v13 optimizer requires its exact policy")
    encoder, heads = _parameter_names(policy)
    partition = M03RV13OptimizerPartition(
        setting_id=policy.v13_setting.setting_id,
        encoder_parameter_names=encoder,
        head_parameter_names=heads,
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
                "params": [named[name] for name in heads],
                "lr": partition.head_learning_rate,
                "weight_decay": partition.weight_decay,
                "group_name": "direct-h3-heads",
            },
        )
    )
    return optimizer, partition


def validate_m03r_v13_optimizer(
    policy: Top2000M03RV13PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV13OptimizerPartition,
) -> None:
    partition.validate()
    names = _parameter_names(policy)
    expected_names = (
        partition.encoder_parameter_names,
        partition.head_parameter_names,
    )
    if (
        partition.setting_id != policy.v13_setting.setting_id
        or names != expected_names
        or len(optimizer.param_groups) != 2
    ):
        raise M03RV13OptimizerError("v13 optimizer inventory drifted")
    named = dict(policy.named_parameters())
    expected = (
        ("encoder", partition.encoder_learning_rate, expected_names[0]),
        ("direct-h3-heads", partition.head_learning_rate, expected_names[1]),
    )
    for group, (group_name, rate, parameter_names) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        if (
            group.get("group_name") != group_name
            or float(group["lr"]) != rate
            or float(group["weight_decay"]) != partition.weight_decay
            or tuple(id(value) for value in group["params"])
            != tuple(id(named[name]) for name in parameter_names)
        ):
            raise M03RV13OptimizerError("v13 optimizer group drifted")


__all__ = [
    "M03R_V13_ENCODER_LEARNING_RATE",
    "M03R_V13_HEAD_LEARNING_RATE",
    "M03R_V13_OPTIMIZER_SCHEMA",
    "M03R_V13_WEIGHT_DECAY",
    "M03RV13OptimizerError",
    "M03RV13OptimizerPartition",
    "build_m03r_v13_optimizer",
    "validate_m03r_v13_optimizer",
]
