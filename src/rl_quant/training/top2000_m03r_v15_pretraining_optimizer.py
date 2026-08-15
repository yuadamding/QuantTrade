"""Independent encoder, mean, and scale optimizer geometry for M03R-v15."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PROTOCOL_SHA256,
    M03R_V15_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v15_policy import (
    M03R_V15_ENCODER_PARAMETER_PREFIXES,
    Top2000M03RV15PredictivePolicy,
)

M03R_V15_OPTIMIZER_SCHEMA = "rl-quant.top2000-dev.m03r-v15-optimizer-v1"
M03R_V15_ENCODER_LEARNING_RATE = 2.0e-5
M03R_V15_MEAN_HEAD_LEARNING_RATE = 1.0e-4
M03R_V15_SCALE_HEAD_LEARNING_RATE = 1.0e-4
M03R_V15_WEIGHT_DECAY = 1.0e-4
M03R_V15_ENCODER_GRADIENT_CLIP_NORM = 1.0
M03R_V15_MEAN_HEAD_GRADIENT_CLIP_NORM = 1.0
M03R_V15_SCALE_HEAD_GRADIENT_CLIP_NORM = 1.0


class M03RV15OptimizerError(ValueError):
    """The v15 optimizer partition drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV15OptimizerPartition:
    setting_id: str
    encoder_parameter_names: tuple[str, ...]
    mean_head_parameter_names: tuple[str, ...]
    scale_head_parameter_names: tuple[str, ...]
    encoder_learning_rate: float = M03R_V15_ENCODER_LEARNING_RATE
    mean_head_learning_rate: float = M03R_V15_MEAN_HEAD_LEARNING_RATE
    scale_head_learning_rate: float = M03R_V15_SCALE_HEAD_LEARNING_RATE
    weight_decay: float = M03R_V15_WEIGHT_DECAY
    encoder_gradient_clip_norm: float = M03R_V15_ENCODER_GRADIENT_CLIP_NORM
    mean_head_gradient_clip_norm: float = M03R_V15_MEAN_HEAD_GRADIENT_CLIP_NORM
    scale_head_gradient_clip_norm: float = M03R_V15_SCALE_HEAD_GRADIENT_CLIP_NORM
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    schema: str = M03R_V15_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_id not in M03R_V15_SETTING_IDS
            or not self.encoder_parameter_names
            or not self.mean_head_parameter_names
            or not self.scale_head_parameter_names
            or tuple(sorted(self.encoder_parameter_names))
            != self.encoder_parameter_names
            or tuple(sorted(self.mean_head_parameter_names))
            != self.mean_head_parameter_names
            or tuple(sorted(self.scale_head_parameter_names))
            != self.scale_head_parameter_names
            or len(set(self.encoder_parameter_names))
            != len(self.encoder_parameter_names)
            or len(set(self.mean_head_parameter_names))
            != len(self.mean_head_parameter_names)
            or len(set(self.scale_head_parameter_names))
            != len(self.scale_head_parameter_names)
            or set(self.encoder_parameter_names).intersection(
                self.mean_head_parameter_names
            )
            or set(self.encoder_parameter_names).intersection(
                self.scale_head_parameter_names
            )
            or set(self.mean_head_parameter_names).intersection(
                self.scale_head_parameter_names
            )
            or self.encoder_learning_rate != M03R_V15_ENCODER_LEARNING_RATE
            or self.mean_head_learning_rate != M03R_V15_MEAN_HEAD_LEARNING_RATE
            or self.scale_head_learning_rate != M03R_V15_SCALE_HEAD_LEARNING_RATE
            or self.weight_decay != M03R_V15_WEIGHT_DECAY
            or self.encoder_gradient_clip_norm
            != M03R_V15_ENCODER_GRADIENT_CLIP_NORM
            or self.mean_head_gradient_clip_norm
            != M03R_V15_MEAN_HEAD_GRADIENT_CLIP_NORM
            or self.scale_head_gradient_clip_norm
            != M03R_V15_SCALE_HEAD_GRADIENT_CLIP_NORM
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or self.schema != M03R_V15_OPTIMIZER_SCHEMA
            or not all(
                name == "source_policy.core.cash_bias"
                or name.startswith(M03R_V15_ENCODER_PARAMETER_PREFIXES)
                for name in self.encoder_parameter_names
            )
            or not all(
                name.startswith("economic_mean_head.")
                for name in self.mean_head_parameter_names
            )
            or not all(
                name.startswith("economic_scale_head.")
                for name in self.scale_head_parameter_names
            )
        ):
            raise M03RV15OptimizerError("v15 optimizer partition drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _parameter_names(
    policy: Top2000M03RV15PredictivePolicy,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    encoder: list[str] = []
    mean_head: list[str] = []
    scale_head: list[str] = []
    unassigned: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "source_policy.core.cash_bias" or name.startswith(
            M03R_V15_ENCODER_PARAMETER_PREFIXES
        ):
            encoder.append(name)
        elif name.startswith("economic_mean_head."):
            mean_head.append(name)
        elif name.startswith("economic_scale_head."):
            scale_head.append(name)
        else:
            unassigned.append(name)
    if unassigned:
        raise M03RV15OptimizerError(
            f"v15 has unassigned trainable parameters: {tuple(sorted(unassigned))!r}"
        )
    return tuple(sorted(encoder)), tuple(sorted(mean_head)), tuple(sorted(scale_head))


def _uses_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return not (
        name.endswith(".bias")
        or ".norm." in lowered
        or ".layernorm." in lowered
        or ".layer_norm." in lowered
    )


def build_m03r_v15_optimizer(
    policy: Top2000M03RV15PredictivePolicy,
) -> tuple[torch.optim.AdamW, M03RV15OptimizerPartition]:
    if not isinstance(policy, Top2000M03RV15PredictivePolicy):
        raise M03RV15OptimizerError("v15 optimizer requires its exact policy")
    encoder, mean_head, scale_head = _parameter_names(policy)
    partition = M03RV15OptimizerPartition(
        setting_id=policy.v15_setting.setting_id,
        encoder_parameter_names=encoder,
        mean_head_parameter_names=mean_head,
        scale_head_parameter_names=scale_head,
    )
    partition.validate()
    named = dict(policy.named_parameters())
    groups: list[dict[str, object]] = []
    for logical_name, names, rate in (
        ("encoder", encoder, partition.encoder_learning_rate),
        ("mean-head", mean_head, partition.mean_head_learning_rate),
        ("scale-head", scale_head, partition.scale_head_learning_rate),
    ):
        for decay in (True, False):
            selected = tuple(name for name in names if _uses_weight_decay(name) == decay)
            if not selected:
                continue
            groups.append(
                {
                    "params": [named[name] for name in selected],
                    "lr": rate,
                    "base_lr": rate,
                    "weight_decay": partition.weight_decay if decay else 0.0,
                    "group_name": f"{logical_name}-{'decay' if decay else 'no-decay'}",
                    "parameter_names": selected,
                }
            )
    optimizer = torch.optim.AdamW(groups)
    return optimizer, partition


def validate_m03r_v15_optimizer(
    policy: Top2000M03RV15PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV15OptimizerPartition,
) -> None:
    partition.validate()
    names = _parameter_names(policy)
    expected_names = (
        partition.encoder_parameter_names,
        partition.mean_head_parameter_names,
        partition.scale_head_parameter_names,
    )
    if (
        partition.setting_id != policy.v15_setting.setting_id
        or names != expected_names
    ):
        raise M03RV15OptimizerError("v15 optimizer inventory drifted")
    named = dict(policy.named_parameters())
    expected: list[tuple[str, float, float, tuple[str, ...]]] = []
    for logical_name, names, rate in (
        ("encoder", expected_names[0], partition.encoder_learning_rate),
        ("mean-head", expected_names[1], partition.mean_head_learning_rate),
        ("scale-head", expected_names[2], partition.scale_head_learning_rate),
    ):
        for decay in (True, False):
            selected = tuple(name for name in names if _uses_weight_decay(name) == decay)
            if selected:
                expected.append(
                    (
                        f"{logical_name}-{'decay' if decay else 'no-decay'}",
                        rate,
                        partition.weight_decay if decay else 0.0,
                        selected,
                    )
                )
    if len(optimizer.param_groups) != len(expected):
        raise M03RV15OptimizerError("v15 optimizer group count drifted")
    for group, (group_name, rate, decay, parameter_names) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        if (
            group.get("group_name") != group_name
            or float(group.get("base_lr", -1.0)) != rate
            or not 0.0 < float(group["lr"]) <= rate
            or float(group["weight_decay"]) != decay
            or tuple(group.get("parameter_names", ())) != parameter_names
            or tuple(id(value) for value in group["params"])
            != tuple(id(named[name]) for name in parameter_names)
        ):
            raise M03RV15OptimizerError("v15 optimizer group drifted")


__all__ = [
    "M03R_V15_ENCODER_LEARNING_RATE",
    "M03R_V15_MEAN_HEAD_LEARNING_RATE",
    "M03R_V15_SCALE_HEAD_LEARNING_RATE",
    "M03R_V15_OPTIMIZER_SCHEMA",
    "M03R_V15_WEIGHT_DECAY",
    "M03RV15OptimizerError",
    "M03RV15OptimizerPartition",
    "build_m03r_v15_optimizer",
    "validate_m03r_v15_optimizer",
]
