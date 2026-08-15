"""Stage-separated score and uncertainty optimizers for M03R-v16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_OPTIMIZER_RULE,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v15_policy import (
    M03R_V15_ENCODER_PARAMETER_PREFIXES,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)

M03R_V16_OPTIMIZER_SCHEMA = "rl-quant.top2000-dev.m03r-v16-optimizer-v1"
M03RV16TrainingStage = Literal["score", "scale_calibration"]


class M03RV16OptimizerError(ValueError):
    """The V16 stage-specific parameter or optimizer inventory drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def _is_encoder(name: str) -> bool:
    return name == "source_policy.core.cash_bias" or name.startswith(
        M03R_V15_ENCODER_PARAMETER_PREFIXES
    )


def _is_mean(name: str) -> bool:
    return name.startswith(("selection_mean_head.", "timing_mean_head."))


def _is_scale(name: str) -> bool:
    return name.startswith(("selection_scale_head.", "timing_scale_head."))


def _uses_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return not (
        name.endswith(".bias")
        or ".norm." in lowered
        or ".layernorm." in lowered
        or ".layer_norm." in lowered
    )


def configure_m03r_v16_training_stage(
    policy: Top2000M03RV16PredictivePolicy,
    stage: M03RV16TrainingStage,
) -> None:
    """Make the mutation boundary explicit before constructing an optimizer."""

    if not isinstance(policy, Top2000M03RV16PredictivePolicy) or stage not in {
        "score",
        "scale_calibration",
    }:
        raise M03RV16OptimizerError("V16 training stage is invalid")
    unknown: list[str] = []
    for name, parameter in policy.named_parameters():
        known = _is_encoder(name) or _is_mean(name) or _is_scale(name)
        if not known and parameter.requires_grad:
            unknown.append(name)
        should_train = (
            (_is_encoder(name) or _is_mean(name))
            if stage == "score"
            else _is_scale(name)
        )
        parameter.requires_grad_(should_train)
    if unknown:
        raise M03RV16OptimizerError(
            f"V16 has unassigned trainable parameters: {tuple(sorted(unknown))!r}"
        )


@dataclass(frozen=True, slots=True)
class M03RV16OptimizerPartition:
    setting_id: str
    stage: M03RV16TrainingStage
    encoder_parameter_names: tuple[str, ...]
    mean_parameter_names: tuple[str, ...]
    scale_parameter_names: tuple[str, ...]
    optimizer_rule: str = M03R_V16_OPTIMIZER_RULE
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_OPTIMIZER_SCHEMA

    def validate(self) -> None:
        groups = (
            self.encoder_parameter_names,
            self.mean_parameter_names,
            self.scale_parameter_names,
        )
        if (
            self.setting_id not in M03R_V16_SETTING_IDS
            or self.stage not in {"score", "scale_calibration"}
            or any(tuple(sorted(group)) != group for group in groups)
            or len(set().union(*map(set, groups))) != sum(map(len, groups))
            or (self.stage == "score" and (not groups[0] or not groups[1] or groups[2]))
            or (self.stage == "scale_calibration" and (groups[0] or groups[1] or not groups[2]))
            or not all(_is_encoder(name) for name in groups[0])
            or not all(_is_mean(name) for name in groups[1])
            or not all(_is_scale(name) for name in groups[2])
            or self.optimizer_rule != M03R_V16_OPTIMIZER_RULE
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_OPTIMIZER_SCHEMA
        ):
            raise M03RV16OptimizerError("V16 optimizer partition drifted")

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        self.validate()
        return (
            *self.encoder_parameter_names,
            *self.mean_parameter_names,
            *self.scale_parameter_names,
        )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _partition(
    policy: Top2000M03RV16PredictivePolicy,
    stage: M03RV16TrainingStage,
) -> M03RV16OptimizerPartition:
    configure_m03r_v16_training_stage(policy, stage)
    encoder: list[str] = []
    means: list[str] = []
    scales: list[str] = []
    unknown: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_encoder(name):
            encoder.append(name)
        elif _is_mean(name):
            means.append(name)
        elif _is_scale(name):
            scales.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise M03RV16OptimizerError(
            f"V16 has unassigned parameters: {tuple(sorted(unknown))!r}"
        )
    result = M03RV16OptimizerPartition(
        setting_id=policy.v16_setting.setting_id,
        stage=stage,
        encoder_parameter_names=tuple(sorted(encoder)),
        mean_parameter_names=tuple(sorted(means)),
        scale_parameter_names=tuple(sorted(scales)),
    )
    result.validate()
    return result


def build_m03r_v16_optimizer(
    policy: Top2000M03RV16PredictivePolicy,
    stage: M03RV16TrainingStage,
) -> tuple[torch.optim.AdamW, M03RV16OptimizerPartition]:
    partition = _partition(policy, stage)
    named = dict(policy.named_parameters())
    spec = M03R_V16_PREDICTIVE_SPEC
    logical: tuple[tuple[str, tuple[str, ...], float], ...]
    if stage == "score":
        logical = (
            ("encoder", partition.encoder_parameter_names, spec.score_learning_rates[0]),
            ("mean", partition.mean_parameter_names, spec.score_learning_rates[1]),
        )
    else:
        logical = (
            ("scale", partition.scale_parameter_names, spec.scale_calibration_learning_rate),
        )
    groups: list[dict[str, object]] = []
    for logical_name, names, learning_rate in logical:
        for decay in (True, False):
            selected = tuple(name for name in names if _uses_weight_decay(name) == decay)
            if selected:
                groups.append(
                    {
                        "params": [named[name] for name in selected],
                        "lr": learning_rate,
                        "base_lr": learning_rate,
                        "weight_decay": spec.weight_decay if decay else 0.0,
                        "group_name": (
                            f"{logical_name}-{'decay' if decay else 'no-decay'}"
                        ),
                        "parameter_names": selected,
                    }
                )
    return torch.optim.AdamW(groups), partition


def validate_m03r_v16_optimizer(
    policy: Top2000M03RV16PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
) -> None:
    partition.validate()
    observed = _partition(policy, partition.stage)
    if observed != partition:
        raise M03RV16OptimizerError("V16 optimizer parameter inventory drifted")
    named = dict(policy.named_parameters())
    spec = M03R_V16_PREDICTIVE_SPEC
    logical = (
        (
            ("encoder", partition.encoder_parameter_names, spec.score_learning_rates[0]),
            ("mean", partition.mean_parameter_names, spec.score_learning_rates[1]),
        )
        if partition.stage == "score"
        else (("scale", partition.scale_parameter_names, spec.scale_calibration_learning_rate),)
    )
    expected: list[tuple[str, float, float, tuple[str, ...]]] = []
    for logical_name, names, rate in logical:
        for decay in (True, False):
            selected = tuple(name for name in names if _uses_weight_decay(name) == decay)
            if selected:
                expected.append(
                    (
                        f"{logical_name}-{'decay' if decay else 'no-decay'}",
                        rate,
                        spec.weight_decay if decay else 0.0,
                        selected,
                    )
                )
    if len(optimizer.param_groups) != len(expected):
        raise M03RV16OptimizerError("V16 optimizer group count drifted")
    for group, (name, rate, expected_decay, names) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        if (
            group.get("group_name") != name
            or float(group.get("base_lr", -1.0)) != rate
            or not 0.0 < float(group["lr"]) <= rate
            or float(group["weight_decay"]) != expected_decay
            or tuple(group.get("parameter_names", ())) != names
            or tuple(id(parameter) for parameter in group["params"])
            != tuple(id(named[value]) for value in names)
        ):
            raise M03RV16OptimizerError("V16 optimizer group drifted")


__all__ = [
    "M03R_V16_OPTIMIZER_SCHEMA",
    "M03RV16OptimizerError",
    "M03RV16OptimizerPartition",
    "M03RV16TrainingStage",
    "build_m03r_v16_optimizer",
    "configure_m03r_v16_training_stage",
    "validate_m03r_v16_optimizer",
]
