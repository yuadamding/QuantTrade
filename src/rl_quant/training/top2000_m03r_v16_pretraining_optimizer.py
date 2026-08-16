"""Selection-only optimizer with module-aware decay for M03R-v16."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_OPTIMIZER_RULE,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTING_IDS,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256 as _sha256
from rl_quant.training.top2000_m03r_v15_policy import (
    M03R_V15_ENCODER_PARAMETER_PREFIXES,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)

M03R_V16_OPTIMIZER_SCHEMA = "rl-quant.top2000-dev.m03r-v16-optimizer-v3"
M03RV16TrainingStage = Literal["score"]


class M03RV16OptimizerError(ValueError):
    """The V16 selection parameter or optimizer inventory drifted."""


def _is_encoder(name: str) -> bool:
    return name == "source_policy.core.cash_bias" or name.startswith(
        M03R_V15_ENCODER_PARAMETER_PREFIXES
    )


def _is_selection_head(name: str) -> bool:
    return name.startswith("selection_score_head.")


def _no_decay_parameter_ids(
    policy: Top2000M03RV16PredictivePolicy,
) -> set[int]:
    result: set[int] = set()
    for module in policy.modules():
        if isinstance(module, nn.LayerNorm):
            result.update(
                id(parameter) for parameter in module.parameters(recurse=False)
            )
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter):
            result.add(id(bias))
    result.add(id(policy.source_policy.core.cash_bias))
    return result


def configure_m03r_v16_training_stage(
    policy: Top2000M03RV16PredictivePolicy,
    stage: M03RV16TrainingStage,
) -> None:
    """Expose only the encoder and one selection head to mutation."""

    if not isinstance(policy, Top2000M03RV16PredictivePolicy) or stage != "score":
        raise M03RV16OptimizerError("V16 supports selection score training only")
    unknown: list[str] = []
    for name, parameter in policy.named_parameters():
        known = _is_encoder(name) or _is_selection_head(name)
        if not known and parameter.requires_grad:
            unknown.append(name)
        parameter.requires_grad_(known)
    if unknown:
        raise M03RV16OptimizerError(
            f"V16 has unassigned trainable parameters: {tuple(sorted(unknown))!r}"
        )


@dataclass(frozen=True, slots=True)
class M03RV16OptimizerPartition:
    setting_id: str
    stage: M03RV16TrainingStage
    encoder_parameter_names: tuple[str, ...]
    selection_head_parameter_names: tuple[str, ...]
    optimizer_rule: str = M03R_V16_OPTIMIZER_RULE
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_OPTIMIZER_SCHEMA

    @property
    def mean_parameter_names(self) -> tuple[str, ...]:
        return self.selection_head_parameter_names

    def validate(self) -> None:
        groups = (self.encoder_parameter_names, self.selection_head_parameter_names)
        if (
            self.setting_id not in M03R_V16_SETTING_IDS
            or self.stage != "score"
            or any(tuple(sorted(group)) != group or not group for group in groups)
            or set(groups[0]).intersection(groups[1])
            or not all(_is_encoder(name) for name in groups[0])
            or not all(_is_selection_head(name) for name in groups[1])
            or self.optimizer_rule != M03R_V16_OPTIMIZER_RULE
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_OPTIMIZER_SCHEMA
        ):
            raise M03RV16OptimizerError("V16 optimizer partition drifted")

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
    selection: list[str] = []
    unknown: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_encoder(name):
            encoder.append(name)
        elif _is_selection_head(name):
            selection.append(name)
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
        selection_head_parameter_names=tuple(sorted(selection)),
    )
    result.validate()
    return result


def _expected_groups(
    policy: Top2000M03RV16PredictivePolicy,
    partition: M03RV16OptimizerPartition,
) -> list[tuple[str, float, float, tuple[str, ...]]]:
    named = dict(policy.named_parameters())
    no_decay_ids = _no_decay_parameter_ids(policy)
    spec = M03R_V16_PREDICTIVE_SPEC
    logical = (
        ("encoder", partition.encoder_parameter_names, spec.score_learning_rates[0]),
        (
            "selection-head",
            partition.selection_head_parameter_names,
            spec.score_learning_rates[1],
        ),
    )
    expected: list[tuple[str, float, float, tuple[str, ...]]] = []
    for logical_name, names, rate in logical:
        for decay in (True, False):
            selected = tuple(
                name for name in names if (id(named[name]) not in no_decay_ids) == decay
            )
            if selected:
                expected.append(
                    (
                        f"{logical_name}-{'decay' if decay else 'no-decay'}",
                        rate,
                        spec.weight_decay if decay else 0.0,
                        selected,
                    )
                )
    return expected


def build_m03r_v16_optimizer(
    policy: Top2000M03RV16PredictivePolicy,
    stage: M03RV16TrainingStage = "score",
) -> tuple[torch.optim.AdamW, M03RV16OptimizerPartition]:
    partition = _partition(policy, stage)
    named = dict(policy.named_parameters())
    groups: list[dict[str, object]] = []
    for group_name, rate, weight_decay, names in _expected_groups(policy, partition):
        groups.append(
            {
                "params": [named[name] for name in names],
                "lr": rate,
                "base_lr": rate,
                "weight_decay": weight_decay,
                "group_name": group_name,
                "parameter_names": names,
            }
        )
    return torch.optim.AdamW(groups), partition


def validate_m03r_v16_optimizer(
    policy: Top2000M03RV16PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
) -> None:
    partition.validate()
    if _partition(policy, "score") != partition:
        raise M03RV16OptimizerError("V16 optimizer parameter inventory drifted")
    named = dict(policy.named_parameters())
    expected = _expected_groups(policy, partition)
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
