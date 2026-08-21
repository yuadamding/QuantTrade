"""Dimensionless selection-only output for the corrected M03R-v16 screen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
    M03RV16PredictiveSetting,
    resolve_m03r_v16_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.top2000_m03r_v15_policy import (
    M03R_V15_ENCODER_PARAMETER_PREFIXES,
)
from rl_quant.training.top2000_m03r_v7_dev import Top2000M03RV7DevelopmentPolicy

M03R_V16_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v16-selection-policy-v3"
M03R_V16_RAW_OUTPUT_SCHEMA = "rl-quant.top2000-dev.m03r-v16-raw-selection-z-v3"
M03R_V16_OUTPUT_CONTRACT_SHA256 = hashlib.sha256(
    (
        "raw-dimensionless-selection-z-only;no-timing-scale-rank-or-execution-alias;"
        "action-operator-creates-executable-z;setting-scale-restores-economic-units-v2"
    ).encode("ascii")
).hexdigest()


class M03RV16PolicyError(ValueError):
    """The V16 predictive output or model identity drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def m03r_v16_score_component_state_sha256(
    policy: Top2000M03RV16PredictivePolicy,
) -> str:
    """Hash the complete selection model only at artifact boundaries."""

    if not isinstance(policy, Top2000M03RV16PredictivePolicy):
        raise M03RV16PolicyError("V16 score-state identity requires its policy")
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16HeadIdentity:
    setting_id: str
    selection_target: str
    numerical_target_support_sessions: int
    selection_target_scale: float
    selection_score_head_state_sha256: str
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    output_contract_sha256: str = M03R_V16_OUTPUT_CONTRACT_SHA256
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256

    def validate(self) -> None:
        expected = {
            value.setting_id: (
                value.selection_target,
                value.numerical_target_support_sessions,
                value.selection_target_scale,
            )
            for value in M03R_V16_SETTINGS
        }
        if (
            self.setting_id not in expected
            or (
                self.selection_target,
                self.numerical_target_support_sessions,
                self.selection_target_scale,
            )
            != expected.get(self.setting_id)
            or not all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (
                    self.selection_score_head_state_sha256,
                    self.output_contract_sha256,
                    self.protocol_sha256,
                )
            )
            or self.output_contract_sha256 != M03R_V16_OUTPUT_CONTRACT_SHA256
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
        ):
            raise M03RV16PolicyError("V16 head identity drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(asdict(self), separators=(",", ":"), sort_keys=True).encode(
                "ascii"
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16RawSelectionPrediction:
    raw_selection_score_z: torch.Tensor
    selection_target: str
    numerical_target_support_sessions: int
    selection_target_scale: float
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_RAW_OUTPUT_SCHEMA

    def validate(self) -> None:
        expected = {
            value.selection_target: (
                value.numerical_target_support_sessions,
                value.selection_target_scale,
            )
            for value in M03R_V16_SETTINGS
        }
        score = self.raw_selection_score_z
        if (
            not isinstance(score, torch.Tensor)
            or score.ndim != 2
            or not score.is_floating_point()
            or not bool(torch.isfinite(score).all())
            or self.selection_target not in expected
            or (
                self.numerical_target_support_sessions,
                self.selection_target_scale,
            )
            != expected.get(self.selection_target)
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.schema != M03R_V16_RAW_OUTPUT_SCHEMA
        ):
            raise M03RV16PolicyError("V16 raw selection output drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": self.schema,
                    "protocol_sha256": self.protocol_sha256,
                    "selection_target": self.selection_target,
                    "numerical_target_support_sessions": (
                        self.numerical_target_support_sessions
                    ),
                    "selection_target_scale": self.selection_target_scale,
                    "hold_target_sessions": self.hold_target_sessions,
                    "hold_target_spec_sha256": self.hold_target_spec_sha256,
                    "raw_selection_score_z_sha256": _tensor_sha256(
                        self.raw_selection_score_z
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


class Top2000M03RV16PredictivePolicy(nn.Module):
    """Daily encoder with one dimensionless holding-aligned selection head."""

    protocol_sha256 = M03R_V16_PROTOCOL_SHA256
    schema = M03R_V16_POLICY_SCHEMA

    def __init__(
        self,
        setting: int | str,
        *,
        token_dim: int = 128,
        raw_stock_chunk: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.v16_setting: M03RV16PredictiveSetting = resolve_m03r_v16_setting(setting)
        self.source_policy = Top2000M03RV7DevelopmentPolicy(
            M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
            token_dim=token_dim,
            raw_stock_chunk=raw_stock_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        for parameter in self.source_policy.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name == "source_policy.core.cash_bias" or name.startswith(
                M03R_V15_ENCODER_PARAMETER_PREFIXES
            ):
                parameter.requires_grad_(True)
        source_alpha = self.source_policy.core.alpha_head
        if source_alpha is None:
            raise M03RV16PolicyError("reviewed encoder omitted its alpha head")
        for parameter in source_alpha.auxiliary_head.parameters():
            parameter.requires_grad_(False)

        self.selection_score_head = nn.Linear(token_dim, 1)
        nn.init.xavier_uniform_(
            self.selection_score_head.weight,
            gain=M03R_V16_PREDICTIVE_SPEC.selection_head_initialization_gain,
        )
        nn.init.zeros_(self.selection_score_head.bias)

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def predictive_output(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV16RawSelectionPrediction:
        if (
            not isinstance(state_t, torch.Tensor)
            or state_t.ndim != 3
            or state_t.shape[-1] != self.token_dim
            or not state_t.is_floating_point()
            or not bool(torch.isfinite(state_t).all())
            or not isinstance(available, torch.Tensor)
            or tuple(available.shape) != tuple(state_t.shape[:2])
            or available.dtype != torch.bool
            or available.device != state_t.device
        ):
            raise M03RV16PolicyError("V16 state or availability axes drifted")
        with torch.autocast(
            device_type=state_t.device.type,
            dtype=torch.bfloat16,
            enabled=state_t.device.type == "cuda" or state_t.dtype == torch.bfloat16,
        ):
            hidden, _mask = self.source_policy.core._allocator_hidden(  # type: ignore[no-untyped-call]
                state_t,
                torch.zeros(
                    state_t.shape[:2], device=state_t.device, dtype=state_t.dtype
                ),
                available,
            )
            score_z = self.selection_score_head(hidden).squeeze(-1).float()
        risky = available.clone()
        risky[:, 0] = False
        result = M03RV16RawSelectionPrediction(
            raw_selection_score_z=torch.where(
                risky, score_z, torch.zeros_like(score_z)
            ),
            selection_target=self.v16_setting.selection_target,
            numerical_target_support_sessions=(
                self.v16_setting.numerical_target_support_sessions
            ),
            selection_target_scale=self.v16_setting.selection_target_scale,
        )
        result.validate()
        return result

    def v16_head_identity(self) -> M03RV16HeadIdentity:
        digest = hashlib.sha256()
        for name, value in sorted(self.selection_score_head.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(_tensor_sha256(value).encode("ascii"))
        result = M03RV16HeadIdentity(
            setting_id=self.v16_setting.setting_id,
            selection_target=self.v16_setting.selection_target,
            numerical_target_support_sessions=(
                self.v16_setting.numerical_target_support_sessions
            ),
            selection_target_scale=self.v16_setting.selection_target_scale,
            selection_score_head_state_sha256=digest.hexdigest(),
        )
        result.validate()
        return result


__all__ = [
    "M03R_V16_OUTPUT_CONTRACT_SHA256",
    "M03R_V16_POLICY_SCHEMA",
    "M03R_V16_RAW_OUTPUT_SCHEMA",
    "M03RV16HeadIdentity",
    "M03RV16PolicyError",
    "M03RV16RawSelectionPrediction",
    "Top2000M03RV16PredictivePolicy",
    "m03r_v16_score_component_state_sha256",
]
