"""Separate long-horizon selection and h3 timing outputs for M03R-v16."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
    M03R_V16_TIMING_HORIZON_SESSIONS,
    M03RV16PredictiveSetting,
    resolve_m03r_v16_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.training.top2000_m03r_v15_policy import (
    M03R_V15_ENCODER_PARAMETER_PREFIXES,
)
from rl_quant.training.top2000_m03r_v7_dev import Top2000M03RV7DevelopmentPolicy

M03R_V16_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v16-predictive-policy-v1"
M03R_V16_RAW_OUTPUT_SCHEMA = "rl-quant.top2000-dev.m03r-v16-raw-alpha-output-v1"
M03R_V16_OUTPUT_CONTRACT_SHA256 = hashlib.sha256(
    (
        "raw-selection-and-h3-timing-mean-scale;no-rank-or-execution-alias;"
        "action-operator-creates-executable-scores;scale-hidden-detached;"
        "common-setting-neutral-head-initialization-v1"
    ).encode("ascii")
).hexdigest()


class M03RV16PolicyError(ValueError):
    """The V16 predictive output or model identity drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16HeadIdentity:
    setting_id: str
    selection_target: str
    selection_support_sessions: int
    selection_mean_head_state_sha256: str
    selection_scale_head_state_sha256: str
    timing_mean_head_state_sha256: str
    timing_scale_head_state_sha256: str
    output_contract_sha256: str = M03R_V16_OUTPUT_CONTRACT_SHA256
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256

    def validate(self) -> None:
        expected = {
            value.setting_id: (
                value.selection_target,
                value.selection_support_sessions,
            )
            for value in M03R_V16_SETTINGS
        }
        if (
            self.setting_id not in expected
            or (
                self.selection_target,
                self.selection_support_sessions,
            )
            != expected.get(self.setting_id)
            or not all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (
                    self.selection_mean_head_state_sha256,
                    self.selection_scale_head_state_sha256,
                    self.timing_mean_head_state_sha256,
                    self.timing_scale_head_state_sha256,
                    self.output_contract_sha256,
                    self.protocol_sha256,
                )
            )
            or self.output_contract_sha256 != M03R_V16_OUTPUT_CONTRACT_SHA256
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
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
    """Hash encoder and mean bytes while excluding calibrating scale heads."""

    if not isinstance(policy, Top2000M03RV16PredictivePolicy):
        raise M03RV16PolicyError("V16 score-state identity requires its policy")
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if name.startswith(("selection_scale_head.", "timing_scale_head.")):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16RawAlphaPrediction:
    raw_selection_mean: torch.Tensor
    raw_selection_log_scale: torch.Tensor
    raw_timing_mean: torch.Tensor
    raw_timing_log_scale: torch.Tensor
    selection_target: str
    selection_support_sessions: int
    timing_horizon_sessions: int = M03R_V16_TIMING_HORIZON_SESSIONS
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_RAW_OUTPUT_SCHEMA

    def validate(self) -> None:
        expected_support = {
            "h21-cumulative-factor-residual": 21,
            "h30-cumulative-factor-residual": 30,
            "survival-weighted-1-30-mean-factor-residual": 30,
        }
        tensors = (
            self.raw_selection_mean,
            self.raw_selection_log_scale,
            self.raw_timing_mean,
            self.raw_timing_log_scale,
        )
        if (
            not isinstance(self.raw_selection_mean, torch.Tensor)
            or self.raw_selection_mean.ndim != 2
            or not self.raw_selection_mean.is_floating_point()
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(self.raw_selection_mean.shape)
                or value.dtype != self.raw_selection_mean.dtype
                or value.device != self.raw_selection_mean.device
                or not bool(torch.isfinite(value).all())
                for value in tensors
            )
            or self.selection_target not in expected_support
            or self.selection_support_sessions
            != expected_support.get(self.selection_target)
            or self.timing_horizon_sessions != M03R_V16_TIMING_HORIZON_SESSIONS
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_RAW_OUTPUT_SCHEMA
        ):
            raise M03RV16PolicyError("V16 raw alpha output drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": self.schema,
                    "protocol_sha256": self.protocol_sha256,
                    "selection_target": self.selection_target,
                    "selection_support_sessions": self.selection_support_sessions,
                    "timing_horizon_sessions": self.timing_horizon_sessions,
                    "raw_selection_mean_sha256": _tensor_sha256(
                        self.raw_selection_mean
                    ),
                    "raw_selection_log_scale_sha256": _tensor_sha256(
                        self.raw_selection_log_scale
                    ),
                    "raw_timing_mean_sha256": _tensor_sha256(self.raw_timing_mean),
                    "raw_timing_log_scale_sha256": _tensor_sha256(
                        self.raw_timing_log_scale
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


class Top2000M03RV16PredictivePolicy(nn.Module):
    """Daily control model with distinct selection and timing distributions."""

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
        self.v16_setting: M03RV16PredictiveSetting = resolve_m03r_v16_setting(
            setting
        )
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

        self.selection_mean_head = nn.Linear(token_dim, 1)
        self.selection_scale_head = nn.Linear(token_dim, 1)
        self.timing_mean_head = nn.Linear(token_dim, 1)
        self.timing_scale_head = nn.Linear(token_dim, 1)
        for head in (self.selection_mean_head, self.timing_mean_head):
            nn.init.xavier_uniform_(head.weight, gain=0.025)
            nn.init.zeros_(head.bias)
        for head in (self.selection_scale_head, self.timing_scale_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(0.02))

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def predictive_output(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV16RawAlphaPrediction:
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
            selection_mean = self.selection_mean_head(hidden).squeeze(-1).float()
            timing_mean = self.timing_mean_head(hidden).squeeze(-1).float()
            detached = hidden.detach()
            selection_log_scale = (
                self.selection_scale_head(detached).squeeze(-1).float().clamp(-8.0, 2.0)
            )
            timing_log_scale = (
                self.timing_scale_head(detached).squeeze(-1).float().clamp(-8.0, 2.0)
            )
        risky = available.clone()
        risky[:, 0] = False
        zero = torch.zeros_like(selection_mean)
        result = M03RV16RawAlphaPrediction(
            raw_selection_mean=torch.where(risky, selection_mean, zero),
            raw_selection_log_scale=torch.where(risky, selection_log_scale, zero),
            raw_timing_mean=torch.where(risky, timing_mean, zero),
            raw_timing_log_scale=torch.where(risky, timing_log_scale, zero),
            selection_target=self.v16_setting.selection_target,
            selection_support_sessions=(
                self.v16_setting.selection_support_sessions
            ),
        )
        result.validate()
        return result

    def v16_head_identity(self) -> M03RV16HeadIdentity:
        def head_sha256(head: nn.Linear) -> str:
            digest = hashlib.sha256()
            for name, value in sorted(head.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(_tensor_sha256(value).encode("ascii"))
            return digest.hexdigest()

        result = M03RV16HeadIdentity(
            setting_id=self.v16_setting.setting_id,
            selection_target=self.v16_setting.selection_target,
            selection_support_sessions=self.v16_setting.selection_support_sessions,
            selection_mean_head_state_sha256=head_sha256(self.selection_mean_head),
            selection_scale_head_state_sha256=head_sha256(self.selection_scale_head),
            timing_mean_head_state_sha256=head_sha256(self.timing_mean_head),
            timing_scale_head_state_sha256=head_sha256(self.timing_scale_head),
        )
        result.validate()
        return result


__all__ = [
    "M03R_V16_OUTPUT_CONTRACT_SHA256",
    "M03R_V16_POLICY_SCHEMA",
    "M03R_V16_RAW_OUTPUT_SCHEMA",
    "M03RV16PolicyError",
    "M03RV16HeadIdentity",
    "M03RV16RawAlphaPrediction",
    "Top2000M03RV16PredictivePolicy",
    "m03r_v16_score_component_state_sha256",
]
