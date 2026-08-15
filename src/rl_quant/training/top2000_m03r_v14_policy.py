"""Single direct-to-execution three-session policy output for M03R-v14."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SELECTED_HORIZON_SESSIONS,
    M03RV14PredictiveSetting,
    resolve_m03r_v14_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.training.top2000_m03r_v7_dev import Top2000M03RV7DevelopmentPolicy

M03R_V14_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v14-predictive-policy-v1"
M03R_V14_OUTPUT_SCHEMA = "rl-quant.top2000-dev.m03r-v14-direct-h3-output-v1"
M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256 = hashlib.sha256(
    (
        "one-three-session-economic-mean-is-rank-and-execution-score;"
        "one-three-session-log-scale;no-separate-rank-head;cash-zero;"
        "mean-head-xavier-gain-0.025;scale-head-does-not-update-encoder-v1"
    ).encode("ascii")
).hexdigest()
M03R_V14_ENCODER_PARAMETER_PREFIXES = (
    "source_policy.core.raw_encoder.",
    "source_policy.core.news_agg.",
    "source_policy.core.token_proj.",
    "source_policy.core.temporal.",
    "source_policy.core.alloc_in.",
    "source_policy.core.attn.",
)


class M03RV14PolicyError(ValueError):
    """The v14 direct predictive output or identity drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    return hashlib.sha256(
        json.dumps(
            tuple(
                (name, _tensor_sha256(value))
                for name, value in sorted(module.state_dict().items())
            ),
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV14AlphaDistribution:
    economic_mean: torch.Tensor
    economic_log_scale: torch.Tensor
    selected_horizon_sessions: int = M03R_V14_SELECTED_HORIZON_SESSIONS
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_OUTPUT_SCHEMA

    def validate(self) -> None:
        if (
            not isinstance(self.economic_mean, torch.Tensor)
            or self.economic_mean.ndim != 2
            or not self.economic_mean.is_floating_point()
            or not bool(torch.isfinite(self.economic_mean).all())
            or not isinstance(self.economic_log_scale, torch.Tensor)
            or tuple(self.economic_log_scale.shape) != tuple(self.economic_mean.shape)
            or self.economic_log_scale.dtype != self.economic_mean.dtype
            or self.economic_log_scale.device != self.economic_mean.device
            or not bool(torch.isfinite(self.economic_log_scale).all())
            or self.selected_horizon_sessions
            != M03R_V14_SELECTED_HORIZON_SESSIONS
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_OUTPUT_SCHEMA
        ):
            raise M03RV14PolicyError("v14 direct alpha distribution drifted")

    @property
    def rank_score(self) -> torch.Tensor:
        """Return the exact score optimized by rank loss and used by execution."""

        self.validate()
        return self.economic_mean

    @property
    def economic_scale(self) -> torch.Tensor:
        self.validate()
        return torch.exp(self.economic_log_scale)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": self.schema,
                    "protocol_sha256": self.protocol_sha256,
                    "selected_horizon_sessions": self.selected_horizon_sessions,
                    "economic_mean_sha256": _tensor_sha256(self.economic_mean),
                    "economic_log_scale_sha256": _tensor_sha256(
                        self.economic_log_scale
                    ),
                    "rank_score_aliases_economic_mean": (
                        self.rank_score.data_ptr() == self.economic_mean.data_ptr()
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV14HeadIdentity:
    setting_id: str
    selected_alpha_horizon: int
    economic_mean_head_state_sha256: str
    economic_scale_head_state_sha256: str
    output_contract_sha256: str = M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256

    def validate(self) -> None:
        resolve_m03r_v14_setting(self.setting_id)
        for value in (
            self.economic_mean_head_state_sha256,
            self.economic_scale_head_state_sha256,
            self.output_contract_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise M03RV14PolicyError("v14 head identity has a malformed digest")
        if (
            self.selected_alpha_horizon != M03R_V14_SELECTED_HORIZON_SESSIONS
            or self.output_contract_sha256
            != M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256
        ):
            raise M03RV14PolicyError("v14 head identity drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(asdict(self), separators=(",", ":"), sort_keys=True).encode(
                "ascii"
            )
        ).hexdigest()


class Top2000M03RV14PredictivePolicy(nn.Module):
    """Reviewed daily encoder with one h3 mean used for rank and execution."""

    protocol_sha256 = M03R_V14_PROTOCOL_SHA256
    schema = M03R_V14_POLICY_SCHEMA

    def __init__(
        self,
        setting: int | str,
        *,
        selected_horizon_sessions: int,
        token_dim: int = 128,
        raw_stock_chunk: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.v14_setting: M03RV14PredictiveSetting = resolve_m03r_v14_setting(setting)
        if (
            isinstance(selected_horizon_sessions, bool)
            or selected_horizon_sessions != M03R_V14_SELECTED_HORIZON_SESSIONS
        ):
            raise M03RV14PolicyError("v14 model selected horizon drifted")
        self.selected_horizon_sessions = selected_horizon_sessions
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
                M03R_V14_ENCODER_PARAMETER_PREFIXES
            ):
                parameter.requires_grad_(True)
        source_alpha = self.source_policy.core.alpha_head
        if source_alpha is None:
            raise M03RV14PolicyError("reviewed source policy omitted its alpha head")
        for parameter in source_alpha.auxiliary_head.parameters():
            parameter.requires_grad_(False)
        self.economic_mean_head = nn.Linear(token_dim, 1)
        self.economic_scale_head = nn.Linear(token_dim, 1)
        nn.init.xavier_uniform_(self.economic_mean_head.weight, gain=0.025)
        nn.init.zeros_(self.economic_mean_head.bias)
        nn.init.zeros_(self.economic_scale_head.weight)
        nn.init.constant_(
            self.economic_scale_head.bias,
            math.log(0.02 * math.sqrt(M03R_V14_SELECTED_HORIZON_SESSIONS)),
        )

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def predictive_output(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV14AlphaDistribution:
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
            raise M03RV14PolicyError("v14 state and availability axes are invalid")
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
            mean = self.economic_mean_head(hidden).squeeze(-1).float()
            log_scale = (
                self.economic_scale_head(hidden.detach())
                .squeeze(-1)
                .float()
                .clamp(-8.0, 2.0)
            )
        risky = available.clone()
        risky[:, 0] = False
        mean = torch.where(risky, mean, torch.zeros_like(mean))
        log_scale = torch.where(risky, log_scale, torch.zeros_like(log_scale))
        result = M03RV14AlphaDistribution(mean, log_scale)
        result.validate()
        return result

    def v14_head_identity(self) -> M03RV14HeadIdentity:
        result = M03RV14HeadIdentity(
            setting_id=self.v14_setting.setting_id,
            selected_alpha_horizon=self.selected_horizon_sessions,
            economic_mean_head_state_sha256=_module_state_sha256(
                self.economic_mean_head
            ),
            economic_scale_head_state_sha256=_module_state_sha256(
                self.economic_scale_head
            ),
        )
        result.validate()
        return result


__all__ = [
    "M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256",
    "M03R_V14_ENCODER_PARAMETER_PREFIXES",
    "M03R_V14_OUTPUT_SCHEMA",
    "M03R_V14_POLICY_SCHEMA",
    "M03RV14AlphaDistribution",
    "M03RV14HeadIdentity",
    "M03RV14PolicyError",
    "Top2000M03RV14PredictivePolicy",
]
