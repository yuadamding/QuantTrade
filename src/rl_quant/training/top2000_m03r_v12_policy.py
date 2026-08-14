"""Explicit selected-horizon predictive model for M03R-v12."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_ELIGIBLE_EXECUTION_HORIZONS,
    M03R_V12_HORIZONS,
    M03R_V12_PROTOCOL_SHA256,
    M03RV12PredictiveSetting,
    resolve_m03r_v12_setting,
)
from rl_quant.training.top2000_m03r_v7_dev import Top2000M03RV7DevelopmentPolicy

M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256 = hashlib.sha256(
    b"m03r-v12-five-horizon-explicit-selection-separate-rank-mean-scale-v2"
).hexdigest()
M03R_V12_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v12-predictive-policy-v2"


class M03RV12PolicyError(ValueError):
    """The v12 policy or its selected-horizon output drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12AlphaDistribution:
    """Five supervised horizons with one constructor-bound economic horizon."""

    mean_by_horizon: torch.Tensor
    log_scale_by_horizon: torch.Tensor
    selected_horizon_sessions: int
    selected_mean: torch.Tensor
    selected_scale: torch.Tensor
    contract_sha256: str = M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256

    def validate(self) -> None:
        mean = self.mean_by_horizon
        log_scale = self.log_scale_by_horizon
        if (
            not isinstance(mean, torch.Tensor)
            or mean.ndim != 3
            or mean.shape[-1] != len(M03R_V12_HORIZONS)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all())
            or not isinstance(log_scale, torch.Tensor)
            or tuple(log_scale.shape) != tuple(mean.shape)
            or log_scale.dtype != mean.dtype
            or log_scale.device != mean.device
            or not bool(torch.isfinite(log_scale).all())
            or self.selected_horizon_sessions
            not in M03R_V12_ELIGIBLE_EXECUTION_HORIZONS
            or not isinstance(self.selected_mean, torch.Tensor)
            or tuple(self.selected_mean.shape) != tuple(mean.shape[:2])
            or self.selected_mean.dtype != mean.dtype
            or self.selected_mean.device != mean.device
            or not isinstance(self.selected_scale, torch.Tensor)
            or tuple(self.selected_scale.shape) != tuple(mean.shape[:2])
            or self.selected_scale.dtype != mean.dtype
            or self.selected_scale.device != mean.device
            or not bool(torch.isfinite(self.selected_scale).all())
            or bool((self.selected_scale <= 0.0).any())
            or self.contract_sha256 != M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256
        ):
            raise M03RV12PolicyError("v12 alpha distribution is invalid")
        index = M03R_V12_HORIZONS.index(self.selected_horizon_sessions)
        if not torch.equal(self.selected_mean, mean[..., index]):
            raise M03RV12PolicyError("v12 selected mean is not its bound head")
        if not torch.equal(self.selected_scale, torch.exp(log_scale[..., index])):
            raise M03RV12PolicyError("v12 selected scale is not its bound head")


@dataclass(frozen=True, slots=True)
class M03RV12PredictiveOutput:
    economic_distribution: M03RV12AlphaDistribution
    rank_score_by_horizon: torch.Tensor
    contract_sha256: str = M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256

    def validate(self) -> None:
        self.economic_distribution.validate()
        mean = self.economic_distribution.mean_by_horizon
        rank = self.rank_score_by_horizon
        if (
            not isinstance(rank, torch.Tensor)
            or rank.shape != mean.shape
            or rank.dtype != mean.dtype
            or rank.device != mean.device
            or not bool(torch.isfinite(rank).all())
            or self.contract_sha256 != M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256
            or rank.data_ptr() == mean.data_ptr()
        ):
            raise M03RV12PolicyError("v12 separated predictive output drifted")


@dataclass(frozen=True, slots=True)
class M03RV12HeadIdentity:
    setting_id: str
    selected_alpha_horizon: int
    economic_mean_head_state_sha256: str
    economic_scale_head_state_sha256: str
    rank_score_head_state_sha256: str
    output_contract_sha256: str = M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256

    def validate(self) -> None:
        for value in (
            self.economic_mean_head_state_sha256,
            self.economic_scale_head_state_sha256,
            self.rank_score_head_state_sha256,
            self.output_contract_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise M03RV12PolicyError("v12 head identity contains a bad digest")
        if (
            self.setting_id
            not in {resolve_m03r_v12_setting(index).setting_id for index in range(3)}
            or self.selected_alpha_horizon not in M03R_V12_ELIGIBLE_EXECUTION_HORIZONS
            or self.output_contract_sha256 != M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256
        ):
            raise M03RV12PolicyError("v12 head identity drifted")


class Top2000M03RV12PredictivePolicy(nn.Module):
    """Reviewed raw encoder plus explicit five-horizon predictive heads."""

    protocol_sha256 = M03R_V12_PROTOCOL_SHA256
    schema = M03R_V12_POLICY_SCHEMA

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
        self.v12_setting: M03RV12PredictiveSetting = resolve_m03r_v12_setting(setting)
        if (
            isinstance(selected_horizon_sessions, bool)
            or selected_horizon_sessions not in M03R_V12_ELIGIBLE_EXECUTION_HORIZONS
        ):
            raise M03RV12PolicyError("v12 model selected horizon drifted")
        self.selected_horizon_sessions = selected_horizon_sessions
        self.source_policy = Top2000M03RV7DevelopmentPolicy(
            M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
            token_dim=token_dim,
            raw_stock_chunk=raw_stock_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        source_alpha = self.source_policy.core.alpha_head
        if source_alpha is None:
            raise M03RV12PolicyError("reviewed source policy omitted its alpha head")
        for parameter in source_alpha.auxiliary_head.parameters():
            parameter.requires_grad_(False)
        self.economic_mean_head = nn.Linear(token_dim, len(M03R_V12_HORIZONS))
        self.economic_scale_head = nn.Linear(token_dim, len(M03R_V12_HORIZONS))
        self.rank_score_head = nn.Linear(token_dim, len(M03R_V12_HORIZONS))
        nn.init.xavier_uniform_(self.economic_mean_head.weight, gain=0.25)
        nn.init.zeros_(self.economic_mean_head.bias)
        nn.init.zeros_(self.economic_scale_head.weight)
        with torch.no_grad():
            self.economic_scale_head.bias.copy_(
                torch.tensor(
                    [
                        math.log(0.02 * math.sqrt(horizon))
                        for horizon in M03R_V12_HORIZONS
                    ],
                    dtype=self.economic_scale_head.bias.dtype,
                )
            )
        nn.init.xavier_uniform_(self.rank_score_head.weight, gain=0.25)
        nn.init.zeros_(self.rank_score_head.bias)

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def predictive_output(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV12PredictiveOutput:
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
            raise M03RV12PolicyError("v12 state and availability axes are invalid")
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
            mean = self.economic_mean_head(hidden).float()
            log_scale = self.economic_scale_head(hidden).float().clamp(-8.0, 2.0)
            rank = self.rank_score_head(hidden).float()
        risky = available.clone()
        risky[:, 0] = False
        risky_horizon = risky.unsqueeze(-1)
        mean = torch.where(risky_horizon, mean, torch.zeros_like(mean))
        log_scale = torch.where(risky_horizon, log_scale, torch.zeros_like(log_scale))
        rank = torch.where(risky_horizon, rank, torch.zeros_like(rank))
        horizon_index = M03R_V12_HORIZONS.index(self.selected_horizon_sessions)
        distribution = M03RV12AlphaDistribution(
            mean_by_horizon=mean,
            log_scale_by_horizon=log_scale,
            selected_horizon_sessions=self.selected_horizon_sessions,
            selected_mean=mean[..., horizon_index],
            selected_scale=torch.exp(log_scale[..., horizon_index]),
        )
        result = M03RV12PredictiveOutput(distribution, rank)
        result.validate()
        return result

    def v12_head_identity(self) -> M03RV12HeadIdentity:
        result = M03RV12HeadIdentity(
            setting_id=self.v12_setting.setting_id,
            selected_alpha_horizon=self.selected_horizon_sessions,
            economic_mean_head_state_sha256=_module_state_sha256(
                self.economic_mean_head
            ),
            economic_scale_head_state_sha256=_module_state_sha256(
                self.economic_scale_head
            ),
            rank_score_head_state_sha256=_module_state_sha256(self.rank_score_head),
        )
        result.validate()
        return result


__all__ = [
    "M03R_V12_ALPHA_OUTPUT_CONTRACT_SHA256",
    "M03R_V12_POLICY_SCHEMA",
    "M03RV12AlphaDistribution",
    "M03RV12HeadIdentity",
    "M03RV12PolicyError",
    "M03RV12PredictiveOutput",
    "Top2000M03RV12PredictivePolicy",
]
