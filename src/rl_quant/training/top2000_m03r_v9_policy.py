"""One predictive alpha distribution for M03R-v9 training and execution."""

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
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256,
    M03R_V9_HORIZONS,
    M03R_V9_PROTOCOL_SHA256,
    M03RV9HorizonBinding,
    M03RV9PredictiveSetting,
    resolve_m03r_v9_setting,
)
from rl_quant.training.top2000_m03r_v7_dev import Top2000M03RV7DevelopmentPolicy

M03R_V9_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v9-predictive-policy-v1"


class M03RV9PolicyError(ValueError):
    """The V9 predictive policy or alpha distribution is malformed."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV9AlphaDistribution:
    """Four-horizon mean/scale tensors with one explicitly selected horizon."""

    mean_by_horizon: torch.Tensor  # [batch, asset, 4]
    log_scale_by_horizon: torch.Tensor  # [batch, asset, 4]
    selected_horizon_sessions: int
    selected_mean: torch.Tensor  # [batch, asset]
    selected_scale: torch.Tensor  # [batch, asset]
    contract_sha256: str = M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256

    def validate(self) -> None:
        mean = self.mean_by_horizon
        log_scale = self.log_scale_by_horizon
        if (
            not isinstance(mean, torch.Tensor)
            or mean.ndim != 3
            or mean.shape[-1] != len(M03R_V9_HORIZONS)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all())
            or not isinstance(log_scale, torch.Tensor)
            or tuple(log_scale.shape) != tuple(mean.shape)
            or log_scale.dtype != mean.dtype
            or log_scale.device != mean.device
            or not bool(torch.isfinite(log_scale).all())
            or self.selected_horizon_sessions not in {21, 30}
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
            or self.contract_sha256 != M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256
        ):
            raise M03RV9PolicyError("V9 alpha distribution is invalid")
        horizon_index = M03R_V9_HORIZONS.index(self.selected_horizon_sessions)
        if not torch.equal(self.selected_mean, mean[..., horizon_index]):
            raise M03RV9PolicyError("selected mean is not the bound horizon tensor")
        expected_scale = torch.exp(log_scale[..., horizon_index])
        if not torch.equal(self.selected_scale, expected_scale):
            raise M03RV9PolicyError("selected scale is not the bound horizon tensor")


@dataclass(frozen=True, slots=True)
class M03RV9AlphaHeadIdentity:
    selected_alpha_horizon: int
    alpha_mean_head_state_sha256: str
    alpha_scale_head_state_sha256: str
    alpha_distribution_contract_sha256: str
    horizon_binding_sha256: str

    def validate(self) -> None:
        for value in (
            self.alpha_mean_head_state_sha256,
            self.alpha_scale_head_state_sha256,
            self.alpha_distribution_contract_sha256,
            self.horizon_binding_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise M03RV9PolicyError(
                    "alpha-head identity contains an invalid digest"
                )
        if (
            self.selected_alpha_horizon not in {21, 30}
            or self.alpha_distribution_contract_sha256
            != M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256
        ):
            raise M03RV9PolicyError("alpha-head identity contract drifted")


class Top2000M03RV9PredictivePolicy(nn.Module):
    """V9 adapter over the reviewed raw encoder and four-horizon alpha mean head."""

    protocol_sha256 = M03R_V9_PROTOCOL_SHA256
    schema = M03R_V9_POLICY_SCHEMA

    def __init__(
        self,
        setting: int | str,
        horizon_binding: M03RV9HorizonBinding,
        *,
        token_dim: int = 128,
        raw_stock_chunk: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.setting: M03RV9PredictiveSetting = resolve_m03r_v9_setting(setting)
        if not isinstance(horizon_binding, M03RV9HorizonBinding):
            raise M03RV9PolicyError("V9 policy requires a typed horizon binding")
        horizon_binding.__post_init__()
        self.horizon_binding = horizon_binding
        self.source_policy = Top2000M03RV7DevelopmentPolicy(
            M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
            token_dim=token_dim,
            raw_stock_chunk=raw_stock_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        self.alpha_scale_head = nn.Linear(token_dim, len(M03R_V9_HORIZONS))
        nn.init.zeros_(self.alpha_scale_head.weight)
        with torch.no_grad():
            self.alpha_scale_head.bias.copy_(
                torch.tensor(
                    [
                        math.log(0.02 * math.sqrt(horizon))
                        for horizon in M03R_V9_HORIZONS
                    ],
                    dtype=self.alpha_scale_head.bias.dtype,
                )
            )

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def alpha_distribution(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV9AlphaDistribution:
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
            raise M03RV9PolicyError("V9 state and availability axes are invalid")
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
            alpha_head = self.source_policy.core.alpha_head
            if alpha_head is None:
                raise M03RV9PolicyError("reviewed source policy omitted its alpha head")
            mean = alpha_head.auxiliary_head(hidden).float()
            log_scale = self.alpha_scale_head(hidden).float().clamp(-8.0, 2.0)
        risky = available.clone()
        risky[:, 0] = False
        mean = torch.where(risky.unsqueeze(-1), mean, torch.zeros_like(mean))
        log_scale = torch.where(
            risky.unsqueeze(-1), log_scale, torch.zeros_like(log_scale)
        )
        index = M03R_V9_HORIZONS.index(self.horizon_binding.economic_execution_horizon)
        result = M03RV9AlphaDistribution(
            mean_by_horizon=mean,
            log_scale_by_horizon=log_scale,
            selected_horizon_sessions=self.horizon_binding.economic_execution_horizon,
            selected_mean=mean[..., index],
            selected_scale=torch.exp(log_scale[..., index]),
        )
        result.validate()
        return result

    def alpha_head_identity(self) -> M03RV9AlphaHeadIdentity:
        alpha_head = self.source_policy.core.alpha_head
        if alpha_head is None:
            raise M03RV9PolicyError("reviewed source policy omitted its alpha head")
        identity = M03RV9AlphaHeadIdentity(
            selected_alpha_horizon=self.horizon_binding.economic_execution_horizon,
            alpha_mean_head_state_sha256=_module_state_sha256(
                alpha_head.auxiliary_head
            ),
            alpha_scale_head_state_sha256=_module_state_sha256(self.alpha_scale_head),
            alpha_distribution_contract_sha256=(
                M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256
            ),
            horizon_binding_sha256=self.horizon_binding.receipt_sha256,
        )
        identity.validate()
        return identity


__all__ = [
    "M03R_V9_POLICY_SCHEMA",
    "M03RV9AlphaDistribution",
    "M03RV9AlphaHeadIdentity",
    "M03RV9PolicyError",
    "Top2000M03RV9PredictivePolicy",
]
