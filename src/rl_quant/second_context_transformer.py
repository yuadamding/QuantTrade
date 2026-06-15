from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from os import PathLike
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.features.stock_second_context import validate_second_context_payload
from rl_quant.hourly_transformer import _validate_action_return_contract


@dataclass
class SecondContextDataSplit:
    name: str
    decision_timestamps: list[str]
    next_timestamps: list[str]
    action_names: list[str]
    feature_names: dict[str, list[str]]
    market_context: torch.Tensor
    market_context_mask: torch.Tensor
    action_features: torch.Tensor
    action_returns: torch.Tensor
    action_valid_mask: torch.Tensor
    action_cost_bps: torch.Tensor
    portfolio_state: torch.Tensor
    constraint_state: torch.Tensor
    valid_start_indices: torch.Tensor
    valid_index_mask: torch.Tensor
    market_mean: torch.Tensor
    market_std: torch.Tensor
    action_feature_mean: torch.Tensor
    action_feature_std: torch.Tensor
    periods_per_year: float

    def to(self, device: torch.device | str) -> "SecondContextDataSplit":
        return replace(
            self,
            market_context=self.market_context.to(device),
            market_context_mask=self.market_context_mask.to(device),
            action_features=self.action_features.to(device),
            action_returns=self.action_returns.to(device),
            action_valid_mask=self.action_valid_mask.to(device),
            action_cost_bps=self.action_cost_bps.to(device),
            portfolio_state=self.portfolio_state.to(device),
            constraint_state=self.constraint_state.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            valid_index_mask=self.valid_index_mask.to(device),
            market_mean=self.market_mean.to(device),
            market_std=self.market_std.to(device),
            action_feature_mean=self.action_feature_mean.to(device),
            action_feature_std=self.action_feature_std.to(device),
        )

    def state(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.market_context[indices],
            self.market_context_mask[indices],
            self.action_features[indices],
            self.portfolio_state[indices],
            self.constraint_state[indices],
        )


def _load_payload(path: str | bytes | PathLike[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_second_context_payload(payload)
    return payload


def _masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.unsqueeze(-1).to(features.dtype)
    count = valid.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (features * valid).sum(dim=(0, 1)) / count
    variance = (((features - mean) * valid) ** 2).sum(dim=(0, 1)) / count
    return mean, variance.sqrt().clamp_min(1e-6)


def _assert_increasing(values: list[str], *, name: str) -> None:
    for left, right in zip(values, values[1:]):
        if _parse_utc_timestamp(right) <= _parse_utc_timestamp(left):
            raise ValueError(f"{name} must be strictly increasing; got {left!r} before {right!r}.")


def _parse_utc_timestamp(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Timestamp {value!r} is not valid ISO format.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} must include timezone information.")
    return parsed.astimezone(timezone.utc)


def _build_split(
    *,
    name: str,
    payload: dict[str, Any],
    start: str | None = None,
    start_after: str | None = None,
    end: str | None = None,
    end_before: str | None = None,
    market_mean: torch.Tensor | None = None,
    market_std: torch.Tensor | None = None,
    action_feature_mean: torch.Tensor | None = None,
    action_feature_std: torch.Tensor | None = None,
) -> SecondContextDataSplit:
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    _assert_increasing(decisions, name="decision_timestamps")
    decision_dt = [_parse_utc_timestamp(value) for value in decisions]
    start_dt = None if start is None else _parse_utc_timestamp(start)
    start_after_dt = None if start_after is None else _parse_utc_timestamp(start_after)
    end_dt = None if end is None else _parse_utc_timestamp(end)
    end_before_dt = None if end_before is None else _parse_utc_timestamp(end_before)
    selected = [
        index
        for index, timestamp_dt in enumerate(decision_dt)
        if (start_dt is None or timestamp_dt >= start_dt)
        and (start_after_dt is None or timestamp_dt > start_after_dt)
        and (end_dt is None or timestamp_dt <= end_dt)
        and (end_before_dt is None or timestamp_dt < end_before_dt)
    ]
    if not selected:
        raise ValueError(f"No rows selected for second-context split {name!r}.")

    raw_market = payload["market_context"].float()[selected]
    market_mask = payload["market_context_mask"].bool()[selected]
    raw_action_features = payload["action_features"].float()[selected]
    action_returns = payload["action_returns"].float()[selected]
    action_valid_mask = payload["action_valid_mask"].bool()[selected]
    _validate_action_return_contract(action_returns, action_valid_mask)
    action_cost_bps = payload["action_cost_bps"].float()[selected]
    portfolio_state = payload["portfolio_state"].float()[selected]
    constraint_state = payload["constraint_state"].float()[selected]

    if market_mean is None or market_std is None:
        market_mean, market_std = _masked_mean_std(raw_market, market_mask)
    if action_feature_mean is None:
        action_feature_mean = raw_action_features.mean(dim=(0, 1))
    if action_feature_std is None:
        action_feature_std = raw_action_features.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

    market = ((raw_market - market_mean) / market_std).clamp_(-8.0, 8.0)
    market = market.masked_fill(~market_mask.unsqueeze(-1), 0.0)
    action_features = ((raw_action_features - action_feature_mean) / action_feature_std).clamp_(-8.0, 8.0)
    valid_indices = [index for index in range(len(selected)) if bool(action_valid_mask[index].any().item())]
    if not valid_indices:
        raise ValueError(f"No valid action rows remain for split {name!r}.")
    valid_start_indices = torch.tensor(valid_indices, dtype=torch.long)
    valid_index_mask = torch.zeros(len(selected), dtype=torch.bool)
    valid_index_mask[valid_start_indices] = True
    manifest = payload.get("dataset_manifest", {})
    decision_interval = str(manifest.get("decision_interval", "15m"))
    periods_per_day = {"5m": 78.0, "15m": 26.0, "30m": 13.0, "60m": 6.0}.get(decision_interval, 26.0)
    return SecondContextDataSplit(
        name=name,
        decision_timestamps=[decisions[i] for i in selected],
        next_timestamps=[next_timestamps[i] for i in selected],
        action_names=list(payload["action_names"]),
        feature_names=dict(payload["feature_names"]),
        market_context=market,
        market_context_mask=market_mask,
        action_features=action_features,
        action_returns=action_returns,
        action_valid_mask=action_valid_mask,
        action_cost_bps=action_cost_bps,
        portfolio_state=portfolio_state,
        constraint_state=constraint_state,
        valid_start_indices=valid_start_indices,
        valid_index_mask=valid_index_mask,
        market_mean=market_mean,
        market_std=market_std,
        action_feature_mean=action_feature_mean,
        action_feature_std=action_feature_std,
        periods_per_year=252.0 * periods_per_day,
    )


def build_second_context_splits(
    *,
    dataset_path,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> tuple[SecondContextDataSplit, SecondContextDataSplit, SecondContextDataSplit]:
    payload = _load_payload(dataset_path)
    train = _build_split(name="train", payload=payload, start=train_start, end=train_end)
    val = _build_split(
        name="val",
        payload=payload,
        start_after=train_end,
        end=val_end,
        end_before=test_start,
        market_mean=train.market_mean,
        market_std=train.market_std,
        action_feature_mean=train.action_feature_mean,
        action_feature_std=train.action_feature_std,
    )
    test = _build_split(
        name="test",
        payload=payload,
        start=test_start,
        end=test_end,
        market_mean=train.market_mean,
        market_std=train.market_std,
        action_feature_mean=train.action_feature_mean,
        action_feature_std=train.action_feature_std,
    )
    assert_matching_second_context_schema(train, val, test)
    return train, val, test


def assert_matching_second_context_schema(*splits: SecondContextDataSplit) -> None:
    if not splits:
        return
    reference = splits[0]
    for split in splits[1:]:
        if split.action_names != reference.action_names:
            raise ValueError(f"Action names/order differ between {reference.name!r} and {split.name!r}.")
        if split.feature_names != reference.feature_names:
            raise ValueError(f"Feature names/order differ between {reference.name!r} and {split.name!r}.")
        if split.market_context.shape[1:] != reference.market_context.shape[1:]:
            raise ValueError(f"Market context shape differs between {reference.name!r} and {split.name!r}.")
        if split.action_features.shape[1:] != reference.action_features.shape[1:]:
            raise ValueError(f"Action feature shape differs between {reference.name!r} and {split.name!r}.")


class SecondContextTransformerQNetwork(nn.Module):
    """Action-conditioned Q-network for compact second-derived decision datasets."""

    def __init__(
        self,
        *,
        market_feature_dim: int,
        action_feature_dim: int,
        portfolio_state_dim: int,
        constraint_state_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        temporal_layers: int = 2,
        feedforward_dim: int = 384,
        dropout: float = 0.10,
        max_lookback_blocks: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if max_lookback_blocks <= 0:
            raise ValueError("max_lookback_blocks must be positive.")
        self.max_lookback_blocks = int(max_lookback_blocks)
        self.market_proj = nn.Sequential(nn.Linear(market_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.position = nn.Parameter(torch.zeros(max_lookback_blocks, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.portfolio_encoder = nn.Linear(portfolio_state_dim, d_model)
        self.constraint_encoder = nn.Linear(constraint_state_dim, d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.action_encoder = nn.Sequential(nn.Linear(action_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim // 2),
            nn.GELU(),
            nn.Linear(feedforward_dim // 2, 1),
        )

    def forward(
        self,
        market_context: torch.Tensor,
        market_context_mask: torch.Tensor,
        action_features: torch.Tensor,
        portfolio_state: torch.Tensor,
        constraint_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, blocks, _ = market_context.shape
        if blocks > self.max_lookback_blocks:
            raise ValueError("market_context exceeds max_lookback_blocks.")
        x = self.market_proj(market_context)
        x = x + self.position[:blocks][None, :, :]
        valid = market_context_mask.bool()
        padding_mask = ~valid
        empty_rows = ~valid.any(dim=1)
        if bool(empty_rows.any().item()):
            padding_mask = padding_mask.clone()
            padding_mask[empty_rows, 0] = False
        encoded = self.market_encoder(x, src_key_padding_mask=padding_mask)
        last_valid = valid.long().sum(dim=1).clamp_min(1) - 1
        market_token = encoded[torch.arange(batch, device=encoded.device), last_valid]
        state_token = self.state_norm(
            market_token
            + self.portfolio_encoder(portfolio_state.float())
            + self.constraint_encoder(constraint_state.float())
        )
        action_token = self.action_encoder(action_features.float())
        state_expanded = state_token[:, None, :].expand(-1, action_token.shape[1], -1)
        pair = torch.cat([state_expanded, action_token, state_expanded * action_token], dim=-1)
        return self.scorer(pair).squeeze(-1)


def masked_contextual_q_loss(
    q_values: torch.Tensor,
    target_returns: torch.Tensor,
    action_valid_mask: torch.Tensor,
    *,
    action_cost_bps: torch.Tensor | None = None,
    reward_scale: float = 10_000.0,
) -> torch.Tensor:
    targets = target_returns.float()
    if action_cost_bps is not None:
        targets = targets - action_cost_bps.float() / 10_000.0
    targets = targets * float(reward_scale)
    valid = action_valid_mask.bool()
    if not bool(valid.any().item()):
        raise ValueError("At least one valid action is required to compute loss.")
    return F.smooth_l1_loss(q_values[valid], targets[valid])


@torch.no_grad()
def evaluate_second_context_policy(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
) -> dict[str, float | int | None]:
    data = split.to(device)
    model.eval()
    q_values = model(
        data.market_context,
        data.market_context_mask,
        data.action_features,
        data.portfolio_state,
        data.constraint_state,
    )
    masked_q = q_values.masked_fill(~data.action_valid_mask, -1e9)
    actions = masked_q.argmax(dim=1)
    rewards = data.action_returns[torch.arange(data.action_returns.shape[0], device=device), actions]
    costs = data.action_cost_bps[torch.arange(data.action_cost_bps.shape[0], device=device), actions] / 10_000.0
    net_returns = (rewards - costs).detach().cpu()
    finite = torch.isfinite(net_returns)
    net_returns = net_returns[finite]
    total_return = float(torch.prod(1.0 + net_returns).item() - 1.0) if net_returns.numel() else 0.0
    avg = float(net_returns.mean().item()) if net_returns.numel() else 0.0
    std = float(net_returns.std(unbiased=False).item()) if net_returns.numel() > 1 else 0.0
    sharpe = None if std <= 0 else avg / std * (split.periods_per_year ** 0.5)
    cash_share = float((actions.detach().cpu() == 0).float().mean().item()) if actions.numel() else 1.0
    return {
        "rows": int(net_returns.numel()),
        "total_return": total_return,
        "mean_return": avg,
        "sharpe": sharpe,
        "cash_action_share": cash_share,
    }
