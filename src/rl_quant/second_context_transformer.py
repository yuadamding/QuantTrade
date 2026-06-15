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
    market_context_available_timestamps_ms: torch.Tensor
    action_features: torch.Tensor
    action_returns: torch.Tensor
    action_valid_mask: torch.Tensor
    action_cost_bps: torch.Tensor
    entry_execution_timestamps_ms: torch.Tensor
    exit_execution_timestamps_ms: torch.Tensor
    entry_price_source: str
    exit_price_source: str
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
            market_context_available_timestamps_ms=self.market_context_available_timestamps_ms.to(device),
            action_features=self.action_features.to(device),
            action_returns=self.action_returns.to(device),
            action_valid_mask=self.action_valid_mask.to(device),
            action_cost_bps=self.action_cost_bps.to(device),
            entry_execution_timestamps_ms=self.entry_execution_timestamps_ms.to(device),
            exit_execution_timestamps_ms=self.exit_execution_timestamps_ms.to(device),
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
    next_dt = [_parse_utc_timestamp(value) for value in next_timestamps]
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
        and (end_dt is None or next_dt[index] <= end_dt)
    ]
    if not selected:
        raise ValueError(f"No rows selected for second-context split {name!r}.")

    raw_market = payload["market_context"].float()[selected]
    market_mask = payload["market_context_mask"].bool()[selected]
    market_context_available_timestamps_ms = payload["market_context_available_timestamps_ms"].long()[selected]
    raw_action_features = payload["action_features"].float()[selected]
    action_returns = payload["action_returns"].float()[selected]
    action_valid_mask = payload["action_valid_mask"].bool()[selected]
    _validate_action_return_contract(action_returns, action_valid_mask)
    action_cost_bps = payload["action_cost_bps"].float()[selected]
    entry_execution_timestamps_ms = payload["entry_execution_timestamps_ms"].long()[selected]
    exit_execution_timestamps_ms = payload["exit_execution_timestamps_ms"].long()[selected]
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
        market_context_available_timestamps_ms=market_context_available_timestamps_ms,
        action_features=action_features,
        action_returns=action_returns,
        action_valid_mask=action_valid_mask,
        action_cost_bps=action_cost_bps,
        entry_execution_timestamps_ms=entry_execution_timestamps_ms,
        exit_execution_timestamps_ms=exit_execution_timestamps_ms,
        entry_price_source=str(payload.get("entry_price_source", "")),
        exit_price_source=str(payload.get("exit_price_source", "")),
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
        action_count: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if max_lookback_blocks <= 0:
            raise ValueError("max_lookback_blocks must be positive.")
        if action_count is not None and action_count <= 0:
            raise ValueError("action_count must be positive when supplied.")
        self.max_lookback_blocks = int(max_lookback_blocks)
        self.action_count = None if action_count is None else int(action_count)
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
        self.action_id_embedding = None if action_count is None else nn.Embedding(int(action_count), d_model)
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
        action_ids: torch.Tensor | None = None,
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
        if self.action_id_embedding is not None:
            action_count = action_token.shape[1]
            if self.action_count is not None and action_count > self.action_count:
                raise ValueError("action_features include more actions than action_count.")
            if action_ids is None:
                action_ids = torch.arange(action_count, device=action_token.device).expand(batch, -1)
            action_token = action_token + self.action_id_embedding(action_ids.long())
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


def _timestamp_ms_to_iso(timestamp_ms: int) -> str:
    if timestamp_ms < 0:
        return ""
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat()


def _summarize_returns(returns: torch.Tensor, *, periods_per_year: float) -> dict[str, float | int | None]:
    finite = returns[torch.isfinite(returns)]
    total_return = float(torch.prod(1.0 + finite).item() - 1.0) if finite.numel() else 0.0
    avg = float(finite.mean().item()) if finite.numel() else 0.0
    std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
    sharpe = None if std <= 0 else avg / std * (periods_per_year ** 0.5)
    return {
        "rows": int(finite.numel()),
        "total_return": total_return,
        "mean_return": avg,
        "sharpe": sharpe,
    }


def _trade_legs(previous_action: int, action: int) -> float:
    if action == previous_action:
        return 0.0
    return 1.0 if previous_action == 0 or action == 0 else 2.0


def _evaluate_action_path(
    split: SecondContextDataSplit,
    actions: torch.Tensor,
    *,
    q_values: torch.Tensor | None = None,
    cost_bps_override: float | None = None,
    initial_action: int = 0,
    return_decision_logs: bool = False,
) -> dict[str, Any]:
    previous_action = int(initial_action)
    net_returns: list[float] = []
    active_net_returns: list[float] = []
    active_gross_returns: list[float] = []
    decision_logs: list[dict[str, Any]] = []
    switches = 0
    cash_actions = 0
    equity = 1.0
    for row in range(actions.numel()):
        requested_action = int(actions[row].item())
        action = requested_action
        if action < 0 or action >= len(split.action_names) or not bool(split.action_valid_mask[row, action].item()):
            action = 0
        gross = float(split.action_returns[row, action].item())
        if not torch.isfinite(split.action_returns[row, action]).item():
            action = 0
            gross = 0.0
        legs = _trade_legs(previous_action, action)
        traded_notional = legs
        if legs > 0:
            previous_cost_bps = (
                float(split.action_cost_bps[row, previous_action].item())
                if 0 <= previous_action < split.action_cost_bps.shape[1]
                else 0.0
            )
            selected_cost_bps = float(split.action_cost_bps[row, action].item())
            if cost_bps_override is not None:
                cost_bps = float(cost_bps_override)
                cost = traded_notional * cost_bps / 10_000.0
            elif previous_action == 0:
                cost_bps = selected_cost_bps
                cost = selected_cost_bps / 10_000.0
            elif action == 0:
                cost_bps = previous_cost_bps
                cost = previous_cost_bps / 10_000.0
            else:
                total_leg_cost_bps = previous_cost_bps + selected_cost_bps
                cost_bps = total_leg_cost_bps / traded_notional
                cost = total_leg_cost_bps / 10_000.0
            switches += 1
        else:
            cost_bps = 0.0
            cost = 0.0
        net = gross - cost
        equity *= 1.0 + net
        net_returns.append(net)
        if action != 0:
            active_gross_returns.append(gross)
            active_net_returns.append(net)
        cash_actions += int(action == 0)
        if return_decision_logs:
            q_row = None if q_values is None else q_values[row].detach().cpu()
            q_map = None
            q_edge_vs_cash = None
            q_edge_vs_current = None
            if q_row is not None:
                q_map = {name: float(q_row[index].item()) for index, name in enumerate(split.action_names)}
                q_edge_vs_cash = float(q_row[action].item() - q_row[0].item())
                q_edge_vs_current = float(q_row[action].item() - q_row[previous_action].item())
            valid_mask = split.action_valid_mask[row].detach().cpu()
            context_available_until = int(split.market_context_available_timestamps_ms[row].max().item())
            decision_logs.append(
                {
                    "decision_ts": split.decision_timestamps[row],
                    "context_available_until": _timestamp_ms_to_iso(context_available_until),
                    "entry_execution_ts": _timestamp_ms_to_iso(int(split.entry_execution_timestamps_ms[row, action].item())),
                    "reward_end_ts": split.next_timestamps[row],
                    "exit_execution_ts": _timestamp_ms_to_iso(int(split.exit_execution_timestamps_ms[row, action].item())),
                    "entry_price_source": split.entry_price_source,
                    "exit_price_source": split.exit_price_source,
                    "previous_action": split.action_names[previous_action],
                    "selected_action": split.action_names[action],
                    "target_weight": 0.0 if action == 0 else 1.0,
                    "order_legs": legs,
                    "traded_notional": traded_notional,
                    "q_values": q_map,
                    "q_edge_vs_cash": q_edge_vs_cash,
                    "q_edge_vs_current": q_edge_vs_current,
                    "action_mask": {name: bool(valid_mask[index].item()) for index, name in enumerate(split.action_names)},
                    "mask_reasons": {
                        name: None if bool(valid_mask[index].item()) else "action_invalid_or_missing_return"
                        for index, name in enumerate(split.action_names)
                    },
                    "data_quality_score": float(split.constraint_state[row, 0].item()) if split.constraint_state.shape[1] else None,
                    "readiness_score": float(split.constraint_state[row, 0].item()) if split.constraint_state.shape[1] else None,
                    "entry_price": None,
                    "exit_price": None,
                    "gross_return": gross,
                    "cost_bps": cost_bps,
                    "net_return": net,
                    "equity_after": equity,
                }
            )
        previous_action = action
    returns = torch.tensor(net_returns, dtype=torch.float32)
    metrics = _summarize_returns(returns, periods_per_year=split.periods_per_year)
    active_returns = torch.tensor(active_net_returns, dtype=torch.float32)
    active_gross = torch.tensor(active_gross_returns, dtype=torch.float32)
    active_metrics = _summarize_returns(active_returns, periods_per_year=split.periods_per_year)
    metrics.update(
        {
            "cash_action_share": cash_actions / len(net_returns) if net_returns else 1.0,
            "switches": switches,
            "switch_rate": switches / len(net_returns) if net_returns else 0.0,
            "active_window_diagnostics": {
                "active_bars": int(active_returns.numel()),
                "cash_bars": int(len(net_returns) - active_returns.numel()),
                "active_gross_return": float(torch.prod(1.0 + active_gross).item() - 1.0) if active_gross.numel() else 0.0,
                "active_net_return": active_metrics["total_return"],
                "active_mean_return": active_metrics["mean_return"],
            },
        }
    )
    if return_decision_logs:
        metrics["decision_logs"] = decision_logs
    return metrics


@torch.no_grad()
def evaluate_second_context_action_scorer(
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
    metrics = _summarize_returns(net_returns, periods_per_year=split.periods_per_year)
    cash_share = float((actions.detach().cpu() == 0).float().mean().item()) if actions.numel() else 1.0
    metrics.update(
        {
            "cash_action_share": cash_share,
            "diagnostic_only": True,
            "cost_model": "rowwise_from_cash_cost_each_decision",
        }
    )
    return metrics


@torch.no_grad()
def evaluate_second_context_trading_policy(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
    initial_action: int = 0,
    min_hold_bars: int = 1,
    cooldown_bars: int = 0,
    return_decision_logs: bool = False,
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
    previous_action = int(initial_action)
    bars_held = 0
    cooldown_remaining = 0
    selected_actions: list[int] = []
    for row in range(q_values.shape[0]):
        valid = data.action_valid_mask[row].clone()
        if not bool(valid.any().item()):
            continue
        if min_hold_bars > 1 and bars_held < min_hold_bars and previous_action < valid.numel():
            hold_only = torch.zeros_like(valid)
            hold_only[previous_action] = valid[previous_action]
            valid = hold_only if bool(hold_only.any().item()) else valid
        if cooldown_remaining > 0 and previous_action < valid.numel():
            hold_only = torch.zeros_like(valid)
            hold_only[previous_action] = valid[previous_action]
            valid = hold_only if bool(hold_only.any().item()) else valid
        masked_q = q_values[row].masked_fill(~valid, -1e9)
        action = int(masked_q.argmax().item())
        if not torch.isfinite(data.action_returns[row, action]).item():
            action = 0
        if action != previous_action:
            bars_held = 0
            cooldown_remaining = int(cooldown_bars)
        else:
            bars_held += 1
            cooldown_remaining = max(0, cooldown_remaining - 1)
        previous_action = action
        selected_actions.append(action)
    metrics = _evaluate_action_path(
        split,
        torch.tensor(selected_actions, dtype=torch.long),
        q_values=q_values[: len(selected_actions)],
        initial_action=initial_action,
        return_decision_logs=return_decision_logs,
    )
    metrics.update({"diagnostic_only": False, "cost_model": "sequential_switch_only_cost"})
    return metrics


def fixed_rollout_cost_stress(
    split: SecondContextDataSplit,
    actions: torch.Tensor,
    *,
    cost_bps_values: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    initial_action: int = 0,
) -> dict[str, dict[str, Any]]:
    return {
        f"{cost:g}_bps": _evaluate_action_path(split, actions.long(), cost_bps_override=cost, initial_action=initial_action)
        for cost in cost_bps_values
    }


def evaluate_second_context_baselines(
    split: SecondContextDataSplit,
    *,
    reference_actions: torch.Tensor | None = None,
    seed: int = 17,
) -> dict[str, dict[str, Any]]:
    row_count = len(split.decision_timestamps)
    baselines: dict[str, dict[str, Any]] = {
        "CASH": _evaluate_action_path(split, torch.zeros(row_count, dtype=torch.long)),
        "PreviousActionNoTrade": _evaluate_action_path(split, torch.zeros(row_count, dtype=torch.long)),
    }
    for name in ("QQQ", "SPY"):
        if name in split.action_names:
            action_index = split.action_names.index(name)
            baselines[f"BuyAndHold_{name}"] = _evaluate_action_path(split, torch.full((row_count,), action_index, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    if reference_actions is None:
        valid_counts = split.action_valid_mask.float().sum(dim=0)
        distribution = valid_counts / valid_counts.sum().clamp_min(1.0)
        switch_count = max(0, row_count // 10)
    else:
        reference_actions = reference_actions.long().cpu()
        distribution = torch.bincount(reference_actions.clamp_min(0), minlength=len(split.action_names)).float()
        distribution = distribution / distribution.sum().clamp_min(1.0)
        switch_count = int((reference_actions[1:] != reference_actions[:-1]).sum().item()) if reference_actions.numel() > 1 else 0
    sampled = torch.multinomial(distribution, row_count, replacement=True, generator=generator)
    baselines["RandomSameActionDistribution"] = _evaluate_action_path(split, sampled)
    turnover_actions = torch.zeros(row_count, dtype=torch.long)
    if row_count:
        current = int(torch.multinomial(distribution, 1, replacement=True, generator=generator).item())
        turnover_actions[0] = current
        switch_positions = (
            torch.randperm(max(row_count - 1, 0), generator=generator)[:switch_count].add(1).sort().values.tolist()
        )
        switch_set = set(int(value) for value in switch_positions)
        for index in range(1, row_count):
            if index in switch_set:
                valid = split.action_valid_mask[index].clone()
                if 0 <= current < valid.numel():
                    valid[current] = False
                candidates = torch.nonzero(valid, as_tuple=False).flatten()
                if candidates.numel():
                    current = int(candidates[torch.randint(candidates.numel(), (1,), generator=generator)].item())
            turnover_actions[index] = current
    baselines["RandomSameTurnover"] = _evaluate_action_path(split, turnover_actions)
    return baselines


def evaluate_second_context_policy(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
) -> dict[str, float | int | None]:
    return evaluate_second_context_action_scorer(split, model, device=device, reward_scale=reward_scale)
