from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from os import PathLike
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.action_risk import (
    EXPOSURE_FEATURE_DIM,
    EXPOSURE_FEATURE_NAMES,
    RISK_AWARE_POLICY_MODEL_VERSION,
    ActionMeta,
    ExposureConstraintConfig,
    action_is_inverse_tensor,
    action_is_leveraged_tensor,
    action_leverage_tensor,
    action_metadata_to_dicts,
    action_weight_tensor,
    apply_exposure_masks,
    build_action_metadata,
    group_ids_for_actions,
    make_exposure_features,
    stable_action_metadata_hash,
    stable_action_risk_config_hash,
    trade_notional,
)
from rl_quant.core import (
    CudaVramReservation,
    DQNLearningConfig,
    TensorDictReplayBuffer,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    dqn_td_target,
    epsilon_by_step,
    fractional_max_drawdown,
    make_grad_scaler,
    safe_next_row_indices,
)
from rl_quant.trading_constraints import (
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
    TradingConstraintConfig,
    apply_notional_aware_hysteresis,
    build_action_mask,
    make_constraint_features,
    sample_valid_actions,
    trade_legs,
)

HOURLY_CONSTRAINT_FEATURE_NAMES = [*CONSTRAINT_FEATURE_NAMES, *EXPOSURE_FEATURE_NAMES]
HOURLY_CONSTRAINT_FEATURE_DIM = CONSTRAINT_FEATURE_DIM + EXPOSURE_FEATURE_DIM


@dataclass
class HourlyDataSplit:
    name: str
    timestamps: list[str]
    next_timestamps: list[str]
    feature_names: list[str]
    action_names: list[str]
    features: torch.Tensor
    action_returns: torch.Tensor
    session_dates: list[str] | None
    valid_start_indices: torch.Tensor
    valid_index_mask: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    lookback: int
    periods_per_year: float = 252.0 * 6.5
    bar_interval: str = "1h"
    action_valid_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "HourlyDataSplit":
        return replace(
            self,
            features=self.features.to(device),
            action_returns=self.action_returns.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            valid_index_mask=self.valid_index_mask.to(device),
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
            action_valid_mask=self.action_valid_mask.to(device) if self.action_valid_mask is not None else None,
        )

    def state_windows(self, indices: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.lookback, device=indices.device, dtype=torch.long)
        window_indices = indices.unsqueeze(1) - (self.lookback - 1) + offsets.unsqueeze(0)
        return self.features[window_indices]

    def valid_actions(self, indices: torch.Tensor) -> torch.Tensor:
        if self.action_valid_mask is None:
            return torch.ones(
                (indices.shape[0], self.action_returns.shape[1]),
                dtype=torch.bool,
                device=indices.device,
            )
        return self.action_valid_mask[indices]


def _load_payload(path: str | bytes | PathLike[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"timestamps", "next_timestamps", "feature_names", "action_names", "features", "action_returns"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Transformer dataset is missing required keys: {sorted(missing)}")
    return payload


def _assert_increasing(values: list[str], *, name: str) -> None:
    for left, right in zip(values, values[1:]):
        if _parse_utc_timestamp(right) <= _parse_utc_timestamp(left):
            raise ValueError(f"{name} must be strictly increasing; got {left!r} before {right!r}.")


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Timestamp {value!r} is not valid ISO format.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} must include timezone information.")
    return parsed.astimezone(timezone.utc)


def _optional_utc_timestamp(value: str | None) -> datetime | None:
    return None if value is None else _parse_utc_timestamp(value)


def _validate_action_return_contract(action_returns: torch.Tensor, action_valid_mask: torch.Tensor | None) -> None:
    if action_valid_mask is None:
        if not bool(torch.isfinite(action_returns).all().item()):
            raise ValueError("action_returns must be finite when no action_valid_mask is provided.")
        return
    valid_returns = action_returns[action_valid_mask]
    if valid_returns.numel() and not bool(torch.isfinite(valid_returns).all().item()):
        raise ValueError("Valid action_returns must be finite.")
    invalid_returns = action_returns[~action_valid_mask]
    if invalid_returns.numel() and not bool(torch.isnan(invalid_returns).all().item()):
        raise ValueError("Invalid action_returns must be NaN when action_valid_mask is false.")


def _build_split(
    *,
    name: str,
    payload: dict[str, Any],
    lookback: int,
    start_ts: str | None = None,
    end_ts: str | None = None,
    reward_start_ts: str | None = None,
    reward_after_ts: str | None = None,
    reward_end_ts: str | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> HourlyDataSplit:
    all_timestamps = list(payload["timestamps"])
    if not all_timestamps:
        raise ValueError("Transformer dataset has no timestamps.")
    _assert_increasing(all_timestamps, name="timestamps")
    all_timestamp_dt = [_parse_utc_timestamp(ts) for ts in all_timestamps]
    all_next_timestamps = list(payload["next_timestamps"])
    if len(all_next_timestamps) != len(all_timestamps):
        raise ValueError("next_timestamps length must match timestamps length.")
    all_next_timestamp_dt = [_parse_utc_timestamp(ts) for ts in all_next_timestamps]
    for ts, ts_dt, next_ts, next_dt in zip(all_timestamps, all_timestamp_dt, all_next_timestamps, all_next_timestamp_dt):
        if next_dt <= ts_dt:
            raise ValueError(f"next_timestamps must be after timestamps; got {ts!r} -> {next_ts!r}.")
    all_features = payload["features"].float()
    all_returns = payload["action_returns"].float()
    if all_features.shape[0] != len(all_timestamps):
        raise ValueError("features row count must match timestamps length.")
    if all_returns.shape[0] != len(all_timestamps):
        raise ValueError("action_returns row count must match timestamps length.")
    all_action_valid = payload.get("action_valid_mask")
    if all_action_valid is not None:
        all_action_valid = all_action_valid.bool()
        if tuple(all_action_valid.shape) != tuple(all_returns.shape):
            raise ValueError("action_valid_mask shape must match action_returns shape.")
    _validate_action_return_contract(all_returns, all_action_valid)
    all_session_dates = payload.get("session_dates")
    start_dt = _optional_utc_timestamp(start_ts)
    end_dt = _optional_utc_timestamp(end_ts)
    selected = [
        i
        for i, ts_dt in enumerate(all_timestamp_dt)
        if (start_dt is None or ts_dt >= start_dt) and (end_dt is None or ts_dt <= end_dt)
    ]
    if len(selected) < lookback + 2:
        raise ValueError(f"Need at least lookback + 2 rows for split {name!r}, got {len(selected)}.")

    timestamps = [all_timestamps[i] for i in selected]
    next_timestamps = [all_next_timestamps[i] for i in selected]
    timestamp_dt = [all_timestamp_dt[i] for i in selected]
    next_timestamp_dt = [all_next_timestamp_dt[i] for i in selected]
    session_dates = [all_session_dates[i] for i in selected] if all_session_dates is not None else None
    raw_features = all_features[selected]
    action_returns = all_returns[selected]
    action_valid_mask = all_action_valid[selected] if all_action_valid is not None else None
    if feature_mean is None:
        feature_mean = raw_features.mean(dim=0)
    if feature_std is None:
        feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    features = ((raw_features - feature_mean) / feature_std).clamp_(-8.0, 8.0)
    valid: list[int] = []
    require_same_session = bool(payload.get("require_same_session_lookback", False))
    reward_after_dt = _optional_utc_timestamp(reward_after_ts)
    reward_start_dt = _optional_utc_timestamp(reward_start_ts)
    reward_end_dt = _optional_utc_timestamp(reward_end_ts)
    for index in range(lookback - 1, len(timestamps) - 1):
        reward_dt = timestamp_dt[index]
        next_reward_dt = next_timestamp_dt[index]
        if reward_after_dt is not None and reward_dt <= reward_after_dt:
            continue
        if reward_start_dt is not None and reward_dt < reward_start_dt:
            continue
        if reward_end_dt is not None and next_reward_dt > reward_end_dt:
            continue
        if require_same_session and session_dates is not None:
            window_dates = session_dates[index - lookback + 1 : index + 1]
            if any(date != window_dates[-1] for date in window_dates):
                continue
        valid.append(index)
    if not valid:
        raise ValueError(f"No valid reward indices remain for split {name!r}.")
    valid_start_indices = torch.tensor(valid, dtype=torch.long)
    valid_index_mask = torch.zeros(len(timestamps), dtype=torch.bool)
    valid_index_mask[valid_start_indices] = True

    return HourlyDataSplit(
        name=name,
        timestamps=timestamps,
        next_timestamps=next_timestamps,
        feature_names=list(payload["feature_names"]),
        action_names=list(payload["action_names"]),
        features=features,
        action_returns=action_returns,
        session_dates=session_dates,
        valid_start_indices=valid_start_indices,
        valid_index_mask=valid_index_mask,
        feature_mean=feature_mean,
        feature_std=feature_std,
        lookback=lookback,
        periods_per_year=float(payload.get("periods_per_year", 252.0 * 6.5)),
        bar_interval=str(payload.get("bar_interval", "1h")),
        action_valid_mask=action_valid_mask,
    )


def build_hourly_splits(
    *,
    dataset_path,
    lookback: int,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> tuple[HourlyDataSplit, HourlyDataSplit, HourlyDataSplit]:
    payload = _load_payload(dataset_path)
    train = _build_split(
        name="train",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=train_end,
        reward_end_ts=train_end,
    )
    val = _build_split(
        name="val",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=val_end,
        reward_after_ts=train_end,
        reward_end_ts=val_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    test = _build_split(
        name="test",
        payload=payload,
        lookback=lookback,
        start_ts=train_start,
        end_ts=test_end,
        reward_start_ts=test_start,
        reward_end_ts=test_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    return train, val, test


def assert_matching_hourly_schema(*splits: HourlyDataSplit) -> None:
    if not splits:
        return
    reference = splits[0]
    for split in splits[1:]:
        if split.feature_names != reference.feature_names:
            raise ValueError(f"Feature names/order differ between {reference.name!r} and {split.name!r}.")
        if split.action_names != reference.action_names:
            raise ValueError(f"Action names/order differ between {reference.name!r} and {split.name!r}.")
        if split.features.shape[1] != reference.features.shape[1]:
            raise ValueError(f"Feature dimensions differ between {reference.name!r} and {split.name!r}.")
        if split.action_returns.shape[1] != reference.action_returns.shape[1]:
            raise ValueError(f"Action dimensions differ between {reference.name!r} and {split.name!r}.")
        if (split.action_valid_mask is None) != (reference.action_valid_mask is None):
            raise ValueError("Splits must agree on whether action_valid_mask is present.")
        if split.action_valid_mask is not None and split.action_valid_mask.shape[1] != reference.action_returns.shape[1]:
            raise ValueError(f"Action-valid mask dimensions differ for split {split.name!r}.")
        if split.bar_interval != reference.bar_interval:
            raise ValueError(f"Bar intervals differ between {reference.name!r} and {split.name!r}.")


class CausalTransformerQNetwork(nn.Module):
    """Causal Q-network over bar-based market context windows."""

    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        action_count: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        require_constraint_features: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.lookback = int(lookback)
        self.action_count = int(action_count)
        self.constraint_feature_dim = int(constraint_feature_dim)
        self.require_constraint_features = bool(require_constraint_features)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(lookback, d_model))
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.previous_action_proj = nn.Linear(action_embedding_dim, d_model)
        self.constraint_proj = nn.Linear(self.constraint_feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, action_count),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((length, length), torch.finfo(torch.float32).min, device=device),
                diagonal=1,
            )
            self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        state_windows: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = state_windows.shape[1]
        if length > self.lookback:
            raise ValueError(f"Window length {length} exceeds configured lookback {self.lookback}.")
        if constraint_features is None:
            if self.require_constraint_features:
                raise ValueError("constraint_features are required for constrained policy inference.")
            constraint_features = torch.zeros(
                state_windows.shape[0],
                self.constraint_feature_dim,
                dtype=state_windows.dtype,
                device=state_windows.device,
            )
        if constraint_features.shape[-1] != self.constraint_feature_dim:
            raise ValueError(
                f"constraint_features must have last dimension {self.constraint_feature_dim}; "
                f"got {constraint_features.shape[-1]}."
            )
        x = self.input_proj(state_windows)
        x = x + self.position_embedding[-length:][None, :, :]
        action_context = self.previous_action_proj(self.previous_action_embedding(previous_actions.long()))
        constraint_context = self.constraint_proj(constraint_features.float())
        x = x + action_context[:, None, :] + constraint_context[:, None, :]
        x = self.encoder(x, mask=self._causal_mask(length, x.device))
        return self.head(self.out_norm(x[:, -1, :]))


@dataclass
class HourlyEnvConfig:
    lookback: int
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    switch_cost_bps: float = 1.0
    initial_action: int = 0
    constraints: TradingConstraintConfig = field(default_factory=TradingConstraintConfig)
    exposure_constraints: ExposureConstraintConfig = field(default_factory=ExposureConstraintConfig)

    def __post_init__(self) -> None:
        if self.switch_cost_bps != 1.0 and self.constraints.one_way_cost_bps == 1.0:
            self.constraints = replace(self.constraints, one_way_cost_bps=self.switch_cost_bps)


@dataclass
class HourlyTransformerTrainingConfig:
    env: HourlyEnvConfig
    learning: DQNLearningConfig
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.05
    action_embedding_dim: int = 32
    target_vram_gb: float | None = None
    vram_safety_gb: float = 0.12


class VectorizedHourlyAllocationEnv:
    def __init__(self, data: HourlyDataSplit, config: HourlyEnvConfig, device: torch.device) -> None:
        if config.lookback != data.lookback:
            raise ValueError("HourlyEnvConfig.lookback must match the loaded split.")
        if not (0 <= config.initial_action < len(data.action_names)):
            raise ValueError("initial_action is outside the action space.")
        self.data = data if data.features.device == device else data.to(device)
        self.config = config
        self.device = device
        self.start_indices = self._build_start_index_pool()
        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.previous_actions = torch.full(
            (config.num_envs,),
            int(config.initial_action),
            dtype=torch.long,
            device=device,
        )
        self.bars_held = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.cooldown_remaining = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_episode = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.order_legs_today = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.order_legs_episode = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.action_meta = build_action_metadata(self.data.action_names)
        self.action_weights = action_weight_tensor(
            self.action_meta,
            device=device,
            max_effective_leverage=config.exposure_constraints.max_effective_leverage,
        )
        self.action_leverage = action_leverage_tensor(self.action_meta, device=device)
        self.action_is_leveraged = action_is_leveraged_tensor(self.action_meta, device=device)
        self.action_is_inverse = action_is_inverse_tensor(self.action_meta, device=device)
        self.action_group_ids, self.action_groups = group_ids_for_actions(self.action_meta, device=device)
        self.steps_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.leveraged_bars_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.consecutive_leveraged_bars = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.group_counts_today = torch.zeros(
            (config.num_envs, len(self.action_groups)),
            dtype=torch.long,
            device=device,
        )
        self.steps = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        starts = self.data.valid_start_indices
        starts = starts[starts + 1 < self.data.action_returns.shape[0]]
        if starts.numel() == 0:
            raise ValueError("No valid bar start indices remain.")
        return starts.to(self.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.config.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count == 0:
            return
        random_ids = torch.randint(0, self.start_indices.shape[0], (count,), device=self.device)
        self.indices[mask] = self.start_indices[random_ids]
        self.previous_actions[mask] = int(self.config.initial_action)
        self.bars_held[mask] = int(self.config.constraints.min_hold_bars)
        self.cooldown_remaining[mask] = 0
        self.switches_today[mask] = 0
        self.switches_episode[mask] = 0
        self.order_legs_today[mask] = 0.0
        self.order_legs_episode[mask] = 0.0
        self.steps_today[mask] = 0
        self.leveraged_bars_today[mask] = 0
        self.consecutive_leveraged_bars[mask] = 0
        self.group_counts_today[mask] = 0
        self.steps[mask] = 0

    def _date_labels(self, indices: torch.Tensor) -> list[str]:
        if self.data.session_dates is not None:
            source = self.data.session_dates
        else:
            source = [timestamp[:10] for timestamp in self.data.timestamps]
        return [source[int(index.item())] for index in indices.detach().cpu()]

    def action_mask(self) -> torch.Tensor:
        constraint_mask = build_action_mask(
            current_action=self.previous_actions,
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            max_switches_per_day=self.config.constraints.max_switches_per_day,
            min_hold_bars=self.config.constraints.min_hold_bars,
            action_count=len(self.data.action_names),
            switches_episode=self.switches_episode,
            max_switches_per_episode=self.config.constraints.max_switches_per_episode,
            order_legs_today=self.order_legs_today,
            max_order_legs_per_day=self.config.constraints.max_order_legs_per_day,
            order_legs_episode=self.order_legs_episode,
            max_order_legs_per_episode=self.config.constraints.max_order_legs_per_episode,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = self.data.valid_actions(self.indices)
        availability_mask[:, int(self.config.constraints.cash_index)] = True
        mask = constraint_mask & availability_mask
        mask = apply_exposure_masks(
            mask,
            current_action=self.previous_actions,
            action_leverage=self.action_leverage,
            action_weights=self.action_weights,
            action_is_leveraged=self.action_is_leveraged,
            action_is_inverse=self.action_is_inverse,
            action_group_ids=self.action_group_ids,
            group_counts_today=self.group_counts_today,
            steps_today=self.steps_today,
            leveraged_bars_today=self.leveraged_bars_today,
            consecutive_leveraged_bars=self.consecutive_leveraged_bars,
            constraints=self.config.exposure_constraints,
            cash_index=self.config.constraints.cash_index,
        )
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, int(self.config.constraints.cash_index)] = True
        return mask

    def constraint_features(self) -> torch.Tensor:
        base = make_constraint_features(
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            switches_episode=self.switches_episode,
            constraints=self.config.constraints,
            episode_length=self.config.episode_length,
            order_legs_today=self.order_legs_today,
            order_legs_episode=self.order_legs_episode,
        )
        exposure = make_exposure_features(
            current_action=self.previous_actions,
            action_leverage=self.action_leverage,
            action_weights=self.action_weights,
            action_is_leveraged=self.action_is_leveraged,
            action_group_ids=self.action_group_ids,
            group_counts_today=self.group_counts_today,
            steps_today=self.steps_today,
            leveraged_bars_today=self.leveraged_bars_today,
            consecutive_leveraged_bars=self.consecutive_leveraged_bars,
            constraints=self.config.exposure_constraints,
            episode_length=self.config.episode_length,
        )
        return torch.cat([base, exposure], dim=1)

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data.state_windows(self.indices), self.previous_actions, self.constraint_features(), self.action_mask()

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        constraint_features = self.constraint_features()
        action_mask = self.action_mask()
        actions = actions.long()
        selected_valid = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        fallback_actions = torch.argmax(action_mask.long(), dim=1)
        actions = torch.where(selected_valid, actions, fallback_actions)
        raw_returns = self.data.action_returns[current_indices, actions]
        position_weights = self.action_weights[actions]
        gross_returns = position_weights * raw_returns
        legs = trade_legs(
            previous_actions,
            actions,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        traded_notional = trade_notional(
            previous_actions,
            actions,
            self.action_weights,
            cash_index=self.config.constraints.cash_index,
        )
        is_switch = actions != previous_actions
        per_notional_cost_bps = float(self.config.constraints.one_way_cost_bps)
        per_notional_cost_bps += is_switch.float() * float(self.config.constraints.extra_switch_penalty_bps)
        cost_bps = traded_notional * per_notional_cost_bps
        net_returns = gross_returns - cost_bps / 10_000.0
        rewards = net_returns * float(self.config.reward_scale)
        next_indices = current_indices + 1

        self.indices = next_indices
        self.previous_actions = actions.long()
        self.bars_held = torch.where(is_switch, torch.ones_like(self.bars_held), self.bars_held + 1)
        self.cooldown_remaining = torch.where(
            is_switch,
            torch.full_like(self.cooldown_remaining, int(self.config.constraints.cooldown_bars)),
            torch.clamp_min(self.cooldown_remaining - 1, 0),
        )
        self.switches_today = self.switches_today + is_switch.long()
        self.switches_episode = self.switches_episode + is_switch.long()
        self.order_legs_today = self.order_legs_today + legs
        self.order_legs_episode = self.order_legs_episode + legs
        selected_groups = self.action_group_ids[actions]
        self.group_counts_today.scatter_add_(
            1,
            selected_groups.unsqueeze(1),
            torch.ones((actions.shape[0], 1), dtype=torch.long, device=self.device),
        )
        selected_leveraged = self.action_is_leveraged[actions]
        self.steps_today = self.steps_today + 1
        self.leveraged_bars_today = self.leveraged_bars_today + selected_leveraged.long()
        self.consecutive_leveraged_bars = torch.where(
            selected_leveraged,
            self.consecutive_leveraged_bars + 1,
            torch.zeros_like(self.consecutive_leveraged_bars),
        )
        self.steps = self.steps + 1
        in_bounds = next_indices + 1 < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        # Separate a TRUE terminal (no valid next row) from an episode-length TRUNCATION (a rollout
        # boundary whose next row is a real continuation). `resets` ends the episode (terminal OR
        # truncation) and drives env reset; only `terminated` may zero the TD bootstrap (see
        # core.dqn_td_target). Naming matches strategy_dqn/intraday_dqn: terminated vs resets.
        terminated = ~next_valid
        truncated = self.steps >= int(self.config.episode_length)
        resets = terminated | truncated
        if bool(in_bounds.any().item()):
            old_dates = self._date_labels(current_indices[in_bounds])
            new_dates = self._date_labels(next_indices[in_bounds])
            reset_today = torch.tensor(
                [old != new for old, new in zip(old_dates, new_dates)],
                dtype=torch.bool,
                device=self.device,
            )
            valid_positions = torch.where(in_bounds)[0]
            reset_positions = valid_positions[reset_today]
            self.switches_today[reset_positions] = 0
            self.order_legs_today[reset_positions] = 0.0
            self.steps_today[reset_positions] = 0
            self.leveraged_bars_today[reset_positions] = 0
            self.consecutive_leveraged_bars[reset_positions] = 0
            self.group_counts_today[reset_positions] = 0
        next_action_mask = self.action_mask()
        return {
            "indices": current_indices,
            "previous_actions": previous_actions,
            "constraint_features": constraint_features,
            "action_mask": action_mask,
            "actions": actions,
            "rewards": rewards,
            "next_indices": next_indices,
            "next_previous_actions": self.previous_actions,
            "next_constraint_features": self.constraint_features(),
            "next_action_mask": next_action_mask,
            "resets": resets.float(),
            "terminated": terminated.float(),
            "legs": legs,
            "raw_action_returns": raw_returns,
            "position_weights": position_weights,
            "gross_returns": gross_returns,
            "traded_notional": traded_notional,
        }


@dataclass
class HourlyEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    total_switches: int
    market_order_legs: float
    total_traded_notional: float
    max_drawdown: float
    hourly_sharpe: float | None
    rollout_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "total_switches": self.total_switches,
            "market_order_legs": self.market_order_legs,
            "total_traded_notional": self.total_traded_notional,
            "max_drawdown": self.max_drawdown,
            "hourly_sharpe": self.hourly_sharpe,
            "annualized_sharpe": self.hourly_sharpe,
            "rollout_records": self.rollout_records,
        }


def _constraint_features_for_model(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    expected_dim = getattr(model, "constraint_feature_dim", features.shape[1])
    if expected_dim == features.shape[1]:
        return features
    if expected_dim < features.shape[1]:
        return features[:, :expected_dim]
    padding = torch.zeros(
        (features.shape[0], int(expected_dim) - features.shape[1]),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, padding], dim=1)


@torch.no_grad()
def evaluate_hourly_policy(
    data: HourlyDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    switch_cost_bps: float = 1.0,
    constraints: TradingConstraintConfig | None = None,
    exposure_constraints: ExposureConstraintConfig | None = None,
    action_meta: list[ActionMeta] | None = None,
    episode_length: int | None = None,
    capture_rollout: bool = False,
) -> HourlyEvaluationResult:
    constraints = constraints or TradingConstraintConfig(one_way_cost_bps=switch_cost_bps)
    exposure_constraints = exposure_constraints or ExposureConstraintConfig()
    data = data if data.features.device == device else data.to(device)
    action_meta = action_meta or build_action_metadata(data.action_names)
    action_weights = action_weight_tensor(
        action_meta,
        device=device,
        max_effective_leverage=exposure_constraints.max_effective_leverage,
    )
    action_leverage = action_leverage_tensor(action_meta, device=device)
    action_is_leveraged = action_is_leveraged_tensor(action_meta, device=device)
    action_is_inverse = action_is_inverse_tensor(action_meta, device=device)
    action_group_ids, action_groups = group_ids_for_actions(action_meta, device=device)
    group_counts_today = torch.zeros((1, len(action_groups)), dtype=torch.long, device=device)
    model.eval()
    previous_action = int(initial_action)
    bars_held = int(constraints.min_hold_bars)
    cooldown_remaining = 0
    switches_today = 0
    switches_episode = 0
    order_legs_today = 0.0
    order_legs_episode = 0.0
    steps_today = 0
    leveraged_bars_today = 0
    consecutive_leveraged_bars = 0
    equity = 1.0
    equity_curve = [equity]
    bar_returns: list[float] = []
    total_reward_bps = 0.0
    switches = 0
    order_legs = 0.0
    traded_notional_total = 0.0
    records: list[dict[str, Any]] = []
    previous_index: int | None = None
    previous_date: str | None = None
    episode_steps = 0
    for index in data.valid_start_indices.detach().cpu().tolist():
        current_date = data.session_dates[index] if data.session_dates is not None else data.timestamps[index][:10]
        segment_reset = previous_index is None or index != previous_index + 1
        if segment_reset:
            previous_action = int(initial_action)
            bars_held = int(constraints.min_hold_bars)
            cooldown_remaining = 0
            switches_today = 0
            switches_episode = 0
            order_legs_today = 0.0
            order_legs_episode = 0.0
            steps_today = 0
            leveraged_bars_today = 0
            consecutive_leveraged_bars = 0
            group_counts_today.zero_()
            episode_steps = 0
        elif previous_date is not None and current_date != previous_date:
            switches_today = 0
            order_legs_today = 0.0
            steps_today = 0
            leveraged_bars_today = 0
            consecutive_leveraged_bars = 0
            group_counts_today.zero_()
        if episode_length is not None and episode_steps >= int(episode_length):
            switches_episode = 0
            order_legs_episode = 0.0
            episode_steps = 0
        index_tensor = torch.tensor([index], dtype=torch.long, device=device)
        previous_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        base_constraint_features = make_constraint_features(
            bars_held=torch.tensor([bars_held], dtype=torch.long, device=device),
            cooldown_remaining=torch.tensor([cooldown_remaining], dtype=torch.long, device=device),
            switches_today=torch.tensor([switches_today], dtype=torch.long, device=device),
            switches_episode=torch.tensor([switches_episode], dtype=torch.long, device=device),
            constraints=constraints,
            episode_length=int(episode_length or max(int(data.valid_start_indices.numel()), 1)),
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
        )
        exposure_features = make_exposure_features(
            current_action=previous_tensor,
            action_leverage=action_leverage,
            action_weights=action_weights,
            action_is_leveraged=action_is_leveraged,
            action_group_ids=action_group_ids,
            group_counts_today=group_counts_today,
            steps_today=torch.tensor([steps_today], dtype=torch.long, device=device),
            leveraged_bars_today=torch.tensor([leveraged_bars_today], dtype=torch.long, device=device),
            consecutive_leveraged_bars=torch.tensor([consecutive_leveraged_bars], dtype=torch.long, device=device),
            constraints=exposure_constraints,
            episode_length=int(episode_length or max(int(data.valid_start_indices.numel()), 1)),
        )
        constraint_features = torch.cat([base_constraint_features, exposure_features], dim=1)
        constraint_mask = build_action_mask(
            current_action=previous_tensor,
            bars_held=torch.tensor([bars_held], dtype=torch.long, device=device),
            cooldown_remaining=torch.tensor([cooldown_remaining], dtype=torch.long, device=device),
            switches_today=torch.tensor([switches_today], dtype=torch.long, device=device),
            max_switches_per_day=constraints.max_switches_per_day,
            min_hold_bars=constraints.min_hold_bars,
            action_count=len(data.action_names),
            switches_episode=torch.tensor([switches_episode], dtype=torch.long, device=device),
            max_switches_per_episode=constraints.max_switches_per_episode,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            max_order_legs_per_day=constraints.max_order_legs_per_day,
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
            max_order_legs_per_episode=constraints.max_order_legs_per_episode,
            cash_index=constraints.cash_index,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = data.valid_actions(index_tensor)
        availability_mask[:, int(constraints.cash_index)] = True
        pre_exposure_mask = constraint_mask & availability_mask
        action_mask = apply_exposure_masks(
            pre_exposure_mask,
            current_action=previous_tensor,
            action_leverage=action_leverage,
            action_weights=action_weights,
            action_is_leveraged=action_is_leveraged,
            action_is_inverse=action_is_inverse,
            action_group_ids=action_group_ids,
            group_counts_today=group_counts_today,
            steps_today=torch.tensor([steps_today], dtype=torch.long, device=device),
            leveraged_bars_today=torch.tensor([leveraged_bars_today], dtype=torch.long, device=device),
            consecutive_leveraged_bars=torch.tensor([consecutive_leveraged_bars], dtype=torch.long, device=device),
            constraints=exposure_constraints,
            cash_index=constraints.cash_index,
        )
        if not bool(action_mask.any().item()):
            action_mask[:, int(constraints.cash_index)] = True
        model_constraint_features = _constraint_features_for_model(model, constraint_features)
        q_values = model(data.state_windows(index_tensor), previous_tensor, model_constraint_features)
        action = int(
            apply_notional_aware_hysteresis(
                q_values,
                previous_tensor,
                action_mask,
                action_weights=action_weights,
                one_way_cost_bps=constraints.one_way_cost_bps,
                extra_switch_penalty_bps=constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=constraints.q_switch_margin_bps,
                cash_index=constraints.cash_index,
            )[0].item()
        )
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        position_weight = float(action_weights[action].item())
        effective_leverage = float((action_weights[action] * action_leverage[action]).item())
        legs = float(
            trade_legs(
                previous_tensor,
                action_tensor,
                cash_index=constraints.cash_index,
                count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            )[0].item()
        )
        traded_notional = float(
            trade_notional(
                previous_tensor,
                action_tensor,
                action_weights,
                cash_index=constraints.cash_index,
            )[0].item()
        )
        raw_action_return = float(data.action_returns[index, action].item())
        gross_return = position_weight * raw_action_return
        is_switch = action != previous_action
        per_notional_cost_bps = float(constraints.one_way_cost_bps)
        per_notional_cost_bps += float(is_switch) * float(constraints.extra_switch_penalty_bps)
        cost_bps = traded_notional * per_notional_cost_bps
        net_return = gross_return - cost_bps / 10_000.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        bar_returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        if is_switch:
            switches += 1
        order_legs += legs
        traded_notional_total += traded_notional
        if capture_rollout:
            q_row = q_values[0].detach().cpu()
            final_mask = action_mask[0].detach().cpu()
            constraint_row = constraint_mask[0].detach().cpu()
            availability_row = availability_mask[0].detach().cpu()
            pre_exposure_row = pre_exposure_mask[0].detach().cpu()
            candidate_actions = torch.arange(len(data.action_names), dtype=torch.long, device=device)
            previous_candidates = torch.full_like(candidate_actions, previous_action)
            candidate_notional = trade_notional(
                previous_candidates,
                candidate_actions,
                action_weights,
                cash_index=constraints.cash_index,
            )
            candidate_is_switch = candidate_actions.ne(previous_action)
            candidate_cost_bps = candidate_notional * (
                float(constraints.one_way_cost_bps)
                + candidate_is_switch.float() * float(constraints.extra_switch_penalty_bps)
            )
            candidate_costs = candidate_cost_bps.detach().cpu()
            q_values_by_action: dict[str, float] = {}
            candidates: dict[str, dict[str, Any]] = {}
            mask_reasons: dict[str, str] = {}
            for action_id, action_name in enumerate(data.action_names):
                valid_candidate = bool(final_mask[action_id].item())
                if valid_candidate:
                    reason = None
                elif not bool(availability_row[action_id].item()):
                    reason = "not_tradable_at_timestamp"
                elif not bool(constraint_row[action_id].item()):
                    reason = "trading_constraint"
                elif bool(pre_exposure_row[action_id].item()):
                    reason = "exposure_constraint"
                else:
                    reason = "masked"
                q_value = round(float(q_row[action_id].item()), 8)
                q_values_by_action[action_name] = q_value
                candidates[action_name] = {
                    "valid": valid_candidate,
                    "q_value": q_value,
                    "expected_cost_bps": round(float(candidate_costs[action_id].item()), 8),
                    "risk_bucket": action_meta[action_id].asset_class,
                    "reason": reason,
                }
                if reason is not None:
                    mask_reasons[action_name] = reason
            records.append(
                {
                    "decision_id": f"{data.name}:{data.timestamps[index]}",
                    "timestamp": data.timestamps[index],
                    "decision_ts": data.timestamps[index],
                    "next_timestamp": data.next_timestamps[index],
                    "model_id": f"{data.bar_interval}_causal_transformer_v{RISK_AWARE_POLICY_MODEL_VERSION}",
                    "action": action,
                    "asset": data.action_names[action],
                    "selected_action": data.action_names[action],
                    "previous_action": previous_action,
                    "previous_asset": data.action_names[previous_action],
                    "segment_reset": int(segment_reset),
                    "market_order_legs": legs,
                    "traded_notional": round(traded_notional, 8),
                    "position_weight": round(position_weight, 8),
                    "effective_leverage": round(effective_leverage, 8),
                    "raw_action_return": round(raw_action_return, 8),
                    "gross_return": round(gross_return, 8),
                    "cost_bps": round(cost_bps, 8),
                    "bar_interval": data.bar_interval,
                    "bar_return": round(net_return, 8),
                    "net_return": round(net_return, 8),
                    "hourly_return": round(net_return, 8),
                    "equity": round(equity, 8),
                    "action_mask_reasons": mask_reasons,
                    "q_values": q_values_by_action,
                    "risk_checks": {
                        "has_valid_action": bool(final_mask.any().item()),
                        "selected_action_valid": bool(final_mask[action].item()),
                        "selected_action_available": bool(availability_row[action].item()),
                    },
                    "expected_cost_bps": round(cost_bps, 8),
                    "data_quality_score": 1.0,
                    "readiness_score": 1.0 if bool(final_mask[action].item()) else 0.0,
                    "readiness_config_hash": stable_action_risk_config_hash(exposure_constraints),
                    "candidates": candidates,
                }
            )
        if is_switch:
            bars_held = 1
            cooldown_remaining = int(constraints.cooldown_bars)
            switches_today += 1
            switches_episode += 1
        else:
            bars_held += 1
            cooldown_remaining = max(cooldown_remaining - 1, 0)
        order_legs_today += legs
        order_legs_episode += legs
        selected_group = int(action_group_ids[action].item())
        group_counts_today[0, selected_group] += 1
        selected_leveraged = bool(action_is_leveraged[action].item())
        steps_today += 1
        leveraged_bars_today += int(selected_leveraged)
        consecutive_leveraged_bars = consecutive_leveraged_bars + 1 if selected_leveraged else 0
        previous_action = action
        previous_index = index
        previous_date = current_date
        episode_steps += 1
    return HourlyEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        total_switches=switches,
        market_order_legs=order_legs,
        total_traded_notional=traded_notional_total,
        max_drawdown=fractional_max_drawdown(equity_curve),
        hourly_sharpe=annualized_sharpe(bar_returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
    )


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_hourly_transformer_dqn(
    train_data: HourlyDataSplit,
    val_data: HourlyDataSplit,
    *,
    device: torch.device,
    config: HourlyTransformerTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    train_data = train_data if train_data.features.device == device else train_data.to(device)
    val_data = val_data if val_data.features.device == device else val_data.to(device)
    assert_matching_hourly_schema(train_data, val_data)
    action_count = len(train_data.action_names)
    action_meta = build_action_metadata(train_data.action_names)
    action_weights = action_weight_tensor(
        action_meta,
        device=device,
        max_effective_leverage=config.env.exposure_constraints.max_effective_leverage,
    )

    q_network = CausalTransformerQNetwork(
        feature_dim=train_data.features.shape[1],
        lookback=config.env.lookback,
        action_count=action_count,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        action_embedding_dim=config.action_embedding_dim,
        constraint_feature_dim=HOURLY_CONSTRAINT_FEATURE_DIM,
    ).to(device)
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp, config.learning.amp_dtype)
    replay = TensorDictReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields={
            "indices": ((), torch.long),
            "previous_actions": ((), torch.long),
            "constraint_features": ((HOURLY_CONSTRAINT_FEATURE_DIM,), torch.float32),
            "action_mask": ((action_count,), torch.bool),
            "actions": ((), torch.long),
            "rewards": ((), torch.float32),
            "next_indices": ((), torch.long),
            "next_previous_actions": ((), torch.long),
            "next_constraint_features": ((HOURLY_CONSTRAINT_FEATURE_DIM,), torch.float32),
            "next_action_mask": ((action_count,), torch.bool),
            "terminated": ((), torch.float32),
        },
    )
    env = VectorizedHourlyAllocationEnv(train_data, config.env, device)
    reservation = CudaVramReservation(
        target_gb=config.target_vram_gb,
        safety_gb=config.vram_safety_gb,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_return = -float("inf")
    best_val_switches = 10**12
    best_state = _state_dict_to_cpu(q_network)
    loss_trace: list[float] = []
    reward_trace: list[float] = []
    valid_action_count_trace: list[float] = []
    eval_trace: list[dict[str, float | int | None | str]] = []

    for step in range(1, config.learning.train_steps + 1):
        states, previous_actions, constraint_features, action_mask = env.observe()
        valid_action_count_trace.append(float(action_mask.sum(dim=1).float().mean().item()))
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q_values = q_network(states, previous_actions, constraint_features)
            greedy_actions = apply_notional_aware_hysteresis(
                q_values,
                previous_actions,
                action_mask,
                action_weights=action_weights,
                one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                cash_index=config.env.constraints.cash_index,
                reward_scale=config.env.reward_scale,
            )
            random_actions = sample_valid_actions(action_mask)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        transition = env.step(actions)
        replay.add(**transition)
        reward_trace.append(float(transition["rewards"].mean().item()))
        env.reset(transition["resets"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            # Clamp next_indices for the state lookup: a true terminal can store an out-of-data next
            # row (its bootstrap is zeroed below); a no-op for non-terminal in-range transitions.
            n_rows = int(train_data.action_returns.shape[0])
            safe_next_indices = safe_next_row_indices(batch["next_indices"], batch["terminated"], n_rows)
            current_states = train_data.state_windows(batch["indices"])
            next_states = train_data.state_windows(safe_next_indices)
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                chosen_q = q_network(
                    current_states,
                    batch["previous_actions"],
                    batch["constraint_features"],
                ).gather(
                    1,
                    batch["actions"].unsqueeze(1),
                ).squeeze(1)
                with torch.no_grad():
                    next_online = q_network(
                        next_states,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    )
                    next_actions = apply_notional_aware_hysteresis(
                        next_online,
                        batch["next_previous_actions"],
                        batch["next_action_mask"],
                        action_weights=action_weights,
                        one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                        extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                        q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                        cash_index=config.env.constraints.cash_index,
                        reward_scale=config.env.reward_scale,
                    )
                    next_q = target_network(
                        next_states,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    ).gather(
                        1,
                        next_actions.unsqueeze(1),
                    ).squeeze(1)
                    # Bootstrap through episode-length truncations; zero only on true terminals.
                    # Shared float32 target with the minute-to-hour trainer (AMP-safe). Using the
                    # reset mask here would wrongly treat truncation as terminal.
                    target_q = dqn_td_target(batch["rewards"], config.learning.gamma, batch["terminated"], next_q)
                loss = F.smooth_l1_loss(chosen_q.float(), target_q)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.learning.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.learning.grad_clip)
                optimizer.step()
            loss_trace.append(float(loss.item()))
            reservation.maybe_reserve(device)

        if step % config.learning.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if step % config.learning.eval_interval == 0 or step == config.learning.train_steps:
            val_result = evaluate_hourly_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                switch_cost_bps=config.env.switch_cost_bps,
                constraints=config.env.constraints,
                exposure_constraints=config.env.exposure_constraints,
                action_meta=action_meta,
                episode_length=config.env.episode_length,
            )
            # evaluate_hourly_policy() puts the shared q_network in eval() mode; restore train()
            # so the rest of training keeps dropout active (otherwise dropout is silently disabled
            # from the first eval onward).
            q_network.train()
            avg_loss = sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1)
            avg_reward = sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1)
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_switches": val_result.total_switches,
                    "val_order_legs": val_result.market_order_legs,
                    "val_sharpe": val_result.hourly_sharpe,
                    "average_loss": avg_loss,
                    "average_train_reward": avg_reward,
                    "average_valid_action_count": sum(valid_action_count_trace[-200:])
                    / max(len(valid_action_count_trace[-200:]), 1),
                }
            )
            if (
                val_result.total_return > best_val_return
                or (
                    abs(val_result.total_return - best_val_return) <= 1e-12
                    and val_result.total_switches < best_val_switches
                )
            ):
                best_val_return = val_result.total_return
                best_val_switches = val_result.total_switches
                best_state = _state_dict_to_cpu(q_network)

    q_network.load_state_dict(best_state)
    artifacts: dict[str, object] = {
        "best_val_return": best_val_return,
        "best_val_switches": best_val_switches,
        "amp_enabled": scaler.is_enabled(),
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "valid_action_count_trace": valid_action_count_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
        "model_version": RISK_AWARE_POLICY_MODEL_VERSION,
        "uses_constraint_features": True,
        "constraint_feature_names": HOURLY_CONSTRAINT_FEATURE_NAMES,
        "action_metadata": action_metadata_to_dicts(action_meta),
        "action_metadata_hash": stable_action_metadata_hash(action_meta),
        "exposure_constraints": asdict(config.env.exposure_constraints),
        "action_risk_config_hash": stable_action_risk_config_hash(config.env.exposure_constraints),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        artifacts.update(
            {
                "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
                "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 4),
                "cuda_device_used_end_gb": round((total - free) / 1024**3, 4),
                "cuda_device_free_end_gb": round(free / 1024**3, 4),
            }
        )
    return q_network, artifacts


def action_index(action_names: list[str], action_name: str) -> int:
    try:
        return action_names.index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action_name!r}") from exc
