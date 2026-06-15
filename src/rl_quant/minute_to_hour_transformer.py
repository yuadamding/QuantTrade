from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from os import PathLike
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rl_quant.core import (
    CudaVramReservation,
    DQNLearningConfig,
    annualized_sharpe,
    autocast_context,
    configure_torch_runtime,
    epsilon_by_step,
    fractional_max_drawdown,
    make_grad_scaler,
)
from rl_quant.hourly_transformer import _validate_action_return_contract
from rl_quant.trading_constraints import (
    CONSTRAINED_POLICY_MODEL_VERSION,
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
    TradingConstraintConfig,
    apply_leg_aware_hysteresis,
    build_action_mask,
    make_constraint_features,
    sample_valid_actions,
    trade_legs,
)

DEFAULT_HOUR_DECISION_GRID_MINUTES = 60
DEFAULT_MINUTE_SOURCE_INTERVAL = "1m"
DEFAULT_SECOND_SOURCE_INTERVAL = "1s"
DEFAULT_MAX_SUBHOUR_TOKENS = 512
DEFAULT_SECOND_BAR_LATENCY_MS = 1000


class TensorDictReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device,
        fields: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.storage = {
            name: torch.zeros((capacity, *shape), dtype=dtype, device=device)
            for name, (shape, dtype) in fields.items()
        }
        self.size = 0
        self.cursor = 0

    def add(self, **transition: torch.Tensor) -> None:
        missing = set(self.storage) - set(transition)
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        first_value = next(iter(transition.values()))
        count = int(first_value.shape[0])
        if count == 0:
            return
        if count >= self.capacity:
            for name in self.storage:
                transition[name] = transition[name][-self.capacity :]
            count = self.capacity
        first = min(count, self.capacity - self.cursor)
        second = count - first
        for name, target in self.storage.items():
            values = transition[name].to(device=self.device, dtype=target.dtype)
            target[self.cursor : self.cursor + first] = values[:first]
            if second:
                target[:second] = values[first:]
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_ids = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {name: values[batch_ids] for name, values in self.storage.items()}


def default_minute_to_hour_constraints() -> TradingConstraintConfig:
    return TradingConstraintConfig(max_switches_per_day=2, q_switch_margin_bps=3.0)


@dataclass
class HourFromMinuteDataSplit:
    name: str
    decision_timestamps: list[str]
    next_timestamps: list[str]
    minute_feature_names: list[str]
    hour_feature_names: list[str]
    action_names: list[str]
    minute_features: torch.Tensor
    minute_mask: torch.Tensor
    hour_features: torch.Tensor
    action_returns: torch.Tensor
    valid_start_indices: torch.Tensor
    valid_index_mask: torch.Tensor
    minute_feature_mean: torch.Tensor
    minute_feature_std: torch.Tensor
    hour_feature_mean: torch.Tensor
    hour_feature_std: torch.Tensor
    hours_lookback: int
    minutes_per_hour: int
    decision_grid_minutes: int = DEFAULT_HOUR_DECISION_GRID_MINUTES
    periods_per_year: float = 252.0 * 6.0
    action_valid_mask: torch.Tensor | None = None
    label_valid_mask: torch.Tensor | None = None
    source_bar_interval: str = DEFAULT_MINUTE_SOURCE_INTERVAL
    context_bars_per_hour: int | None = None

    @property
    def effective_context_bars_per_hour(self) -> int:
        return int(self.context_bars_per_hour or self.minutes_per_hour)

    def to(self, device: torch.device | str) -> "HourFromMinuteDataSplit":
        return replace(
            self,
            minute_features=self.minute_features.to(device),
            minute_mask=self.minute_mask.to(device),
            hour_features=self.hour_features.to(device),
            action_returns=self.action_returns.to(device),
            action_valid_mask=self.action_valid_mask.to(device) if self.action_valid_mask is not None else None,
            label_valid_mask=self.label_valid_mask.to(device) if self.label_valid_mask is not None else None,
            valid_start_indices=self.valid_start_indices.to(device),
            valid_index_mask=self.valid_index_mask.to(device),
            minute_feature_mean=self.minute_feature_mean.to(device),
            minute_feature_std=self.minute_feature_std.to(device),
            hour_feature_mean=self.hour_feature_mean.to(device),
            hour_feature_std=self.hour_feature_std.to(device),
        )

    def state(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.minute_features[indices], self.minute_mask[indices], self.hour_features[indices]

    def valid_actions(self, indices: torch.Tensor) -> torch.Tensor:
        if self.action_valid_mask is None:
            return torch.ones(
                (indices.shape[0], self.action_returns.shape[1]),
                dtype=torch.bool,
                device=indices.device,
            )
        return self.action_valid_mask[indices]

    def label_valid_actions(self, indices: torch.Tensor) -> torch.Tensor:
        if self.label_valid_mask is not None:
            return self.label_valid_mask[indices]
        return torch.isfinite(self.action_returns[indices])


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


def source_interval_seconds(interval: str) -> int:
    text = interval.strip().lower()
    if text.endswith("s"):
        value = int(text[:-1])
    elif text.endswith("m"):
        value = int(text[:-1]) * 60
    else:
        raise ValueError(f"Unsupported source_bar_interval {interval!r}; expected values like 1s, 5s, or 1m.")
    if value <= 0:
        raise ValueError("source_bar_interval must be positive.")
    return value


def expected_context_bars_per_hour(source_bar_interval: str) -> int:
    seconds = source_interval_seconds(source_bar_interval)
    hour_seconds = DEFAULT_HOUR_DECISION_GRID_MINUTES * 60
    if hour_seconds % seconds != 0:
        raise ValueError("source_bar_interval must divide one hourly decision window exactly.")
    return hour_seconds // seconds


def _assert_alias_compatible(payload: dict[str, Any], *, canonical: str, legacy: str) -> None:
    left = payload[canonical]
    right = payload[legacy]
    if torch.is_tensor(left) and torch.is_tensor(right):
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            raise ValueError(f"{canonical} and {legacy} aliases must have matching shape and dtype.")
        if left.dtype.is_floating_point or right.dtype.is_floating_point:
            same_values = torch.allclose(left, right, equal_nan=True)
        else:
            same_values = torch.equal(left, right)
        if not bool(same_values):
            raise ValueError(f"{canonical} and {legacy} aliases must contain the same values.")
        return
    if left != right:
        raise ValueError(f"{canonical} and {legacy} aliases must contain the same values.")


def _canonicalize_subhour_payload(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    for canonical, legacy in (
        ("subhour_timestamp_grid", "minute_timestamp_grid"),
        ("subhour_feature_names", "minute_feature_names"),
        ("subhour_features", "minute_features"),
        ("subhour_mask", "minute_mask"),
    ):
        has_canonical = canonical in resolved
        has_legacy = legacy in resolved
        if has_canonical and has_legacy:
            _assert_alias_compatible(resolved, canonical=canonical, legacy=legacy)
        elif has_canonical:
            resolved[legacy] = resolved[canonical]
        elif has_legacy:
            resolved[canonical] = resolved[legacy]
    return resolved


def validate_minute_timestamp_grid(payload: dict[str, Any]) -> None:
    payload = _canonicalize_subhour_payload(payload)
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    grid = payload["minute_timestamp_grid"]
    mask = payload["minute_mask"].bool()
    source_interval = str(payload.get("source_bar_interval", DEFAULT_MINUTE_SOURCE_INTERVAL))
    default_latency_ms = DEFAULT_SECOND_BAR_LATENCY_MS if source_interval == DEFAULT_SECOND_SOURCE_INTERVAL else 0
    bar_latency_ms = int(payload.get("bar_latency_ms", default_latency_ms))
    if bar_latency_ms < 0:
        raise ValueError("bar_latency_ms must be non-negative.")
    if source_interval == DEFAULT_SECOND_SOURCE_INTERVAL and bar_latency_ms < DEFAULT_SECOND_BAR_LATENCY_MS:
        raise ValueError("One-second aggregate context requires bar_latency_ms >= 1000.")
    latency_delta = timedelta(milliseconds=bar_latency_ms)
    if len(grid) != len(decisions):
        raise ValueError("minute_timestamp_grid length must match decision_timestamps length.")
    if mask.shape[0] != len(decisions):
        raise ValueError("minute_mask row count must match decision_timestamps length.")
    for row_id, (decision_ts, next_ts) in enumerate(zip(decisions, next_timestamps)):
        decision_dt = _parse_utc_timestamp(decision_ts)
        next_dt = _parse_utc_timestamp(next_ts)
        if decision_dt >= next_dt:
            raise ValueError(f"decision_timestamps must be before next_timestamps at row {row_id}.")
        if len(grid[row_id]) != mask.shape[1]:
            raise ValueError(f"minute_timestamp_grid hour count does not match minute_mask at row {row_id}.")
        for hour_id, hour in enumerate(grid[row_id]):
            if len(hour) != mask.shape[2]:
                raise ValueError(
                    f"minute_timestamp_grid minute count does not match minute_mask at row {row_id}, hour {hour_id}."
                )
            for minute_id, minute_ts in enumerate(hour):
                if not minute_ts:
                    continue
                minute_dt = _parse_utc_timestamp(str(minute_ts))
                if bool(mask[row_id, hour_id, minute_id]) and minute_dt + latency_delta > decision_dt:
                    raise ValueError(
                        "Subhour context leakage at "
                        f"row={row_id}, hour={hour_id}, minute={minute_id}: "
                        f"{minute_ts} available after {decision_ts}."
                    )


def validate_hour_level_decision_grid(payload: dict[str, Any]) -> None:
    explicit_stride = payload.get("decision_stride_minutes", payload.get("decision_grid_minutes"))
    if explicit_stride is not None and int(explicit_stride) != DEFAULT_HOUR_DECISION_GRID_MINUTES:
        raise ValueError("Subhour-to-hour datasets must use an hourly decision grid with 60-minute rewards.")
    source_interval = str(payload.get("source_bar_interval", DEFAULT_MINUTE_SOURCE_INTERVAL))
    expected_bars = expected_context_bars_per_hour(source_interval)
    explicit_context_bars = payload.get("context_bars_per_hour", payload.get("minutes_per_hour"))
    if explicit_context_bars is not None and int(explicit_context_bars) != expected_bars:
        raise ValueError(
            "Subhour-to-hour datasets must encode exactly one hour of source bars per hour token; "
            f"{source_interval} expects {expected_bars} bars."
        )

    for row_id, (decision_ts, next_ts) in enumerate(zip(payload["decision_timestamps"], payload["next_timestamps"])):
        delta_minutes = (
            _parse_utc_timestamp(next_ts) - _parse_utc_timestamp(decision_ts)
        ).total_seconds() / 60.0
        if abs(delta_minutes - DEFAULT_HOUR_DECISION_GRID_MINUTES) > 1e-9:
            raise ValueError(
                "Minute-to-hour datasets must use an hourly decision grid; "
                f"row {row_id} has {delta_minutes:g} minutes between decision and reward."
            )


def _load_payload(path: str | bytes | PathLike[str]) -> dict[str, Any]:
    payload = _canonicalize_subhour_payload(torch.load(path, map_location="cpu", weights_only=True))
    required = {
        "decision_timestamps",
        "next_timestamps",
        "minute_timestamp_grid",
        "minute_feature_names",
        "hour_feature_names",
        "action_names",
        "minute_features",
        "minute_mask",
        "hour_features",
        "action_returns",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Minute-to-hour dataset is missing required keys: {sorted(missing)}")
    validate_hour_level_decision_grid(payload)
    validate_minute_timestamp_grid(payload)
    return payload


def _masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.unsqueeze(-1).to(features.dtype)
    count = valid.sum(dim=(0, 1, 2)).clamp_min(1.0)
    mean = (features * valid).sum(dim=(0, 1, 2)) / count
    variance = (((features - mean) * valid) ** 2).sum(dim=(0, 1, 2)) / count
    return mean, variance.sqrt().clamp_min(1e-6)


def _build_split(
    *,
    name: str,
    payload: dict[str, Any],
    start_ts: str | None = None,
    end_ts: str | None = None,
    reward_start_ts: str | None = None,
    reward_after_ts: str | None = None,
    reward_end_ts: str | None = None,
    minute_feature_mean: torch.Tensor | None = None,
    minute_feature_std: torch.Tensor | None = None,
    hour_feature_mean: torch.Tensor | None = None,
    hour_feature_std: torch.Tensor | None = None,
) -> HourFromMinuteDataSplit:
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    if not decisions:
        raise ValueError("Minute-to-hour dataset has no decision_timestamps.")
    _assert_increasing(decisions, name="decision_timestamps")
    decision_dt = [_parse_utc_timestamp(ts) for ts in decisions]
    if len(next_timestamps) != len(decisions):
        raise ValueError("next_timestamps length must match decision_timestamps length.")
    next_dt = [_parse_utc_timestamp(ts) for ts in next_timestamps]
    for decision_ts, current_dt, next_ts, following_dt in zip(decisions, decision_dt, next_timestamps, next_dt):
        if following_dt <= current_dt:
            raise ValueError(f"next_timestamps must be after decisions; got {decision_ts!r} -> {next_ts!r}.")

    all_minute_features = payload["minute_features"].float()
    all_minute_mask = payload["minute_mask"].bool()
    all_hour_features = payload["hour_features"].float()
    all_returns = payload["action_returns"].float()
    raw_action_valid = payload.get("action_valid_mask")
    raw_decision_valid = payload.get("decision_action_valid_mask", raw_action_valid)
    raw_label_valid = payload.get("label_valid_mask", payload.get("action_label_valid_mask", raw_action_valid))
    all_action_valid = raw_decision_valid
    if all_action_valid is not None:
        all_action_valid = all_action_valid.bool()
        if tuple(all_action_valid.shape) != tuple(all_returns.shape):
            raise ValueError("action_valid_mask shape must match action_returns shape.")
    all_label_valid = raw_label_valid
    if all_label_valid is not None:
        all_label_valid = all_label_valid.bool()
        if tuple(all_label_valid.shape) != tuple(all_returns.shape):
            raise ValueError("label_valid_mask shape must match action_returns shape.")
        if all_action_valid is not None and bool((all_label_valid & ~all_action_valid).any().item()):
            raise ValueError("label_valid_mask must be a subset of decision action validity.")
    _validate_action_return_contract(all_returns, all_label_valid if raw_label_valid is not None else all_action_valid)
    row_count = len(decisions)
    if all_minute_features.shape[0] != row_count or all_minute_mask.shape[0] != row_count:
        raise ValueError("minute feature/mask row counts must match decision_timestamps length.")
    if all_hour_features.shape[0] != row_count or all_returns.shape[0] != row_count:
        raise ValueError("hour_features and action_returns rows must match decision_timestamps length.")

    start_dt = None if start_ts is None else _parse_utc_timestamp(start_ts)
    end_dt = None if end_ts is None else _parse_utc_timestamp(end_ts)
    selected = [
        i
        for i, ts_dt in enumerate(decision_dt)
        if (start_dt is None or ts_dt >= start_dt) and (end_dt is None or ts_dt <= end_dt)
    ]
    if not selected:
        raise ValueError(f"No rows selected for split {name!r}.")

    decision_subset = [decisions[i] for i in selected]
    next_subset = [next_timestamps[i] for i in selected]
    decision_subset_dt = [decision_dt[i] for i in selected]
    next_subset_dt = [next_dt[i] for i in selected]
    raw_minute = all_minute_features[selected]
    raw_mask = all_minute_mask[selected]
    raw_hour = all_hour_features[selected]
    returns = all_returns[selected]
    action_valid_mask = all_action_valid[selected] if all_action_valid is not None else None
    label_valid_mask = all_label_valid[selected] if all_label_valid is not None else None

    reward_after_dt = None if reward_after_ts is None else _parse_utc_timestamp(reward_after_ts)
    reward_start_dt = None if reward_start_ts is None else _parse_utc_timestamp(reward_start_ts)
    reward_end_dt = None if reward_end_ts is None else _parse_utc_timestamp(reward_end_ts)
    valid: list[int] = []
    for index, current_dt in enumerate(decision_subset_dt):
        following_dt = next_subset_dt[index]
        if reward_after_dt is not None and current_dt <= reward_after_dt:
            continue
        if reward_start_dt is not None and current_dt < reward_start_dt:
            continue
        if reward_end_dt is not None and following_dt > reward_end_dt:
            continue
        valid.append(index)
    if not valid:
        raise ValueError(f"No valid reward indices remain for split {name!r}.")

    if minute_feature_mean is None or minute_feature_std is None:
        minute_feature_mean, minute_feature_std = _masked_mean_std(raw_minute, raw_mask)
    if hour_feature_mean is None:
        hour_feature_mean = raw_hour.mean(dim=(0, 1))
    if hour_feature_std is None:
        hour_feature_std = raw_hour.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

    minute = ((raw_minute - minute_feature_mean) / minute_feature_std).clamp_(-8.0, 8.0)
    minute = minute.masked_fill(~raw_mask.unsqueeze(-1), 0.0)
    hour = ((raw_hour - hour_feature_mean) / hour_feature_std).clamp_(-8.0, 8.0)
    valid_start_indices = torch.tensor(valid, dtype=torch.long)
    valid_index_mask = torch.zeros(len(decision_subset), dtype=torch.bool)
    valid_index_mask[valid_start_indices] = True

    return HourFromMinuteDataSplit(
        name=name,
        decision_timestamps=decision_subset,
        next_timestamps=next_subset,
        minute_feature_names=list(payload["minute_feature_names"]),
        hour_feature_names=list(payload["hour_feature_names"]),
        action_names=list(payload["action_names"]),
        minute_features=minute,
        minute_mask=raw_mask,
        hour_features=hour,
        action_returns=returns,
        action_valid_mask=action_valid_mask,
        label_valid_mask=label_valid_mask,
        valid_start_indices=valid_start_indices,
        valid_index_mask=valid_index_mask,
        minute_feature_mean=minute_feature_mean,
        minute_feature_std=minute_feature_std,
        hour_feature_mean=hour_feature_mean,
        hour_feature_std=hour_feature_std,
        hours_lookback=int(payload.get("hours_lookback", raw_minute.shape[1])),
        minutes_per_hour=int(payload.get("minutes_per_hour", raw_minute.shape[2])),
        decision_grid_minutes=int(payload.get("decision_grid_minutes", payload.get("decision_stride_minutes", DEFAULT_HOUR_DECISION_GRID_MINUTES))),
        periods_per_year=float(payload.get("periods_per_year", 252.0 * 6.0)),
        source_bar_interval=str(payload.get("source_bar_interval", DEFAULT_MINUTE_SOURCE_INTERVAL)),
        context_bars_per_hour=int(payload.get("context_bars_per_hour", payload.get("minutes_per_hour", raw_minute.shape[2]))),
    )


def build_hour_from_minute_splits(
    *,
    dataset_path,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> tuple[HourFromMinuteDataSplit, HourFromMinuteDataSplit, HourFromMinuteDataSplit]:
    payload = _load_payload(dataset_path)
    train = _build_split(
        name="train",
        payload=payload,
        start_ts=train_start,
        end_ts=train_end,
        reward_end_ts=train_end,
    )
    val = _build_split(
        name="val",
        payload=payload,
        start_ts=train_start,
        end_ts=val_end,
        reward_after_ts=train_end,
        reward_end_ts=val_end,
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
    )
    test = _build_split(
        name="test",
        payload=payload,
        start_ts=train_start,
        end_ts=test_end,
        reward_start_ts=test_start,
        reward_end_ts=test_end,
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
    )
    assert_matching_hour_from_minute_schema(train, val, test)
    return train, val, test


def assert_matching_hour_from_minute_schema(*splits: HourFromMinuteDataSplit) -> None:
    if not splits:
        return
    reference = splits[0]
    for split in splits[1:]:
        if split.minute_feature_names != reference.minute_feature_names:
            raise ValueError(f"Minute feature names/order differ between {reference.name!r} and {split.name!r}.")
        if split.hour_feature_names != reference.hour_feature_names:
            raise ValueError(f"Hour feature names/order differ between {reference.name!r} and {split.name!r}.")
        if split.action_names != reference.action_names:
            raise ValueError(f"Action names/order differ between {reference.name!r} and {split.name!r}.")
        if split.decision_grid_minutes != reference.decision_grid_minutes:
            raise ValueError(f"Decision grid minutes differ between {reference.name!r} and {split.name!r}.")
        if split.source_bar_interval != reference.source_bar_interval:
            raise ValueError(f"Source bar interval differs between {reference.name!r} and {split.name!r}.")
        if split.effective_context_bars_per_hour != reference.effective_context_bars_per_hour:
            raise ValueError(f"Context bars per hour differ between {reference.name!r} and {split.name!r}.")
        if (split.action_valid_mask is None) != (reference.action_valid_mask is None):
            raise ValueError("Splits must agree on whether action_valid_mask is present.")
        if split.action_valid_mask is not None and split.action_valid_mask.shape[1] != reference.action_returns.shape[1]:
            raise ValueError(f"Action-valid mask dimensions differ for split {split.name!r}.")
        if (split.label_valid_mask is None) != (reference.label_valid_mask is None):
            raise ValueError("Splits must agree on whether label_valid_mask is present.")
        if split.label_valid_mask is not None and split.label_valid_mask.shape[1] != reference.action_returns.shape[1]:
            raise ValueError(f"Label-valid mask dimensions differ for split {split.name!r}.")
        if split.minute_features.shape[1:] != reference.minute_features.shape[1:]:
            raise ValueError(f"Subhour tensor shape differs between {reference.name!r} and {split.name!r}.")
        if split.hour_features.shape[1:] != reference.hour_features.shape[1:]:
            raise ValueError(f"Hour tensor shape differs between {reference.name!r} and {split.name!r}.")


class MinuteToHourCausalTransformerQNetwork(nn.Module):
    def __init__(
        self,
        *,
        minute_feature_dim: int,
        hour_feature_dim: int,
        action_count: int,
        hours_lookback: int,
        minutes_per_hour: int,
        d_model: int = 256,
        n_heads: int = 8,
        minute_layers: int = 2,
        hour_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        max_subhour_tokens: int | None = DEFAULT_MAX_SUBHOUR_TOKENS,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if max_subhour_tokens is not None and int(max_subhour_tokens) <= 0:
            raise ValueError("max_subhour_tokens must be positive when provided.")
        self.hours_lookback = int(hours_lookback)
        self.minutes_per_hour = int(minutes_per_hour)
        self.max_subhour_tokens = None if max_subhour_tokens is None else int(max_subhour_tokens)
        self.action_count = int(action_count)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.minute_proj = nn.Sequential(nn.Linear(minute_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.minute_pos = nn.Parameter(torch.zeros(minutes_per_hour, d_model))
        minute_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.minute_encoder = nn.TransformerEncoder(minute_layer, num_layers=minute_layers)
        self.hour_proj = nn.Sequential(
            nn.Linear(d_model + hour_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.hour_pos = nn.Parameter(torch.zeros(hours_lookback, d_model))
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.action_context = nn.Linear(action_embedding_dim, d_model)
        self.constraint_context = nn.Linear(constraint_feature_dim, d_model)
        hour_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hour_encoder = nn.TransformerEncoder(hour_layer, num_layers=hour_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, action_count),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
            self._mask_cache[key] = mask
        return mask

    def _compress_subhour_tokens(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        if self.max_subhour_tokens is None or length <= self.max_subhour_tokens:
            return tokens, valid_mask
        chunk_size = int(math.ceil(length / float(self.max_subhour_tokens)))
        chunk_count = int(math.ceil(length / float(chunk_size)))
        padded_length = chunk_count * chunk_size
        if padded_length != length:
            pad_tokens = torch.zeros(
                (tokens.shape[0], padded_length - length, tokens.shape[2]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            pad_mask = torch.zeros(
                (valid_mask.shape[0], padded_length - length),
                dtype=valid_mask.dtype,
                device=valid_mask.device,
            )
            tokens = torch.cat([tokens, pad_tokens], dim=1)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=1)
        grouped_tokens = tokens.reshape(tokens.shape[0], chunk_count, chunk_size, tokens.shape[2])
        grouped_mask = valid_mask.reshape(valid_mask.shape[0], chunk_count, chunk_size)
        weights = grouped_mask.to(tokens.dtype).unsqueeze(-1)
        counts = weights.sum(dim=2).clamp_min(1.0)
        compressed = (grouped_tokens * weights).sum(dim=2) / counts
        compressed_mask = grouped_mask.any(dim=2)
        return compressed, compressed_mask

    def forward(
        self,
        minute_features: torch.Tensor,
        minute_mask: torch.Tensor,
        hour_features: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, hours, minutes, _ = minute_features.shape
        if hours > self.hours_lookback or minutes > self.minutes_per_hour:
            raise ValueError("Input context exceeds configured hours_lookback or minutes_per_hour.")
        x = self.minute_proj(minute_features)
        x = x + self.minute_pos[:minutes][None, None, :, :]
        x = x.reshape(batch * hours, minutes, -1)
        flat_mask = minute_mask.reshape(batch * hours, minutes).bool()
        x, flat_mask = self._compress_subhour_tokens(x, flat_mask)
        minutes = x.shape[1]
        safe_padding_mask = ~flat_mask
        empty_rows = ~flat_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            safe_padding_mask[empty_rows, 0] = False
        minute_context = self.minute_encoder(
            x,
            mask=self._causal_mask(minutes, x.device),
            src_key_padding_mask=safe_padding_mask,
        )
        valid_positions = torch.arange(minutes, device=x.device).expand(batch * hours, -1)
        last_valid = torch.where(flat_mask, valid_positions, torch.full_like(valid_positions, -1)).max(dim=1).values
        last_valid = last_valid.clamp_min(0)
        hour_context = minute_context[
            torch.arange(batch * hours, device=x.device),
            last_valid,
        ].reshape(batch, hours, -1)
        hour_context = hour_context.masked_fill(empty_rows.reshape(batch, hours, 1), 0.0)

        hour_tokens = self.hour_proj(torch.cat([hour_context, hour_features], dim=-1))
        hour_tokens = hour_tokens + self.hour_pos[:hours][None, :, :]
        action_ctx = self.action_context(self.previous_action_embedding(previous_actions.long()))
        constraint_ctx = self.constraint_context(constraint_features.float())
        hour_tokens = hour_tokens + action_ctx[:, None, :] + constraint_ctx[:, None, :]
        encoded = self.hour_encoder(hour_tokens, mask=self._causal_mask(hours, x.device))
        return self.head(encoded[:, -1, :])


@dataclass
class MinuteToHourEnvConfig:
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    initial_action: int = 0
    constraints: TradingConstraintConfig = field(default_factory=default_minute_to_hour_constraints)


@dataclass
class MinuteToHourTrainingConfig:
    env: MinuteToHourEnvConfig
    learning: DQNLearningConfig
    d_model: int = 256
    n_heads: int = 8
    minute_layers: int = 2
    hour_layers: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.05
    action_embedding_dim: int = 32
    target_vram_gb: float | None = None
    vram_safety_gb: float = 0.12
    warm_start_model: str | bytes | PathLike[str] | None = None
    max_subhour_tokens: int | None = DEFAULT_MAX_SUBHOUR_TOKENS


class VectorizedMinuteToHourEnv:
    def __init__(self, data: HourFromMinuteDataSplit, config: MinuteToHourEnvConfig, device: torch.device) -> None:
        if not (0 <= config.initial_action < len(data.action_names)):
            raise ValueError("initial_action is outside the action space.")
        self.data = data if data.minute_features.device == device else data.to(device)
        self.config = config
        self.device = device
        self.start_indices = self._build_start_index_pool()
        self.indices = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.previous_actions = torch.full((config.num_envs,), int(config.initial_action), dtype=torch.long, device=device)
        self.bars_held = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.cooldown_remaining = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_today = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.switches_episode = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.order_legs_today = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.order_legs_episode = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.steps = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.reset()

    def _build_start_index_pool(self) -> torch.Tensor:
        starts = self.data.valid_start_indices
        starts = starts[starts + 1 < self.data.action_returns.shape[0]]
        if starts.numel() == 0:
            raise ValueError("No valid minute-to-hour start indices remain.")
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
        self.steps[mask] = 0

    def constraint_features(self) -> torch.Tensor:
        return make_constraint_features(
            bars_held=self.bars_held,
            cooldown_remaining=self.cooldown_remaining,
            switches_today=self.switches_today,
            switches_episode=self.switches_episode,
            constraints=self.config.constraints,
            episode_length=self.config.episode_length,
            order_legs_today=self.order_legs_today,
            order_legs_episode=self.order_legs_episode,
        )

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
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, int(self.config.constraints.cash_index)] = True
        return mask

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        minute, mask, hour = self.data.state(self.indices)
        return minute, mask, hour, self.previous_actions, self.constraint_features(), self.action_mask()

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        constraint_features = self.constraint_features()
        action_mask = self.action_mask()
        actions = actions.long()
        selected_valid = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        fallback_actions = torch.argmax(action_mask.long(), dim=1)
        actions = torch.where(selected_valid, actions, fallback_actions)
        label_mask = self.data.label_valid_actions(current_indices)
        selected_label_valid = label_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        cash_actions = torch.full_like(actions, int(self.config.constraints.cash_index))
        actions = torch.where(selected_label_valid, actions, cash_actions)
        raw_returns = self.data.action_returns[current_indices, actions]
        legs = trade_legs(
            previous_actions,
            actions,
            cash_index=self.config.constraints.cash_index,
            count_etf_to_etf_as_two_legs=self.config.constraints.count_etf_to_etf_as_two_legs,
        )
        is_switch = actions != previous_actions
        cost_bps = legs * float(self.config.constraints.one_way_cost_bps)
        cost_bps = cost_bps + is_switch.float() * float(self.config.constraints.extra_switch_penalty_bps)
        rewards = raw_returns * float(self.config.reward_scale) - cost_bps * float(self.config.reward_scale) / 10_000.0

        next_indices = current_indices + 1
        self.indices = next_indices
        self.previous_actions = actions
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
        self.steps = self.steps + 1

        in_bounds = next_indices + 1 < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        dones = (~next_valid) | (self.steps >= int(self.config.episode_length))
        if bool(in_bounds.any().item()):
            old_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in current_indices[in_bounds].detach().cpu()]
            new_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in next_indices[in_bounds].detach().cpu()]
            reset_today = torch.tensor([old != new for old, new in zip(old_dates, new_dates)], dtype=torch.bool, device=self.device)
            valid_positions = torch.where(in_bounds)[0]
            self.switches_today[valid_positions[reset_today]] = 0
            self.order_legs_today[valid_positions[reset_today]] = 0.0

        next_constraint_features = self.constraint_features()
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
            "next_constraint_features": next_constraint_features,
            "next_action_mask": next_action_mask,
            "dones": dones.float(),
            "legs": legs,
        }


@dataclass
class MinuteToHourEvaluationResult:
    split_name: str
    total_return: float
    total_reward_bps: float
    allocation_switches: int
    market_order_legs: float
    max_drawdown: float
    annualized_sharpe: float | None
    rollout_records: list[dict[str, float | str | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "total_return": self.total_return,
            "after_cost_return": self.total_return,
            "total_reward_bps": self.total_reward_bps,
            "allocation_switches": self.allocation_switches,
            "market_order_legs": self.market_order_legs,
            "max_drawdown": self.max_drawdown,
            "annualized_sharpe": self.annualized_sharpe,
            "rollout_records": self.rollout_records,
        }


@torch.no_grad()
def evaluate_minute_to_hour_policy(
    data: HourFromMinuteDataSplit,
    model: nn.Module,
    *,
    device: torch.device,
    initial_action: int = 0,
    constraints: TradingConstraintConfig | None = None,
    episode_length: int | None = None,
    reward_scale: float = 10_000.0,
    capture_rollout: bool = False,
) -> MinuteToHourEvaluationResult:
    constraints = constraints or default_minute_to_hour_constraints()
    constraint_episode_length = int(episode_length or max(int(data.valid_start_indices.numel()), 1))
    data = data if data.minute_features.device == device else data.to(device)
    model.eval()
    previous_action = int(initial_action)
    bars_held = int(constraints.min_hold_bars)
    cooldown_remaining = 0
    switches_today = 0
    switches_episode = 0
    order_legs_today = 0.0
    order_legs_episode = 0.0
    previous_index: int | None = None
    previous_date: str | None = None
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    total_reward_bps = 0.0
    allocation_switches = 0
    order_legs = 0.0
    records: list[dict[str, float | str | int]] = []
    episode_steps = 0
    for index in data.valid_start_indices.detach().cpu().tolist():
        current_date = data.decision_timestamps[index][:10]
        segment_reset = previous_index is None or index != previous_index + 1
        if segment_reset:
            previous_action = int(initial_action)
            bars_held = int(constraints.min_hold_bars)
            cooldown_remaining = 0
            switches_today = 0
            switches_episode = 0
            order_legs_today = 0.0
            order_legs_episode = 0.0
            episode_steps = 0
        elif previous_date is not None and current_date != previous_date:
            switches_today = 0
            order_legs_today = 0.0
        if episode_steps >= constraint_episode_length:
            switches_episode = 0
            order_legs_episode = 0.0
            episode_steps = 0
        minute, mask, hour = data.state(torch.tensor([index], dtype=torch.long, device=device))
        prev_tensor = torch.tensor([previous_action], dtype=torch.long, device=device)
        bars_tensor = torch.tensor([bars_held], dtype=torch.long, device=device)
        cooldown_tensor = torch.tensor([cooldown_remaining], dtype=torch.long, device=device)
        switches_today_tensor = torch.tensor([switches_today], dtype=torch.long, device=device)
        switches_episode_tensor = torch.tensor([switches_episode], dtype=torch.long, device=device)
        constraints_tensor = make_constraint_features(
            bars_held=bars_tensor,
            cooldown_remaining=cooldown_tensor,
            switches_today=switches_today_tensor,
            switches_episode=switches_episode_tensor,
            constraints=constraints,
            episode_length=constraint_episode_length,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
        )
        action_mask = build_action_mask(
            current_action=prev_tensor,
            bars_held=bars_tensor,
            cooldown_remaining=cooldown_tensor,
            switches_today=switches_today_tensor,
            max_switches_per_day=constraints.max_switches_per_day,
            min_hold_bars=constraints.min_hold_bars,
            action_count=len(data.action_names),
            switches_episode=switches_episode_tensor,
            max_switches_per_episode=constraints.max_switches_per_episode,
            order_legs_today=torch.tensor([order_legs_today], dtype=torch.float32, device=device),
            max_order_legs_per_day=constraints.max_order_legs_per_day,
            order_legs_episode=torch.tensor([order_legs_episode], dtype=torch.float32, device=device),
            max_order_legs_per_episode=constraints.max_order_legs_per_episode,
            cash_index=constraints.cash_index,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
        )
        availability_mask = data.valid_actions(torch.tensor([index], dtype=torch.long, device=device))
        availability_mask[:, int(constraints.cash_index)] = True
        action_mask = action_mask & availability_mask
        if not bool(action_mask.any().item()):
            action_mask[:, int(constraints.cash_index)] = True
        q_values = model(minute, mask, hour, prev_tensor, constraints_tensor)
        action = int(
            apply_leg_aware_hysteresis(
                q_values,
                prev_tensor,
                action_mask,
                one_way_cost_bps=constraints.one_way_cost_bps,
                extra_switch_penalty_bps=constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=constraints.q_switch_margin_bps,
                cash_index=constraints.cash_index,
                reward_scale=reward_scale,
                count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            )[0].item()
        )
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        label_mask = data.label_valid_actions(torch.tensor([index], dtype=torch.long, device=device))
        if not bool(label_mask[0, action].item()) or not bool(torch.isfinite(data.action_returns[index, action]).item()):
            action = int(constraints.cash_index)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        legs = float(
            trade_legs(
                prev_tensor,
                action_tensor,
                cash_index=constraints.cash_index,
                count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            )[0].item()
        )
        is_switch = action != previous_action
        cost_bps = legs * float(constraints.one_way_cost_bps)
        cost_bps += float(is_switch) * float(constraints.extra_switch_penalty_bps)
        gross_return = float(data.action_returns[index, action].item())
        net_return = gross_return - cost_bps / 10_000.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        returns.append(net_return)
        total_reward_bps += net_return * 10_000.0
        allocation_switches += int(is_switch)
        order_legs += legs
        if capture_rollout:
            records.append(
                {
                    "decision_timestamp": data.decision_timestamps[index],
                    "next_timestamp": data.next_timestamps[index],
                    "action": action,
                    "asset": data.action_names[action],
                    "previous_action": previous_action,
                    "segment_reset": int(segment_reset),
                    "market_order_legs": legs,
                    "net_return": round(net_return, 8),
                    "equity": round(equity, 8),
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
        previous_action = action
        previous_index = index
        previous_date = current_date
        episode_steps += 1

    return MinuteToHourEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        allocation_switches=allocation_switches,
        market_order_legs=order_legs,
        max_drawdown=fractional_max_drawdown(equity_curve),
        annualized_sharpe=annualized_sharpe(returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
    )


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_checkpoint_schema(
    checkpoint: dict[str, Any],
    *,
    minute_feature_names: list[str],
    hour_feature_names: list[str],
    action_names: list[str],
) -> None:
    expected = {
        "minute_feature_names": minute_feature_names,
        "hour_feature_names": hour_feature_names,
        "action_names": action_names,
    }
    for key, expected_values in expected.items():
        actual = checkpoint.get(key)
        if actual is None:
            raise ValueError(f"Warm-start checkpoint is missing {key}; refusing unverified fine-tune.")
        if list(actual) != list(expected_values):
            raise ValueError(f"Warm-start checkpoint {key} does not match the current dataset schema.")

    constraint_names = checkpoint.get("constraint_feature_names")
    if constraint_names is None:
        raise ValueError("Warm-start checkpoint is missing constraint_feature_names; refusing unverified fine-tune.")
    if list(constraint_names) != list(CONSTRAINT_FEATURE_NAMES):
        raise ValueError("Warm-start checkpoint constraint feature schema does not match current code.")


def load_minute_to_hour_warm_start(
    model: nn.Module,
    *,
    checkpoint_path: str | bytes | PathLike[str],
    train_data: HourFromMinuteDataSplit,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Warm-start checkpoint must be a saved minute-to-hour model artifact with model_state_dict.")
    _assert_checkpoint_schema(
        checkpoint,
        minute_feature_names=train_data.minute_feature_names,
        hour_feature_names=train_data.hour_feature_names,
        action_names=train_data.action_names,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Warm-start checkpoint architecture does not match current model hyperparameters.") from exc
    return {
        "loaded": True,
        "path": str(checkpoint_path),
        "model_version": checkpoint.get("model_version"),
        "uses_constraint_features": checkpoint.get("uses_constraint_features"),
    }


def train_minute_to_hour_dqn(
    train_data: HourFromMinuteDataSplit,
    val_data: HourFromMinuteDataSplit,
    *,
    device: torch.device,
    config: MinuteToHourTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    configure_torch_runtime(device)
    train_data = train_data if train_data.minute_features.device == device else train_data.to(device)
    val_data = val_data if val_data.minute_features.device == device else val_data.to(device)
    assert_matching_hour_from_minute_schema(train_data, val_data)
    action_count = len(train_data.action_names)
    q_network = MinuteToHourCausalTransformerQNetwork(
        minute_feature_dim=train_data.minute_features.shape[-1],
        hour_feature_dim=train_data.hour_features.shape[-1],
        action_count=action_count,
        hours_lookback=train_data.hours_lookback,
        minutes_per_hour=train_data.minutes_per_hour,
        d_model=config.d_model,
        n_heads=config.n_heads,
        minute_layers=config.minute_layers,
        hour_layers=config.hour_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        action_embedding_dim=config.action_embedding_dim,
        max_subhour_tokens=config.max_subhour_tokens,
    ).to(device)
    warm_start_info: dict[str, object] | None = None
    if config.warm_start_model is not None:
        warm_start_info = load_minute_to_hour_warm_start(
            q_network,
            checkpoint_path=config.warm_start_model,
            train_data=train_data,
        )
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=config.learning.weight_decay,
    )
    scaler = make_grad_scaler(device, config.learning.use_amp)
    replay = TensorDictReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields={
            "indices": ((), torch.long),
            "previous_actions": ((), torch.long),
            "constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
            "action_mask": ((action_count,), torch.bool),
            "actions": ((), torch.long),
            "rewards": ((), torch.float32),
            "next_indices": ((), torch.long),
            "next_previous_actions": ((), torch.long),
            "next_constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
            "next_action_mask": ((action_count,), torch.bool),
            "dones": ((), torch.float32),
        },
    )
    env = VectorizedMinuteToHourEnv(train_data, config.env, device)
    reservation = CudaVramReservation(target_gb=config.target_vram_gb, safety_gb=config.vram_safety_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_return = -float("inf")
    best_val_legs = float("inf")
    best_state = _state_dict_to_cpu(q_network)
    loss_trace: list[float] = []
    reward_trace: list[float] = []
    valid_action_count_trace: list[float] = []
    eval_trace: list[dict[str, float | int | None | str]] = []
    for step in range(1, config.learning.train_steps + 1):
        minute, mask, hour, previous_actions, constraint_features, action_mask = env.observe()
        valid_action_count_trace.append(float(action_mask.sum(dim=1).float().mean().item()))
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp):
                q_values = q_network(minute, mask, hour, previous_actions, constraint_features)
            greedy_actions = apply_leg_aware_hysteresis(
                q_values,
                previous_actions,
                action_mask,
                one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                cash_index=config.env.constraints.cash_index,
                reward_scale=config.env.reward_scale,
                count_etf_to_etf_as_two_legs=config.env.constraints.count_etf_to_etf_as_two_legs,
            )
            random_actions = sample_valid_actions(action_mask)
            explore = torch.rand(greedy_actions.shape, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)
        transition = env.step(actions)
        replay.add(**{key: value for key, value in transition.items() if key in replay.storage})
        reward_trace.append(float(transition["rewards"].mean().item()))
        env.reset(transition["dones"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            current_minute, current_mask, current_hour = train_data.state(batch["indices"])
            next_minute, next_mask, next_hour = train_data.state(batch["next_indices"])
            with autocast_context(device, config.learning.use_amp):
                q = q_network(current_minute, current_mask, current_hour, batch["previous_actions"], batch["constraint_features"])
                chosen_q = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online = q_network(
                        next_minute,
                        next_mask,
                        next_hour,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    )
                    next_actions = apply_leg_aware_hysteresis(
                        next_online,
                        batch["next_previous_actions"],
                        batch["next_action_mask"],
                        one_way_cost_bps=config.env.constraints.one_way_cost_bps,
                        extra_switch_penalty_bps=config.env.constraints.extra_switch_penalty_bps,
                        q_switch_margin_bps=config.env.constraints.q_switch_margin_bps,
                        cash_index=config.env.constraints.cash_index,
                        reward_scale=config.env.reward_scale,
                        count_etf_to_etf_as_two_legs=config.env.constraints.count_etf_to_etf_as_two_legs,
                    )
                    next_target = target_network(
                        next_minute,
                        next_mask,
                        next_hour,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                    )
                    next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = batch["rewards"] + config.learning.gamma * (1.0 - batch["dones"]) * next_q
                loss = F.smooth_l1_loss(chosen_q, target_q)
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
            val_result = evaluate_minute_to_hour_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                constraints=config.env.constraints,
                episode_length=config.env.episode_length,
                reward_scale=config.env.reward_scale,
            )
            eval_trace.append(
                {
                    "step": step,
                    "epsilon": epsilon,
                    "val_return": val_result.total_return,
                    "val_order_legs": val_result.market_order_legs,
                    "val_sharpe": val_result.annualized_sharpe,
                    "average_loss": sum(loss_trace[-200:]) / max(len(loss_trace[-200:]), 1),
                    "average_train_reward": sum(reward_trace[-200:]) / max(len(reward_trace[-200:]), 1),
                    "average_valid_action_count": sum(valid_action_count_trace[-200:])
                    / max(len(valid_action_count_trace[-200:]), 1),
                }
            )
            if val_result.total_return > best_val_return or (
                abs(val_result.total_return - best_val_return) <= 1e-12
                and val_result.market_order_legs < best_val_legs
            ):
                best_val_return = val_result.total_return
                best_val_legs = val_result.market_order_legs
                best_state = _state_dict_to_cpu(q_network)

    q_network.load_state_dict(best_state)
    artifacts: dict[str, object] = {
        "best_val_return": best_val_return,
        "best_val_order_legs": best_val_legs,
        "amp_enabled": scaler.is_enabled(),
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "valid_action_count_trace": valid_action_count_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
        "model_version": CONSTRAINED_POLICY_MODEL_VERSION,
        "uses_constraint_features": True,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "warm_start": warm_start_info or {"loaded": False},
        "source_bar_interval": train_data.source_bar_interval,
        "context_bars_per_hour": train_data.effective_context_bars_per_hour,
        "max_subhour_tokens": config.max_subhour_tokens,
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
