"""Datasets layer: hourly bar-context dataset split + builders/validators (extracted from rl_quant.hourly_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from os import PathLike
from typing import Any

import torch

from rl_quant.action_risk import (
    EXPOSURE_FEATURE_DIM,
    EXPOSURE_FEATURE_NAMES,
)
from rl_quant.trading_constraints import (
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
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
