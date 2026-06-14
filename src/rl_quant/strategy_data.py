from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import torch


@dataclass
class StrategyDataSplit:
    name: str
    dates: list[str]
    feature_names: list[str]
    action_names: list[str]
    features: torch.Tensor
    action_returns: torch.Tensor
    valid_start_indices: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    lookback: int

    def to(self, device: torch.device | str) -> "StrategyDataSplit":
        return replace(
            self,
            features=self.features.to(device),
            action_returns=self.action_returns.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
        )

    def state_windows(self, indices: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.lookback, device=indices.device, dtype=torch.long)
        window_indices = indices.unsqueeze(1) - (self.lookback - 1) + offsets.unsqueeze(0)
        return self.features[window_indices]

    def next_action_returns(self, indices: torch.Tensor) -> torch.Tensor:
        return self.action_returns[indices + 1]


def _float_or_zero(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _read_numeric_table(path: Path) -> tuple[list[str], list[str], dict[str, list[float]]]:
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if not header or header[0] != "Date":
            raise ValueError(f"{path} must have Date as the first column")
        names = header[1:]
        dates: list[str] = []
        rows: dict[str, list[float]] = {}
        for row in reader:
            if not row:
                continue
            date = row[0]
            dates.append(date)
            rows[date] = [_float_or_zero(row[i + 1] if i + 1 < len(row) else "") for i in range(len(names))]
    return dates, names, rows


def build_strategy_split(
    *,
    name: str,
    state_features_path: Path,
    action_returns_path: Path,
    lookback: int,
    start_date: str | None = None,
    end_date: str | None = None,
    reward_start_date: str | None = None,
    reward_after_date: str | None = None,
    reward_end_date: str | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> StrategyDataSplit:
    state_dates, feature_names, state_rows = _read_numeric_table(state_features_path)
    _action_dates, action_names, action_rows = _read_numeric_table(action_returns_path)

    dates = [
        date
        for date in state_dates
        if date in action_rows
        and (start_date is None or date >= start_date)
        and (end_date is None or date <= end_date)
    ]
    if len(dates) < lookback + 1:
        raise ValueError(
            f"Need at least lookback + 1 aligned rows, got {len(dates)} rows for split {name!r}."
        )

    raw_features = torch.tensor([state_rows[date] for date in dates], dtype=torch.float32)
    action_returns = torch.tensor([action_rows[date] for date in dates], dtype=torch.float32)

    if feature_mean is None:
        feature_mean = raw_features.mean(dim=0)
    if feature_std is None:
        feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    features = ((raw_features - feature_mean) / feature_std).clamp_(-8.0, 8.0)
    valid_indices: list[int] = []
    # The reward for index i is action_returns[i + 1], so filter on that
    # realized reward date while allowing state lookback to use earlier rows.
    for index in range(lookback - 1, len(dates) - 1):
        reward_date = dates[index + 1]
        if reward_after_date is not None and reward_date <= reward_after_date:
            continue
        if reward_start_date is not None and reward_date < reward_start_date:
            continue
        if reward_end_date is not None and reward_date > reward_end_date:
            continue
        valid_indices.append(index)
    if not valid_indices:
        raise ValueError(f"No valid reward indices remain for split {name!r}.")
    valid_start_indices = torch.tensor(valid_indices, dtype=torch.long)

    return StrategyDataSplit(
        name=name,
        dates=dates,
        feature_names=feature_names,
        action_names=action_names,
        features=features,
        action_returns=action_returns,
        valid_start_indices=valid_start_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
        lookback=lookback,
    )


def build_strategy_splits(
    *,
    state_features_path: Path,
    action_returns_path: Path,
    lookback: int,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> tuple[StrategyDataSplit, StrategyDataSplit, StrategyDataSplit]:
    train = build_strategy_split(
        name="train",
        state_features_path=state_features_path,
        action_returns_path=action_returns_path,
        lookback=lookback,
        start_date=train_start,
        end_date=train_end,
        reward_end_date=train_end,
    )
    val = build_strategy_split(
        name="val",
        state_features_path=state_features_path,
        action_returns_path=action_returns_path,
        lookback=lookback,
        start_date=train_start,
        end_date=val_end,
        reward_after_date=train_end,
        reward_end_date=val_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    test = build_strategy_split(
        name="test",
        state_features_path=state_features_path,
        action_returns_path=action_returns_path,
        lookback=lookback,
        start_date=train_start,
        end_date=test_end,
        reward_start_date=test_start,
        reward_end_date=test_end,
        feature_mean=train.feature_mean,
        feature_std=train.feature_std,
    )
    return train, val, test


def action_index(action_names: Sequence[str], action_name: str) -> int:
    try:
        return list(action_names).index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action_name!r}") from exc
