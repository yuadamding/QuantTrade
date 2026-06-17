"""Datasets layer: compact second-derived decision dataset split + builders/validators (extracted from rl_quant.second_context_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from os import PathLike
from typing import Any

import torch

from rl_quant.features.stock_second_context import validate_second_context_payload
from rl_quant.datasets.hourly import _validate_action_return_contract


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
    action_target_weights: torch.Tensor
    entry_execution_timestamps_ms: torch.Tensor
    exit_execution_timestamps_ms: torch.Tensor
    entry_price_source: str
    exit_price_source: str
    execution_model: str
    portfolio_state: torch.Tensor
    constraint_state: torch.Tensor
    segment_ids: torch.Tensor
    session_ids: list[str]
    valid_start_indices: torch.Tensor
    valid_index_mask: torch.Tensor
    market_mean: torch.Tensor
    market_std: torch.Tensor
    action_feature_mean: torch.Tensor
    action_feature_std: torch.Tensor
    periods_per_year: float
    label_valid_mask: torch.Tensor | None = None

    @property
    def decision_action_valid_mask(self) -> torch.Tensor:
        return self.action_valid_mask

    @property
    def supervised_action_valid_mask(self) -> torch.Tensor:
        if self.label_valid_mask is None:
            return self.action_valid_mask
        return self.action_valid_mask & self.label_valid_mask

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
            action_target_weights=self.action_target_weights.to(device),
            entry_execution_timestamps_ms=self.entry_execution_timestamps_ms.to(device),
            exit_execution_timestamps_ms=self.exit_execution_timestamps_ms.to(device),
            portfolio_state=self.portfolio_state.to(device),
            constraint_state=self.constraint_state.to(device),
            segment_ids=self.segment_ids.to(device),
            valid_start_indices=self.valid_start_indices.to(device),
            valid_index_mask=self.valid_index_mask.to(device),
            market_mean=self.market_mean.to(device),
            market_std=self.market_std.to(device),
            action_feature_mean=self.action_feature_mean.to(device),
            action_feature_std=self.action_feature_std.to(device),
            label_valid_mask=self.label_valid_mask.to(device) if self.label_valid_mask is not None else None,
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
    action_valid_mask = payload.get("decision_action_valid_mask", payload["action_valid_mask"]).bool()[selected]
    label_valid_mask = payload.get("label_valid_mask", payload["action_valid_mask"]).bool()[selected]
    _validate_action_return_contract(action_returns, label_valid_mask)
    action_cost_bps = payload["action_cost_bps"].float()[selected]
    if "action_target_weights" in payload:
        action_target_weights = payload["action_target_weights"].float()[selected]
    else:
        action_target_weights = torch.ones_like(action_returns)
        action_target_weights[:, 0] = 0.0
    entry_execution_timestamps_ms = payload["entry_execution_timestamps_ms"].long()[selected]
    exit_execution_timestamps_ms = payload["exit_execution_timestamps_ms"].long()[selected]
    portfolio_state = payload["portfolio_state"].float()[selected]
    constraint_state = payload["constraint_state"].float()[selected]
    if "segment_ids" in payload:
        segment_ids = payload["segment_ids"].long()[selected]
    else:
        segment_ids = torch.zeros(len(selected), dtype=torch.long)
    if "session_ids" in payload:
        payload_session_ids = list(payload["session_ids"])
        session_ids = [payload_session_ids[i] for i in selected]
    else:
        session_ids = ["" for _ in selected]

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
        action_target_weights=action_target_weights,
        entry_execution_timestamps_ms=entry_execution_timestamps_ms,
        exit_execution_timestamps_ms=exit_execution_timestamps_ms,
        entry_price_source=str(payload.get("entry_price_source", "")),
        exit_price_source=str(payload.get("exit_price_source", "")),
        execution_model=str(payload.get("execution_model", payload.get("dataset_manifest", {}).get("execution_model", ""))),
        portfolio_state=portfolio_state,
        constraint_state=constraint_state,
        segment_ids=segment_ids,
        session_ids=session_ids,
        valid_start_indices=valid_start_indices,
        valid_index_mask=valid_index_mask,
        market_mean=market_mean,
        market_std=market_std,
        action_feature_mean=action_feature_mean,
        action_feature_std=action_feature_std,
        periods_per_year=252.0 * periods_per_day,
        label_valid_mask=label_valid_mask,
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
