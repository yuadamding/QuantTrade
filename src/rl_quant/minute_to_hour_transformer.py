from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
from torch import nn

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
from rl_quant.hourly_transformer import _validate_action_return_contract
from rl_quant.trading_constraints import (
    CONSTRAINED_POLICY_MODEL_VERSION,
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
    POSITION_AWARE_POLICY_MODEL_VERSION,
    TRANSITION_FEATURE_DIM,
    TRANSITION_FEATURE_NAMES,
    TRANSITION_FEATURE_SCHEMA_VERSION,
    TradingConstraintConfig,
    DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION,
    DYNAMIC_TRANSITION_FEATURE_DIM,
    DYNAMIC_TRANSITION_FEATURE_NAMES,
    DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION,
    apply_leg_aware_hysteresis,
    build_action_mask,
    build_dynamic_transition_features,
    build_transition_feature_table,
    make_constraint_features,
    sample_valid_actions,
    trade_legs,
)

DEFAULT_HOUR_DECISION_GRID_MINUTES = 60
DEFAULT_MINUTE_SOURCE_INTERVAL = "1m"
DEFAULT_SECOND_SOURCE_INTERVAL = "1s"
DEFAULT_MAX_SUBHOUR_TOKENS = 512
DEFAULT_SECOND_BAR_LATENCY_MS = 1000
DEFAULT_EXCHANGE_CALENDAR_ID = "XNYS_decision_timestamp_sessions_America_New_York_v1"
_EASTERN = ZoneInfo("America/New_York")


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
    dataset_reportable: bool = True
    dataset_reportability_errors: list[str] = field(default_factory=list)
    action_features: torch.Tensor | None = None
    action_feature_names: list[str] = field(default_factory=list)
    action_feature_mean: torch.Tensor | None = None
    action_feature_std: torch.Tensor | None = None
    action_feature_groups: dict[str, list[int]] = field(default_factory=dict)
    split_policy: dict[str, object] = field(default_factory=dict)
    # Audit of the missing-selectable-label row filter (see _build_split): how many time-eligible rows
    # were dropped, and whether the drop removed the split's LATEST reward row(s) -- which, for the test
    # split, shrinks it below the full latest period and is gated as non-reportable.
    excluded_missing_label_rows: int = 0
    filter_removed_latest_reward_rows: bool = False

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
            action_features=self.action_features.to(device) if self.action_features is not None else None,
            valid_start_indices=self.valid_start_indices.to(device),
            valid_index_mask=self.valid_index_mask.to(device),
            minute_feature_mean=self.minute_feature_mean.to(device),
            minute_feature_std=self.minute_feature_std.to(device),
            hour_feature_mean=self.hour_feature_mean.to(device),
            hour_feature_std=self.hour_feature_std.to(device),
            action_feature_mean=self.action_feature_mean.to(device) if self.action_feature_mean is not None else None,
            action_feature_std=self.action_feature_std.to(device) if self.action_feature_std is not None else None,
        )

    def state(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.minute_features[indices], self.minute_mask[indices], self.hour_features[indices]

    def action_feature_state(self, indices: torch.Tensor) -> torch.Tensor | None:
        return None if self.action_features is None else self.action_features[indices]

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


def _timestamp_to_epoch_ms(value: str) -> int:
    parsed = _parse_utc_timestamp(value)
    return int(parsed.timestamp() * 1000)


def _file_sha256(path: str | bytes | PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_decision_timestamps_ms(payload: dict[str, Any]) -> torch.Tensor:
    if "decision_timestamps_ms" in payload:
        return torch.as_tensor(payload["decision_timestamps_ms"], dtype=torch.long)
    return torch.tensor([_timestamp_to_epoch_ms(value) for value in payload["decision_timestamps"]], dtype=torch.long)


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _session_ids_from_payload(payload: dict[str, Any]) -> tuple[list[str], str]:
    decisions = list(payload["decision_timestamps"])
    for key in ("session_ids", "session_dates"):
        raw = payload.get(key)
        if raw is None:
            continue
        sessions = [str(value) for value in raw]
        if len(sessions) != len(decisions):
            raise ValueError(f"{key} length must match decision_timestamps length.")
        return sessions, f"payload.{key}"
    sessions = [_parse_utc_timestamp(value).astimezone(_EASTERN).date().isoformat() for value in decisions]
    return sessions, "derived_from_decision_timestamps"


def _session_ids_for_timestamps(timestamps: list[str]) -> list[str]:
    return [_parse_utc_timestamp(value).astimezone(_EASTERN).date().isoformat() for value in timestamps]


def _split_policy_reportability_errors(split_policy: dict[str, object] | None) -> list[str]:
    if not split_policy:
        return []
    errors = split_policy.get("reportability_errors", [])
    return [str(error) for error in errors]


def _block_bounds(
    *,
    block_name: str,
    block_sessions: list[str],
    decisions: list[str],
    next_timestamps: list[str],
    decision_sessions: list[str],
    next_sessions: list[str],
) -> dict[str, object]:
    block = set(block_sessions)
    selected = [index for index, session in enumerate(decision_sessions) if session in block]
    usable = [
        index
        for index in selected
        if next_sessions[index] in block and _parse_utc_timestamp(next_timestamps[index]) > _parse_utc_timestamp(decisions[index])
    ]
    if not selected:
        raise ValueError(f"No decision rows selected for {block_name} split.")
    if not usable:
        raise ValueError(f"No reward-complete rows selected for {block_name} split.")
    return {
        "sessions": list(block_sessions),
        "session_count": len(block_sessions),
        "start": decisions[min(selected)],
        "end": decisions[max(selected)],
        "reward_start": decisions[min(usable)],
        "reward_end": max(next_timestamps[index] for index in usable),
        "valid_decision_start": decisions[min(usable)],
        "valid_decision_end": decisions[max(usable)],
        "valid_reward_end": max(next_timestamps[index] for index in usable),
        "selected_rows": len(selected),
        "valid_rows": len(usable),
    }


def infer_latest_holdout_split_policy(
    payload: dict[str, Any],
    *,
    val_sessions: int,
    test_sessions: int,
    embargo_sessions: int = 1,
    min_train_sessions: int = 60,
) -> dict[str, object]:
    if val_sessions <= 0 or test_sessions <= 0 or min_train_sessions <= 0:
        raise ValueError("val_sessions, test_sessions, and min_train_sessions must be positive.")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")

    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    decision_sessions, session_source = _session_ids_from_payload(payload)
    next_sessions = _session_ids_for_timestamps(next_timestamps)
    complete_indices = [
        index
        for index, next_ts in enumerate(next_timestamps)
        if next_sessions[index] == decision_sessions[index]
        and _parse_utc_timestamp(next_ts) > _parse_utc_timestamp(decisions[index])
    ]
    complete_sessions = _unique_in_order([decision_sessions[index] for index in complete_indices])
    required = min_train_sessions + val_sessions + test_sessions + 2 * embargo_sessions
    if len(complete_sessions) < required:
        raise ValueError(
            "Not enough complete sessions for reportable latest_holdout split: "
            f"need {required}, got {len(complete_sessions)}."
        )

    test_block = complete_sessions[-test_sessions:]
    val_end_idx = len(complete_sessions) - test_sessions - embargo_sessions
    val_start_idx = val_end_idx - val_sessions
    train_end_idx = val_start_idx - embargo_sessions
    train_block = complete_sessions[:train_end_idx]
    val_block = complete_sessions[val_start_idx:val_end_idx]
    if len(train_block) < min_train_sessions:
        raise ValueError("Empty or too-small training split after validation/test/embargo allocation.")

    train = _block_bounds(
        block_name="train",
        block_sessions=train_block,
        decisions=decisions,
        next_timestamps=next_timestamps,
        decision_sessions=decision_sessions,
        next_sessions=next_sessions,
    )
    val = _block_bounds(
        block_name="val",
        block_sessions=val_block,
        decisions=decisions,
        next_timestamps=next_timestamps,
        decision_sessions=decision_sessions,
        next_sessions=next_sessions,
    )
    test = _block_bounds(
        block_name="test",
        block_sessions=test_block,
        decisions=decisions,
        next_timestamps=next_timestamps,
        decision_sessions=decision_sessions,
        next_sessions=next_sessions,
    )
    max_decision = max((decisions[index] for index in complete_indices), key=_parse_utc_timestamp)
    max_reward = max((next_timestamps[index] for index in complete_indices), key=_parse_utc_timestamp)
    test_uses_latest = test["valid_decision_end"] == max_decision and test["valid_reward_end"] == max_reward
    errors: list[str] = [] if test_uses_latest else ["latest_holdout_test_does_not_use_latest_complete_period"]
    return {
        "split_mode": "latest_holdout",
        "calendar_id": DEFAULT_EXCHANGE_CALENDAR_ID,
        "session_id_source": session_source,
        "train_sessions": train["session_count"],
        "val_sessions": val["session_count"],
        "test_sessions": test["session_count"],
        "embargo_sessions": int(embargo_sessions),
        "min_train_sessions": int(min_train_sessions),
        "train_start": train["valid_decision_start"],
        "train_end": train["valid_decision_end"],
        "train_reward_end": train["valid_reward_end"],
        "val_start": val["valid_decision_start"],
        "val_end": val["valid_decision_end"],
        "val_reward_end": val["valid_reward_end"],
        "test_start": test["valid_decision_start"],
        "test_end": test["valid_decision_end"],
        "test_reward_end": test["valid_reward_end"],
        "max_dataset_decision_timestamp": max_decision,
        "max_dataset_reward_end_timestamp": max_reward,
        "test_uses_latest_complete_period": bool(test_uses_latest),
        "manual_split_used": False,
        "reportable": not errors,
        "reportability_errors": errors,
        "blocks": {"train": train, "val": val, "test": test},
    }


def infer_latest_rows_smoke_split_policy(
    payload: dict[str, Any],
    *,
    val_rows: int,
    test_rows: int,
    min_train_rows: int = 1,
) -> dict[str, object]:
    if val_rows <= 0 or test_rows <= 0 or min_train_rows <= 0:
        raise ValueError("val_rows, test_rows, and min_train_rows must be positive.")
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    required = min_train_rows + val_rows + test_rows
    if len(decisions) < required:
        raise ValueError(f"Not enough rows for latest_rows_smoke split: need {required}, got {len(decisions)}.")
    train_end = len(decisions) - val_rows - test_rows
    val_end = len(decisions) - test_rows
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, len(decisions)))

    def row_block(name: str, indices: list[int]) -> dict[str, object]:
        return {
            "sessions": [],
            "session_count": 0,
            "start": decisions[indices[0]],
            "end": decisions[indices[-1]],
            "reward_start": decisions[indices[0]],
            "reward_end": next_timestamps[indices[-1]],
            "valid_decision_start": decisions[indices[0]],
            "valid_decision_end": decisions[indices[-1]],
            "valid_reward_end": next_timestamps[indices[-1]],
            "selected_rows": len(indices),
            "valid_rows": len(indices),
        }

    train = row_block("train", train_indices)
    val = row_block("val", val_indices)
    test = row_block("test", test_indices)
    return {
        "split_mode": "latest_rows_smoke",
        "calendar_id": DEFAULT_EXCHANGE_CALENDAR_ID,
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "test_rows": len(test_indices),
        "train_start": train["valid_decision_start"],
        "train_end": train["valid_decision_end"],
        "train_reward_end": train["valid_reward_end"],
        "val_start": val["valid_decision_start"],
        "val_end": val["valid_decision_end"],
        "val_reward_end": val["valid_reward_end"],
        "test_start": test["valid_decision_start"],
        "test_end": test["valid_decision_end"],
        "test_reward_end": test["valid_reward_end"],
        "max_dataset_decision_timestamp": decisions[-1],
        "max_dataset_reward_end_timestamp": next_timestamps[-1],
        "test_uses_latest_complete_period": True,
        "manual_split_used": False,
        "reportable": False,
        "reportability_errors": ["smoke_row_based_split"],
        "blocks": {"train": train, "val": val, "test": test},
    }


def manual_split_policy(
    payload: dict[str, Any],
    *,
    train_end: str,
    val_end: str,
    test_start: str,
    train_start: str | None = None,
    test_end: str | None = None,
) -> dict[str, object]:
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    max_decision = max(decisions, key=_parse_utc_timestamp)
    max_reward = max(next_timestamps, key=_parse_utc_timestamp)
    test_end_text = test_end or max_reward
    test_uses_latest = (
        _parse_utc_timestamp(test_start) <= _parse_utc_timestamp(max_decision)
        and _parse_utc_timestamp(test_end_text) >= _parse_utc_timestamp(max_reward)
    )
    errors: list[str] = [] if test_uses_latest else ["manual_split_skips_latest_complete_period"]
    return {
        "split_mode": "manual",
        "calendar_id": DEFAULT_EXCHANGE_CALENDAR_ID,
        "train_start": train_start,
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "max_dataset_decision_timestamp": max_decision,
        "max_dataset_reward_end_timestamp": max_reward,
        "test_uses_latest_complete_period": bool(test_uses_latest),
        "manual_split_used": True,
        "reportable": not errors,
        "reportability_errors": errors,
    }


def _merge_action_covariate_sidecar(
    dataset_path: str | bytes | PathLike[str],
    payload: dict[str, Any],
    *,
    action_covariate_sidecar: str | bytes | PathLike[str] = "auto",
) -> dict[str, Any]:
    if str(action_covariate_sidecar) == "none":
        return payload
    if str(action_covariate_sidecar) in {"auto", "required"}:
        sidecar_path = Path(dataset_path).with_name("action_covariates.pt")
    else:
        sidecar_path = Path(action_covariate_sidecar)
    if not sidecar_path.exists():
        if str(action_covariate_sidecar) == "required":
            raise FileNotFoundError(f"Required action covariate sidecar does not exist: {sidecar_path}")
        return payload
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if not isinstance(sidecar, dict):
        raise ValueError(f"Action covariate sidecar must contain a dictionary: {sidecar_path}")
    if sidecar.get("base_dataset_file_name") not in {None, Path(dataset_path).name}:
        raise ValueError(f"Action covariate sidecar base_dataset_file_name does not match dataset: {sidecar_path}")
    expected_base_sha = sidecar.get("base_dataset_sha256")
    if expected_base_sha is None:
        raise ValueError(f"Action covariate sidecar missing base_dataset_sha256; rebuild sidecar: {sidecar_path}")
    if str(expected_base_sha) != _file_sha256(dataset_path):
        raise ValueError(f"Action covariate sidecar base_dataset_sha256 does not match dataset: {sidecar_path}")
    payload_identity = {
        "payload_hash": payload.get("payload_hash"),
        "feature_schema_hash": payload.get("feature_schema_hash"),
        "action_schema_hash": payload.get("action_schema_hash", payload.get("action_metadata_hash")),
    }
    for sidecar_key, payload_key, label in (
        ("base_dataset_payload_hash", "payload_hash", "payload_hash"),
        ("base_dataset_feature_schema_hash", "feature_schema_hash", "feature_schema_hash"),
        ("base_dataset_action_schema_hash", "action_schema_hash", "action_schema_hash"),
    ):
        sidecar_value = sidecar.get(sidecar_key)
        payload_value = payload_identity.get(payload_key)
        if payload_value is not None:
            if sidecar_value is None:
                raise ValueError(f"Action covariate sidecar missing {label}: {sidecar_path}")
            if str(sidecar_value) != str(payload_value):
                raise ValueError(f"Action covariate sidecar {label} does not match dataset: {sidecar_path}")
    if list(sidecar.get("action_names", [])) != list(payload.get("action_names", [])):
        raise ValueError(f"Action covariate sidecar action_names do not match dataset: {sidecar_path}")
    if list(sidecar.get("decision_timestamps", [])) != list(payload.get("decision_timestamps", [])):
        raise ValueError(f"Action covariate sidecar decision_timestamps do not match dataset: {sidecar_path}")
    merged = dict(payload)
    for key, value in sidecar.items():
        if key in {
            "base_dataset_file_name",
            "base_dataset_sha256",
            "base_dataset_payload_hash",
            "base_dataset_feature_schema_hash",
            "base_dataset_action_schema_hash",
        }:
            continue
        if key in merged and key not in {"decision_timestamps", "decision_timestamps_ms", "action_names"}:
            raise ValueError(f"Action covariate sidecar key already exists in dataset payload: {key}")
        merged[key] = value
    merged["action_covariate_sidecar_path"] = str(sidecar_path)
    errors = list(merged.get("dataset_reportability_errors", []))
    errors.extend(sidecar.get("action_covariate_reportability_errors", []))
    errors = list(dict.fromkeys(errors))
    merged["dataset_reportability_errors"] = errors
    merged["dataset_reportable"] = bool(merged.get("dataset_reportable", merged.get("reportable", True))) and not errors
    return merged


def _merge_news_llm_sidecar(
    dataset_path: str | bytes | PathLike[str],
    payload: dict[str, Any],
    *,
    news_llm_sidecar: str | bytes | PathLike[str] = "none",
) -> dict[str, Any]:
    if str(news_llm_sidecar) == "none":
        return payload
    if str(news_llm_sidecar) in {"auto", "required"}:
        sidecar_path = Path(dataset_path).with_name("action_news_llm_covariates.pt")
    else:
        sidecar_path = Path(news_llm_sidecar)
    if not sidecar_path.exists():
        if str(news_llm_sidecar) == "required":
            raise FileNotFoundError(f"Required news LLM sidecar does not exist: {sidecar_path}")
        return payload
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if not isinstance(sidecar, dict):
        raise ValueError(f"News LLM sidecar must contain a dictionary: {sidecar_path}")
    if sidecar.get("base_dataset_file_name") not in {None, Path(dataset_path).name}:
        raise ValueError(f"News LLM sidecar base_dataset_file_name does not match dataset: {sidecar_path}")
    expected_base_sha = sidecar.get("base_dataset_sha256")
    if expected_base_sha is None:
        raise ValueError(f"News LLM sidecar missing base_dataset_sha256; rebuild sidecar: {sidecar_path}")
    if str(expected_base_sha) != _file_sha256(dataset_path):
        raise ValueError(f"News LLM sidecar base_dataset_sha256 does not match dataset: {sidecar_path}")
    payload_identity = {
        "payload_hash": payload.get("payload_hash"),
        "feature_schema_hash": payload.get("feature_schema_hash"),
        "action_schema_hash": payload.get("action_schema_hash", payload.get("action_metadata_hash")),
    }
    for sidecar_key, payload_key, label in (
        ("base_dataset_payload_hash", "payload_hash", "payload_hash"),
        ("base_dataset_feature_schema_hash", "feature_schema_hash", "feature_schema_hash"),
        ("base_dataset_action_schema_hash", "action_schema_hash", "action_schema_hash"),
    ):
        sidecar_value = sidecar.get(sidecar_key)
        payload_value = payload_identity.get(payload_key)
        if payload_value is not None:
            if sidecar_value is None:
                raise ValueError(f"News LLM sidecar missing {label}: {sidecar_path}")
            if str(sidecar_value) != str(payload_value):
                raise ValueError(f"News LLM sidecar {label} does not match dataset: {sidecar_path}")
    if list(sidecar.get("action_names", [])) != list(payload.get("action_names", [])):
        raise ValueError(f"News LLM sidecar action_names do not match dataset: {sidecar_path}")
    if list(sidecar.get("decision_timestamps", [])) != list(payload.get("decision_timestamps", [])):
        raise ValueError(f"News LLM sidecar decision_timestamps do not match dataset: {sidecar_path}")
    sidecar_features = sidecar.get("action_features")
    sidecar_available = sidecar.get("action_feature_available_timestamps_ms")
    sidecar_feature_names = list(sidecar.get("action_feature_names", []))
    if not torch.is_tensor(sidecar_features) or sidecar_features.ndim != 3:
        raise ValueError(f"News LLM sidecar action_features missing or invalid: {sidecar_path}")
    if tuple(sidecar_features.shape[:2]) != tuple(payload["action_returns"].shape):
        raise ValueError(f"News LLM sidecar action_features shape does not match dataset: {sidecar_path}")
    if len(sidecar_feature_names) != int(sidecar_features.shape[-1]):
        raise ValueError(f"News LLM sidecar action_feature_names length mismatch: {sidecar_path}")
    if not torch.is_tensor(sidecar_available) or tuple(sidecar_available.shape) != tuple(sidecar_features.shape):
        raise ValueError(f"News LLM sidecar action_feature_available_timestamps_ms shape mismatch: {sidecar_path}")
    merged = dict(payload)
    existing_features = merged.get("action_features")
    existing_width = 0
    if existing_features is None:
        merged["action_features"] = sidecar_features.float()
        merged["action_feature_names"] = sidecar_feature_names
        merged["action_feature_available_timestamps_ms"] = sidecar_available.long()
    else:
        existing_features = existing_features.float()
        existing_width = int(existing_features.shape[-1])
        if tuple(existing_features.shape[:2]) != tuple(sidecar_features.shape[:2]):
            raise ValueError(f"News LLM sidecar action_features shape cannot be appended: {sidecar_path}")
        existing_names = list(merged.get("action_feature_names", merged.get("feature_names", {}).get("action_features", [])))
        if len(existing_names) != existing_width:
            raise ValueError("Existing action_feature_names length does not match action_features width.")
        existing_available = merged.get("action_feature_available_timestamps_ms")
        if existing_available is None:
            raise ValueError("Existing action features require per-feature availability before appending news LLM.")
        existing_available = existing_available.long()
        if tuple(existing_available.shape) != tuple(existing_features.shape):
            raise ValueError("Existing action feature availability shape does not match action_features.")
        merged["action_features"] = torch.cat([existing_features, sidecar_features.float()], dim=-1)
        merged["action_feature_names"] = [*existing_names, *sidecar_feature_names]
        merged["action_feature_available_timestamps_ms"] = torch.cat(
            [existing_available, sidecar_available.long()],
            dim=-1,
        )
    known = merged["action_feature_available_timestamps_ms"] >= 0
    row_available = torch.where(
        known,
        merged["action_feature_available_timestamps_ms"],
        torch.full_like(merged["action_feature_available_timestamps_ms"], -1),
    ).amax(dim=-1)
    merged["action_features_available_timestamps_ms"] = row_available
    merged["action_features_any_available_timestamps_ms"] = row_available
    groups = {
        str(key): [int(bounds[0]), int(bounds[1])]
        for key, bounds in dict(merged.get("action_feature_groups", {})).items()
    }
    for key, bounds in dict(sidecar.get("action_feature_groups", {})).items():
        groups[str(key)] = [int(bounds[0]) + existing_width, int(bounds[1]) + existing_width]
    merged["action_feature_groups"] = groups
    feature_names = {key: list(value) for key, value in dict(merged.get("feature_names", {})).items()}
    feature_names["action_features"] = list(merged["action_feature_names"])
    merged["feature_names"] = feature_names
    merged["feature_names_by_tensor"] = feature_names
    for key, value in sidecar.items():
        if key in {
            "base_dataset_file_name",
            "base_dataset_sha256",
            "base_dataset_payload_hash",
            "base_dataset_feature_schema_hash",
            "base_dataset_action_schema_hash",
            "decision_timestamps",
            "decision_timestamps_ms",
            "action_names",
            "action_features",
            "action_feature_names",
            "action_feature_available_timestamps_ms",
            "action_features_available_timestamps_ms",
            "action_features_any_available_timestamps_ms",
            "action_feature_groups",
        }:
            continue
        if key in merged:
            raise ValueError(f"News LLM sidecar key already exists in dataset payload: {key}")
        merged[key] = value
    merged["action_news_llm_sidecar_path"] = str(sidecar_path)
    merged["action_features_augmented_with_news_llm"] = True
    errors = list(merged.get("dataset_reportability_errors", []))
    errors.extend(sidecar.get("action_news_llm_reportability_errors", []))
    errors = list(dict.fromkeys(errors))
    merged["dataset_reportability_errors"] = errors
    merged["dataset_reportable"] = bool(merged.get("dataset_reportable", merged.get("reportable", True))) and not errors
    return merged


def validate_action_feature_tensors(payload: dict[str, Any]) -> None:
    if "action_features" not in payload:
        return
    action_features = payload["action_features"].float()
    action_returns = payload["action_returns"].float()
    if action_features.ndim != 3 or tuple(action_features.shape[:2]) != tuple(action_returns.shape):
        raise ValueError("action_features must have shape [rows, actions, features].")
    feature_names = list(payload.get("action_feature_names", payload.get("feature_names", {}).get("action_features", [])))
    if len(feature_names) != int(action_features.shape[-1]):
        raise ValueError("action_feature_names length must match action_features width.")
    available = payload.get("action_feature_available_timestamps_ms")
    if available is not None:
        available = torch.as_tensor(available, dtype=torch.long)
        if tuple(available.shape) != tuple(action_features.shape):
            raise ValueError("action_feature_available_timestamps_ms shape must match action_features.")
        decision_ms = _row_decision_timestamps_ms(payload).view(-1, 1, 1).expand_as(available)
        known = available >= 0
        if bool((available[known] > decision_ms[known]).any().item()):
            raise ValueError("action feature availability timestamp exceeds decision timestamp.")
    if not bool(torch.isfinite(action_features).all().item()):
        raise ValueError("action_features must be finite.")


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


def _load_payload(
    path: str | bytes | PathLike[str],
    *,
    action_covariate_sidecar: str | bytes | PathLike[str] = "auto",
    news_llm_sidecar: str | bytes | PathLike[str] = "none",
) -> dict[str, Any]:
    payload = _canonicalize_subhour_payload(torch.load(path, map_location="cpu", weights_only=True))
    payload = _merge_action_covariate_sidecar(path, payload, action_covariate_sidecar=action_covariate_sidecar)
    payload = _merge_news_llm_sidecar(path, payload, news_llm_sidecar=news_llm_sidecar)
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
    validate_action_feature_tensors(payload)
    return payload


def _masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Finite-safe: include finiteness in the valid set and zero out invalid positions with
    # torch.where before summing. Multiplying by a 0/1 mask is NOT safe because NaN * 0 == NaN,
    # so a single NaN in a masked-out position would poison the whole channel's mean/std.
    valid = mask.unsqueeze(-1).bool() & torch.isfinite(features)
    raw_count = valid.to(features.dtype).sum(dim=(0, 1, 2))
    count = raw_count.clamp_min(1.0)
    clean = torch.where(valid, features, torch.zeros_like(features))
    mean = clean.sum(dim=(0, 1, 2)) / count
    centered = torch.where(valid, features - mean, torch.zeros_like(features))
    variance = (centered * centered).sum(dim=(0, 1, 2)) / count
    std = variance.sqrt().clamp_min(1e-6)
    # A channel with fewer than two valid observations cannot estimate a stable mean/std; leave it
    # unnormalized (mean 0, std 1) so a single value (or none) is not amplified by a near-zero std.
    insufficient = raw_count < 2.0
    mean = torch.where(insufficient, torch.zeros_like(mean), mean)
    std = torch.where(insufficient, torch.ones_like(std), std)
    return mean, std


def _action_feature_mean_std(
    features: torch.Tensor,
    feature_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=(0, 1))
    std = features.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    if not feature_names:
        return mean, std

    def _is_keep_raw(name: str) -> bool:
        return (
            name.startswith("stock_covariates_v1_mask.")
            or name.startswith("stock_news_llm_v1_mask.")
            or name.endswith("_missing_flag")
            or name.startswith("stock_covariates_v1_type.")
            or name.endswith(".is_common_stock")
            or name.endswith(".is_adr_or_foreign")
            or name.endswith(".is_active_reference_record")
        )

    name_to_index = {name: index for index, name in enumerate(feature_names)}
    # Value channels (news / covariates) carry a sibling mask channel and are zero-filled where
    # unavailable (and for CASH). Computing mean/std over those zero/masked entries distorts the
    # normalizer, so for any value channel with a matching mask channel, fit statistics only over
    # mask-true, finite entries (this also excludes CASH, whose covariate/news masks are False).
    mask_prefixes = {"stock_news_llm_v1.": "stock_news_llm_v1_mask.", "stock_covariates_v1.": "stock_covariates_v1_mask."}
    for index, name in enumerate(feature_names):
        if _is_keep_raw(name):
            mean[index] = 0.0
            std[index] = 1.0
            continue
        mask_name = None
        for value_prefix, mask_prefix in mask_prefixes.items():
            if name.startswith(value_prefix):
                mask_name = mask_prefix + name[len(value_prefix) :]
                break
        mask_index = name_to_index.get(mask_name) if mask_name is not None else None
        if mask_index is None:
            continue
        valid = (features[:, :, mask_index] > 0.5) & torch.isfinite(features[:, :, index])
        valid_values = features[:, :, index][valid]
        if valid_values.numel() < 2:
            # Match _masked_mean_std: a value channel with fewer than two valid (mask-true, finite)
            # observations cannot estimate a stable mean/std. Leave it unnormalized (mean 0, std 1) so
            # a single value is not amplified ~1e6x by an std clamped to 1e-6 (and none -> identity).
            mean[index] = 0.0
            std[index] = 1.0
        else:
            mean[index] = valid_values.mean()
            std[index] = valid_values.std(unbiased=False).clamp_min(1e-6)
    return mean, std


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
    action_feature_mean: torch.Tensor | None = None,
    action_feature_std: torch.Tensor | None = None,
    split_policy: dict[str, object] | None = None,
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
    all_action_features = payload.get("action_features")
    action_feature_names = list(payload.get("action_feature_names", []))
    action_feature_groups = {
        str(key): [int(bounds[0]), int(bounds[1])]
        for key, bounds in dict(payload.get("action_feature_groups", {})).items()
    }
    raw_action_valid = payload.get("action_valid_mask")
    has_explicit_decision_mask = "decision_action_valid_mask" in payload
    has_explicit_label_mask = "label_valid_mask" in payload or "action_label_valid_mask" in payload
    raw_decision_valid = payload.get("decision_action_valid_mask", raw_action_valid)
    raw_label_valid = payload.get("label_valid_mask", payload.get("action_label_valid_mask", raw_action_valid))
    dataset_reportability_errors = list(payload.get("dataset_reportability_errors", []))
    dataset_reportability_errors.extend(_split_policy_reportability_errors(split_policy))
    if raw_action_valid is not None and (not has_explicit_decision_mask or not has_explicit_label_mask):
        dataset_reportability_errors.append("legacy_action_valid_mask_semantics_ambiguous")
    dataset_reportability_errors = list(dict.fromkeys(dataset_reportability_errors))
    dataset_reportable = bool(payload.get("dataset_reportable", payload.get("reportable", True))) and not dataset_reportability_errors
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
    if all_action_features is not None:
        all_action_features = all_action_features.float()
        if all_action_features.ndim != 3 or tuple(all_action_features.shape[:2]) != tuple(all_returns.shape):
            raise ValueError("action_features must have shape [rows, actions, features].")
        if len(action_feature_names) != int(all_action_features.shape[-1]):
            raise ValueError("action_feature_names length must match action_features width.")

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
    raw_action_features = all_action_features[selected] if all_action_features is not None else None
    returns = all_returns[selected]
    action_valid_mask = all_action_valid[selected] if all_action_valid is not None else None
    label_valid_mask = all_label_valid[selected] if all_label_valid is not None else None

    if action_valid_mask is None:
        scorable_decision_valid = torch.ones_like(returns, dtype=torch.bool)
    else:
        scorable_decision_valid = action_valid_mask.bool()
    if label_valid_mask is None:
        scorable_label_ok = torch.isfinite(returns)
    else:
        scorable_label_ok = label_valid_mask.bool() & torch.isfinite(returns)
    non_cash_actions = torch.ones(returns.shape[1], dtype=torch.bool, device=returns.device)
    action_names = list(payload["action_names"])
    cash_index = action_names.index("CASH") if "CASH" in action_names else 0
    if 0 <= cash_index < int(non_cash_actions.numel()):
        non_cash_actions[cash_index] = False
    rows_with_selectable_missing_labels = (
        scorable_decision_valid & non_cash_actions.unsqueeze(0) & ~scorable_label_ok
    ).any(dim=1)

    reward_after_dt = None if reward_after_ts is None else _parse_utc_timestamp(reward_after_ts)
    reward_start_dt = None if reward_start_ts is None else _parse_utc_timestamp(reward_start_ts)
    reward_end_dt = None if reward_end_ts is None else _parse_utc_timestamp(reward_end_ts)
    valid: list[int] = []
    time_eligible: list[int] = []
    for index, current_dt in enumerate(decision_subset_dt):
        following_dt = next_subset_dt[index]
        if reward_after_dt is not None and current_dt <= reward_after_dt:
            continue
        if reward_start_dt is not None and current_dt < reward_start_dt:
            continue
        if reward_end_dt is not None and following_dt > reward_end_dt:
            continue
        # Time-eligible row (its reward window lies inside the split). Tracked separately from `valid`
        # so the missing-selectable-label filter below cannot SILENTLY shrink the split's latest reward
        # coverage without it being recorded and (for the test split) gated.
        time_eligible.append(index)
        if bool(rows_with_selectable_missing_labels[index].item()):
            continue
        valid.append(index)
    if not valid:
        raise ValueError(f"No valid reward indices remain for split {name!r}.")
    # Audit the filter: indices are chronological, so the LAST time-eligible row carries the latest
    # reward. If the filter dropped it, the split no longer covers the latest complete period -- it is a
    # "filtered fully-scorable universe", not the full latest period. For the TEST split that breaks the
    # latest-period reportability contract, so fail closed (train/val record the count but are not gated).
    excluded_missing_label_rows = len(time_eligible) - len(valid)
    filter_removed_latest_reward_rows = bool(time_eligible) and valid[-1] != time_eligible[-1]
    if name == "test" and filter_removed_latest_reward_rows:
        dataset_reportability_errors = list(
            dict.fromkeys([*dataset_reportability_errors, "test_filter_removed_latest_reward_rows"])
        )
        dataset_reportable = False

    if minute_feature_mean is None or minute_feature_std is None:
        minute_feature_mean, minute_feature_std = _masked_mean_std(raw_minute, raw_mask)
    if hour_feature_mean is None:
        hour_feature_mean = raw_hour.mean(dim=(0, 1))
    if hour_feature_std is None:
        hour_feature_std = raw_hour.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    action_features = None
    if raw_action_features is not None:
        if action_feature_mean is None or action_feature_std is None:
            action_feature_mean, action_feature_std = _action_feature_mean_std(raw_action_features, action_feature_names)
        action_features = ((raw_action_features - action_feature_mean) / action_feature_std).clamp_(-8.0, 8.0)

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
        dataset_reportable=dataset_reportable,
        dataset_reportability_errors=dataset_reportability_errors,
        action_features=action_features,
        action_feature_names=action_feature_names,
        action_feature_mean=action_feature_mean,
        action_feature_std=action_feature_std,
        action_feature_groups=action_feature_groups,
        split_policy=dict(split_policy or {}),
        excluded_missing_label_rows=excluded_missing_label_rows,
        filter_removed_latest_reward_rows=filter_removed_latest_reward_rows,
    )


def build_hour_from_minute_splits(
    *,
    dataset_path,
    split_mode: str = "latest_holdout",
    train_end: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    train_start: str | None = None,
    test_end: str | None = None,
    val_sessions: int = 10,
    test_sessions: int = 20,
    embargo_sessions: int = 1,
    min_train_sessions: int = 60,
    val_rows: int = 20,
    test_rows: int = 20,
    min_train_rows: int = 1,
    action_covariate_sidecar: str | bytes | PathLike[str] = "auto",
    news_llm_sidecar: str | bytes | PathLike[str] = "none",
) -> tuple[HourFromMinuteDataSplit, HourFromMinuteDataSplit, HourFromMinuteDataSplit]:
    payload = _load_payload(
        dataset_path,
        action_covariate_sidecar=action_covariate_sidecar,
        news_llm_sidecar=news_llm_sidecar,
    )
    if split_mode == "latest_holdout":
        if any(value is not None for value in (train_start, train_end, val_end, test_start, test_end)):
            raise ValueError("Manual timestamp cutoffs require split_mode='manual'.")
        split_policy = infer_latest_holdout_split_policy(
            payload,
            val_sessions=val_sessions,
            test_sessions=test_sessions,
            embargo_sessions=embargo_sessions,
            min_train_sessions=min_train_sessions,
        )
        blocks = split_policy["blocks"]
        train_block = dict(blocks["train"])
        val_block = dict(blocks["val"])
        test_block = dict(blocks["test"])
        train = _build_split(
            name="train",
            payload=payload,
            start_ts=str(train_block["start"]),
            end_ts=str(train_block["end"]),
            reward_end_ts=str(train_block["reward_end"]),
            split_policy=split_policy,
        )
        val = _build_split(
            name="val",
            payload=payload,
            start_ts=str(val_block["start"]),
            end_ts=str(val_block["end"]),
            reward_start_ts=str(val_block["reward_start"]),
            reward_end_ts=str(val_block["reward_end"]),
            minute_feature_mean=train.minute_feature_mean,
            minute_feature_std=train.minute_feature_std,
            hour_feature_mean=train.hour_feature_mean,
            hour_feature_std=train.hour_feature_std,
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
        )
        test = _build_split(
            name="test",
            payload=payload,
            start_ts=str(test_block["start"]),
            end_ts=str(test_block["end"]),
            reward_start_ts=str(test_block["reward_start"]),
            reward_end_ts=str(test_block["reward_end"]),
            minute_feature_mean=train.minute_feature_mean,
            minute_feature_std=train.minute_feature_std,
            hour_feature_mean=train.hour_feature_mean,
            hour_feature_std=train.hour_feature_std,
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
        )
    elif split_mode == "latest_rows_smoke":
        if any(value is not None for value in (train_start, train_end, val_end, test_start, test_end)):
            raise ValueError("Manual timestamp cutoffs cannot be combined with latest_rows_smoke.")
        split_policy = infer_latest_rows_smoke_split_policy(
            payload,
            val_rows=val_rows,
            test_rows=test_rows,
            min_train_rows=min_train_rows,
        )
        blocks = split_policy["blocks"]
        train_block = dict(blocks["train"])
        val_block = dict(blocks["val"])
        test_block = dict(blocks["test"])
        train = _build_split(
            name="train",
            payload=payload,
            start_ts=str(train_block["start"]),
            end_ts=str(train_block["end"]),
            reward_end_ts=str(train_block["reward_end"]),
            split_policy=split_policy,
        )
        val = _build_split(
            name="val",
            payload=payload,
            start_ts=str(val_block["start"]),
            end_ts=str(val_block["end"]),
            reward_start_ts=str(val_block["reward_start"]),
            reward_end_ts=str(val_block["reward_end"]),
            minute_feature_mean=train.minute_feature_mean,
            minute_feature_std=train.minute_feature_std,
            hour_feature_mean=train.hour_feature_mean,
            hour_feature_std=train.hour_feature_std,
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
        )
        test = _build_split(
            name="test",
            payload=payload,
            start_ts=str(test_block["start"]),
            end_ts=str(test_block["end"]),
            reward_start_ts=str(test_block["reward_start"]),
            reward_end_ts=str(test_block["reward_end"]),
            minute_feature_mean=train.minute_feature_mean,
            minute_feature_std=train.minute_feature_std,
            hour_feature_mean=train.hour_feature_mean,
            hour_feature_std=train.hour_feature_std,
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
        )
    elif split_mode == "manual":
        if train_end is None or val_end is None or test_start is None:
            raise ValueError("Manual split mode requires train_end, val_end, and test_start.")
        split_policy = manual_split_policy(
            payload,
            train_start=train_start,
            train_end=train_end,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        train = _build_split(
            name="train",
            payload=payload,
            start_ts=train_start,
            end_ts=train_end,
            reward_end_ts=train_end,
            split_policy=split_policy,
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
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
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
            action_feature_mean=train.action_feature_mean,
            action_feature_std=train.action_feature_std,
            split_policy=split_policy,
        )
    else:
        raise ValueError(f"Unsupported split_mode {split_mode!r}.")
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
        if (split.action_features is None) != (reference.action_features is None):
            raise ValueError("Splits must agree on whether action_features are present.")
        if split.action_feature_names != reference.action_feature_names:
            raise ValueError(f"Action feature names/order differ between {reference.name!r} and {split.name!r}.")
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
        if split.action_features is not None and split.action_features.shape[1:] != reference.action_features.shape[1:]:
            raise ValueError(f"Action feature tensor shape differs between {reference.name!r} and {split.name!r}.")


def minute_to_hour_missing_label_report(
    split: HourFromMinuteDataSplit,
    *,
    row_indices: torch.Tensor | list[int] | None = None,
    requested_actions: torch.Tensor | list[int] | None = None,
    executed_actions: torch.Tensor | list[int] | None = None,
    cash_index: int = 0,
) -> dict[str, object]:
    errors = list(split.dataset_reportability_errors)
    if not split.dataset_reportable and not errors:
        errors.append("dataset_marked_non_reportable")
    if row_indices is None:
        rows = split.valid_start_indices.detach().cpu().long()
    else:
        rows = torch.as_tensor(row_indices, dtype=torch.long).detach().cpu()
    if rows.numel() == 0:
        return {
            "evaluation_reportable": not errors,
            "reportability_errors": errors,
            "selectable_missing_label_count": 0,
            "rows_with_any_selectable_missing_label": 0,
            "requested_action_missing_label_count": 0 if requested_actions is not None else None,
            "executed_action_missing_label_count": 0 if executed_actions is not None else None,
            "policy_unscorable_rows": 0 if requested_actions is not None else None,
        }

    action_returns = split.action_returns.detach().cpu()
    selected_returns = action_returns[rows]
    if split.action_valid_mask is None:
        decision_valid = torch.ones((rows.numel(), action_returns.shape[1]), dtype=torch.bool)
    else:
        decision_valid = split.action_valid_mask.detach().cpu()[rows].bool()
    if split.label_valid_mask is None:
        label_valid = torch.isfinite(selected_returns)
    else:
        label_valid = split.label_valid_mask.detach().cpu()[rows].bool()
    finite_returns = torch.isfinite(selected_returns)
    label_ok = label_valid & finite_returns
    non_cash = torch.ones(decision_valid.shape[1], dtype=torch.bool)
    if 0 <= int(cash_index) < non_cash.shape[0]:
        non_cash[int(cash_index)] = False
    selectable_missing = decision_valid & non_cash.unsqueeze(0) & ~label_ok
    selectable_missing_count = int(selectable_missing.sum().item())
    rows_with_missing = int(selectable_missing.any(dim=1).sum().item())

    requested_missing_count: int | None = None
    if requested_actions is not None:
        requested = torch.as_tensor(requested_actions, dtype=torch.long).detach().cpu()
        requested_missing_count = 0
        for position, action_value in enumerate(requested.tolist()):
            if position >= rows.numel() or action_value == int(cash_index):
                continue
            if not (0 <= int(action_value) < label_ok.shape[1]):
                requested_missing_count += 1
                continue
            requested_missing_count += int(not bool(label_ok[position, int(action_value)].item()))

    executed_missing_count: int | None = None
    if executed_actions is not None:
        executed = torch.as_tensor(executed_actions, dtype=torch.long).detach().cpu()
        executed_missing_count = 0
        for position, action_value in enumerate(executed.tolist()):
            if position >= rows.numel() or action_value == int(cash_index):
                continue
            if not (0 <= int(action_value) < label_ok.shape[1]):
                executed_missing_count += 1
                continue
            executed_missing_count += int(not bool(label_ok[position, int(action_value)].item()))

    if selectable_missing_count > 0:
        errors.append("selectable_actions_with_missing_reward_labels")
    if requested_missing_count:
        errors.append("requested_actions_with_missing_reward_labels")
    errors = list(dict.fromkeys(errors))
    return {
        "evaluation_reportable": not errors,
        "reportability_errors": errors,
        "selectable_missing_label_count": selectable_missing_count,
        "rows_with_any_selectable_missing_label": rows_with_missing,
        "requested_action_missing_label_count": requested_missing_count,
        "executed_action_missing_label_count": executed_missing_count,
        "policy_unscorable_rows": requested_missing_count,
    }


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
        action_feature_dim: int = 0,
        transition_feature_dim: int = 0,
        transition_table: torch.Tensor | None = None,
        dynamic_feature_dim: int = 0,
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
        self.action_feature_dim = int(action_feature_dim)
        self.transition_feature_dim = int(transition_feature_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
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
        if self.action_feature_dim > 0:
            self.action_id_embedding = nn.Embedding(action_count, d_model)
            self.action_feature_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.action_feature_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, 1),
            )
        else:
            self.action_id_embedding = None
            self.action_feature_encoder = None
            self.action_feature_head = None
        # Position-aware transition features (opt-in). When enabled, a static [A, A, F] table of
        # (previous_action, candidate_action) features is gathered by previous_action id inside forward
        # and fed per-candidate into the Q head. The encoders are ZERO-INITIALISED so a freshly-built
        # transition-aware model scores identically to the pre-feature model until trained (and a model
        # built with transition_feature_dim=0 has no new params at all -> existing checkpoints load strict).
        if self.transition_feature_dim > 0:
            if transition_table is None:
                raise ValueError("transition_feature_dim > 0 requires a transition_table [A, A, F].")
            self.register_buffer("transition_table", transition_table.float())
            if self.action_feature_dim > 0:
                self.transition_encoder = nn.Sequential(
                    nn.Linear(self.transition_feature_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                nn.init.zeros_(self.transition_encoder[0].weight)
                nn.init.zeros_(self.transition_encoder[0].bias)
                self.transition_bias = None
            else:
                # Fallback head emits Q[B, A] from one Linear over context, with no per-candidate tokens
                # to add to; inject transition awareness as an additive per-candidate [B, A] bias instead.
                self.transition_encoder = None
                self.transition_bias = nn.Linear(self.transition_feature_dim, 1)
                nn.init.zeros_(self.transition_bias.weight)
                nn.init.zeros_(self.transition_bias.bias)
        else:
            self.transition_table = None
            self.transition_encoder = None
            self.transition_bias = None
        # Dynamic position-state features (opt-in, PR-D). A per-env [B, dynamic_feature_dim] vector of the
        # HELD position's realized-P&L excursion is passed into forward() and injected per-candidate
        # (broadcast across candidates, since it is position-level not candidate-level). Encoders are
        # ZERO-INITIALISED so a freshly-built dynamic-aware model scores identically until trained; and
        # dynamic_feature_dim=0 registers no params at all -> existing checkpoints load strict.
        if self.dynamic_feature_dim > 0:
            # Build the (zero-init) dynamic submodule WITHOUT perturbing the construction RNG of the rest of
            # the network (hour_encoder/head are built below). Its random init is immediately overwritten by
            # zeros_, so saving/restoring the RNG makes the shared backbone's init identical whether the flag
            # is off or on -> a freshly built dynamic-aware model is a CLEAN perturbation of the non-dynamic
            # one (same backbone init + zero-init dynamic head => identical until trained), so the D4 A/B
            # isolates the feature rather than a different random initialization.
            _rng_state = torch.get_rng_state()
            if self.action_feature_dim > 0:
                self.dynamic_encoder = nn.Sequential(
                    nn.Linear(self.dynamic_feature_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                nn.init.zeros_(self.dynamic_encoder[0].weight)
                nn.init.zeros_(self.dynamic_encoder[0].bias)
                self.dynamic_bias = None
            else:
                self.dynamic_encoder = None
                self.dynamic_bias = nn.Linear(self.dynamic_feature_dim, 1)
                nn.init.zeros_(self.dynamic_bias.weight)
                nn.init.zeros_(self.dynamic_bias.bias)
            torch.set_rng_state(_rng_state)
        else:
            self.dynamic_encoder = None
            self.dynamic_bias = None
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

    def _transition_rows(self, previous_actions: torch.Tensor) -> torch.Tensor:
        # Gather the [B, A, F] per-candidate transition features for the held positions. Validate the
        # ids up front so an out-of-range previous_action raises a clear error instead of silently
        # wrapping (negative index) or tripping a cryptic CUDA assert deep in the gather.
        ids = previous_actions.long()
        if bool(((ids < 0) | (ids >= self.action_count)).any().item()):
            raise ValueError(
                f"previous_actions must be valid action ids in [0, {self.action_count}); "
                "got an out-of-range id (CASH=0 is the expected reset state)."
            )
        return self.transition_table[ids]

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
        action_features: torch.Tensor | None = None,
        dynamic_state: torch.Tensor | None = None,
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
        context = encoded[:, -1, :]
        if self.action_feature_encoder is None:
            out = self.head(context)
            if self.transition_bias is not None:
                # Per-candidate transition bias gathered by the held position id (zero at init).
                out = out + self.transition_bias(self._transition_rows(previous_actions)).squeeze(-1)
            if self.dynamic_bias is not None and dynamic_state is not None:
                # Per-env dynamic position-state bias, broadcast across candidates ([B,1]->[B,A]; zero at init).
                out = out + self.dynamic_bias(dynamic_state.float())
            return out
        if action_features is None:
            raise ValueError("Model was configured with action_feature_dim > 0 but action_features were not provided.")
        if action_features.shape[1] != self.action_count or action_features.shape[2] != self.action_feature_dim:
            raise ValueError("action_features shape does not match configured action count/feature dimension.")
        action_ids = torch.arange(self.action_count, device=action_features.device)
        action_tokens = self.action_feature_encoder(action_features.float())
        action_tokens = action_tokens + self.action_id_embedding(action_ids)[None, :, :]
        if self.transition_encoder is not None:
            # Add a per-candidate token encoding the cost/risk of moving from the held position
            # (previous_actions) to each candidate. Gathered from the static table; zero at init.
            action_tokens = action_tokens + self.transition_encoder(self._transition_rows(previous_actions))
        if self.dynamic_encoder is not None and dynamic_state is not None:
            # Add the held position's dynamic state (P&L excursion) as a per-env token, broadcast across
            # candidates ([B,d_model]->[B,1,d_model]); zero at init so an untrained dynamic model is identical.
            action_tokens = action_tokens + self.dynamic_encoder(dynamic_state.float())[:, None, :]
        q_tokens = context[:, None, :] + action_tokens
        return self.action_feature_head(q_tokens).squeeze(-1)


@dataclass
class MinuteToHourEnvConfig:
    num_envs: int
    episode_length: int
    reward_scale: float = 10_000.0
    initial_action: int = 0
    cash_idle_penalty_bps: float = 0.0
    constraints: TradingConstraintConfig = field(default_factory=default_minute_to_hour_constraints)


@dataclass
class RecencyWeightConfig:
    """Recency-focus weighting of TRAINING transitions. ``mode='none'`` -> uniform (default).

    With ``mode='exponential'`` a training row with decision timestamp ``t`` gets weight
        ``min_weight + (1 - min_weight) * exp(-ln2 * age_days / half_life_days)``
    where ``age_days`` is measured relative to the VALIDATION start (never the test block), so
    older training rows are down-weighted toward the recent pre-validation regime but never fully
    ignored (weight stays >= ``min_weight``). The test split is never passed to the trainer, so
    recency weighting is structurally incapable of touching it.
    """

    mode: str = "none"
    half_life_days: float = 120.0
    min_weight: float = 0.05


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
    resume_training_state: str | bytes | PathLike[str] | None = None
    checkpoint_training_state: str | bytes | PathLike[str] | None = None
    checkpoint_every_steps: int = 0
    max_subhour_tokens: int | None = DEFAULT_MAX_SUBHOUR_TOKENS
    recency: RecencyWeightConfig = field(default_factory=RecencyWeightConfig)
    # Opt-in position-aware transition features (default off -> no new model params, existing
    # checkpoints load unchanged). When True, the Q-network scores each candidate with the cost/risk of
    # moving from the held position to it (see build_transition_feature_table).
    use_transition_features: bool = False
    # Opt-in PR-D dynamic position-state features (default off -> byte-identical, existing checkpoints load
    # strict). When True, the Q-network also scores each candidate with the HELD position's realized-P&L
    # excursion (unrealized_pnl / MAE / MFE / drawdown / runup), threaded from the env through replay. This
    # MOVES training numbers when on, so it ships behind this flag and flips to default only after a
    # latest-period A/B (no default flip here).
    use_dynamic_transition_features: bool = False


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
        # PR-D / D0 dynamic position bookkeeping: maintained as internal env state but NOT yet consumed by
        # the reward, the model forward, the replay buffer, or the step() dict -- so training is byte-identical
        # (see pr_d_dynamic_state_design.md). This is a RETURN-based env (no prices/target weights), so we track
        # the entry row, the compounded return since entry, and the max adverse/favorable excursion since entry.
        self.entry_index = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.unrealized_pnl = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        self.mae = torch.zeros(config.num_envs, dtype=torch.float32, device=device)  # max adverse excursion (<= 0)
        self.mfe = torch.zeros(config.num_envs, dtype=torch.float32, device=device)  # max favorable excursion (>= 0)
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
        # D0 dynamic bookkeeping resets with the episode (entry starts at the freshly-drawn start row).
        self.entry_index[mask] = self.indices[mask]
        self.unrealized_pnl[mask] = 0.0
        self.mae[mask] = 0.0
        self.mfe[mask] = 0.0

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

    def action_mask(self, row_indices: torch.Tensor | None = None) -> torch.Tensor:
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
        if row_indices is None:
            row_indices = self.indices
        if row_indices.shape != self.previous_actions.shape:
            raise ValueError(
                "row_indices must have the same shape as the vectorized environment state; "
                f"got {tuple(row_indices.shape)} and {tuple(self.previous_actions.shape)}."
            )
        row_count = int(self.data.action_returns.shape[0])
        in_bounds = (row_indices >= 0) & (row_indices < row_count)
        safe_indices = row_indices.clamp(0, max(row_count - 1, 0))
        availability_mask = self.data.valid_actions(safe_indices)
        if bool((~in_bounds).any().item()):
            availability_mask[~in_bounds] = False
        availability_mask[:, int(self.config.constraints.cash_index)] = True
        mask = constraint_mask & availability_mask
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any().item()):
            mask[empty_rows, int(self.config.constraints.cash_index)] = True
        return mask

    def observe(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        minute, mask, hour = self.data.state(self.indices)
        return (
            minute,
            mask,
            hour,
            self.data.action_feature_state(self.indices),
            self.previous_actions,
            self.constraint_features(),
            self.action_mask(),
        )

    def dynamic_state(self) -> torch.Tensor:
        """Per-env [B, DYNAMIC_TRANSITION_FEATURE_DIM] dynamic position-state features (PR-D) of the position
        held entering the current decision. Fed to the Q-network only when use_dynamic_transition_features."""
        return build_dynamic_transition_features(
            unrealized_pnl=self.unrealized_pnl, mae=self.mae, mfe=self.mfe
        )

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        current_indices = self.indices.clone()
        previous_actions = self.previous_actions.clone()
        # PR-D: snapshot the dynamic state of the position held ENTERING this decision (before the update
        # below). Always computed and returned (the replay add() filters unknown keys, so it is harmless when
        # use_dynamic_transition_features is off); consumed only when the flag is on.
        position_dynamic = self.dynamic_state()
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
        cash_idle_penalty_bps = (
            (actions == int(self.config.constraints.cash_index)).float() * float(self.config.cash_idle_penalty_bps)
        )
        rewards = raw_returns * float(self.config.reward_scale) - (
            cost_bps + cash_idle_penalty_bps
        ) * float(self.config.reward_scale) / 10_000.0

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
        # D0 dynamic bookkeeping (computed from existing state only; not fed to reward/model/replay): on a HOLD
        # compound this step's return into the position held since entry and extend MAE/MFE; on a SWITCH the new
        # position starts fresh this step (entry row = the current decision row). Held across a day boundary is
        # still held, so unlike the daily switch/leg counters below, this state is NOT reset on a new day.
        held = ~is_switch
        cum = torch.where(held, (1.0 + self.unrealized_pnl) * (1.0 + raw_returns) - 1.0, raw_returns)
        zeros = torch.zeros_like(cum)
        self.entry_index = torch.where(is_switch, current_indices, self.entry_index)
        self.mae = torch.where(held, torch.minimum(self.mae, cum), torch.minimum(zeros, cum))
        self.mfe = torch.where(held, torch.maximum(self.mfe, cum), torch.maximum(zeros, cum))
        self.unrealized_pnl = cum
        next_position_dynamic = self.dynamic_state()  # PR-D: post-action dynamic state (enters the next bar)

        in_bounds = next_indices < self.data.action_returns.shape[0]
        next_valid = torch.zeros_like(in_bounds)
        if bool(in_bounds.any().item()):
            next_valid[in_bounds] = self.data.valid_index_mask[next_indices[in_bounds]]
        # Distinguish a TRUE terminal (no valid next row -> nothing to bootstrap from) from a mere
        # episode-length TRUNCATION (a rollout boundary whose next row is a real continuation). DQN
        # must bootstrap through truncations; only `terminated` may zero the TD bootstrap. `resets`
        # ends the episode (terminal OR truncation) and drives env reset (matches strategy/intraday).
        terminated = ~next_valid
        truncated = self.steps >= int(self.config.episode_length)
        resets = terminated | truncated
        if bool(in_bounds.any().item()):
            old_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in current_indices[in_bounds].detach().cpu()]
            new_dates = [self.data.decision_timestamps[int(i.item())][:10] for i in next_indices[in_bounds].detach().cpu()]
            reset_today = torch.tensor([old != new for old, new in zip(old_dates, new_dates)], dtype=torch.bool, device=self.device)
            valid_positions = torch.where(in_bounds)[0]
            self.switches_today[valid_positions[reset_today]] = 0
            self.order_legs_today[valid_positions[reset_today]] = 0.0

        next_constraint_features = self.constraint_features()
        next_action_mask = self.action_mask(next_indices)
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
            "resets": resets.float(),
            "terminated": terminated.float(),
            "legs": legs,
            "position_dynamic": position_dynamic,
            "next_position_dynamic": next_position_dynamic,
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
    evaluation_reportable: bool = True
    reportability_errors: list[str] = field(default_factory=list)
    selectable_missing_label_count: int = 0
    rows_with_any_selectable_missing_label: int = 0
    requested_action_missing_label_count: int = 0
    executed_action_missing_label_count: int = 0
    policy_unscorable_rows: int = 0

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
            "evaluation_reportable": self.evaluation_reportable,
            "reportability_errors": self.reportability_errors,
            "selectable_missing_label_count": self.selectable_missing_label_count,
            "rows_with_any_selectable_missing_label": self.rows_with_any_selectable_missing_label,
            "requested_action_missing_label_count": self.requested_action_missing_label_count,
            "executed_action_missing_label_count": self.executed_action_missing_label_count,
            "policy_unscorable_rows": self.policy_unscorable_rows,
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
    evaluated_rows: list[int] = []
    requested_actions: list[int] = []
    executed_actions: list[int] = []
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
        action_features = data.action_feature_state(torch.tensor([index], dtype=torch.long, device=device))
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
        if action_features is None:
            q_values = model(minute, mask, hour, prev_tensor, constraints_tensor)
        else:
            q_values = model(minute, mask, hour, prev_tensor, constraints_tensor, action_features=action_features)
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
        requested_action = action
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        label_mask = data.label_valid_actions(torch.tensor([index], dtype=torch.long, device=device))
        requested_label_missing = (
            action != int(constraints.cash_index)
            and (not bool(label_mask[0, action].item()) or not bool(torch.isfinite(data.action_returns[index, action]).item()))
        )
        if not bool(label_mask[0, action].item()) or not bool(torch.isfinite(data.action_returns[index, action]).item()):
            action = int(constraints.cash_index)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        evaluated_rows.append(int(index))
        requested_actions.append(int(requested_action))
        executed_actions.append(int(action))
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
                    "requested_action": requested_action,
                    "executed_action": action,
                    "asset": data.action_names[action],
                    "requested_asset": data.action_names[requested_action],
                    "executed_asset": data.action_names[action],
                    "previous_action": previous_action,
                    "segment_reset": int(segment_reset),
                    "fallback_due_to_missing_label": int(requested_label_missing),
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

    report = minute_to_hour_missing_label_report(
        data,
        row_indices=evaluated_rows,
        requested_actions=requested_actions,
        executed_actions=executed_actions,
        cash_index=int(constraints.cash_index),
    )
    return MinuteToHourEvaluationResult(
        split_name=data.name,
        total_return=equity - 1.0,
        total_reward_bps=total_reward_bps,
        allocation_switches=allocation_switches,
        market_order_legs=order_legs,
        max_drawdown=fractional_max_drawdown(equity_curve),
        annualized_sharpe=annualized_sharpe(returns, periods_per_year=data.periods_per_year),
        rollout_records=records,
        evaluation_reportable=bool(report["evaluation_reportable"]),
        reportability_errors=list(report["reportability_errors"]),
        selectable_missing_label_count=int(report["selectable_missing_label_count"]),
        rows_with_any_selectable_missing_label=int(report["rows_with_any_selectable_missing_label"]),
        requested_action_missing_label_count=int(report["requested_action_missing_label_count"] or 0),
        executed_action_missing_label_count=int(report["executed_action_missing_label_count"] or 0),
        policy_unscorable_rows=int(report["policy_unscorable_rows"] or 0),
    )


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _tensor_dict_to_cpu(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return move(optimizer.state_dict())


def _replay_state_to_cpu(replay: TensorDictReplayBuffer) -> dict[str, object]:
    return {
        "capacity": int(replay.capacity),
        "size": int(replay.size),
        "cursor": int(replay.cursor),
        "storage": _tensor_dict_to_cpu(replay.storage),
    }


def _load_replay_state(replay: TensorDictReplayBuffer, state: dict[str, object], device: torch.device) -> None:
    if int(state.get("capacity", -1)) != int(replay.capacity):
        raise ValueError("Resume checkpoint replay capacity does not match the current training config.")
    storage = state.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("Resume checkpoint is missing replay storage.")
    if set(storage) != set(replay.storage):
        raise ValueError("Resume checkpoint replay fields do not match the current training config.")
    for key, target in replay.storage.items():
        value = storage[key]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Resume checkpoint replay field {key!r} has an incompatible shape.")
        target.copy_(value.to(device=device, dtype=target.dtype))
    replay.size = int(state.get("size", 0))
    replay.cursor = int(state.get("cursor", 0))


def _env_state_to_cpu(env: VectorizedMinuteToHourEnv) -> dict[str, torch.Tensor]:
    return {
        "indices": env.indices.detach().cpu().clone(),
        "previous_actions": env.previous_actions.detach().cpu().clone(),
        "bars_held": env.bars_held.detach().cpu().clone(),
        "cooldown_remaining": env.cooldown_remaining.detach().cpu().clone(),
        "switches_today": env.switches_today.detach().cpu().clone(),
        "switches_episode": env.switches_episode.detach().cpu().clone(),
        "order_legs_today": env.order_legs_today.detach().cpu().clone(),
        "order_legs_episode": env.order_legs_episode.detach().cpu().clone(),
        "steps": env.steps.detach().cpu().clone(),
    }


def _load_env_state(env: VectorizedMinuteToHourEnv, state: dict[str, torch.Tensor], device: torch.device) -> None:
    for key in _env_state_to_cpu(env):
        value = state.get(key)
        target = getattr(env, key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Resume checkpoint env field {key!r} has an incompatible shape.")
        target.copy_(value.to(device=device, dtype=target.dtype))


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object], device: torch.device) -> None:
    torch_rng_state = state.get("torch_rng_state")
    if torch.is_tensor(torch_rng_state):
        torch.set_rng_state(torch_rng_state.cpu())
    cuda_state = state.get("cuda_rng_state_all")
    if device.type == "cuda" and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all([item.cpu() if torch.is_tensor(item) else item for item in cuda_state])


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _save_minute_to_hour_training_state(
    path: Path,
    *,
    step: int,
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: TensorDictReplayBuffer,
    env: VectorizedMinuteToHourEnv,
    best_val_return: float,
    best_val_legs: float,
    best_state: dict[str, torch.Tensor],
    loss_trace: list[float],
    reward_trace: list[float],
    valid_action_count_trace: list[float],
    eval_trace: list[dict[str, float | int | None | str]],
    device: torch.device,
) -> None:
    _atomic_torch_save(
        {
            "checkpoint_kind": "minute_to_hour_dqn_training_state",
            "checkpoint_version": 1,
            "step": int(step),
            "q_network_state_dict": _state_dict_to_cpu(q_network),
            "target_network_state_dict": _state_dict_to_cpu(target_network),
            "optimizer_state_dict": _optimizer_state_to_cpu(optimizer),
            "scaler_state_dict": scaler.state_dict(),
            "replay": _replay_state_to_cpu(replay),
            "env": _env_state_to_cpu(env),
            "best_val_return": float(best_val_return),
            "best_val_legs": float(best_val_legs),
            "best_state": _tensor_dict_to_cpu(best_state),
            "loss_trace": list(loss_trace),
            "train_reward_trace": list(reward_trace),
            "valid_action_count_trace": list(valid_action_count_trace),
            "eval_trace": list(eval_trace),
            "rng_state": _capture_rng_state(device),
        },
        path,
    )


def _assert_checkpoint_schema(
    checkpoint: dict[str, Any],
    *,
    minute_feature_names: list[str],
    hour_feature_names: list[str],
    action_names: list[str],
    action_feature_names: list[str],
    transition_feature_dim: int = 0,
    dynamic_feature_dim: int = 0,
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

    checkpoint_action_features = checkpoint.get("action_feature_names", [])
    if action_feature_names or checkpoint_action_features:
        if list(checkpoint_action_features) != list(action_feature_names):
            raise ValueError("Warm-start checkpoint action_feature_names does not match the current dataset schema.")

    constraint_names = checkpoint.get("constraint_feature_names")
    if constraint_names is None:
        raise ValueError("Warm-start checkpoint is missing constraint_feature_names; refusing unverified fine-tune.")
    if list(constraint_names) != list(CONSTRAINT_FEATURE_NAMES):
        raise ValueError("Warm-start checkpoint constraint feature schema does not match current code.")

    # Transition (position-aware) schema must match the model being warm-started into, in BOTH
    # directions: a v3 transition checkpoint cannot load a v2 (transition-off) model and vice versa,
    # and a schema-version/name drift is rejected with a clear message rather than a cryptic strict-load
    # state_dict error.
    expected_transition = list(TRANSITION_FEATURE_NAMES) if transition_feature_dim > 0 else []
    checkpoint_transition = list(checkpoint.get("transition_feature_names", []))
    if checkpoint_transition != expected_transition:
        raise ValueError(
            "Warm-start checkpoint transition feature schema does not match the current model "
            f"(use_transition_features mismatch or schema drift): checkpoint={checkpoint_transition}, "
            f"expected={expected_transition}."
        )

    # Same bidirectional guard for the PR-D dynamic position-state schema: a dynamic-aware checkpoint
    # (wider input) cannot warm-start a non-dynamic model and vice versa.
    expected_dynamic = list(DYNAMIC_TRANSITION_FEATURE_NAMES) if dynamic_feature_dim > 0 else []
    checkpoint_dynamic = list(checkpoint.get("dynamic_transition_feature_names", []))
    if checkpoint_dynamic != expected_dynamic:
        raise ValueError(
            "Warm-start checkpoint dynamic transition feature schema does not match the current model "
            f"(use_dynamic_transition_features mismatch or schema drift): checkpoint={checkpoint_dynamic}, "
            f"expected={expected_dynamic}."
        )


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
        action_feature_names=train_data.action_feature_names,
        transition_feature_dim=int(getattr(model, "transition_feature_dim", 0)),
        dynamic_feature_dim=int(getattr(model, "dynamic_feature_dim", 0)),
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
        "uses_transition_features": checkpoint.get("uses_transition_features"),
    }


def compute_recency_weights(
    decision_timestamps: list[str],
    validation_start_ms: int,
    *,
    mode: str,
    half_life_days: float,
    min_weight: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-row recency weights for training transitions (see :class:`RecencyWeightConfig`).

    Ages are clamped at 0 and measured against ``validation_start_ms`` (the earliest validation
    decision), so a training row is weighted only by how far it sits BEFORE validation -- the test
    block is never referenced. Returns a float32 tensor of shape ``(len(decision_timestamps),)``.
    ``mode='none'`` returns all ones (so a weighted mean is identical to an unweighted mean).
    """
    count = len(decision_timestamps)
    if mode == "none":
        return torch.ones(count, dtype=torch.float32, device=device)
    if mode != "exponential":
        raise ValueError(f"Unsupported recency weighting mode: {mode!r}")
    if half_life_days <= 0.0:
        raise ValueError("recency half_life_days must be positive.")
    if not 0.0 < min_weight <= 1.0:
        # A zero floor lets a batch of only-old rows collapse to ~0 weight (unstable loss scale after
        # the clamp_min denominator), and contradicts "older regimes are never fully ignored".
        raise ValueError("recency min_weight must be in (0, 1].")
    day_ms = 86_400_000.0
    decay = math.log(2.0) / float(half_life_days)
    weights = torch.empty(count, dtype=torch.float32, device=device)
    for index, timestamp in enumerate(decision_timestamps):
        age_days = max(0.0, (validation_start_ms - _timestamp_to_epoch_ms(timestamp)) / day_ms)
        weights[index] = min_weight + (1.0 - min_weight) * math.exp(-decay * age_days)
    return weights


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
    # Recency weighting is anchored to the earliest VALIDATION decision; the test split is never
    # passed to this function, so older training rows can be down-weighted without any risk of
    # touching the held-out test block. mode='none' yields uniform weights (no behavior change).
    validation_start_ms = (
        min(_timestamp_to_epoch_ms(ts) for ts in val_data.decision_timestamps)
        if val_data.decision_timestamps
        else None
    )
    recency_mode = config.recency.mode if validation_start_ms is not None else "none"
    if recency_mode != "none" and validation_start_ms is not None and train_data.decision_timestamps:
        # Recency weighting is precisely where the train/validation boundary should be re-asserted:
        # a training row at/after validation start would silently get weight 1.0 (age clamped to 0)
        # and mask an upstream split bug. Fail loudly instead.
        train_max_ms = max(_timestamp_to_epoch_ms(ts) for ts in train_data.decision_timestamps)
        if train_max_ms >= validation_start_ms:
            raise ValueError(
                "train split overlaps validation start; refusing recency-weighted training "
                f"(train_max_ms={train_max_ms} >= validation_start_ms={validation_start_ms})."
            )
    train_recency_weights = compute_recency_weights(
        train_data.decision_timestamps,
        validation_start_ms or 0,
        mode=recency_mode,
        half_life_days=config.recency.half_life_days,
        min_weight=config.recency.min_weight,
        device=device,
    )
    recency_active = recency_mode != "none"
    action_count = len(train_data.action_names)
    transition_feature_dim = 0
    transition_table = None
    if config.use_transition_features:
        from rl_quant.action_risk import action_leverage_tensor, build_action_metadata, group_ids_for_actions

        action_meta = build_action_metadata(train_data.action_names)
        action_group_ids, _ = group_ids_for_actions(action_meta, device=device)
        cons = config.env.constraints
        # Use the env's cash_index / leg convention so the table's legs/cost columns match realized cost.
        transition_table = build_transition_feature_table(
            action_count=action_count,
            cash_index=int(cons.cash_index),
            one_way_cost_bps=cons.one_way_cost_bps,
            extra_switch_penalty_bps=cons.extra_switch_penalty_bps,
            count_etf_to_etf_as_two_legs=cons.count_etf_to_etf_as_two_legs,
            action_leverage=action_leverage_tensor(action_meta, device=device),
            action_group_ids=action_group_ids,
            device=device,
        )
        transition_feature_dim = TRANSITION_FEATURE_DIM
    dynamic_feature_dim = DYNAMIC_TRANSITION_FEATURE_DIM if config.use_dynamic_transition_features else 0
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
        action_feature_dim=0 if train_data.action_features is None else int(train_data.action_features.shape[-1]),
        transition_feature_dim=transition_feature_dim,
        transition_table=transition_table,
        dynamic_feature_dim=dynamic_feature_dim,
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
    scaler = make_grad_scaler(device, config.learning.use_amp, config.learning.amp_dtype)
    replay_fields = {
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
        "terminated": ((), torch.float32),
    }
    if config.use_dynamic_transition_features:
        # Only declared when the flag is on -> storage has exactly the 11 legacy keys otherwise, so the
        # buffer (and a flag-off resume) is byte-identical. The step dict always carries these keys; the
        # add() call below filters to declared fields, so they are silently dropped when the flag is off.
        replay_fields["position_dynamic"] = ((DYNAMIC_TRANSITION_FEATURE_DIM,), torch.float32)
        replay_fields["next_position_dynamic"] = ((DYNAMIC_TRANSITION_FEATURE_DIM,), torch.float32)
    replay = TensorDictReplayBuffer(
        capacity=config.learning.replay_capacity,
        device=device,
        fields=replay_fields,
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
    resume_info: dict[str, object] = {"loaded": False}
    start_step = 1
    resume_path = Path(config.resume_training_state) if config.resume_training_state is not None else None
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_kind") != "minute_to_hour_dqn_training_state":
            raise ValueError("Resume checkpoint is not a minute-to-hour DQN training state.")
        q_network.load_state_dict(checkpoint["q_network_state_dict"], strict=True)
        target_network.load_state_dict(checkpoint["target_network_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        replay_state = checkpoint.get("replay")
        env_state = checkpoint.get("env")
        if not isinstance(replay_state, dict) or not isinstance(env_state, dict):
            raise ValueError("Resume checkpoint is missing replay or environment state.")
        _load_replay_state(replay, replay_state, device)
        _load_env_state(env, env_state, device)
        best_val_return = float(checkpoint.get("best_val_return", best_val_return))
        best_val_legs = float(checkpoint.get("best_val_legs", best_val_legs))
        raw_best_state = checkpoint.get("best_state")
        if not isinstance(raw_best_state, dict):
            raise ValueError("Resume checkpoint is missing best_state.")
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in raw_best_state.items()
            if torch.is_tensor(value)
        }
        loss_trace = [float(item) for item in checkpoint.get("loss_trace", [])]
        reward_trace = [float(item) for item in checkpoint.get("train_reward_trace", [])]
        valid_action_count_trace = [float(item) for item in checkpoint.get("valid_action_count_trace", [])]
        eval_trace = list(checkpoint.get("eval_trace", []))
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state, device)
        resumed_step = int(checkpoint.get("step", 0))
        start_step = min(resumed_step + 1, config.learning.train_steps + 1)
        resume_info = {
            "loaded": True,
            "path": str(resume_path),
            "resumed_from_step": resumed_step,
            "start_step": start_step,
        }

    checkpoint_path = Path(config.checkpoint_training_state) if config.checkpoint_training_state is not None else None
    checkpoint_every_steps = max(0, int(config.checkpoint_every_steps))
    for step in range(start_step, config.learning.train_steps + 1):
        minute, mask, hour, action_features, previous_actions, constraint_features, action_mask = env.observe()
        valid_action_count_trace.append(float(action_mask.sum(dim=1).float().mean().item()))
        epsilon = epsilon_by_step(
            step=step,
            train_steps=config.learning.train_steps,
            start=config.learning.epsilon_start,
            end=config.learning.epsilon_end,
        )
        with torch.no_grad():
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q_values = q_network(
                    minute,
                    mask,
                    hour,
                    previous_actions,
                    constraint_features,
                    action_features=action_features,
                    dynamic_state=env.dynamic_state() if config.use_dynamic_transition_features else None,
                )
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
        env.reset(transition["resets"].bool())

        if replay.size >= max(config.learning.warmup_steps, config.learning.batch_size):
            batch = replay.sample(config.learning.batch_size)
            # Clamp next_indices for the state lookup: a TRUE terminal transition can store an
            # out-of-data next row, whose bootstrapped value is zeroed below via `terminated` anyway.
            # For non-terminal transitions next_indices is always in range, so this is a no-op there.
            n_rows = int(train_data.action_returns.shape[0])
            # min_index = 0: state() is plain row indexing (each row carries its own self-contained
            # window), so there is no rolling-window floor to respect and no tail-wrap to avoid.
            # valid_index_mask is the same tensor the env uses to DEFINE terminated (terminated =
            # ~valid_index_mask[next]), so every non-terminal next is mask-True by construction --
            # passing it rejects nothing legitimate and turns a mask/terminated mismatch into a loud error.
            safe_next_indices = safe_next_row_indices(
                batch["next_indices"],
                batch["terminated"],
                min_index=0,
                max_index=n_rows - 1,
                valid_index_mask=train_data.valid_index_mask,
            )
            current_minute, current_mask, current_hour = train_data.state(batch["indices"])
            next_minute, next_mask, next_hour = train_data.state(safe_next_indices)
            current_action_features = train_data.action_feature_state(batch["indices"])
            next_action_features = train_data.action_feature_state(safe_next_indices)
            with autocast_context(device, config.learning.use_amp, config.learning.amp_dtype):
                q = q_network(
                    current_minute,
                    current_mask,
                    current_hour,
                    batch["previous_actions"],
                    batch["constraint_features"],
                    action_features=current_action_features,
                    dynamic_state=batch.get("position_dynamic"),
                )
                chosen_q = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online = q_network(
                        next_minute,
                        next_mask,
                        next_hour,
                        batch["next_previous_actions"],
                        batch["next_constraint_features"],
                        action_features=next_action_features,
                        dynamic_state=batch.get("next_position_dynamic"),
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
                        action_features=next_action_features,
                        dynamic_state=batch.get("next_position_dynamic"),
                    )
                    next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    # fp32 TD target/loss under AMP: with reward_scale=10_000 the bootstrapped
                    # targets reach magnitudes where fp16 precision is comparable to per-step
                    # rewards, so compute the target and smooth_l1 loss in float32.
                    # Bootstrap through episode-length TRUNCATIONS; zero the bootstrap only on TRUE
                    # terminals. Shared with the hourly trainer via core.dqn_td_target.
                    target_q = dqn_td_target(batch["rewards"], config.learning.gamma, batch["terminated"], next_q)
                if recency_active:
                    # Recency-weighted smooth_l1: per-sample loss scaled by each transition's source
                    # training row weight (looked up via the replay-stored decision-row `indices`).
                    per_sample_loss = F.smooth_l1_loss(chosen_q.float(), target_q, reduction="none")
                    sample_weights = train_recency_weights[batch["indices"]]
                    loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
                else:
                    # Default (uniform) path: identical fused mean reduction as before recency support,
                    # so disabling recency is a byte-for-byte no-op on the training objective.
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
            val_result = evaluate_minute_to_hour_policy(
                val_data,
                q_network,
                device=device,
                initial_action=config.env.initial_action,
                constraints=config.env.constraints,
                episode_length=config.env.episode_length,
                reward_scale=config.env.reward_scale,
            )
            # Restore train() mode: the evaluator puts the shared q_network in eval(), which would
            # otherwise leave dropout disabled for all subsequent gradient steps.
            q_network.train()
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
        if checkpoint_path is not None and checkpoint_every_steps > 0 and (
            step % checkpoint_every_steps == 0 or step == config.learning.train_steps
        ):
            _save_minute_to_hour_training_state(
                checkpoint_path,
                step=step,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                scaler=scaler,
                replay=replay,
                env=env,
                best_val_return=best_val_return,
                best_val_legs=best_val_legs,
                best_state=best_state,
                loss_trace=loss_trace,
                reward_trace=reward_trace,
                valid_action_count_trace=valid_action_count_trace,
                eval_trace=eval_trace,
                device=device,
            )

    q_network.load_state_dict(best_state)
    recency_policy: dict[str, object] = {
        "mode": recency_mode,
        "half_life_days": config.recency.half_life_days,
        "min_weight": config.recency.min_weight,
        "validation_start_ms": validation_start_ms,
        # The trainer only ever receives train_data + val_data; the test split is never visible here.
        "test_used_for_recency_selection": False,
    }
    if train_recency_weights.numel() > 0:
        recency_policy["weight_min"] = float(train_recency_weights.min().item())
        recency_policy["weight_max"] = float(train_recency_weights.max().item())
        recency_policy["weight_mean"] = float(train_recency_weights.mean().item())
    artifacts: dict[str, object] = {
        "best_val_return": best_val_return,
        "best_val_order_legs": best_val_legs,
        "recency_policy": recency_policy,
        "amp_enabled": scaler.is_enabled(),
        "loss_trace": loss_trace,
        "train_reward_trace": reward_trace,
        "valid_action_count_trace": valid_action_count_trace,
        "eval_trace": eval_trace,
        "vram_reservation": reservation.report,
        "cash_idle_penalty_bps": float(config.env.cash_idle_penalty_bps),
        "model_version": (
            DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION
            if config.use_dynamic_transition_features
            else POSITION_AWARE_POLICY_MODEL_VERSION
            if config.use_transition_features
            else CONSTRAINED_POLICY_MODEL_VERSION
        ),
        "uses_constraint_features": True,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "uses_transition_features": bool(config.use_transition_features),
        "transition_feature_names": list(TRANSITION_FEATURE_NAMES) if config.use_transition_features else [],
        "transition_feature_dim": TRANSITION_FEATURE_DIM if config.use_transition_features else 0,
        "transition_feature_schema_version": TRANSITION_FEATURE_SCHEMA_VERSION if config.use_transition_features else 0,
        "uses_dynamic_transition_features": bool(config.use_dynamic_transition_features),
        "dynamic_transition_feature_names": (
            list(DYNAMIC_TRANSITION_FEATURE_NAMES) if config.use_dynamic_transition_features else []
        ),
        "dynamic_transition_feature_dim": (
            DYNAMIC_TRANSITION_FEATURE_DIM if config.use_dynamic_transition_features else 0
        ),
        "dynamic_transition_feature_schema_version": (
            DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION if config.use_dynamic_transition_features else 0
        ),
        "warm_start": warm_start_info or {"loaded": False},
        "resume": resume_info,
        "last_completed_step": int(config.learning.train_steps if start_step <= config.learning.train_steps else start_step - 1),
        "checkpoint_training_state": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_every_steps": checkpoint_every_steps,
        "source_bar_interval": train_data.source_bar_interval,
        "context_bars_per_hour": train_data.effective_context_bars_per_hour,
        "max_subhour_tokens": config.max_subhour_tokens,
        "split_policy": train_data.split_policy,
        "action_feature_names": train_data.action_feature_names,
        "action_feature_dim": 0 if train_data.action_features is None else int(train_data.action_features.shape[-1]),
        "action_feature_groups": train_data.action_feature_groups,
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
