"""Datasets layer: hour-from-sub-hour dataset split + builders / validators / split-policy inference (extracted from rl_quant.minute_to_hour_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from rl_quant.datasets.hourly import _validate_action_return_contract
# The action-return basis contract lives in the protocol layer so the loader, the DatasetManifest, and the
# reportability validators share ONE definition. Re-exported here so existing importers of these names from this
# module keep working unchanged (the canonical home is rl_quant.protocol.action_return_basis).
from rl_quant.protocol.action_return_basis import (  # noqa: F401  (re-export for back-compat)
    ALLOWED_ACTION_RETURN_WEIGHT_SEMANTICS,
    ReturnBasis,
    _RETURN_BASIS_FIELD_KEYS,
    _basis_value_differs,
    return_basis_agreement_errors,
)
from rl_quant.trading_constraints import (
    TradingConstraintConfig,
)


DEFAULT_HOUR_DECISION_GRID_MINUTES = 60
DEFAULT_MINUTE_SOURCE_INTERVAL = "1m"
DEFAULT_SECOND_SOURCE_INTERVAL = "1s"
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
    # PR-4 gate (see docs/execution_wiring_design.md §3): the weight basis of action_returns. The execution
    # shadow prices turnover with action_metadata.max_weight, which is correct ONLY for
    # "metadata_weighted_portfolio_returns"; for "full_capital_single_slot_returns" leveraged turnover is
    # undercharged. None / "unresolved" means the gold builder has not declared it, which fail-closes
    # use_execution_env_reward (PR-4) -- it must NOT be trained on until this is resolved AND metadata complete.
    action_return_weight_semantics: str | None = None
    # The rest of the action-return BASIS (recorded by the builder alongside the weight label). Carried so the
    # run-semantics fingerprint can capture the FULL basis: two datasets sharing the weight label but differing
    # in return formula / clip bounds / semantics version are NOT economically equivalent. None on legacy
    # payloads that predate these fields (and then they do not perturb the fingerprint -- backward-compatible).
    action_return_formula: str | None = None
    action_return_clip_min: float | None = None
    action_return_clip_max: float | None = None
    action_return_semantics_version: str | None = None
    # The fill convention: the price/timing assumption under which a position is realized (e.g.
    # "next_bar_open", "decision_bar_close"). Part of the canonical ReturnBasis. None on payloads that predate
    # the field; it is recorded and participates in basis agreement, but its absence does not break loading.
    action_return_fill_convention: str | None = None

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


# The action-return BASIS detail (recorded by the builder alongside the weight label). Must be ALL present or
# ALL absent: a partial basis (e.g. a formula with no clip bounds) is under-specified -- the run-semantics
# fingerprint and any cost-basis reasoning would be ambiguous. Absent ENTIRELY is allowed (legacy payloads
# predate these fields; their execution-reward variant is gated separately via action_return_weight_semantics).
_ACTION_RETURN_BASIS_DETAIL_KEYS = (
    "action_return_formula",
    "action_return_clip_min",
    "action_return_clip_max",
    "action_return_semantics_version",
)


def validate_action_return_basis(payload: dict[str, Any]) -> None:
    """Fail closed on a PARTIALLY-populated action-return basis (some of formula / clip_min / clip_max /
    semantics_version present, others missing). All-present or all-absent only."""
    present = [k for k in _ACTION_RETURN_BASIS_DETAIL_KEYS if payload.get(k) is not None]
    if present and len(present) != len(_ACTION_RETURN_BASIS_DETAIL_KEYS):
        missing = [k for k in _ACTION_RETURN_BASIS_DETAIL_KEYS if payload.get(k) is None]
        raise ValueError(
            f"partial action-return basis metadata: present={present} but missing={missing}; the basis "
            "detail must be all-present or all-absent."
        )


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
    validate_action_return_basis(payload)
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
    # CASH is a hard invariant (forced safety fallback, cash-idle, zero exposure); a silent fallback to index 0
    # would corrupt missing-label filtering / reportability before the env later rejects the split.
    if "CASH" not in action_names:
        raise ValueError("hour-from-subhour dataset requires an explicit 'CASH' action in action_names.")
    cash_index = action_names.index("CASH")
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
        # PR-4 gate: carried from the gold payload (None = the builder has not declared it -> fail-closed).
        action_return_weight_semantics=payload.get("action_return_weight_semantics"),
        action_return_formula=payload.get("action_return_formula"),
        action_return_clip_min=payload.get("action_return_clip_min"),
        action_return_clip_max=payload.get("action_return_clip_max"),
        action_return_semantics_version=payload.get("action_return_semantics_version"),
        action_return_fill_convention=payload.get("action_return_fill_convention"),
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
