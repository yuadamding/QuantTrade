"""Fail-closed post-screen pipeline for the development-only TOP2000 PPO study.

The screen, confirmation, refit, and 2026 evaluation stages are deliberately
separate immutable artifacts.  Model selection can fail; no stage silently
falls back to a merely best-ranked non-positive setting.  The TOP2000 universe
is future-selected, so every artifact and verdict remains development-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import torch

from rl_quant.envs import HistoricalMarketData, PortfolioConstraints, VectorPortfolioEnv
from rl_quant.execution import FixedTurnoverTargetWeightExecution
from rl_quant.rl import ActionBatch, PPOConfig, RecurrentPPO
from rl_quant.workflows import top2000_ppo as ppo


SCHEMA_VERSION = 1
SCREEN_SELECTION_KIND = "screen-top2-selection"
SCREEN_RECEIPT_SET_KIND = "screen-receipt-set"
CONFIRMATION_PLAN_TYPE = "top2000-confirmation"
CONFIRMATION_RECEIPT_KIND = "confirmation-test-only"
CONFIRMATION_RECEIPT_SET_KIND = "confirmation-receipt-set"
CONFIRMATION_WINNER_KIND = "confirmation-winner"
REFIT_PLAN_TYPE = "top2000-refit"
REFIT_MEMBER_KIND = "final-refit-member"
REFIT_RECEIPT_SET_KIND = "refit-receipt-set"
ENSEMBLE_KIND = "final-confirmed-ensemble"
CONFIRMATION_SEEDS = (17, 29, 43, 71)
RUNTIME_BINDING_KEYS = frozenset(
    {"image_ref", "source_manifest_sha256", "orchestration_manifest_sha256"}
)
COST_LADDER_KEYS = ("gross_0bp", "base", "stress_20bp", "stress_40bp")
_COST_BY_KEY = {
    "gross_0bp": 0.0,
    "base": 10.0,
    "stress_20bp": 20.0,
    "stress_40bp": 40.0,
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PinnedJsonArtifact:
    path: Path
    raw: bytes
    sha256: str
    payload: dict[str, Any]


class PipelineValidationError(ValueError):
    """An immutable post-screen artifact violated the protocol contract."""


def _canonical_json(value: Any) -> bytes:
    return ppo._canonical_json(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return ppo._sha256_file(path)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PipelineValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, location: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PipelineValidationError(f"non-finite JSON number {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineValidationError(f"{location} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise PipelineValidationError(f"{location} must contain a JSON object")
    return value


def _read_pinned_json(path: str | Path, expected_sha256: str, location: str) -> dict[str, Any]:
    _require_digest(expected_sha256, f"{location} SHA-256")
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise PipelineValidationError(f"{location} must be an existing non-symlink regular file")
    raw = artifact.read_bytes()
    actual = _sha256_bytes(raw)
    if not hmac.compare_digest(actual, expected_sha256):
        raise PipelineValidationError(
            f"{location} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return _decode_json(raw, location)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PipelineValidationError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise PipelineValidationError(f"{location} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise PipelineValidationError(
            f"{location} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PipelineValidationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise PipelineValidationError(f"{location} must be a bounded non-empty string")
    return value


def _require_integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineValidationError(f"{location} must be an integer >= {minimum}")
    return value


def _require_development_header(value: Mapping[str, Any], *, kind: str, location: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION or value.get("artifact_kind") != kind:
        raise PipelineValidationError(f"{location} has the wrong schema or artifact kind")
    if (
        value.get("label") != ppo.DEVELOPMENT_LABEL
        or value.get("development_only") is not True
        or value.get("bars_only") is not True
    ):
        raise PipelineValidationError(f"{location} must remain development-only and bars-only")


def _validated_runtime(value: Any, location: str = "runtime") -> dict[str, str]:
    runtime = _require_mapping(value, location)
    _require_exact_keys(runtime, set(RUNTIME_BINDING_KEYS), location)
    image_ref = _require_nonempty_string(runtime["image_ref"], f"{location}.image_ref")
    source = _require_digest(runtime["source_manifest_sha256"], f"{location}.source_manifest_sha256")
    orchestration = _require_digest(
        runtime["orchestration_manifest_sha256"],
        f"{location}.orchestration_manifest_sha256",
    )
    return {
        "image_ref": image_ref,
        "source_manifest_sha256": source,
        "orchestration_manifest_sha256": orchestration,
    }


def _validated_trial_config(value: Any, location: str) -> dict[str, Any]:
    config = _require_mapping(value, location)
    try:
        trial = ppo._trial_from_mapping(config)
    except (TypeError, ValueError) as error:
        raise PipelineValidationError(f"{location} is invalid: {error}") from error
    return asdict(trial)


def _validate_fold_descriptors(value: Any, location: str = "folds") -> list[dict[str, Any]]:
    folds = _require_list(value, location)
    if len(folds) != 3:
        raise PipelineValidationError(f"{location} must contain exactly three folds")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(folds):
        fold = dict(_require_mapping(raw, f"{location}[{index}]"))
        if fold.get("fold_index") != index or not isinstance(fold.get("fold_id"), str):
            raise PipelineValidationError(f"{location}[{index}] has an invalid identity")
        normalized.append(fold)
    return normalized


def _validate_metric_payload(value: Any, location: str) -> dict[str, Any]:
    metric = dict(_require_mapping(value, location))
    required = {
        "observations",
        "decision_coverage",
        "net_total_return",
        "net_annualized_sharpe",
        "max_drawdown",
        "mean_total_one_way_turnover",
        "cost_bps",
        "daily_net_returns",
        "daily_total_one_way_turnover",
        "daily_risky_available",
    }
    missing = sorted(required - set(metric))
    if missing:
        raise PipelineValidationError(f"{location} is missing metric fields {missing}")
    observations = _require_integer(metric["observations"], f"{location}.observations", minimum=1)
    for name in (
        "decision_coverage",
        "net_total_return",
        "net_annualized_sharpe",
        "max_drawdown",
        "mean_total_one_way_turnover",
        "cost_bps",
    ):
        raw = metric[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise PipelineValidationError(f"{location}.{name} must be finite")
    if not 0.0 <= float(metric["decision_coverage"]) <= 1.0:
        raise PipelineValidationError(f"{location}.decision_coverage must lie in [0, 1]")
    if not 0.0 <= float(metric["max_drawdown"]) <= 1.0:
        raise PipelineValidationError(f"{location}.max_drawdown must lie in [0, 1]")
    if float(metric["mean_total_one_way_turnover"]) < 0.0:
        raise PipelineValidationError(f"{location}.mean_total_one_way_turnover cannot be negative")
    series_names = ("daily_net_returns", "daily_total_one_way_turnover")
    for name in series_names:
        series = _require_list(metric[name], f"{location}.{name}")
        if len(series) != observations:
            raise PipelineValidationError(
                f"{location}.{name} length does not equal observations"
            )
        for item in series:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise PipelineValidationError(f"{location}.{name} must contain finite numbers")
    if any(float(item) <= -1.0 for item in metric["daily_net_returns"]):
        raise PipelineValidationError(f"{location}.daily_net_returns contains a <= -100% return")
    if any(float(item) < 0.0 for item in metric["daily_total_one_way_turnover"]):
        raise PipelineValidationError(f"{location}.daily turnover contains a negative value")
    availability = _require_list(
        metric["daily_risky_available"], f"{location}.daily_risky_available"
    )
    if len(availability) != observations or any(not isinstance(item, bool) for item in availability):
        raise PipelineValidationError(
            f"{location}.daily_risky_available must contain one bool per observation"
        )
    recomputed = _series_metrics(
        [float(value) for value in metric["daily_net_returns"]],
        [float(value) for value in metric["daily_total_one_way_turnover"]],
        risky_available=[bool(value) for value in availability],
        cost_bps=float(metric["cost_bps"]),
    )
    for name in (
        "decision_coverage",
        "net_total_return",
        "net_annualized_sharpe",
        "max_drawdown",
        "mean_total_one_way_turnover",
    ):
        if not math.isclose(
            float(metric[name]),
            float(recomputed[name]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise PipelineValidationError(
                f"{location}.{name} does not recompute from its pinned daily series"
            )
    return metric


def _validate_cost_ladder(value: Any, location: str) -> dict[str, dict[str, Any]]:
    ladder = _require_mapping(value, location)
    _require_exact_keys(ladder, set(COST_LADDER_KEYS), location)
    result: dict[str, dict[str, Any]] = {}
    for key in COST_LADDER_KEYS:
        metric = _validate_metric_payload(ladder[key], f"{location}.{key}")
        if float(metric["cost_bps"]) != _COST_BY_KEY[key]:
            raise PipelineValidationError(f"{location}.{key} has the wrong cost_bps")
        result[key] = metric
    return result


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= -1.0:
            raise PipelineValidationError("daily returns must be finite and greater than -100%")
        equity *= 1.0 + numeric
    return equity - 1.0


def _series_metrics(
    returns: Sequence[float],
    turnovers: Sequence[float],
    *,
    risky_available: Sequence[bool],
    cost_bps: float,
) -> dict[str, Any]:
    if (
        not returns
        or len(returns) != len(turnovers)
        or len(returns) != len(risky_available)
    ):
        raise PipelineValidationError("metric series must be non-empty and have equal lengths")
    daily = torch.tensor(list(returns), dtype=torch.float64)
    turnover = torch.tensor(list(turnovers), dtype=torch.float64)
    mean = daily.mean()
    std = daily.std(unbiased=False)
    sharpe = 0.0 if float(std.item()) == 0.0 else float((mean / std * math.sqrt(252.0)).item())
    curve = torch.cat((torch.ones(1, dtype=torch.float64), torch.cumprod(1.0 + daily, dim=0)))
    peaks = torch.cummax(curve, dim=0).values
    drawdown = float((1.0 - curve / peaks.clamp_min(1e-12)).max().item())
    return {
        "observations": len(returns),
        "decision_coverage": sum(bool(value) for value in risky_available) / len(risky_available),
        "net_total_return": float(curve[-1].item() - 1.0),
        "net_annualized_sharpe": sharpe,
        "max_drawdown": drawdown,
        "mean_total_one_way_turnover": float(turnover.mean().item()),
        "cost_bps": float(cost_bps),
        "daily_net_returns": [float(value) for value in returns],
        "daily_total_one_way_turnover": [float(value) for value in turnovers],
        "daily_risky_available": [bool(value) for value in risky_available],
    }


def moving_block_bootstrap_mean_ci(
    returns: Sequence[float],
    *,
    block_length: int = 5,
    samples: int = 2000,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Deterministic, bounded moving-block 95% CI for the mean daily return."""

    if not returns:
        raise PipelineValidationError("bootstrap requires a non-empty return series")
    if block_length <= 0 or samples <= 0 or block_length > len(returns):
        raise PipelineValidationError("bootstrap block_length/samples are invalid")
    values = torch.tensor(list(returns), dtype=torch.float64)
    n = values.numel()
    blocks = math.ceil(n / block_length)
    starts = torch.randint(
        0,
        n,
        (samples, blocks),
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    offsets = torch.arange(block_length).view(1, 1, -1)
    indexes = (starts.unsqueeze(-1) + offsets) % n
    sampled = values[indexes.reshape(samples, -1)[:, :n]].mean(dim=1)
    lower, upper = torch.quantile(sampled, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "method": "deterministic-circular-moving-block-bootstrap",
        "statistic": "mean_daily_net_return",
        "confidence": 0.95,
        "block_length": block_length,
        "samples": samples,
        "seed": seed,
        "estimate": float(values.mean().item()),
        "lower": float(lower.item()),
        "upper": float(upper.item()),
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    ppo._write_exclusive_json(path, payload)


def _load_kind(
    path: str | Path,
    *,
    expected_sha256: str,
    kind: str,
    location: str,
) -> dict[str, Any]:
    value = _read_pinned_json(path, expected_sha256, location)
    _require_development_header(value, kind=kind, location=location)
    return value


def load_screen_selection(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    value = _load_kind(
        path,
        expected_sha256=expected_sha256,
        kind=SCREEN_SELECTION_KIND,
        location="screen selection",
    )
    selected = _require_list(value.get("selected_settings"), "selected_settings")
    if len(selected) != 2:
        raise PipelineValidationError("screen selection must contain exactly two settings")
    _check_artifact_identity(value, "selection_identity", "screen selection")
    return value


def load_confirmation_winner(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    value = _load_kind(
        path,
        expected_sha256=expected_sha256,
        kind=CONFIRMATION_WINNER_KIND,
        location="confirmation winner",
    )
    _require_mapping(value.get("winning_setting"), "winning_setting")
    _check_artifact_identity(value, "winner_identity", "confirmation winner")
    return value


def load_confirmation_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise PipelineValidationError("confirmation receipt must be a non-symlink file")
    raw = artifact.read_bytes()
    actual = _sha256_bytes(raw)
    if expected_sha256 is not None and not hmac.compare_digest(
        actual, _require_digest(expected_sha256, "confirmation receipt SHA-256")
    ):
        raise PipelineValidationError("confirmation receipt SHA-256 mismatch")
    value = _decode_json(raw, "confirmation receipt")
    _require_development_header(
        value,
        kind=CONFIRMATION_RECEIPT_KIND,
        location="confirmation receipt",
    )
    return value


def load_refit_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise PipelineValidationError("refit receipt must be a non-symlink file")
    raw = artifact.read_bytes()
    actual = _sha256_bytes(raw)
    if expected_sha256 is not None and not hmac.compare_digest(
        actual, _require_digest(expected_sha256, "refit receipt SHA-256")
    ):
        raise PipelineValidationError("refit receipt SHA-256 mismatch")
    value = _decode_json(raw, "refit receipt")
    _require_development_header(value, kind=REFIT_MEMBER_KIND, location="refit receipt")
    return value


def load_ensemble_manifest(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    value = _load_kind(
        path,
        expected_sha256=expected_sha256,
        kind=ENSEMBLE_KIND,
        location="ensemble manifest",
    )
    _check_artifact_identity(value, "ensemble_identity", "ensemble manifest")
    return value


def _artifact_identity(payload: Mapping[str, Any], field: str) -> str:
    without_identity = {key: value for key, value in payload.items() if key != field}
    return _sha256_bytes(_canonical_json(without_identity))


def _check_artifact_identity(payload: Mapping[str, Any], field: str, location: str) -> str:
    claimed = _require_digest(payload.get(field), f"{location}.{field}")
    actual = _artifact_identity(payload, field)
    if not hmac.compare_digest(claimed, actual):
        raise PipelineValidationError(f"{location}.{field} does not bind its canonical payload")
    return claimed


def build_confirmation_plan(
    screen_selection: Mapping[str, Any],
    *,
    screen_selection_sha256: str,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact immutable top-two-by-four-seed confirmation plan."""

    _require_development_header(
        screen_selection,
        kind=SCREEN_SELECTION_KIND,
        location="screen selection",
    )
    _require_digest(screen_selection_sha256, "screen_selection_sha256")
    selection_identity = _check_artifact_identity(
        screen_selection, "selection_identity", "screen selection"
    )
    selected = _require_list(screen_selection.get("selected_settings"), "selected_settings")
    if len(selected) != 2:
        raise PipelineValidationError("confirmation requires exactly two selected settings")
    folds = _validate_fold_descriptors(screen_selection.get("folds"))
    rows: list[dict[str, Any]] = []
    for selected_position, raw_setting in enumerate(selected):
        setting = _require_mapping(raw_setting, f"selected_settings[{selected_position}]")
        setting_index = _require_integer(
            setting.get("setting_index"), f"selected_settings[{selected_position}].setting_index"
        )
        setting_id = _require_nonempty_string(
            setting.get("setting_id"), f"selected_settings[{selected_position}].setting_id"
        )
        base_config = ppo.TrialConfig(
            **_validated_trial_config(
                setting.get("trial_config"),
                f"selected_settings[{selected_position}].trial_config",
            )
        )
        if base_config.cost_bps != 10.0:
            raise PipelineValidationError("the frozen confirmation protocol requires cost_bps=10")
        for seed in CONFIRMATION_SEEDS:
            rows.append(
                {
                    "global_index": len(rows),
                    "setting_index": setting_index,
                    "setting_id": setting_id,
                    "seed": seed,
                    "fold_indexes": [0, 1, 2],
                    "config": asdict(replace(base_config, seed=seed)),
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_type": CONFIRMATION_PLAN_TYPE,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "cache_identity": screen_selection["cache_identity"],
        "cache_sha256": screen_selection["cache_sha256"],
        "search_identity": screen_selection["search_identity"],
        "base_dataset_identity": screen_selection["base_dataset_identity"],
        "lockbox_partition_names_hash": screen_selection["lockbox_partition_names_hash"],
        "screen_selection_sha256": screen_selection_sha256,
        "selection_identity": selection_identity,
        "runtime": _validated_runtime(runtime),
        "folds": folds,
        "trials": rows,
    }
    return validate_confirmation_plan(payload)


def validate_confirmation_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _require_mapping(payload, "confirmation plan")
    expected = {
        "schema_version",
        "plan_type",
        "label",
        "development_only",
        "bars_only",
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "screen_selection_sha256",
        "selection_identity",
        "runtime",
        "folds",
        "trials",
    }
    _require_exact_keys(plan, expected, "confirmation plan")
    if plan["schema_version"] != SCHEMA_VERSION or plan["plan_type"] != CONFIRMATION_PLAN_TYPE:
        raise PipelineValidationError("unsupported confirmation plan schema/type")
    if (
        plan["label"] != ppo.DEVELOPMENT_LABEL
        or plan["development_only"] is not True
        or plan["bars_only"] is not True
    ):
        raise PipelineValidationError("confirmation plan must remain development-only and bars-only")
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "screen_selection_sha256",
        "selection_identity",
    ):
        _require_digest(plan[name], f"confirmation plan.{name}")
    _validated_runtime(plan["runtime"], "confirmation plan.runtime")
    _validate_fold_descriptors(plan["folds"], "confirmation plan.folds")
    trials = _require_list(plan["trials"], "confirmation plan.trials")
    if len(trials) != 2 * len(CONFIRMATION_SEEDS):
        raise PipelineValidationError("confirmation plan must contain exactly eight rows")
    seen_settings: list[tuple[int, str]] = []
    for index, raw in enumerate(trials):
        row = _require_mapping(raw, f"confirmation plan.trials[{index}]")
        _require_exact_keys(
            row,
            {"global_index", "setting_index", "setting_id", "seed", "fold_indexes", "config"},
            f"confirmation plan.trials[{index}]",
        )
        if row["global_index"] != index:
            raise PipelineValidationError("confirmation global indexes must be canonical")
        setting_index = _require_integer(row["setting_index"], f"trials[{index}].setting_index")
        setting_id = _require_nonempty_string(row["setting_id"], f"trials[{index}].setting_id")
        expected_seed = CONFIRMATION_SEEDS[index % len(CONFIRMATION_SEEDS)]
        if row["seed"] != expected_seed or row["fold_indexes"] != [0, 1, 2]:
            raise PipelineValidationError("confirmation seed/fold ordering changed")
        config = _validated_trial_config(row["config"], f"trials[{index}].config")
        if config["seed"] != expected_seed or float(config["cost_bps"]) != 10.0:
            raise PipelineValidationError("confirmation config seed/cost contract changed")
        group = index // len(CONFIRMATION_SEEDS)
        identity = (setting_index, setting_id)
        if group == len(seen_settings):
            seen_settings.append(identity)
        elif seen_settings[group] != identity:
            raise PipelineValidationError("each confirmation setting must occupy one four-seed block")
        first = trials[group * len(CONFIRMATION_SEEDS)]
        first_config = dict(_require_mapping(first["config"], "first config"))
        if {**config, "seed": first_config["seed"]} != first_config:
            raise PipelineValidationError("confirmation seed rows changed non-seed hyperparameters")
    if len(seen_settings) != 2 or seen_settings[0] == seen_settings[1]:
        raise PipelineValidationError("confirmation plan needs two distinct settings")
    return dict(plan)


def build_refit_plan(
    confirmation_winner: Mapping[str, Any],
    *,
    confirmation_winner_sha256: str,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact four-seed all-pre-2026 final-refit plan."""

    _require_development_header(
        confirmation_winner,
        kind=CONFIRMATION_WINNER_KIND,
        location="confirmation winner",
    )
    _require_digest(confirmation_winner_sha256, "confirmation_winner_sha256")
    winner_identity = _check_artifact_identity(
        confirmation_winner, "winner_identity", "confirmation winner"
    )
    setting = _require_mapping(confirmation_winner.get("winning_setting"), "winning_setting")
    setting_index = _require_integer(setting.get("setting_index"), "winning_setting.setting_index")
    setting_id = _require_nonempty_string(setting.get("setting_id"), "winning_setting.setting_id")
    base_config = ppo.TrialConfig(
        **_validated_trial_config(setting.get("trial_config"), "winning_setting.trial_config")
    )
    rows = [
        {
            "global_index": index,
            "setting_index": setting_index,
            "setting_id": setting_id,
            "seed": seed,
            "config": asdict(replace(base_config, seed=seed)),
        }
        for index, seed in enumerate(CONFIRMATION_SEEDS)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_type": REFIT_PLAN_TYPE,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "cache_identity": confirmation_winner["cache_identity"],
        "cache_sha256": confirmation_winner["cache_sha256"],
        "search_identity": confirmation_winner["search_identity"],
        "base_dataset_identity": confirmation_winner["base_dataset_identity"],
        "lockbox_partition_names_hash": confirmation_winner["lockbox_partition_names_hash"],
        "confirmation_winner_sha256": confirmation_winner_sha256,
        "winner_identity": winner_identity,
        "runtime": _validated_runtime(runtime),
        "trials": rows,
    }
    return validate_refit_plan(payload)


def validate_refit_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _require_mapping(payload, "refit plan")
    expected = {
        "schema_version",
        "plan_type",
        "label",
        "development_only",
        "bars_only",
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "confirmation_winner_sha256",
        "winner_identity",
        "runtime",
        "trials",
    }
    _require_exact_keys(plan, expected, "refit plan")
    if plan["schema_version"] != SCHEMA_VERSION or plan["plan_type"] != REFIT_PLAN_TYPE:
        raise PipelineValidationError("unsupported refit plan schema/type")
    if (
        plan["label"] != ppo.DEVELOPMENT_LABEL
        or plan["development_only"] is not True
        or plan["bars_only"] is not True
    ):
        raise PipelineValidationError("refit plan must remain development-only and bars-only")
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "confirmation_winner_sha256",
        "winner_identity",
    ):
        _require_digest(plan[name], f"refit plan.{name}")
    _validated_runtime(plan["runtime"], "refit plan.runtime")
    trials = _require_list(plan["trials"], "refit plan.trials")
    if len(trials) != len(CONFIRMATION_SEEDS):
        raise PipelineValidationError("refit plan must contain exactly four rows")
    identity: tuple[int, str] | None = None
    first_config: dict[str, Any] | None = None
    for index, raw in enumerate(trials):
        row = _require_mapping(raw, f"refit plan.trials[{index}]")
        _require_exact_keys(
            row,
            {"global_index", "setting_index", "setting_id", "seed", "config"},
            f"refit plan.trials[{index}]",
        )
        setting_index = _require_integer(row["setting_index"], f"trials[{index}].setting_index")
        setting_id = _require_nonempty_string(row["setting_id"], f"trials[{index}].setting_id")
        expected_seed = CONFIRMATION_SEEDS[index]
        if row["global_index"] != index or row["seed"] != expected_seed:
            raise PipelineValidationError("refit index/seed ordering changed")
        config = _validated_trial_config(row["config"], f"trials[{index}].config")
        if config["seed"] != expected_seed or float(config["cost_bps"]) != 10.0:
            raise PipelineValidationError("refit config seed/cost contract changed")
        if identity is None:
            identity = (setting_index, setting_id)
            first_config = config
        elif identity != (setting_index, setting_id):
            raise PipelineValidationError("all refit rows must use the winning setting")
        assert first_config is not None
        if {**config, "seed": first_config["seed"]} != first_config:
            raise PipelineValidationError("refit rows changed non-seed hyperparameters")
    return dict(plan)


def _require_acknowledgement(value: str, stage: str) -> None:
    if value != ppo.DEVELOPMENT_ACK:
        raise PipelineValidationError(
            f"{stage} requires --development-ack {ppo.DEVELOPMENT_ACK!r}"
        )


def _resolve_indexed_receipt(root: Path, index: int, filename: str) -> Path:
    candidates = (
        root / f"trial-{index:04d}" / filename,
        root / f"index-{index:04d}" / f"trial-{index:04d}" / filename,
    )
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise PipelineValidationError(
            f"expected exactly one bounded {filename} path for index {index}; found {len(present)}"
        )
    return present[0]


def _load_receipt_set(
    manifest_path: str | Path,
    manifest_sha256: str,
    *,
    root: str | Path,
    kind: str,
    plan_sha256: str,
    expected_count: int,
) -> tuple[list[PinnedJsonArtifact], dict[str, Any]]:
    manifest = _read_pinned_json(manifest_path, manifest_sha256, "receipt-set manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_kind",
            "label",
            "development_only",
            "bars_only",
            "plan_sha256",
            "expected_count",
            "receipts",
            "receipt_set_identity",
        },
        "receipt-set manifest",
    )
    _require_development_header(manifest, kind=kind, location="receipt-set manifest")
    if manifest["plan_sha256"] != plan_sha256 or manifest["expected_count"] != expected_count:
        raise PipelineValidationError("receipt-set manifest plan/count binding is wrong")
    _check_artifact_identity(manifest, "receipt_set_identity", "receipt-set manifest")
    rows = _require_list(manifest["receipts"], "receipt-set manifest.receipts")
    if len(rows) != expected_count:
        raise PipelineValidationError("receipt-set manifest does not have its exact expected count")
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise PipelineValidationError("receipts root must be an existing non-symlink directory")
    resolved_root = root_path.resolve(strict=True)
    artifacts: list[PinnedJsonArtifact] = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"receipt-set manifest.receipts[{index}]")
        _require_exact_keys(
            row,
            {"global_index", "relative_path", "receipt_sha256"},
            f"receipt-set manifest.receipts[{index}]",
        )
        if row["global_index"] != index:
            raise PipelineValidationError("receipt-set manifest indexes must be canonical")
        relative_raw = row["relative_path"]
        if not isinstance(relative_raw, str) or not relative_raw or "\\" in relative_raw:
            raise PipelineValidationError("receipt-set relative_path must be a POSIX relative path")
        if any(part in {"", ".", ".."} for part in relative_raw.split("/")):
            raise PipelineValidationError(
                "receipt-set manifest contains an unsafe/non-canonical path"
            )
        relative = Path(relative_raw)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_raw
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise PipelineValidationError("receipt-set manifest contains an unsafe relative path")
        candidate = root_path / relative
        cursor = root_path
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PipelineValidationError("receipt-set paths may not traverse symlinks")
        if not candidate.is_file():
            raise PipelineValidationError(f"pinned receipt file is absent: {candidate}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise PipelineValidationError("receipt-set path escapes receipts root") from error
        digest = _require_digest(
            row["receipt_sha256"], f"receipt-set manifest.receipts[{index}].receipt_sha256"
        )
        raw_bytes = resolved.read_bytes()
        actual_digest = _sha256_bytes(raw_bytes)
        if not hmac.compare_digest(actual_digest, digest):
            raise PipelineValidationError(f"pinned receipt {index} SHA-256 mismatch")
        artifacts.append(
            PinnedJsonArtifact(
                path=resolved,
                raw=raw_bytes,
                sha256=actual_digest,
                payload=_decode_json(raw_bytes, f"pinned receipt {index}"),
            )
        )
    return artifacts, manifest


_RECEIPT_SET_STAGE = {
    "screen": (
        SCREEN_RECEIPT_SET_KIND,
        "screen-validation-only",
        "screen_plan_sha256",
        "setting_index",
        8,
    ),
    "confirmation": (
        CONFIRMATION_RECEIPT_SET_KIND,
        CONFIRMATION_RECEIPT_KIND,
        "confirmation_plan_sha256",
        "global_index",
        8,
    ),
    "refit": (
        REFIT_RECEIPT_SET_KIND,
        REFIT_MEMBER_KIND,
        "refit_plan_sha256",
        "global_index",
        4,
    ),
}


def build_receipt_set_manifest(
    stage: str,
    plan_sha256: str,
    receipts_root: str | Path,
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Build a canonical manifest from an explicit list; this function never scans."""

    if stage not in _RECEIPT_SET_STAGE:
        raise PipelineValidationError(f"unsupported receipt-set stage {stage!r}")
    kind, receipt_kind, plan_field, index_field, expected_count = _RECEIPT_SET_STAGE[stage]
    _require_digest(plan_sha256, "receipt-set plan_sha256")
    if len(relative_paths) != expected_count:
        raise PipelineValidationError(
            f"{stage} receipt set requires exactly {expected_count} explicit paths"
        )
    root = Path(receipts_root)
    rows: list[dict[str, Any]] = []
    for index, relative in enumerate(relative_paths):
        # Reuse the strict path verifier by first constructing a one-file-like
        # row locally; no glob/rglob or directory enumeration is performed.
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise PipelineValidationError("receipt-set relative paths must be POSIX strings")
        if any(part in {"", ".", ".."} for part in relative.split("/")):
            raise PipelineValidationError(
                "receipt-set builder received an unsafe/non-canonical path"
            )
        candidate_relative = Path(relative)
        if (
            candidate_relative.is_absolute()
            or candidate_relative.as_posix() != relative
            or any(part in {"", ".", ".."} for part in candidate_relative.parts)
        ):
            raise PipelineValidationError("receipt-set builder received an unsafe relative path")
        candidate = root / candidate_relative
        cursor = root
        for part in candidate_relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PipelineValidationError("receipt-set paths may not traverse symlinks")
        if not candidate.is_file():
            raise PipelineValidationError(f"receipt-set input is absent: {candidate}")
        raw_bytes = candidate.read_bytes()
        receipt = _decode_json(raw_bytes, f"{stage} receipt {index}")
        _require_development_header(receipt, kind=receipt_kind, location=f"{stage} receipt {index}")
        if receipt.get(plan_field) != plan_sha256 or receipt.get(index_field) != index:
            raise PipelineValidationError(f"{stage} receipt {index} does not bind its plan/index")
        rows.append(
            {
                "global_index": index,
                "relative_path": candidate_relative.as_posix(),
                "receipt_sha256": _sha256_bytes(raw_bytes),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": kind,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "plan_sha256": plan_sha256,
        "expected_count": expected_count,
        "receipts": rows,
    }
    payload["receipt_set_identity"] = _artifact_identity(payload, "receipt_set_identity")
    return payload


def load_receipt_set_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    stage: str,
    plan_sha256: str,
    receipts_root: str | Path,
) -> dict[str, Any]:
    if stage not in _RECEIPT_SET_STAGE:
        raise PipelineValidationError(f"unsupported receipt-set stage {stage!r}")
    kind, _receipt_kind, _plan_field, _index_field, count = _RECEIPT_SET_STAGE[stage]
    _artifacts, manifest = _load_receipt_set(
        path,
        expected_sha256,
        root=receipts_root,
        kind=kind,
        plan_sha256=plan_sha256,
        expected_count=count,
    )
    return manifest


def _safe_child(parent: Path, name: Any, expected_name: str) -> Path:
    if name != expected_name or Path(str(name)).name != name:
        raise PipelineValidationError(f"unsafe or unexpected child artifact name {name!r}")
    child = parent / str(name)
    if child.is_symlink() or not child.is_file():
        raise PipelineValidationError(f"required child artifact is absent: {child}")
    return child


def _read_pinned_torch(path: Path, expected_sha256: Any, location: str) -> tuple[bytes, Mapping[str, Any]]:
    digest = _require_digest(expected_sha256, f"{location} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise PipelineValidationError(f"{location} must be an existing non-symlink file")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if not hmac.compare_digest(actual, digest):
        raise PipelineValidationError(f"{location} SHA-256 mismatch")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PipelineValidationError(f"{location} is not a valid safe torch checkpoint") from error
    return raw, _require_mapping(payload, location)


def _validate_screen_plan(
    value: Mapping[str, Any],
    *,
    plan_sha256: str,
    cache: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]], dict[str, str]]:
    if (
        value.get("schema_version") != 1
        or value.get("label") != ppo.DEVELOPMENT_LABEL
        or value.get("development_only") is not True
        or value.get("bars_only") is not True
    ):
        raise PipelineValidationError("screen plan has incompatible schema/labels")
    expected = {
        "cache_sha256": cache["cache_sha256"],
        "cache_identity": cache["cache_identity"],
        "search_identity": cache["search_identity"],
    }
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise PipelineValidationError(f"screen plan {name} does not match the pinned cache")
    evidence = _require_mapping(value.get("data_evidence"), "screen plan.data_evidence")
    if evidence.get("base_dataset_identity") != cache["base_dataset_identity"]:
        raise PipelineValidationError("screen plan base dataset identity does not match the cache")
    runtime = _validated_runtime(
        {
            "image_ref": value.get("image_ref"),
            "source_manifest_sha256": value.get("source_manifest_sha256"),
            "orchestration_manifest_sha256": value.get("orchestration_manifest_sha256"),
        },
        "screen plan runtime",
    )
    folds = _validate_fold_descriptors(value.get("folds"), "screen plan.folds")
    actual_folds = [ppo.fold_descriptor(fold) for fold in ppo.walk_forward_folds(cache)]
    if folds != actual_folds:
        raise PipelineValidationError("screen plan folds do not match the pinned cache")
    trials = _require_list(value.get("trials"), "screen plan.trials")
    if len(trials) != 8:
        raise PipelineValidationError("screen plan must contain exactly eight settings")
    normalized: list[Mapping[str, Any]] = []
    for index, raw in enumerate(trials):
        row = _require_mapping(raw, f"screen plan.trials[{index}]")
        _require_exact_keys(
            row,
            {"global_index", "setting_id", "fold_indexes", "config"},
            f"screen plan.trials[{index}]",
        )
        if row["global_index"] != index or row["fold_indexes"] != [0, 1, 2]:
            raise PipelineValidationError("screen plan index/fold contract changed")
        _require_nonempty_string(row["setting_id"], f"screen plan.trials[{index}].setting_id")
        config = _validated_trial_config(row["config"], f"screen plan.trials[{index}].config")
        if float(config["cost_bps"]) != 10.0:
            raise PipelineValidationError("screen plan must use the 10bp base cost")
        normalized.append(row)
    _require_digest(plan_sha256, "screen plan SHA-256")
    return folds, normalized, runtime


def _validate_checkpoint_common(
    checkpoint: Any,
    *,
    expected: Mapping[str, Any],
    location: str,
) -> Mapping[str, Any]:
    value = _require_mapping(checkpoint, location)
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise PipelineValidationError(f"{location}.{name} does not match its immutable receipt")
    state = value.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise PipelineValidationError(f"{location} lacks a non-empty model_state_dict")
    return value


def _screen_candidate(
    receipt: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    frozen_folds: Sequence[Mapping[str, Any]],
    receipt_path: Path,
    receipt_sha256: str,
    plan_sha256: str,
    cache: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    _require_development_header(receipt, kind="screen-validation-only", location="screen receipt")
    expected_top = {
        "screen_plan_sha256": plan_sha256,
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "setting_index": row["global_index"],
        "setting_id": row["setting_id"],
        "trial_config": row["config"],
        "runtime": runtime,
    }
    for name, wanted in expected_top.items():
        if receipt.get(name) != wanted:
            raise PipelineValidationError(f"screen receipt {name} does not match plan/cache")
    folds = _require_list(receipt.get("folds"), "screen receipt.folds")
    if len(folds) != 3:
        raise PipelineValidationError("screen receipt must contain all three folds")
    base_returns: list[float] = []
    stress_returns: list[float] = []
    stress_sharpes: list[float] = []
    stress_daily: list[float] = []
    turnovers: list[float] = []
    coverages: list[float] = []
    for index, raw_fold in enumerate(folds):
        fold = _require_mapping(raw_fold, f"screen receipt.folds[{index}]")
        if fold.get("fold") != frozen_folds[index]:
            raise PipelineValidationError("screen receipt fold descriptor drifted")
        if fold.get("fold_test_status") != "sealed-for-post-selection-confirmation":
            raise PipelineValidationError("screen receipt improperly revealed a fold test")
        if "test_metrics" in fold or "screen_test_metrics" in fold:
            raise PipelineValidationError("screen selection cannot consume fold-test metrics")
        ladder = _validate_cost_ladder(
            fold.get("validation_cost_ladder"),
            f"screen receipt.folds[{index}].validation_cost_ladder",
        )
        if fold.get("validation_metrics") != ladder["base"]:
            raise PipelineValidationError("screen base validation metric is not its 10bp ladder rung")
        expected_observations = int(frozen_folds[index]["validation_stop"]) - int(
            frozen_folds[index]["validation_start"]
        )
        if any(int(metric["observations"]) != expected_observations for metric in ladder.values()):
            raise PipelineValidationError("screen validation series length differs from its fold")
        checkpoint = _safe_child(receipt_path.parent, fold.get("checkpoint"), f"fold-{index:02d}.pt")
        _checkpoint_raw, loaded = _read_pinned_torch(
            checkpoint,
            fold.get("checkpoint_sha256"),
            f"screen fold {index} checkpoint",
        )
        _validate_checkpoint_common(
            loaded,
            expected={
                "schema_version": 1,
                "label": ppo.DEVELOPMENT_LABEL,
                "development_only": True,
                "screen_plan_sha256": plan_sha256,
                "runtime": runtime,
                "cache_identity": cache["cache_identity"],
                "setting_index": row["global_index"],
                "setting_id": row["setting_id"],
                "fold": frozen_folds[index],
                "trial_config": row["config"],
            },
            location=f"screen fold {index} checkpoint",
        )
        base = ladder["base"]
        stress = ladder["stress_20bp"]
        base_returns.append(float(base["net_total_return"]))
        stress_returns.append(float(stress["net_total_return"]))
        stress_sharpes.append(float(stress["net_annualized_sharpe"]))
        stress_daily.extend(float(value) for value in stress["daily_net_returns"])
        turnovers.append(float(base["mean_total_one_way_turnover"]))
        coverages.append(float(base["decision_coverage"]))
    metrics = {
        "minimum_decision_coverage": min(coverages),
        "positive_base_fold_count": sum(value > 0.0 for value in base_returns),
        "mean_20bp_return": sum(stress_returns) / len(stress_returns),
        "pooled_20bp_return": _compound(stress_daily),
        "worst_fold_20bp_sharpe": min(stress_sharpes),
        "mean_20bp_sharpe": sum(stress_sharpes) / len(stress_sharpes),
        "mean_base_return": sum(base_returns) / len(base_returns),
        "mean_base_turnover": sum(turnovers) / len(turnovers),
    }
    gates = {
        "all_three_folds": len(folds) == 3,
        "coverage_at_least_95pct": metrics["minimum_decision_coverage"] >= 0.95,
        "base_positive_in_at_least_two_folds": metrics["positive_base_fold_count"] >= 2,
        "mean_20bp_return_positive": metrics["mean_20bp_return"] > 0.0,
        "pooled_20bp_return_positive": metrics["pooled_20bp_return"] > 0.0,
    }
    return {
        "setting_index": row["global_index"],
        "setting_id": row["setting_id"],
        "trial_config": dict(row["config"]),
        "receipt_sha256": receipt_sha256,
        "metrics": metrics,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def aggregate_screen_receipts(
    cache_path: str | Path,
    cache_sha256: str,
    plan_path: str | Path,
    plan_sha256: str,
    receipts_root: str | Path,
    receipt_set_manifest_path: str | Path,
    receipt_set_manifest_sha256: str,
    output_path: str | Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    """Validate all eight screen receipts and immutably select exactly two."""

    _require_acknowledgement(acknowledgement, "Screen aggregation")
    plan = _read_pinned_json(plan_path, plan_sha256, "screen plan")
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256, device="cpu")
    folds, rows, runtime = _validate_screen_plan(
        plan,
        plan_sha256=plan_sha256,
        cache=cache,
    )
    receipt_artifacts, _receipt_set = _load_receipt_set(
        receipt_set_manifest_path,
        receipt_set_manifest_sha256,
        root=receipts_root,
        kind=SCREEN_RECEIPT_SET_KIND,
        plan_sha256=plan_sha256,
        expected_count=8,
    )
    candidates: list[dict[str, Any]] = []
    receipt_bindings: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        artifact = receipt_artifacts[index]
        receipt_path = artifact.path
        receipt = artifact.payload
        candidate = _screen_candidate(
            receipt,
            row=row,
            frozen_folds=folds,
            receipt_path=receipt_path,
            receipt_sha256=artifact.sha256,
            plan_sha256=plan_sha256,
            cache=cache,
            runtime=runtime,
        )
        candidates.append(candidate)
        receipt_bindings.append(
            {
                "setting_index": index,
                "setting_id": row["setting_id"],
                "receipt_sha256": artifact.sha256,
            }
        )
    eligible = [value for value in candidates if value["eligible"]]
    eligible.sort(
        key=lambda value: (
            -float(value["metrics"]["worst_fold_20bp_sharpe"]),
            -float(value["metrics"]["mean_20bp_sharpe"]),
            -float(value["metrics"]["mean_base_return"]),
            float(value["metrics"]["mean_base_turnover"]),
            str(value["setting_id"]),
        )
    )
    if len(eligible) < 2:
        raise PipelineValidationError(
            f"screen selection requires two eligible settings; only {len(eligible)} passed"
        )
    selected = [
        {
            "setting_index": value["setting_index"],
            "setting_id": value["setting_id"],
            "trial_config": value["trial_config"],
            "ranking_metrics": value["metrics"],
        }
        for value in eligible[:2]
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SCREEN_SELECTION_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "screen_plan_sha256": plan_sha256,
        "receipt_set_manifest_sha256": receipt_set_manifest_sha256,
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "runtime": runtime,
        "folds": folds,
        "receipt_sha256s": receipt_bindings,
        "eligibility_gates": {
            "fold_count": 3,
            "minimum_decision_coverage": 0.95,
            "minimum_positive_base_folds": 2,
            "require_positive_mean_20bp_return": True,
            "require_positive_pooled_20bp_return": True,
        },
        "ranking_rule": [
            "worst_fold_20bp_sharpe_desc",
            "mean_20bp_sharpe_desc",
            "mean_base_return_desc",
            "mean_base_turnover_asc",
            "setting_id_asc",
        ],
        "candidate_summaries": candidates,
        "selected_settings": selected,
    }
    payload["selection_identity"] = _artifact_identity(payload, "selection_identity")
    _write_exclusive_json(Path(output_path), payload)
    return payload


def _evaluation_environment(
    data: HistoricalMarketData,
    trial: ppo.TrialConfig,
    *,
    cost_bps: float,
) -> VectorPortfolioEnv:
    return VectorPortfolioEnv(
        data,
        cash_index=0,
        constraints=PortfolioConstraints(
            max_asset_weight=trial.max_asset_weight,
            max_leverage=1.0,
            max_turnover=trial.max_turnover,
            max_drawdown=trial.max_drawdown,
        ),
        execution_model=FixedTurnoverTargetWeightExecution(cost_bps=cost_bps),
        discount=trial.discount,
        observation_adapter=ppo.BarsOnlyObservationAdapter(cash_index=0),
    )


@torch.no_grad()
def _evaluate_baseline(
    data: HistoricalMarketData,
    trial: ppo.TrialConfig,
    *,
    cost_bps: float,
    equal_weight_available_assets: bool,
) -> dict[str, Any]:
    if data.batch_size != 1:
        raise PipelineValidationError("baseline evaluation requires one chronology")
    environment = _evaluation_environment(data, trial, cost_bps=cost_bps)
    observation, _info = environment.reset()
    returns: list[float] = []
    turnovers: list[float] = []
    risky_available: list[bool] = []
    while True:
        mask = observation.action_mask
        if mask is None:
            raise PipelineValidationError("baseline evaluation requires an action mask")
        requested = torch.zeros(
            (data.batch_size, data.num_assets),
            dtype=data.asset_returns.dtype,
            device=data.device,
        )
        if equal_weight_available_assets:
            risky = mask.clone()
            risky[:, 0] = False
            count = risky.sum(dim=-1, keepdim=True)
            requested = risky.to(dtype=requested.dtype) / count.clamp_min(1).to(requested.dtype)
            requested[:, 0] = (count.squeeze(-1) == 0).to(dtype=requested.dtype)
        else:
            requested[:, 0] = 1.0
        risky_available.extend(bool(value) for value in mask[:, 1:].any(dim=-1).cpu().tolist())
        transition = environment.step(ActionBatch(action=requested))
        returns.extend(float(value) for value in transition.reward.double().cpu().tolist())
        turnovers.extend(
            float(value) for value in transition.info["recent_turnover"].double().cpu().tolist()
        )
        observation = transition.next_observation
        if bool(transition.done.all().item()):
            break
    return _series_metrics(
        returns,
        turnovers,
        risky_available=risky_available,
        cost_bps=cost_bps,
    )


def _baseline_cost_ladder(
    data: HistoricalMarketData,
    trial: ppo.TrialConfig,
    *,
    equal_weight_available_assets: bool,
) -> dict[str, dict[str, Any]]:
    return {
        key: _evaluate_baseline(
            data,
            trial,
            cost_bps=cost,
            equal_weight_available_assets=equal_weight_available_assets,
        )
        for key, cost in _COST_BY_KEY.items()
    }


def _excess_payload(model: Mapping[str, Any], equal_weight: Mapping[str, Any]) -> dict[str, Any]:
    model_daily = [float(value) for value in model["daily_net_returns"]]
    baseline_daily = [float(value) for value in equal_weight["daily_net_returns"]]
    if len(model_daily) != len(baseline_daily):
        raise PipelineValidationError("model and equal-weight return series lengths differ")
    daily = [left - right for left, right in zip(model_daily, baseline_daily, strict=True)]
    values = torch.tensor(daily, dtype=torch.float64)
    std = values.std(unbiased=False)
    sharpe = (
        0.0
        if float(std.item()) == 0.0
        else float((values.mean() / std * math.sqrt(252.0)).item())
    )
    return {
        "observations": len(daily),
        "net_total_return_difference": float(model["net_total_return"])
        - float(equal_weight["net_total_return"]),
        "mean_daily_net_return_excess": float(values.mean().item()),
        "annualized_sharpe_of_daily_excess": sharpe,
        "daily_net_return_excess": daily,
    }


def _excess_ladder(
    model: Mapping[str, Mapping[str, Any]],
    equal_weight: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {key: _excess_payload(model[key], equal_weight[key]) for key in COST_LADDER_KEYS}


def _pooled_ladder(ladders: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    if not ladders:
        raise PipelineValidationError("cannot pool an empty ladder sequence")
    pooled: dict[str, dict[str, Any]] = {}
    for key in COST_LADDER_KEYS:
        returns: list[float] = []
        turnovers: list[float] = []
        risky_available: list[bool] = []
        for ladder in ladders:
            metric = ladder[key]
            returns.extend(float(value) for value in metric["daily_net_returns"])
            turnovers.extend(float(value) for value in metric["daily_total_one_way_turnover"])
            risky_available.extend(bool(value) for value in metric["daily_risky_available"])
        pooled[key] = _series_metrics(
            returns,
            turnovers,
            risky_available=risky_available,
            cost_bps=_COST_BY_KEY[key],
        )
    return pooled


def _confirmation_aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model = _pooled_ladder([fold["test_cost_ladder"] for fold in folds])
    cash = _pooled_ladder([fold["baselines"]["cash"] for fold in folds])
    equal_weight = _pooled_ladder(
        [fold["baselines"]["equal_weight_available_assets"] for fold in folds]
    )
    return {
        "model_cost_ladder": model,
        "baselines": {
            "cash": cash,
            "equal_weight_available_assets": equal_weight,
        },
        "excess_vs_equal_weight": _excess_ladder(model, equal_weight),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_confirmation_worker(
    cache_path: str | Path,
    cache_sha256: str,
    plan_path: str | Path,
    plan_sha256: str,
    screen_selection_path: str | Path,
    screen_selection_sha256: str,
    output_root: str | Path,
    *,
    index: int,
    device: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Train one selected-setting/seed row and reveal only its three fold tests."""

    _require_acknowledgement(acknowledgement, "Confirmation worker")
    plan = validate_confirmation_plan(
        _read_pinned_json(plan_path, plan_sha256, "confirmation plan")
    )
    selection = load_screen_selection(
        screen_selection_path,
        expected_sha256=screen_selection_sha256,
    )
    if (
        plan["screen_selection_sha256"] != screen_selection_sha256
        or plan["selection_identity"] != selection.get("selection_identity")
    ):
        raise PipelineValidationError("confirmation plan does not bind the pinned screen selection")
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256, device=device)
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
    ):
        if plan[name] != cache[name] or selection.get(name) != cache[name]:
            raise PipelineValidationError(f"confirmation {name} does not match the pinned cache")
    trials = plan["trials"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(trials):
        raise PipelineValidationError("confirmation worker index is out of range")
    row = trials[index]
    selected_by_id = {
        (value["setting_index"], value["setting_id"]): value
        for value in selection["selected_settings"]
    }
    selected = selected_by_id.get((row["setting_index"], row["setting_id"]))
    if selected is None:
        raise PipelineValidationError("confirmation row is not one of the pinned top two settings")
    selected_config = dict(selected["trial_config"])
    if {**row["config"], "seed": selected_config["seed"]} != selected_config:
        raise PipelineValidationError("confirmation row hyperparameters differ from screen selection")
    trial = ppo.TrialConfig(**row["config"])
    actual_folds = ppo.walk_forward_folds(cache)
    descriptors = [ppo.fold_descriptor(fold) for fold in actual_folds]
    if descriptors != plan["folds"]:
        raise PipelineValidationError("confirmation fold geometry differs from cache")
    output_dir = Path(output_root) / f"trial-{index:04d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    fold_receipts: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(actual_folds):
        ppo._seed_everything(trial.seed)
        stack, history, sampling = ppo.train_cache_block(
            cache,
            start=fold.train.start_position,
            stop=fold.train.stop_position,
            trial=trial,
        )
        test_data = ppo._market_from_cache_range(
            cache,
            fold.test.start_position,
            fold.test.stop_position,
        )
        model_ladder = ppo.evaluate_cost_ladder(stack.model, test_data, trial)
        cash_ladder = _baseline_cost_ladder(
            test_data,
            trial,
            equal_weight_available_assets=False,
        )
        equal_weight_ladder = _baseline_cost_ladder(
            test_data,
            trial,
            equal_weight_available_assets=True,
        )
        checkpoint_path = output_dir / f"fold-{fold_index:02d}.pt"
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "confirmation-fold-checkpoint",
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "confirmation_plan_sha256": plan_sha256,
            "screen_selection_sha256": screen_selection_sha256,
            "selection_identity": selection["selection_identity"],
            "cache_identity": cache["cache_identity"],
            "global_index": index,
            "setting_index": row["setting_index"],
            "setting_id": row["setting_id"],
            "seed": row["seed"],
            "fold": descriptors[fold_index],
            "trial_config": row["config"],
            "model_state_dict": {
                name: value.detach().cpu() for name, value in stack.model.state_dict().items()
            },
        }
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
        fold_receipts.append(
            {
                "fold": descriptors[fold_index],
                "sampling": sampling,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "test_metrics": model_ladder["base"],
                "test_cost_ladder": model_ladder,
                "baselines": {
                    "cash": cash_ladder,
                    "equal_weight_available_assets": equal_weight_ladder,
                },
                "excess_vs_equal_weight": _excess_ladder(model_ladder, equal_weight_ladder),
                "last_training_metrics": history[-1],
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIRMATION_RECEIPT_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "confirmation_plan_sha256": plan_sha256,
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "screen_selection_sha256": screen_selection_sha256,
        "selection_identity": selection["selection_identity"],
        "runtime": dict(plan["runtime"]),
        "global_index": index,
        "setting_index": row["setting_index"],
        "setting_id": row["setting_id"],
        "seed": row["seed"],
        "trial_config": row["config"],
        "folds": fold_receipts,
        "aggregate": _confirmation_aggregate(fold_receipts),
    }
    _write_exclusive_json(output_dir / "confirmation-receipt.json", receipt)
    return receipt


def _validate_confirmation_receipt_against_plan(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    _require_development_header(
        receipt,
        kind=CONFIRMATION_RECEIPT_KIND,
        location="confirmation receipt",
    )
    expected = {
        "confirmation_plan_sha256": plan_sha256,
        "cache_identity": plan["cache_identity"],
        "cache_sha256": plan["cache_sha256"],
        "search_identity": plan["search_identity"],
        "base_dataset_identity": plan["base_dataset_identity"],
        "lockbox_partition_names_hash": plan["lockbox_partition_names_hash"],
        "screen_selection_sha256": plan["screen_selection_sha256"],
        "selection_identity": plan["selection_identity"],
        "runtime": plan["runtime"],
        "global_index": row["global_index"],
        "setting_index": row["setting_index"],
        "setting_id": row["setting_id"],
        "seed": row["seed"],
        "trial_config": row["config"],
    }
    for name, wanted in expected.items():
        if receipt.get(name) != wanted:
            raise PipelineValidationError(f"confirmation receipt {name} does not match its plan")
    fold_receipts = _require_list(receipt.get("folds"), "confirmation receipt.folds")
    if len(fold_receipts) != 3:
        raise PipelineValidationError("confirmation receipt must contain exactly three fold tests")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(fold_receipts):
        fold = dict(_require_mapping(raw, f"confirmation receipt.folds[{index}]"))
        descriptor = plan["folds"][index]
        if fold.get("fold") != descriptor:
            raise PipelineValidationError("confirmation fold descriptor differs from plan")
        model_ladder = _validate_cost_ladder(
            fold.get("test_cost_ladder"),
            f"confirmation receipt.folds[{index}].test_cost_ladder",
        )
        if fold.get("test_metrics") != model_ladder["base"]:
            raise PipelineValidationError("confirmation base test metric is not the 10bp rung")
        expected_observations = int(descriptor["test_stop"]) - int(descriptor["test_start"])
        if any(
            int(metric["observations"]) != expected_observations
            for metric in model_ladder.values()
        ):
            raise PipelineValidationError("confirmation test series length differs from its fold")
        baselines = _require_mapping(
            fold.get("baselines"), f"confirmation receipt.folds[{index}].baselines"
        )
        _require_exact_keys(
            baselines,
            {"cash", "equal_weight_available_assets"},
            f"confirmation receipt.folds[{index}].baselines",
        )
        cash = _validate_cost_ladder(
            baselines["cash"], f"confirmation receipt.folds[{index}].baselines.cash"
        )
        equal_weight = _validate_cost_ladder(
            baselines["equal_weight_available_assets"],
            f"confirmation receipt.folds[{index}].baselines.equal_weight_available_assets",
        )
        if any(
            int(metric["observations"]) != expected_observations
            for ladder in (cash, equal_weight)
            for metric in ladder.values()
        ):
            raise PipelineValidationError("confirmation baseline series length differs from its fold")
        for metric in cash.values():
            if (
                float(metric["net_total_return"]) != 0.0
                or float(metric["mean_total_one_way_turnover"]) != 0.0
                or any(float(value) != 0.0 for value in metric["daily_net_returns"])
            ):
                raise PipelineValidationError("CASH baseline must be identically zero")
        expected_excess = _excess_ladder(model_ladder, equal_weight)
        if fold.get("excess_vs_equal_weight") != expected_excess:
            raise PipelineValidationError("confirmation excess-vs-equal-weight metrics drifted")
        checkpoint_path = _safe_child(
            receipt_path.parent,
            fold.get("checkpoint"),
            f"fold-{index:02d}.pt",
        )
        _checkpoint_raw, loaded = _read_pinned_torch(
            checkpoint_path,
            fold.get("checkpoint_sha256"),
            f"confirmation fold {index} checkpoint",
        )
        _validate_checkpoint_common(
            loaded,
            expected={
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "confirmation-fold-checkpoint",
                "label": ppo.DEVELOPMENT_LABEL,
                "development_only": True,
                "bars_only": True,
                "confirmation_plan_sha256": plan_sha256,
                "screen_selection_sha256": plan["screen_selection_sha256"],
                "selection_identity": plan["selection_identity"],
                "cache_identity": plan["cache_identity"],
                "global_index": row["global_index"],
                "setting_index": row["setting_index"],
                "setting_id": row["setting_id"],
                "seed": row["seed"],
                "fold": descriptor,
                "trial_config": row["config"],
            },
            location=f"confirmation fold {index} checkpoint",
        )
        fold["test_cost_ladder"] = model_ladder
        fold["baselines"] = {
            "cash": cash,
            "equal_weight_available_assets": equal_weight,
        }
        normalized.append(fold)
    expected_aggregate = _confirmation_aggregate(normalized)
    if receipt.get("aggregate") != expected_aggregate:
        raise PipelineValidationError("confirmation aggregate does not recompute from fold series")
    return {**dict(receipt), "folds": normalized, "aggregate": expected_aggregate}


def _confirmation_candidate(
    setting_rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(setting_rows) != len(CONFIRMATION_SEEDS):
        raise PipelineValidationError("confirmation candidate needs all four seeds")
    setting_id = str(setting_rows[0][0]["setting_id"])
    setting_index = int(setting_rows[0][0]["setting_index"])
    seed_metrics: list[dict[str, Any]] = []
    all_folds: list[Mapping[str, Any]] = []
    for row, receipt in setting_rows:
        all_folds.extend(receipt["folds"])
        metric = receipt["aggregate"]["model_cost_ladder"]["stress_20bp"]
        excess = receipt["aggregate"]["excess_vs_equal_weight"]["stress_20bp"]
        seed_metrics.append(
            {
                "seed": row["seed"],
                "pooled_20bp_return": float(metric["net_total_return"]),
                "pooled_20bp_sharpe": float(metric["net_annualized_sharpe"]),
                "pooled_20bp_turnover": float(metric["mean_total_one_way_turnover"]),
                "pooled_20bp_return_excess_vs_equal_weight": float(
                    excess["net_total_return_difference"]
                ),
            }
        )
    ensemble_daily: list[float] = []
    equal_weight_daily: list[float] = []
    for fold_index in range(3):
        seed_series = [
            [
                float(value)
                for value in receipt["folds"][fold_index]["test_cost_ladder"]["stress_20bp"][
                    "daily_net_returns"
                ]
            ]
            for _row, receipt in setting_rows
        ]
        lengths = {len(values) for values in seed_series}
        if len(lengths) != 1:
            raise PipelineValidationError("confirmation seed return series lengths differ")
        ensemble_daily.extend(
            sum(values[position] for values in seed_series) / len(seed_series)
            for position in range(len(seed_series[0]))
        )
        baseline_series = [
            [
                float(value)
                for value in receipt["folds"][fold_index]["baselines"][
                    "equal_weight_available_assets"
                ]["stress_20bp"]["daily_net_returns"]
            ]
            for _row, receipt in setting_rows
        ]
        if any(values != baseline_series[0] for values in baseline_series[1:]):
            raise PipelineValidationError("equal-weight baselines unexpectedly vary across seeds")
        equal_weight_daily.extend(baseline_series[0])
    seed_material = int(hashlib.sha256(setting_id.encode()).hexdigest()[:8], 16)
    confidence = moving_block_bootstrap_mean_ci(ensemble_daily, seed=seed_material)
    positive_fold_count = sum(
        float(fold["test_cost_ladder"]["stress_20bp"]["net_total_return"]) > 0.0
        for fold in all_folds
    )
    minimum_coverage = min(
        float(fold["test_cost_ladder"]["stress_20bp"]["decision_coverage"])
        for fold in all_folds
    )
    positive_seed_count = sum(
        float(value["pooled_20bp_return"]) > 0.0 for value in seed_metrics
    )
    seed_averaged_pooled_return = _compound(ensemble_daily)
    metrics = {
        "minimum_decision_coverage": minimum_coverage,
        "positive_seed_count_20bp": positive_seed_count,
        "positive_fold_count_20bp": positive_fold_count,
        "seed_averaged_pooled_20bp_return": seed_averaged_pooled_return,
        "seed_averaged_equal_weight_pooled_20bp_return": _compound(equal_weight_daily),
        "seed_averaged_pooled_20bp_return_excess_vs_equal_weight": _compound(ensemble_daily)
        - _compound(equal_weight_daily),
        "worst_seed_pooled_20bp_sharpe": min(
            float(value["pooled_20bp_sharpe"]) for value in seed_metrics
        ),
        "mean_seed_pooled_20bp_sharpe": sum(
            float(value["pooled_20bp_sharpe"]) for value in seed_metrics
        )
        / len(seed_metrics),
        "mean_seed_pooled_20bp_turnover": sum(
            float(value["pooled_20bp_turnover"]) for value in seed_metrics
        )
        / len(seed_metrics),
        "mean_seed_pooled_20bp_return_excess_vs_equal_weight": sum(
            float(value["pooled_20bp_return_excess_vs_equal_weight"])
            for value in seed_metrics
        )
        / len(seed_metrics),
        "bootstrap_95pct_mean_daily_return_20bp": confidence,
    }
    gates = {
        "all_four_seeds_and_three_folds": len(all_folds) == 12,
        "coverage_at_least_95pct": minimum_coverage >= 0.95,
        "at_least_three_positive_seed_pooled_returns_20bp": positive_seed_count >= 3,
        "at_least_eight_positive_fold_returns_20bp": positive_fold_count >= 8,
        "seed_averaged_pooled_return_20bp_positive": seed_averaged_pooled_return > 0.0,
        "bootstrap_lower_mean_daily_return_20bp_positive": float(confidence["lower"]) > 0.0,
    }
    first_config = dict(setting_rows[0][0]["config"])
    first_config["seed"] = CONFIRMATION_SEEDS[0]
    return {
        "setting_index": setting_index,
        "setting_id": setting_id,
        "trial_config": first_config,
        "seed_summaries": seed_metrics,
        "metrics": metrics,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def aggregate_confirmation_receipts(
    cache_path: str | Path,
    cache_sha256: str,
    plan_path: str | Path,
    plan_sha256: str,
    screen_selection_path: str | Path,
    screen_selection_sha256: str,
    receipts_root: str | Path,
    receipt_set_manifest_path: str | Path,
    receipt_set_manifest_sha256: str,
    output_path: str | Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    """Validate exact 2x4 confirmation receipts and select one robust winner."""

    _require_acknowledgement(acknowledgement, "Confirmation aggregation")
    plan = validate_confirmation_plan(
        _read_pinned_json(plan_path, plan_sha256, "confirmation plan")
    )
    selection = load_screen_selection(
        screen_selection_path,
        expected_sha256=screen_selection_sha256,
    )
    if (
        plan["screen_selection_sha256"] != screen_selection_sha256
        or plan["selection_identity"] != selection.get("selection_identity")
    ):
        raise PipelineValidationError("confirmation plan/selection binding mismatch")
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256, device="cpu")
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
    ):
        if plan[name] != cache[name] or selection.get(name) != cache[name]:
            raise PipelineValidationError(f"confirmation aggregate {name} mismatch")
    receipt_artifacts, _receipt_set = _load_receipt_set(
        receipt_set_manifest_path,
        receipt_set_manifest_sha256,
        root=receipts_root,
        kind=CONFIRMATION_RECEIPT_SET_KIND,
        plan_sha256=plan_sha256,
        expected_count=8,
    )
    validated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    bindings: list[dict[str, Any]] = []
    for index, row in enumerate(plan["trials"]):
        artifact = receipt_artifacts[index]
        path = artifact.path
        receipt = artifact.payload
        _require_development_header(
            receipt,
            kind=CONFIRMATION_RECEIPT_KIND,
            location=f"confirmation receipt {index}",
        )
        normalized = _validate_confirmation_receipt_against_plan(
            receipt,
            receipt_path=path,
            row=row,
            plan=plan,
            plan_sha256=plan_sha256,
        )
        validated.append((row, normalized))
        bindings.append(
            {
                "global_index": index,
                "setting_id": row["setting_id"],
                "seed": row["seed"],
                "receipt_sha256": artifact.sha256,
            }
        )
    candidates = [
        _confirmation_candidate(validated[offset : offset + len(CONFIRMATION_SEEDS)])
        for offset in range(0, len(validated), len(CONFIRMATION_SEEDS))
    ]
    eligible = [value for value in candidates if value["eligible"]]
    eligible.sort(
        key=lambda value: (
            -float(value["metrics"]["worst_seed_pooled_20bp_sharpe"]),
            -float(value["metrics"]["mean_seed_pooled_20bp_sharpe"]),
            -float(value["metrics"]["bootstrap_95pct_mean_daily_return_20bp"]["lower"]),
            -float(value["metrics"]["mean_seed_pooled_20bp_return_excess_vs_equal_weight"]),
            float(value["metrics"]["mean_seed_pooled_20bp_turnover"]),
            str(value["setting_id"]),
        )
    )
    if not eligible:
        raise PipelineValidationError("no confirmation setting passed every robust positive-net gate")
    winner = eligible[0]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIRMATION_WINNER_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "confirmation_plan_sha256": plan_sha256,
        "receipt_set_manifest_sha256": receipt_set_manifest_sha256,
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "screen_selection_sha256": screen_selection_sha256,
        "selection_identity": selection["selection_identity"],
        "receipt_sha256s": bindings,
        "robust_gates": {
            "required_seed_count": 4,
            "required_fold_count_per_seed": 3,
            "minimum_decision_coverage": 0.95,
            "minimum_positive_seed_pooled_returns_20bp": 3,
            "minimum_positive_fold_returns_20bp": 8,
            "require_positive_seed_averaged_pooled_return_20bp": True,
            "require_positive_bootstrap_95pct_lower_mean_daily_return_20bp": True,
            "bootstrap_block_length": 5,
            "bootstrap_samples": 2000,
        },
        "ranking_rule": [
            "worst_seed_pooled_20bp_sharpe_desc",
            "mean_seed_pooled_20bp_sharpe_desc",
            "bootstrap_95pct_lower_mean_daily_return_20bp_desc",
            "mean_seed_pooled_20bp_return_excess_vs_equal_weight_desc",
            "mean_seed_pooled_20bp_turnover_asc",
            "setting_id_asc",
        ],
        "candidate_summaries": candidates,
        "winning_setting": {
            "setting_index": winner["setting_index"],
            "setting_id": winner["setting_id"],
            "trial_config": winner["trial_config"],
            "confirmation_metrics": winner["metrics"],
        },
    }
    payload["winner_identity"] = _artifact_identity(payload, "winner_identity")
    _write_exclusive_json(Path(output_path), payload)
    return payload


def run_refit_worker(
    cache_path: str | Path,
    cache_sha256: str,
    plan_path: str | Path,
    plan_sha256: str,
    confirmation_winner_path: str | Path,
    confirmation_winner_sha256: str,
    output_root: str | Path,
    *,
    index: int,
    device: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Refit one confirmed member on every decision in the pre-2026 cache."""

    _require_acknowledgement(acknowledgement, "Refit worker")
    plan = validate_refit_plan(_read_pinned_json(plan_path, plan_sha256, "refit plan"))
    winner = load_confirmation_winner(
        confirmation_winner_path,
        expected_sha256=confirmation_winner_sha256,
    )
    if (
        plan["confirmation_winner_sha256"] != confirmation_winner_sha256
        or plan["winner_identity"] != winner.get("winner_identity")
    ):
        raise PipelineValidationError("refit plan does not bind the pinned confirmation winner")
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256, device=device)
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
    ):
        if plan[name] != cache[name] or winner.get(name) != cache[name]:
            raise PipelineValidationError(f"refit {name} does not match the pinned cache")
    trials = plan["trials"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(trials):
        raise PipelineValidationError("refit worker index is out of range")
    row = trials[index]
    winning = winner["winning_setting"]
    if (row["setting_index"], row["setting_id"]) != (
        winning["setting_index"],
        winning["setting_id"],
    ):
        raise PipelineValidationError("refit row is not the confirmed winning setting")
    winning_config = dict(winning["trial_config"])
    if {**row["config"], "seed": winning_config["seed"]} != winning_config:
        raise PipelineValidationError("refit row hyperparameters differ from the winner")
    trial = ppo.TrialConfig(**row["config"])
    decision_stop = len(cache["exchange_dates"]) - 1
    if decision_stop <= 0:
        raise PipelineValidationError("pre-2026 cache contains no training decisions")
    ppo._seed_everything(trial.seed)
    stack, history, sampling = ppo.train_cache_block(
        cache,
        start=0,
        stop=decision_stop,
        trial=trial,
    )
    output_dir = Path(output_root) / f"trial-{index:04d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REFIT_MEMBER_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "refit_plan_sha256": plan_sha256,
        "confirmation_winner_sha256": confirmation_winner_sha256,
        "winner_identity": winner["winner_identity"],
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "runtime": dict(plan["runtime"]),
        "global_index": index,
        "setting_index": row["setting_index"],
        "setting_id": row["setting_id"],
        "seed": row["seed"],
        "trial_config": row["config"],
        "model_state_dict": {
            name: value.detach().cpu() for name, value in stack.model.state_dict().items()
        },
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REFIT_MEMBER_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "refit_plan_sha256": plan_sha256,
        "confirmation_winner_sha256": confirmation_winner_sha256,
        "winner_identity": winner["winner_identity"],
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "runtime": dict(plan["runtime"]),
        "global_index": index,
        "setting_index": row["setting_index"],
        "setting_id": row["setting_id"],
        "seed": row["seed"],
        "trial_config": row["config"],
        "full_pre2026_training_range": {
            "decision_start": 0,
            "decision_stop": decision_stop,
            "decision_count": decision_stop,
            "first_decision_date": cache["exchange_dates"][0],
            "last_decision_date": cache["exchange_dates"][decision_stop - 1],
            "final_label_support_date": cache["exchange_dates"][decision_stop],
            "cutoff_exclusive": ppo.TEST_START.isoformat(),
        },
        "sampling": sampling,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "last_training_metrics": history[-1],
    }
    _write_exclusive_json(output_dir / "refit-receipt.json", receipt)
    return receipt


def _validated_refit_member(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_sha256: str,
    winner: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, Mapping[str, Any]]:
    _require_development_header(receipt, kind=REFIT_MEMBER_KIND, location="refit receipt")
    expected = {
        "refit_plan_sha256": plan_sha256,
        "confirmation_winner_sha256": plan["confirmation_winner_sha256"],
        "winner_identity": plan["winner_identity"],
        "cache_identity": plan["cache_identity"],
        "cache_sha256": plan["cache_sha256"],
        "search_identity": plan["search_identity"],
        "base_dataset_identity": plan["base_dataset_identity"],
        "lockbox_partition_names_hash": plan["lockbox_partition_names_hash"],
        "runtime": plan["runtime"],
        "global_index": row["global_index"],
        "setting_index": row["setting_index"],
        "setting_id": row["setting_id"],
        "seed": row["seed"],
        "trial_config": row["config"],
    }
    for name, wanted in expected.items():
        if receipt.get(name) != wanted:
            raise PipelineValidationError(f"refit receipt {name} does not match its plan")
    training_range = _require_mapping(
        receipt.get("full_pre2026_training_range"), "full_pre2026_training_range"
    )
    if training_range.get("decision_start") != 0 or training_range.get("cutoff_exclusive") != "2026-01-01":
        raise PipelineValidationError("refit receipt does not prove an all-pre-2026 training range")
    if not str(training_range.get("final_label_support_date", "")) < "2026-01-01":
        raise PipelineValidationError("refit label support reaches the 2026 lockbox")
    checkpoint_path = _safe_child(receipt_path.parent, receipt.get("checkpoint"), "checkpoint.pt")
    checkpoint_sha256 = _require_digest(receipt.get("checkpoint_sha256"), "checkpoint_sha256")
    checkpoint_raw, checkpoint = _read_pinned_torch(
        checkpoint_path,
        checkpoint_sha256,
        "refit checkpoint",
    )
    _validate_checkpoint_common(
        checkpoint,
        expected={
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": REFIT_MEMBER_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            **expected,
        },
        location="refit checkpoint",
    )
    winning = winner["winning_setting"]
    if (row["setting_index"], row["setting_id"]) != (
        winning["setting_index"],
        winning["setting_id"],
    ):
        raise PipelineValidationError("refit member is not the confirmed winner")
    return dict(receipt), checkpoint_raw, checkpoint


def seal_refit_ensemble(
    plan_path: str | Path,
    plan_sha256: str,
    confirmation_winner_path: str | Path,
    confirmation_winner_sha256: str,
    receipts_root: str | Path,
    receipt_set_manifest_path: str | Path,
    receipt_set_manifest_sha256: str,
    output_dir: str | Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    """Validate exactly four refits and atomically publish an immutable ensemble."""

    _require_acknowledgement(acknowledgement, "Ensemble sealing")
    plan = validate_refit_plan(_read_pinned_json(plan_path, plan_sha256, "refit plan"))
    winner = load_confirmation_winner(
        confirmation_winner_path,
        expected_sha256=confirmation_winner_sha256,
    )
    if (
        plan["confirmation_winner_sha256"] != confirmation_winner_sha256
        or plan["winner_identity"] != winner.get("winner_identity")
    ):
        raise PipelineValidationError("refit plan/winner binding mismatch")
    for name in (
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
    ):
        if plan[name] != winner.get(name):
            raise PipelineValidationError(f"refit plan/winner {name} mismatch")
    receipt_artifacts, _receipt_set = _load_receipt_set(
        receipt_set_manifest_path,
        receipt_set_manifest_sha256,
        root=receipts_root,
        kind=REFIT_RECEIPT_SET_KIND,
        plan_sha256=plan_sha256,
        expected_count=4,
    )
    members: list[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]] = []
    for index, row in enumerate(plan["trials"]):
        artifact = receipt_artifacts[index]
        receipt_path = artifact.path
        receipt = artifact.payload
        _require_development_header(
            receipt,
            kind=REFIT_MEMBER_KIND,
            location=f"refit receipt {index}",
        )
        members.append(
            _validated_refit_member(
                receipt,
                receipt_path=receipt_path,
                row=row,
                plan=plan,
                plan_sha256=plan_sha256,
                winner=winner,
            )
        )
    final = Path(output_dir)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable ensemble directory {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    temporary.mkdir(mode=0o750)
    try:
        manifest_members: list[dict[str, Any]] = []
        for index, ((member_receipt, checkpoint_raw, _checkpoint), seed) in enumerate(
            zip(members, CONFIRMATION_SEEDS, strict=True)
        ):
            name = f"member-{index:02d}-seed-{seed}.pt"
            destination = temporary / name
            with destination.open("xb") as member_file:
                member_file.write(checkpoint_raw)
            digest = _sha256_file(destination)
            if digest != member_receipt["checkpoint_sha256"]:
                raise PipelineValidationError("copied refit checkpoint digest changed")
            manifest_members.append(
                {
                    "member_index": index,
                    "name": name,
                    "seed": seed,
                    "checkpoint_sha256": digest,
                    "refit_receipt_sha256": receipt_artifacts[index].sha256,
                }
            )
        winning = winner["winning_setting"]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ENSEMBLE_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "not_reportable": True,
            "refit_plan_sha256": plan_sha256,
            "receipt_set_manifest_sha256": receipt_set_manifest_sha256,
            "confirmation_winner_sha256": confirmation_winner_sha256,
            "winner_identity": winner["winner_identity"],
            "cache_identity": plan["cache_identity"],
            "cache_sha256": plan["cache_sha256"],
            "search_identity": plan["search_identity"],
            "base_dataset_identity": plan["base_dataset_identity"],
            "lockbox_partition_names_hash": plan["lockbox_partition_names_hash"],
            "setting_index": winning["setting_index"],
            "setting_id": winning["setting_id"],
            "trial_config": winning["trial_config"],
            "seeds": list(CONFIRMATION_SEEDS),
            "runtime": dict(plan["runtime"]),
            "members": manifest_members,
            "action_combination": "arithmetic-mean-of-four-deterministic-requested-weight-vectors",
        }
        manifest["ensemble_identity"] = _artifact_identity(manifest, "ensemble_identity")
        _write_exclusive_json(temporary / "ensemble-manifest.json", manifest)
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def _load_ensemble_members_before_lockbox(
    ensemble_dir: Path,
    manifest_sha256: str,
    *,
    action_dim: int,
    device: str,
) -> tuple[dict[str, Any], list[ppo.SharedAssetRecurrentActorCritic], list[ppo.TrialConfig]]:
    manifest_path = ensemble_dir / "ensemble-manifest.json"
    manifest = load_ensemble_manifest(manifest_path, expected_sha256=manifest_sha256)
    _check_artifact_identity(manifest, "ensemble_identity", "ensemble manifest")
    required = {
        "refit_plan_sha256",
        "receipt_set_manifest_sha256",
        "confirmation_winner_sha256",
        "winner_identity",
        "cache_identity",
        "cache_sha256",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
    }
    for name in required:
        _require_digest(manifest.get(name), f"ensemble manifest.{name}")
    if manifest.get("not_reportable") is not True:
        raise PipelineValidationError("ensemble manifest must remain explicitly non-reportable")
    if manifest.get("seeds") != list(CONFIRMATION_SEEDS):
        raise PipelineValidationError("ensemble manifest must contain the exact four confirmation seeds")
    if (
        manifest.get("action_combination")
        != "arithmetic-mean-of-four-deterministic-requested-weight-vectors"
    ):
        raise PipelineValidationError("ensemble requested-weight combination contract changed")
    runtime = _validated_runtime(manifest.get("runtime"), "ensemble manifest.runtime")
    base_config = _validated_trial_config(manifest.get("trial_config"), "ensemble trial_config")
    members = _require_list(manifest.get("members"), "ensemble manifest.members")
    if len(members) != 4:
        raise PipelineValidationError("ensemble manifest must bind exactly four members")
    models: list[ppo.SharedAssetRecurrentActorCritic] = []
    trials: list[ppo.TrialConfig] = []
    for index, (raw_member, seed) in enumerate(zip(members, CONFIRMATION_SEEDS, strict=True)):
        member = _require_mapping(raw_member, f"ensemble manifest.members[{index}]")
        _require_exact_keys(
            member,
            {
                "member_index",
                "name",
                "seed",
                "checkpoint_sha256",
                "refit_receipt_sha256",
            },
            f"ensemble manifest.members[{index}]",
        )
        expected_name = f"member-{index:02d}-seed-{seed}.pt"
        if member["member_index"] != index or member["seed"] != seed:
            raise PipelineValidationError("ensemble member index/seed ordering changed")
        _require_digest(member["refit_receipt_sha256"], f"member {index} refit receipt digest")
        checkpoint_path = _safe_child(ensemble_dir, member["name"], expected_name)
        _raw, checkpoint = _read_pinned_torch(
            checkpoint_path,
            member["checkpoint_sha256"],
            f"ensemble member {index}",
        )
        member_config = _validated_trial_config(
            checkpoint.get("trial_config"), f"ensemble member {index}.trial_config"
        )
        if {**member_config, "seed": base_config["seed"]} != base_config:
            raise PipelineValidationError("ensemble member changed non-seed hyperparameters")
        if member_config["seed"] != seed:
            raise PipelineValidationError("ensemble checkpoint seed does not match its member row")
        expected_checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": REFIT_MEMBER_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "refit_plan_sha256": manifest["refit_plan_sha256"],
            "confirmation_winner_sha256": manifest["confirmation_winner_sha256"],
            "winner_identity": manifest["winner_identity"],
            "cache_identity": manifest["cache_identity"],
            "cache_sha256": manifest["cache_sha256"],
            "search_identity": manifest["search_identity"],
            "base_dataset_identity": manifest["base_dataset_identity"],
            "lockbox_partition_names_hash": manifest["lockbox_partition_names_hash"],
            "runtime": runtime,
            "global_index": index,
            "setting_index": manifest["setting_index"],
            "setting_id": manifest["setting_id"],
            "seed": seed,
            "trial_config": member_config,
        }
        _validate_checkpoint_common(
            checkpoint,
            expected=expected_checkpoint,
            location=f"ensemble member {index}",
        )
        trial = ppo.TrialConfig(**member_config)
        model = ppo.SharedAssetRecurrentActorCritic(
            observation_key=ppo.BarsOnlyObservationAdapter.observation_key,
            asset_feature_dim=ppo.BarsOnlyObservationAdapter.asset_feature_dim,
            hidden_dim=trial.hidden_dim,
            action_dim=action_dim,
            shared_mlp_layers=trial.shared_mlp_layers,
        )
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (RuntimeError, TypeError) as error:
            raise PipelineValidationError(
                f"ensemble member {index} model state is incompatible"
            ) from error
        models.append(model.to(device))
        trials.append(trial)
    return manifest, models, trials


@torch.no_grad()
def _evaluate_requested_weight_ensemble(
    models: Sequence[ppo.SharedAssetRecurrentActorCritic],
    trials: Sequence[ppo.TrialConfig],
    data: HistoricalMarketData,
    *,
    cost_bps: float,
) -> dict[str, Any]:
    if len(models) != 4 or len(trials) != 4 or data.batch_size != 1:
        raise PipelineValidationError("final evaluation requires four models and one chronology")
    trial = trials[0]
    environment = _evaluation_environment(data, trial, cost_bps=cost_bps)
    evaluators = [RecurrentPPO(model, PPOConfig(seed=value.seed)) for model, value in zip(models, trials)]
    observation, _info = environment.reset()
    states = [evaluator.initial_recurrent_state(observation) for evaluator in evaluators]
    returns: list[float] = []
    turnovers: list[float] = []
    risky_available: list[bool] = []
    while True:
        actions = [
            evaluator.act(observation, deterministic=True, recurrent_state=state)
            for evaluator, state in zip(evaluators, states, strict=True)
        ]
        requested = torch.stack([action.action for action in actions], dim=0).mean(dim=0)
        mask = observation.action_mask
        if mask is None:
            raise PipelineValidationError("ensemble evaluation requires an action mask")
        risky_available.extend(bool(value) for value in mask[:, 1:].any(dim=-1).cpu().tolist())
        transition = environment.step(ActionBatch(action=requested))
        returns.extend(float(value) for value in transition.reward.double().cpu().tolist())
        turnovers.extend(
            float(value) for value in transition.info["recent_turnover"].double().cpu().tolist()
        )
        states = [action.recurrent_state for action in actions]
        observation = transition.next_observation
        if bool(transition.done.all().item()):
            break
    return _series_metrics(
        returns,
        turnovers,
        risky_available=risky_available,
        cost_bps=cost_bps,
    )


def _ensemble_cost_ladder(
    models: Sequence[ppo.SharedAssetRecurrentActorCritic],
    trials: Sequence[ppo.TrialConfig],
    data: HistoricalMarketData,
) -> dict[str, dict[str, Any]]:
    return {
        key: _evaluate_requested_weight_ensemble(models, trials, data, cost_bps=cost)
        for key, cost in _COST_BY_KEY.items()
    }


def evaluate_2026_ensemble(
    data_root: str | Path,
    ensemble_dir: str | Path,
    ensemble_manifest_sha256: str,
    output_path: str | Path,
    *,
    bar_seconds: int,
    device: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Open the 2026 lockbox once and evaluate four averaged requested actions."""

    _require_acknowledgement(acknowledgement, "2026 ensemble evaluation")
    root = Path(data_root)
    ensemble = Path(ensemble_dir)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite 2026 evaluation receipt {output}")
    # Validate the ensemble receipt and every checkpoint before any operation
    # resolves/stats/opens a lockbox bars file.
    action_dim = len(ppo.declared_universe_actions(root))
    manifest, models, trials = _load_ensemble_members_before_lockbox(
        ensemble,
        ensemble_manifest_sha256,
        action_dim=action_dim,
        device=device,
    )
    search_plan = ppo.build_search_plan(root, bar_seconds=bar_seconds)
    safe_expected = {
        "search_identity": search_plan.search_identity,
        "base_dataset_identity": search_plan.base_dataset_identity,
        "lockbox_partition_names_hash": search_plan.lockbox_partition_names_hash,
    }
    for name, wanted in safe_expected.items():
        if manifest.get(name) != wanted:
            raise PipelineValidationError(
                f"ensemble {name} does not match the current pre-lockbox namespace"
            )
    access_marker = ensemble / "test-accessed.json"
    marker_payload = {
        "schema_version": SCHEMA_VERSION,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "purpose": "single-use 2026 lockbox access",
        "ensemble_manifest_sha256": ensemble_manifest_sha256,
        "ensemble_identity": manifest["ensemble_identity"],
        "member_checkpoint_sha256s": [value["checkpoint_sha256"] for value in manifest["members"]],
    }
    _write_exclusive_json(access_marker, marker_payload)
    # Only after the irreversible marker may full-content test hashes or test
    # rows be resolved.  A downstream failure consumes the one allowed access.
    evaluation_plan = ppo.build_evaluation_plan(root, bar_seconds=bar_seconds)
    for name, wanted in safe_expected.items():
        if getattr(evaluation_plan, name) != wanted:
            raise PipelineValidationError(f"evaluation plan {name} changed after access marking")
    data, exchange_dates = ppo.load_market_data(
        root,
        evaluation_plan.test,
        bar_seconds=bar_seconds,
        device=device,
        date_start=ppo.TEST_START,
        date_end=ppo.TEST_START.replace(month=12, day=31),
    )
    model_ladder = _ensemble_cost_ladder(models, trials, data)
    cash_ladder = _baseline_cost_ladder(
        data,
        trials[0],
        equal_weight_available_assets=False,
    )
    equal_weight_ladder = _baseline_cost_ladder(
        data,
        trials[0],
        equal_weight_available_assets=True,
    )
    excess = _excess_ladder(model_ladder, equal_weight_ladder)
    confidence = moving_block_bootstrap_mean_ci(
        model_ladder["stress_20bp"]["daily_net_returns"],
        seed=2026,
    )
    positive = (
        float(model_ladder["base"]["net_total_return"]) > 0.0
        and float(model_ladder["stress_20bp"]["net_total_return"]) > 0.0
        and float(confidence["lower"]) > 0.0
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "single-use-2026-ensemble-evaluation",
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "not_reportable": True,
        "profitability_label": (
            "development-only-positive-under-specified-protocol"
            if positive
            else "development-only-not-positive-under-specified-protocol"
        ),
        "ensemble_manifest_sha256": ensemble_manifest_sha256,
        "ensemble_identity": manifest["ensemble_identity"],
        "test_access_marker_sha256": _sha256_file(access_marker),
        "test_identity": evaluation_plan.test_identity,
        "test_exchange_date_range": [exchange_dates[0], exchange_dates[-1]],
        "model_cost_ladder": model_ladder,
        "baselines": {
            "cash": cash_ladder,
            "equal_weight_available_assets": equal_weight_ladder,
        },
        "excess_vs_equal_weight": excess,
        "bootstrap_95pct_mean_daily_return_20bp": confidence,
        "positive_label_gates": {
            "positive_base_10bp_total_return": float(
                model_ladder["base"]["net_total_return"]
            )
            > 0.0,
            "positive_20bp_total_return": float(
                model_ladder["stress_20bp"]["net_total_return"]
            )
            > 0.0,
            "positive_bootstrap_95pct_lower_mean_daily_return_20bp": float(
                confidence["lower"]
            )
            > 0.0,
        },
    }
    _write_exclusive_json(output, payload)
    return payload


def _worker_index(value: int | None) -> int:
    raw: int | str | None = value if value is not None else os.environ.get("JOB_COMPLETION_INDEX")
    if raw is None:
        raise PipelineValidationError("worker needs --index or JOB_COMPLETION_INDEX")
    try:
        index = int(raw)
    except (TypeError, ValueError) as error:
        raise PipelineValidationError("worker index must be an integer") from error
    return index


def _add_cache_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)


def _add_receipt_set_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipts-root", required=True)
    parser.add_argument("--receipt-set-manifest", required=True)
    parser.add_argument("--receipt-set-manifest-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    screen = commands.add_parser("aggregate-screen", help="Select the exact top two screen settings.")
    _add_cache_plan_arguments(screen)
    _add_receipt_set_arguments(screen)
    screen.add_argument("--output", required=True)
    screen.add_argument("--development-ack", required=True)

    confirmation = commands.add_parser(
        "confirmation-worker",
        help="Run one selected-setting/seed row across all three fold tests.",
    )
    _add_cache_plan_arguments(confirmation)
    confirmation.add_argument("--screen-selection", required=True)
    confirmation.add_argument("--screen-selection-sha256", required=True)
    confirmation.add_argument("--output-root", required=True)
    confirmation.add_argument("--index", type=int)
    confirmation.add_argument("--device", default="cuda")
    confirmation.add_argument("--development-ack", required=True)

    confirm_aggregate = commands.add_parser(
        "aggregate-confirmation",
        help="Apply robust gates to the exact two-by-four confirmation set.",
    )
    _add_cache_plan_arguments(confirm_aggregate)
    confirm_aggregate.add_argument("--screen-selection", required=True)
    confirm_aggregate.add_argument("--screen-selection-sha256", required=True)
    _add_receipt_set_arguments(confirm_aggregate)
    confirm_aggregate.add_argument("--output", required=True)
    confirm_aggregate.add_argument("--development-ack", required=True)

    refit = commands.add_parser("refit-worker", help="Refit one final seed on all pre-2026 data.")
    _add_cache_plan_arguments(refit)
    refit.add_argument("--confirmation-winner", required=True)
    refit.add_argument("--confirmation-winner-sha256", required=True)
    refit.add_argument("--output-root", required=True)
    refit.add_argument("--index", type=int)
    refit.add_argument("--device", default="cuda")
    refit.add_argument("--development-ack", required=True)

    seal = commands.add_parser("seal-ensemble", help="Seal the exact four-member refit ensemble.")
    seal.add_argument("--plan", required=True)
    seal.add_argument("--plan-sha256", required=True)
    seal.add_argument("--confirmation-winner", required=True)
    seal.add_argument("--confirmation-winner-sha256", required=True)
    _add_receipt_set_arguments(seal)
    seal.add_argument("--output-dir", required=True)
    seal.add_argument("--development-ack", required=True)

    evaluate = commands.add_parser(
        "evaluate-test",
        help="Consume the single-use 2026 lockbox with the confirmed four-model ensemble.",
    )
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--ensemble-dir", required=True)
    evaluate.add_argument("--ensemble-manifest-sha256", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bar-seconds", type=int, default=300)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--development-ack", required=True)
    return parser


def _summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "artifact_kind",
            "global_index",
            "setting_id",
            "seed",
            "selection_identity",
            "winner_identity",
            "ensemble_identity",
            "profitability_label",
        )
        if key in value
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: Mapping[str, Any]
    if args.command == "aggregate-screen":
        result = aggregate_screen_receipts(
            args.cache,
            args.cache_sha256,
            args.plan,
            args.plan_sha256,
            args.receipts_root,
            args.receipt_set_manifest,
            args.receipt_set_manifest_sha256,
            args.output,
            acknowledgement=args.development_ack,
        )
    elif args.command == "confirmation-worker":
        result = run_confirmation_worker(
            args.cache,
            args.cache_sha256,
            args.plan,
            args.plan_sha256,
            args.screen_selection,
            args.screen_selection_sha256,
            args.output_root,
            index=_worker_index(args.index),
            device=args.device,
            acknowledgement=args.development_ack,
        )
    elif args.command == "aggregate-confirmation":
        result = aggregate_confirmation_receipts(
            args.cache,
            args.cache_sha256,
            args.plan,
            args.plan_sha256,
            args.screen_selection,
            args.screen_selection_sha256,
            args.receipts_root,
            args.receipt_set_manifest,
            args.receipt_set_manifest_sha256,
            args.output,
            acknowledgement=args.development_ack,
        )
    elif args.command == "refit-worker":
        result = run_refit_worker(
            args.cache,
            args.cache_sha256,
            args.plan,
            args.plan_sha256,
            args.confirmation_winner,
            args.confirmation_winner_sha256,
            args.output_root,
            index=_worker_index(args.index),
            device=args.device,
            acknowledgement=args.development_ack,
        )
    elif args.command == "seal-ensemble":
        result = seal_refit_ensemble(
            args.plan,
            args.plan_sha256,
            args.confirmation_winner,
            args.confirmation_winner_sha256,
            args.receipts_root,
            args.receipt_set_manifest,
            args.receipt_set_manifest_sha256,
            args.output_dir,
            acknowledgement=args.development_ack,
        )
    else:
        result = evaluate_2026_ensemble(
            args.data_root,
            args.ensemble_dir,
            args.ensemble_manifest_sha256,
            args.output,
            bar_seconds=args.bar_seconds,
            device=args.device,
            acknowledgement=args.development_ack,
        )
    print(json.dumps(_summary(result), sort_keys=True))
    return 0


__all__ = [
    "CONFIRMATION_PLAN_TYPE",
    "CONFIRMATION_RECEIPT_KIND",
    "CONFIRMATION_RECEIPT_SET_KIND",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_WINNER_KIND",
    "ENSEMBLE_KIND",
    "PipelineValidationError",
    "REFIT_MEMBER_KIND",
    "REFIT_PLAN_TYPE",
    "REFIT_RECEIPT_SET_KIND",
    "RUNTIME_BINDING_KEYS",
    "SCREEN_RECEIPT_SET_KIND",
    "SCREEN_SELECTION_KIND",
    "aggregate_confirmation_receipts",
    "aggregate_screen_receipts",
    "build_confirmation_plan",
    "build_receipt_set_manifest",
    "build_refit_plan",
    "evaluate_2026_ensemble",
    "load_confirmation_receipt",
    "load_confirmation_winner",
    "load_ensemble_manifest",
    "load_receipt_set_manifest",
    "load_refit_receipt",
    "load_screen_selection",
    "main",
    "moving_block_bootstrap_mean_ci",
    "run_confirmation_worker",
    "run_refit_worker",
    "seal_refit_ensemble",
    "validate_confirmation_plan",
    "validate_refit_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
