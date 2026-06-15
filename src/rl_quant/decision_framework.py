from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_quant.research_protocol import ResearchProtocolError, parse_iso_timestamp, stable_json_hash


class DecisionFrameworkError(ResearchProtocolError):
    """Raised when point-in-time decision artifacts violate the framework contract."""


def _require_nonempty(value: str, *, name: str) -> None:
    if not value:
        raise DecisionFrameworkError(f"{name} is required.")


def _validate_unit_interval(value: float, *, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise DecisionFrameworkError(f"{name} must be between 0 and 1.")


def assert_available_at(*, decision_ts: str, available_ts: str, name: str = "row") -> None:
    if parse_iso_timestamp(available_ts) > parse_iso_timestamp(decision_ts):
        raise DecisionFrameworkError(f"{name} is not point-in-time: {available_ts} is after {decision_ts}.")


def filter_point_in_time_rows(
    rows: list[dict[str, Any]],
    *,
    decision_ts: str,
    available_key: str = "available_timestamp",
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if available_key not in row:
            raise DecisionFrameworkError(f"row {index} is missing {available_key!r}.")
        if parse_iso_timestamp(str(row[available_key])) <= parse_iso_timestamp(decision_ts):
            kept.append(row)
    return kept


@dataclass(frozen=True)
class MarketDataManifest:
    dataset_id: str
    created_at: str
    source: str
    source_version: str
    symbols: list[str]
    symbol_id_map_hash: str
    start_ts: str
    end_ts: str
    bar_interval: str
    timezone: str
    corporate_action_policy: str
    calendar_id: str
    raw_payload_hash: str
    quality_report_hash: str

    def validate(self) -> None:
        for name in (
            "dataset_id",
            "created_at",
            "source",
            "source_version",
            "symbol_id_map_hash",
            "start_ts",
            "end_ts",
            "bar_interval",
            "timezone",
            "corporate_action_policy",
            "calendar_id",
            "raw_payload_hash",
            "quality_report_hash",
        ):
            _require_nonempty(str(getattr(self, name)), name=name)
        if not self.symbols:
            raise DecisionFrameworkError("symbols must not be empty.")
        if parse_iso_timestamp(self.start_ts) > parse_iso_timestamp(self.end_ts):
            raise DecisionFrameworkError("start_ts must be <= end_ts.")
        parse_iso_timestamp(self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_json_hash(self.to_dict())


@dataclass(frozen=True)
class FeatureManifest:
    feature_set_id: str
    input_dataset_ids: list[str]
    feature_names: list[str]
    feature_available_ts_rule: str
    fit_start: str | None
    fit_end: str | None
    normalizer_hash: str
    code_version: str

    def validate(self) -> None:
        _require_nonempty(self.feature_set_id, name="feature_set_id")
        _require_nonempty(self.feature_available_ts_rule, name="feature_available_ts_rule")
        _require_nonempty(self.normalizer_hash, name="normalizer_hash")
        _require_nonempty(self.code_version, name="code_version")
        if not self.input_dataset_ids:
            raise DecisionFrameworkError("input_dataset_ids must not be empty.")
        if not self.feature_names:
            raise DecisionFrameworkError("feature_names must not be empty.")
        if (self.fit_start is None) != (self.fit_end is None):
            raise DecisionFrameworkError("fit_start and fit_end must be provided together.")
        if self.fit_start is not None and self.fit_end is not None:
            if parse_iso_timestamp(self.fit_start) > parse_iso_timestamp(self.fit_end):
                raise DecisionFrameworkError("fit_start must be <= fit_end.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    report_id: str
    created_at: str
    row_count: int
    quality_score: float
    missing_bar_rate: float = 0.0
    stale_quote_rate: float = 0.0
    outlier_rate: float = 0.0
    issue_counts: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        _require_nonempty(self.report_id, name="report_id")
        parse_iso_timestamp(self.created_at)
        if self.row_count < 0:
            raise DecisionFrameworkError("row_count must be non-negative.")
        for name in ("quality_score", "missing_bar_rate", "stale_quote_rate", "outlier_rate"):
            _validate_unit_interval(float(getattr(self, name)), name=name)
        for key, value in self.issue_counts.items():
            if value < 0:
                raise DecisionFrameworkError(f"issue_counts[{key!r}] must be non-negative.")

    def should_force_cash(self, *, min_quality_score: float = 0.95) -> bool:
        self.validate()
        return self.quality_score < float(min_quality_score)


@dataclass(frozen=True)
class ActionEligibility:
    symbol_id: str
    decision_ts: str
    tradable: bool
    reason_if_excluded: str | None
    avg_dollar_volume_20d: float
    median_spread_bps_20d: float
    missing_bar_rate_5d: float
    leverage_factor: float
    inverse: bool
    risk_bucket: str

    def validate(self) -> None:
        _require_nonempty(self.symbol_id, name="symbol_id")
        parse_iso_timestamp(self.decision_ts)
        _require_nonempty(self.risk_bucket, name="risk_bucket")
        _validate_unit_interval(self.missing_bar_rate_5d, name="missing_bar_rate_5d")
        if not self.tradable and not self.reason_if_excluded:
            raise DecisionFrameworkError("Non-tradable actions require reason_if_excluded.")
        if self.avg_dollar_volume_20d < 0:
            raise DecisionFrameworkError("avg_dollar_volume_20d must be non-negative.")
        if self.median_spread_bps_20d < 0:
            raise DecisionFrameworkError("median_spread_bps_20d must be non-negative.")


def action_eligibilities_to_mask(
    eligibilities: list[ActionEligibility],
    *,
    cash_index: int = 0,
) -> tuple[torch.Tensor, dict[str, str]]:
    if not eligibilities:
        raise DecisionFrameworkError("eligibilities must not be empty.")
    mask = torch.tensor([item.tradable for item in eligibilities], dtype=torch.bool)
    reasons = {
        item.symbol_id: str(item.reason_if_excluded)
        for item in eligibilities
        if not item.tradable and item.reason_if_excluded
    }
    if not 0 <= int(cash_index) < len(eligibilities):
        raise DecisionFrameworkError("cash_index is outside the eligibility list.")
    mask[int(cash_index)] = True
    return mask, reasons


def apply_data_quality_gate(
    action_mask: torch.Tensor,
    *,
    data_quality_score: float,
    min_quality_score: float = 0.95,
    cash_index: int = 0,
) -> torch.Tensor:
    if action_mask.ndim != 1:
        raise DecisionFrameworkError("action_mask must be one-dimensional.")
    if not 0 <= int(cash_index) < action_mask.shape[0]:
        raise DecisionFrameworkError("cash_index is outside action_mask.")
    _validate_unit_interval(data_quality_score, name="data_quality_score")
    gated = action_mask.clone().bool()
    if data_quality_score < float(min_quality_score):
        gated[:] = False
    gated[int(cash_index)] = True
    return gated


def decision_readiness_score(
    *,
    data_quality: float,
    model_confidence: float,
    ensemble_agreement: float,
    regime_knownness: float,
    cost_score: float,
    liquidity_score: float,
    constraint_budget: float,
    recent_paper_performance: float,
) -> float:
    values = {
        "data_quality": data_quality,
        "model_confidence": model_confidence,
        "ensemble_agreement": ensemble_agreement,
        "regime_knownness": regime_knownness,
        "cost_score": cost_score,
        "liquidity_score": liquidity_score,
        "constraint_budget": constraint_budget,
        "recent_paper_performance": recent_paper_performance,
    }
    for name, value in values.items():
        _validate_unit_interval(value, name=name)
    weights = {
        "data_quality": 0.22,
        "model_confidence": 0.12,
        "ensemble_agreement": 0.12,
        "regime_knownness": 0.10,
        "cost_score": 0.12,
        "liquidity_score": 0.12,
        "constraint_budget": 0.10,
        "recent_paper_performance": 0.10,
    }
    return sum(values[name] * weight for name, weight in weights.items())


def readiness_band(score: float) -> str:
    _validate_unit_interval(score, name="score")
    if score >= 0.90:
        return "normal"
    if score >= 0.75:
        return "reduced_risk"
    if score >= 0.50:
        return "cash_or_hold"
    return "no_trade"


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_ts: str
    instrument_universe_hash: str
    market_state: torch.Tensor
    portfolio_state: torch.Tensor
    action_valid_mask: torch.Tensor
    action_cost_estimate_bps: torch.Tensor
    action_risk_features: torch.Tensor
    data_quality_score: float

    def validate(self) -> None:
        parse_iso_timestamp(self.decision_ts)
        _require_nonempty(self.instrument_universe_hash, name="instrument_universe_hash")
        _validate_unit_interval(self.data_quality_score, name="data_quality_score")
        if self.action_valid_mask.ndim != 1:
            raise DecisionFrameworkError("action_valid_mask must be one-dimensional.")
        if self.action_cost_estimate_bps.shape != self.action_valid_mask.shape:
            raise DecisionFrameworkError("action_cost_estimate_bps shape must match action_valid_mask.")
        if self.action_risk_features.ndim != 2 or self.action_risk_features.shape[0] != self.action_valid_mask.shape[0]:
            raise DecisionFrameworkError("action_risk_features must have shape [actions, features].")
        if not bool(self.action_valid_mask.any().item()):
            raise DecisionFrameworkError("At least one action must be valid.")


@dataclass(frozen=True)
class DecisionDataset:
    snapshots: list[DecisionSnapshot]
    action_returns: torch.Tensor
    action_valid_mask: torch.Tensor
    action_cost_bps: torch.Tensor
    next_timestamps: list[str]
    manifests: list[str]

    def validate(self) -> None:
        if not self.snapshots:
            raise DecisionFrameworkError("snapshots must not be empty.")
        rows = len(self.snapshots)
        if self.action_returns.shape[0] != rows:
            raise DecisionFrameworkError("action_returns row count must match snapshots.")
        if tuple(self.action_valid_mask.shape) != tuple(self.action_returns.shape):
            raise DecisionFrameworkError("action_valid_mask shape must match action_returns.")
        if tuple(self.action_cost_bps.shape) != tuple(self.action_returns.shape):
            raise DecisionFrameworkError("action_cost_bps shape must match action_returns.")
        if len(self.next_timestamps) != rows:
            raise DecisionFrameworkError("next_timestamps length must match snapshots.")
        if not self.manifests:
            raise DecisionFrameworkError("manifests must not be empty.")
        for snapshot, next_ts in zip(self.snapshots, self.next_timestamps):
            snapshot.validate()
            if parse_iso_timestamp(next_ts) <= parse_iso_timestamp(snapshot.decision_ts):
                raise DecisionFrameworkError("next_timestamps must be after decision_ts.")


@dataclass(frozen=True)
class DecisionLog:
    decision_id: str
    decision_ts: str
    model_id: str
    selected_action: str
    previous_action: str
    action_mask_reasons: dict[str, str]
    q_values: dict[str, float]
    risk_checks: dict[str, bool]
    expected_cost_bps: float
    data_quality_score: float
    readiness_score: float

    def validate(self) -> None:
        for name in ("decision_id", "decision_ts", "model_id", "selected_action", "previous_action"):
            _require_nonempty(str(getattr(self, name)), name=name)
        parse_iso_timestamp(self.decision_ts)
        if self.expected_cost_bps < 0:
            raise DecisionFrameworkError("expected_cost_bps must be non-negative.")
        _validate_unit_interval(self.data_quality_score, name="data_quality_score")
        _validate_unit_interval(self.readiness_score, name="readiness_score")
        if self.selected_action not in self.q_values:
            raise DecisionFrameworkError("q_values must include selected_action.")
        if not self.risk_checks:
            raise DecisionFrameworkError("risk_checks must not be empty.")

    def write_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_json_hash(self.to_dict())
