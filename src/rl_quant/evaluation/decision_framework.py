from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_quant.protocol.reportability_contract import (
    canonicalize_baseline_id,
    canonicalize_cost_stress_id,
    validate_baseline_stress_coverage,
)
from rl_quant.research_protocol import ResearchProtocolError, parse_iso_timestamp, stable_json_hash


class DecisionFrameworkError(ResearchProtocolError):
    """Raised when point-in-time decision artifacts violate the framework contract."""


def _require_nonempty(value: str, *, name: str) -> None:
    if not value:
        raise DecisionFrameworkError(f"{name} is required.")


def _validate_unit_interval(value: float, *, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise DecisionFrameworkError(f"{name} must be between 0 and 1.")


def _assert_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise DecisionFrameworkError(f"{name} contains NaN or Inf.")


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
class FeatureManifest:
    feature_set_id: str
    input_dataset_ids: list[str]
    feature_names: list[str]
    feature_available_ts_rule: str
    fit_start: str | None
    fit_end: str | None
    normalizer_hash: str
    code_version: str
    feature_asof: str | None = None

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
            if self.feature_asof is not None:
                if parse_iso_timestamp(self.fit_end) >= parse_iso_timestamp(self.feature_asof):
                    raise DecisionFrameworkError("fit_end must be before feature_asof.")
        elif self.feature_asof is not None:
            parse_iso_timestamp(self.feature_asof)

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
    available_ts: str
    source: str
    source_payload_hash: str
    calculation_window: str
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
        _require_nonempty(self.available_ts, name="available_ts")
        _require_nonempty(self.source, name="source")
        _require_nonempty(self.source_payload_hash, name="source_payload_hash")
        _require_nonempty(self.calculation_window, name="calculation_window")
        assert_available_at(
            decision_ts=self.decision_ts,
            available_ts=self.available_ts,
            name=f"eligibility[{self.symbol_id}]",
        )
        _require_nonempty(self.risk_bucket, name="risk_bucket")
        _validate_unit_interval(self.missing_bar_rate_5d, name="missing_bar_rate_5d")
        if not self.tradable and not self.reason_if_excluded:
            raise DecisionFrameworkError("Non-tradable actions require reason_if_excluded.")
        if self.avg_dollar_volume_20d < 0:
            raise DecisionFrameworkError("avg_dollar_volume_20d must be non-negative.")
        if self.median_spread_bps_20d < 0:
            raise DecisionFrameworkError("median_spread_bps_20d must be non-negative.")
        if self.leverage_factor < 0:
            raise DecisionFrameworkError("leverage_factor must be non-negative.")
        if self.symbol_id.upper() == "CASH" and self.leverage_factor != 0:
            raise DecisionFrameworkError("CASH leverage_factor must be 0.")
        if self.inverse and self.leverage_factor <= 0:
            raise DecisionFrameworkError("inverse instruments must have positive leverage.")


def action_eligibilities_to_mask(
    eligibilities: list[ActionEligibility],
    *,
    cash_index: int = 0,
) -> tuple[torch.Tensor, dict[str, str]]:
    if not eligibilities:
        raise DecisionFrameworkError("eligibilities must not be empty.")
    for item in eligibilities:
        item.validate()
    if not 0 <= int(cash_index) < len(eligibilities):
        raise DecisionFrameworkError("cash_index is outside the eligibility list.")
    if eligibilities[int(cash_index)].symbol_id.upper() != "CASH":
        raise DecisionFrameworkError("cash_index must point to CASH.")
    mask = torch.tensor([item.tradable for item in eligibilities], dtype=torch.bool)
    reasons = {
        item.symbol_id: str(item.reason_if_excluded)
        for item in eligibilities
        if not item.tradable and item.reason_if_excluded
    }
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
class ReadinessConfig:
    weights: dict[str, float]
    thresholds: dict[str, float]
    min_data_quality: float
    min_liquidity_score: float
    min_constraint_budget: float
    version: str

    def validate(self) -> None:
        _require_nonempty(self.version, name="version")
        if not self.weights:
            raise DecisionFrameworkError("weights must not be empty.")
        if not self.thresholds:
            raise DecisionFrameworkError("thresholds must not be empty.")
        for name, value in self.weights.items():
            if float(value) < 0:
                raise DecisionFrameworkError(f"weights[{name!r}] must be non-negative.")
        weight_sum = sum(float(value) for value in self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise DecisionFrameworkError("weights must sum to 1.")
        for name, value in self.thresholds.items():
            _validate_unit_interval(float(value), name=f"thresholds[{name!r}]")
        _validate_unit_interval(self.min_data_quality, name="min_data_quality")
        _validate_unit_interval(self.min_liquidity_score, name="min_liquidity_score")
        _validate_unit_interval(self.min_constraint_budget, name="min_constraint_budget")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        self.validate()
        return stable_json_hash(self.to_dict())


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
    action_names: list[str] | None = None

    def validate(self, *, cash_index: int = 0) -> None:
        parse_iso_timestamp(self.decision_ts)
        _require_nonempty(self.instrument_universe_hash, name="instrument_universe_hash")
        _validate_unit_interval(self.data_quality_score, name="data_quality_score")
        _assert_finite(self.market_state, name="market_state")
        _assert_finite(self.portfolio_state, name="portfolio_state")
        _assert_finite(self.action_cost_estimate_bps, name="action_cost_estimate_bps")
        _assert_finite(self.action_risk_features, name="action_risk_features")
        if self.action_valid_mask.ndim != 1:
            raise DecisionFrameworkError("action_valid_mask must be one-dimensional.")
        if not 0 <= int(cash_index) < self.action_valid_mask.shape[0]:
            raise DecisionFrameworkError("cash_index is outside action_valid_mask.")
        if self.action_names is not None:
            if len(self.action_names) != self.action_valid_mask.shape[0]:
                raise DecisionFrameworkError("action_names length must match action_valid_mask.")
            if self.action_names[int(cash_index)].upper() != "CASH":
                raise DecisionFrameworkError("cash_index must point to CASH.")
        if self.action_cost_estimate_bps.shape != self.action_valid_mask.shape:
            raise DecisionFrameworkError("action_cost_estimate_bps shape must match action_valid_mask.")
        if bool((self.action_cost_estimate_bps < 0).any().item()):
            raise DecisionFrameworkError("action_cost_estimate_bps must be non-negative.")
        if not bool(self.action_valid_mask[int(cash_index)].item()):
            raise DecisionFrameworkError("cash action must be valid.")
        if abs(float(self.action_cost_estimate_bps[int(cash_index)].item())) > 1e-12:
            raise DecisionFrameworkError("cash action cost estimate must be 0.")
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
        _assert_finite(self.action_cost_bps, name="action_cost_bps")
        if bool((self.action_cost_bps < 0).any().item()):
            raise DecisionFrameworkError("action_cost_bps must be non-negative.")
        valid_returns = self.action_returns[self.action_valid_mask]
        if valid_returns.numel() and not bool(torch.isfinite(valid_returns).all().item()):
            raise DecisionFrameworkError("Valid action returns must be finite.")
        invalid_returns = self.action_returns[~self.action_valid_mask]
        if invalid_returns.numel() and not bool(torch.isnan(invalid_returns).all().item()):
            raise DecisionFrameworkError("Invalid action returns must be NaN.")
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
    readiness_config_hash: str
    candidates: dict[str, dict[str, Any]]

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
        _require_nonempty(self.readiness_config_hash, name="readiness_config_hash")
        if not self.candidates:
            raise DecisionFrameworkError("candidates must not be empty.")
        if self.selected_action not in self.candidates:
            raise DecisionFrameworkError("candidates must include selected_action.")
        for action, candidate in self.candidates.items():
            if "valid" not in candidate:
                raise DecisionFrameworkError(f"candidates[{action!r}] is missing valid.")
            if "q_value" not in candidate:
                raise DecisionFrameworkError(f"candidates[{action!r}] is missing q_value.")
            if "expected_cost_bps" not in candidate:
                raise DecisionFrameworkError(f"candidates[{action!r}] is missing expected_cost_bps.")
            if "risk_bucket" not in candidate:
                raise DecisionFrameworkError(f"candidates[{action!r}] is missing risk_bucket.")
            if float(candidate["expected_cost_bps"]) < 0:
                raise DecisionFrameworkError(f"candidates[{action!r}].expected_cost_bps must be non-negative.")
            if not isinstance(candidate["valid"], bool):
                raise DecisionFrameworkError(f"candidates[{action!r}].valid must be boolean.")
            if action not in self.q_values:
                raise DecisionFrameworkError(f"q_values must include candidate {action!r}.")

    def write_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_json_hash(self.to_dict())


def _has_path(payload: dict[str, Any], path: str) -> bool:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def extract_baseline_ids(summary: dict[str, Any]) -> set[str]:
    """Canonical baseline ids actually present in a run summary's ``baselines`` section: each produced name
    (CASH, BuyAndHold_QQQ, RandomSameTurnover, ...) is mapped to its canonical logical id via the protocol name
    map; unrecognized / more-specific variants (e.g. RandomSameTurnoverSameTiming) are dropped. Pure -- this is
    the canonical INPUT to validate_baseline_stress_coverage, not a reportability decision by itself.

    Consumed by validate_reportable_summary below as the canonical baseline-coverage input."""
    baselines = summary.get("baselines")
    if not isinstance(baselines, dict):
        return set()
    return {cid for name in baselines if (cid := canonicalize_baseline_id(str(name))) is not None}


def extract_stress_ids(summary: dict[str, Any]) -> set[str]:
    """Canonical stress ids present in a run summary's ``cost_stress`` section, proven by PARAMETER. Each
    rollout leg (fixed_rollout / adaptive) is a dict of {label: metrics}; a leg entry maps to a canonical stress
    id only when its metrics carry a ``cost_multiplier`` that canonicalize_cost_stress_id recognizes (2x ->
    cost_doubled, 3x -> cost_tripled). A produced rollout-mode NAME never implies a cost level. Pure.

    Consumed by validate_reportable_summary below as the canonical stress-coverage input."""
    cost_stress = summary.get("cost_stress")
    if not isinstance(cost_stress, dict):
        return set()
    ids: set[str] = set()
    for leg in cost_stress.values():
        if not isinstance(leg, dict):
            continue
        for entry in leg.values():
            if isinstance(entry, dict):
                sid = canonicalize_cost_stress_id(entry.get("cost_multiplier"))
                if sid is not None:
                    ids.add(sid)
    return ids


_REQUIRED_COST_STRESS_ROLLOUT_POLICIES = ("fixed_rollout", "adaptive")


def _cost_stress_rollout_completeness(summary: dict[str, Any]) -> list[str]:
    """Producer completeness for cost_doubled: the 2x cost stress must be produced under BOTH rollout policies
    (fixed_rollout AND adaptive). This preserves the legacy requirement that both cost_stress.fixed_rollout and
    cost_stress.adaptive exist, now proven by a 2x cost_multiplier rather than mere key presence."""
    cost_stress = summary.get("cost_stress")
    errors: list[str] = []
    for policy in _REQUIRED_COST_STRESS_ROLLOUT_POLICIES:
        leg = cost_stress.get(policy) if isinstance(cost_stress, dict) else None
        produced = isinstance(leg, dict) and any(
            isinstance(entry, dict) and canonicalize_cost_stress_id(entry.get("cost_multiplier")) == "cost_doubled"
            for entry in leg.values()
        )
        if not produced:
            errors.append(f"missing cost_doubled under cost_stress.{policy}")
    return errors


def validate_reportable_summary(summary: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Return the reportability reasons (empty == reportable). ``strict`` (default False, default-preserving)
    additionally requires the action-return basis to be COMPLETE on both the eval and dataset-manifest sides --
    the stricter posture for a NEW reportable artifact. With ``strict=False`` the basis is checked for AGREEMENT
    and value VALIDITY only (a contradiction, an invalid weight semantics, or a corrupt clip), so a legacy /
    partially-declared basis stays reportable."""
    errors: list[str] = []
    declared_manifest = summary.get("dataset_manifest") if isinstance(summary.get("dataset_manifest"), dict) else {}
    # Structural artifact sections (unchanged). The baseline/stress requirements moved OUT of this list to the
    # canonical protocol contract below (the legacy baselines.CASH / cost_stress.* path checks are replaced).
    required_paths = [
        "dataset_manifest",
        "feature_manifest",
        "model_manifest",
        "data_quality_report",
        "action_eligibility",
        "action_concentration",
        "return_diagnostics",
    ]
    for path in required_paths:
        if not _has_path(summary, path):
            errors.append(f"missing {path}")

    # Canonical baseline/stress COVERAGE is now the reportability verdict source (protocol contract over
    # canonical ids extracted from the summary), replacing the hardcoded legacy dotted-path checks. Coverage
    # requires the produced grid (cash / buy_and_hold / random_action_distribution / same_turnover_random and
    # the cost_doubled stress); _cost_stress_rollout_completeness then preserves the old both-rollout-legs rigor.
    # Quote-conditional spread_impact + buy-and-hold applicability are honored from EXPLICIT declarations
    # (summary or dataset manifest), defaulting to the safe coverage defaults: buy-and-hold required, and
    # spread_impact NOT required while crossable quotes are absent. QuantTrade's current OHLCV-aggregate
    # datasets carry no quotes, so quote_data_available stays False unless a dataset explicitly declares it --
    # at which point a run with crossable quotes can no longer skip the spread/impact stress.
    quote_data_available = bool(
        summary.get("quote_data_available", declared_manifest.get("quote_data_available", False))
    )
    buy_and_hold_applicable = bool(
        summary.get("buy_and_hold_applicable", declared_manifest.get("buy_and_hold_applicable", True))
    )
    ok, issues = validate_baseline_stress_coverage(
        sorted(extract_baseline_ids(summary)),
        sorted(extract_stress_ids(summary)),
        buy_and_hold_applicable=buy_and_hold_applicable,
        quote_data_available=quote_data_available,
    )
    if not ok:
        errors.extend(issues)
    errors.extend(_cost_stress_rollout_completeness(summary))

    if declared_manifest.get("manifest_available") is False:
        errors.append("dataset_manifest_file_missing")

    # Return-basis reportability check: the evaluation's declared basis (summary["return_basis"], canonical
    # fields) and the dataset manifest's declared basis (action_return_* keys) must (a) AGREE -- no contradiction
    # on a jointly-declared field, no invalid weight semantics -- and (b) be value-VALID (no corrupt clip).
    # Default-preserving: a basis that declares nothing (the legacy shape) yields no agreement/validity error.
    # Only ``strict`` additionally requires the basis to be COMPLETE on both sides.
    from rl_quant.protocol.action_return_basis import ReturnBasis, return_basis_agreement_errors

    eval_basis = ReturnBasis.from_canonical(summary.get("return_basis") or {})
    declared_basis = ReturnBasis.from_mapping(declared_manifest)
    errors.extend(return_basis_agreement_errors(eval_basis, declared_basis))
    for label, basis in (("eval", eval_basis), ("dataset_manifest", declared_basis)):
        errors.extend(f"return_basis_invalid[{label}]:{problem}" for problem in basis.validation_errors())
    if strict:
        for label, basis in (("eval", eval_basis), ("dataset_manifest", declared_basis)):
            if not basis.is_complete():
                errors.append(f"return_basis_incomplete[{label}]")

    reportability = summary.get("reportability")
    if isinstance(reportability, dict):
        for reason in reportability.get("reasons", []):
            errors.append(str(reason))

    test_return = summary.get("test_metrics", {}).get("total_return")
    cash_return = summary.get("baselines", {}).get("CASH", {}).get("test", {}).get("total_return")
    qqq_return = summary.get("baselines", {}).get("BuyAndHold_QQQ", {}).get("test", {}).get("total_return")
    if test_return is not None and cash_return is not None and float(test_return) < float(cash_return):
        errors.append("test_return_below_cash")
    if test_return is not None and qqq_return is not None and float(test_return) < float(qqq_return):
        errors.append("test_return_below_buy_and_hold_qqq")

    concentration = summary.get("action_concentration", {})
    if isinstance(concentration, dict):
        if float(concentration.get("max_risky_group_share", concentration.get("max_group_share", 0.0))) > 0.75:
            errors.append("max_group_share_exceeds_limit")
        if float(concentration.get("leveraged_action_share", 0.0)) > 0.50:
            errors.append("leveraged_action_share_exceeds_limit")
    return list(dict.fromkeys(errors))


def classify_reportability(summary: dict[str, Any]) -> dict[str, Any]:
    """Tier a run's reportability so ``reportable: true`` means ONE thing -- a strictly-validated artifact.

    Three tiers:
      * ``strict`` -- passes the base contract AND a COMPLETE, agreeing, valid return basis on both the eval and
        dataset-manifest sides. This is the ONLY tier with ``reportable == True``.
      * ``legacy_diagnostic`` -- base-reportable (structure / coverage / agreement / no below-cash) but the
        return basis is not strictly complete. Readable and classified, but NOT ``reportable`` -- so an old or
        partially-declared run can no longer be mistaken for a strict research result.
      * ``non_reportable`` -- fails the base contract (missing artifacts/baselines/stress, a basis contradiction
        or invalid value, below-cash, over-concentration, ...).

    Default-preserving in spirit: nothing is deleted; a legacy run is downgraded a tier, not broken. The verdict
    is pure (depends only on the summary) and testable without a training run. summary["reportability"] is read
    for the upstream reportability_flags verdict/reasons (a False flag or any flag reason forces non_reportable).
    """
    flags = summary.get("reportability") if isinstance(summary.get("reportability"), dict) else {}
    # The flag channel is permissive by default: an ABSENT / non-dict reportability section (or a missing
    # "reportable" key) defaults to True, so it never alone fails a run -- the base and strict CONTRACTS below
    # still gate everything. A flag that is explicitly False, or carries any reason, does force non_reportable.
    flags_reportable = bool(flags.get("reportable", True))
    base_errors = validate_reportable_summary(summary, strict=False)
    strict_errors = validate_reportable_summary(summary, strict=True)
    base_reportable = flags_reportable and not base_errors
    strict_reportable = flags_reportable and not strict_errors

    if not base_reportable:
        tier, reportable, reasons = "non_reportable", False, list(base_errors)
    elif not strict_reportable:
        # The strict-only gap (what strict adds beyond the base contract), plus an explicit marker.
        strict_only = [e for e in strict_errors if e not in base_errors]
        tier, reportable = "legacy_diagnostic", False
        reasons = ["strict_return_basis_not_enforced", *strict_only]
    else:
        tier, reportable, reasons = "strict", True, []

    return {
        "reportable": reportable,
        "reportable_tier": tier,
        "base_reportable": base_reportable,
        "strict_return_basis": strict_reportable,
        "return_basis_policy": (
            "strict_complete_eval_and_dataset_manifest" if strict_reportable
            else "legacy_agreement_and_validity_only"
        ),
        "reasons": list(dict.fromkeys(reasons)),
    }
