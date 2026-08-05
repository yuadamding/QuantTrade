"""Immutable, soft-persistence checkpoint selection contract for M03R v6.

V6 treats the approximately-30-session holding objective as a soft preference.
Survival and restricted-mean holding metrics remain bound validation evidence,
but cannot make an otherwise profitable, risk-valid checkpoint ineligible.
Economic evidence ranks first; holding preference is only a late tie-breaker.

The authoritative chronological-ledger adapter is deliberately unavailable.
This module can qualify the deterministic gate/rank semantics, but cannot mint
or select a governed checkpoint until all metrics reproduce from one bound
execution ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, NoReturn

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    validate_m03r_v6_artifact_identity,
)

M03R_V6_SELECTION_CONTRACT_SCHEMA = (
    "rl-quant.m03r-v6-soft-persistence-checkpoint-selection-contract-v1"
)
M03R_V6_VALIDATION_METRICS_SCHEMA = (
    "rl-quant.m03r-v6-soft-persistence-validation-metrics-v1"
)
M03R_V6_SELECTION_ADAPTER_SCHEMA = (
    "rl-quant.m03r-v6-authoritative-ledger-selection-adapter-v1"
)
M03R_V6_SELECTION_ADAPTER_AVAILABLE = False
M03R_V6_SELECTION_ADAPTER_BLOCKERS = (
    "authoritative_chronological_cohort_ledger_adapter_not_implemented",
    "cause_typed_discretionary_exit_notional_by_age_not_reproduced",
    "cause_typed_turnover_cost_and_projection_metrics_not_reproduced",
)
M03R_V6_HOLDING_PREFERENCE_FLOOR_TRADING_SESSIONS = 25.0
M03R_V6_EXIT_AGE_BIN_COUNT = 61

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class M03RV6SelectionError(ValueError):
    """A v6 checkpoint metric, contract, or evidence binding is invalid."""


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M03RV6SelectionError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV6SelectionError(
            "v6 selection payload is not canonical-JSON safe"
        ) from exc
    return rendered.encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _finite_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise M03RV6SelectionError(f"{name} must be finite")
    return float(value)


def _nonnegative_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise M03RV6SelectionError(f"{name} cannot be negative")
    return result


@dataclass(frozen=True, order=True, slots=True)
class M03RV6FoldSeed:
    """One exact inner-validation cell in canonical fold/seed order."""

    fold_id: str
    seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fold_id, str)
            or not self.fold_id.strip()
            or self.fold_id != self.fold_id.strip()
        ):
            raise M03RV6SelectionError(
                "fold_id must be non-empty without surrounding whitespace"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise M03RV6SelectionError("seed must be a nonnegative integer")


def _require_canonical_inventory(inventory: tuple[M03RV6FoldSeed, ...]) -> None:
    if not isinstance(inventory, tuple) or not inventory:
        raise M03RV6SelectionError(
            "expected_fold_seed_inventory must be a non-empty tuple"
        )
    if not all(isinstance(cell, M03RV6FoldSeed) for cell in inventory):
        raise M03RV6SelectionError(
            "expected_fold_seed_inventory contains an untyped cell"
        )
    if inventory != tuple(sorted(inventory)):
        raise M03RV6SelectionError(
            "expected_fold_seed_inventory must use canonical fold/seed order"
        )
    if len(set(inventory)) != len(inventory):
        raise M03RV6SelectionError(
            "expected_fold_seed_inventory cannot contain duplicate cells"
        )


@dataclass(frozen=True, slots=True)
class M03RV6ValidationMetrics:
    """Content-bound continuous validation evidence for one checkpoint update."""

    update: int
    net_active_return_20bp: float
    net_active_return_40bp: float
    block_bootstrap_lcb95_net_active_return_20bp: float
    annual_tracking_error: float
    active_market_beta: float
    active_beta_equivalence_upper_bound: float
    notional_survival_at_20_sessions: float
    notional_survival_at_30_sessions: float
    restricted_mean_holding_time_through_60_sessions: float
    early_exit_penalty_paid: float
    discretionary_exit_notional_by_age: tuple[float, ...]
    fold_censored_notional_fraction: float
    requested_executed_projection_distance: float
    forced_turnover_fraction: float
    information_ratio_20bp: float
    total_portfolio_sharpe_20bp: float
    maximum_drawdown_20bp: float
    mean_daily_one_way_discretionary_turnover: float
    discretionary_turnover_cost_20bp: float
    holding_preference_score: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
        ):
            raise M03RV6SelectionError("update must be a positive integer")
        scalar_names = tuple(
            name
            for name in self.__dataclass_fields__
            if name
            not in {
                "update",
                "discretionary_exit_notional_by_age",
                "holding_preference_score",
            }
        )
        for name in scalar_names:
            _finite_float(name, getattr(self, name))

        for name in (
            "annual_tracking_error",
            "active_beta_equivalence_upper_bound",
            "restricted_mean_holding_time_through_60_sessions",
            "early_exit_penalty_paid",
            "requested_executed_projection_distance",
            "maximum_drawdown_20bp",
            "mean_daily_one_way_discretionary_turnover",
            "discretionary_turnover_cost_20bp",
        ):
            _nonnegative_float(name, getattr(self, name))
        for name in (
            "notional_survival_at_20_sessions",
            "notional_survival_at_30_sessions",
            "fold_censored_notional_fraction",
            "forced_turnover_fraction",
            "mean_daily_one_way_discretionary_turnover",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise M03RV6SelectionError(f"{name} must lie in [0,1]")
        if (
            self.notional_survival_at_30_sessions
            > self.notional_survival_at_20_sessions
        ):
            raise M03RV6SelectionError(
                "30-session survival cannot exceed 20-session survival"
            )
        if not 0.0 <= self.restricted_mean_holding_time_through_60_sessions <= 60.0:
            raise M03RV6SelectionError("RMST60 must lie in [0,60] sessions")
        if self.active_beta_equivalence_upper_bound < abs(self.active_market_beta):
            raise M03RV6SelectionError(
                "active-beta equivalence upper bound cannot be below the point estimate"
            )
        ages = self.discretionary_exit_notional_by_age
        if not isinstance(ages, tuple) or len(ages) != M03R_V6_EXIT_AGE_BIN_COUNT:
            raise M03RV6SelectionError(
                "discretionary_exit_notional_by_age must be a canonical 61-bin tuple"
            )
        canonical_ages = tuple(
            _nonnegative_float(
                f"discretionary_exit_notional_by_age[{index}]",
                value,
            )
            for index, value in enumerate(ages)
        )
        object.__setattr__(self, "discretionary_exit_notional_by_age", canonical_ages)
        object.__setattr__(
            self,
            "holding_preference_score",
            -max(
                0.0,
                M03R_V6_HOLDING_PREFERENCE_FLOOR_TRADING_SESSIONS
                - float(self.restricted_mean_holding_time_through_60_sessions),
            ),
        )

    @property
    def discretionary_exit_notional(self) -> float:
        """Total ledger-backed discretionary exits across canonical age bins."""

        return float(math.fsum(self.discretionary_exit_notional_by_age))

    @property
    def rank_key(self) -> tuple[float, ...]:
        """Rank economics first and soft persistence only after turnover/cost."""

        return (
            -float(self.block_bootstrap_lcb95_net_active_return_20bp),
            -float(self.information_ratio_20bp),
            -float(self.total_portfolio_sharpe_20bp),
            float(self.maximum_drawdown_20bp),
            float(self.mean_daily_one_way_discretionary_turnover),
            float(self.discretionary_turnover_cost_20bp),
            -float(self.holding_preference_score),
            float(self.update),
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Return every result-moving metric in a stable typed representation."""

        return {
            "schema": M03R_V6_VALIDATION_METRICS_SCHEMA,
            "update": self.update,
            "net_active_return_20bp": float(self.net_active_return_20bp),
            "net_active_return_40bp": float(self.net_active_return_40bp),
            "block_bootstrap_lcb95_net_active_return_20bp": float(
                self.block_bootstrap_lcb95_net_active_return_20bp
            ),
            "annual_tracking_error": float(self.annual_tracking_error),
            "active_market_beta": float(self.active_market_beta),
            "active_beta_equivalence_upper_bound": float(
                self.active_beta_equivalence_upper_bound
            ),
            "notional_survival_at_20_sessions": float(
                self.notional_survival_at_20_sessions
            ),
            "notional_survival_at_30_sessions": float(
                self.notional_survival_at_30_sessions
            ),
            "restricted_mean_holding_time_through_60_sessions": float(
                self.restricted_mean_holding_time_through_60_sessions
            ),
            "holding_preference_score": float(self.holding_preference_score),
            "early_exit_penalty_paid": float(self.early_exit_penalty_paid),
            "discretionary_exit_notional_by_age": list(
                self.discretionary_exit_notional_by_age
            ),
            "discretionary_exit_notional": self.discretionary_exit_notional,
            "fold_censored_notional_fraction": float(
                self.fold_censored_notional_fraction
            ),
            "requested_executed_projection_distance": float(
                self.requested_executed_projection_distance
            ),
            "forced_turnover_fraction": float(self.forced_turnover_fraction),
            "information_ratio_20bp": float(self.information_ratio_20bp),
            "total_portfolio_sharpe_20bp": float(self.total_portfolio_sharpe_20bp),
            "maximum_drawdown_20bp": float(self.maximum_drawdown_20bp),
            "mean_daily_one_way_discretionary_turnover": float(
                self.mean_daily_one_way_discretionary_turnover
            ),
            "discretionary_turnover_cost_20bp": float(
                self.discretionary_turnover_cost_20bp
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class M03RV6CheckpointSelectionContract:
    """Frozen v6 identity and hard eligibility/data-quality gates."""

    setting_id: str
    expected_fold_seed_inventory: tuple[M03RV6FoldSeed, ...]
    inference_contract_sha256: str
    common_evaluator_inputs_sha256: str
    evaluator_implementation_sha256: str
    maximum_fold_censored_notional_fraction: float
    maximum_requested_executed_projection_distance: float
    maximum_forced_turnover_fraction: float
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID
    bootstrap_confidence_level: float = 0.95

    def __post_init__(self) -> None:
        try:
            validate_m03r_v6_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV6SelectionError(str(exc)) from exc
        _require_canonical_inventory(self.expected_fold_seed_inventory)
        for name in (
            "inference_contract_sha256",
            "common_evaluator_inputs_sha256",
            "evaluator_implementation_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.bootstrap_confidence_level != 0.95:
            raise M03RV6SelectionError(
                "v6 checkpoint selection requires exactly 95% bootstrap confidence"
            )
        for name in (
            "maximum_fold_censored_notional_fraction",
            "maximum_requested_executed_projection_distance",
            "maximum_forced_turnover_fraction",
        ):
            _nonnegative_float(name, getattr(self, name))
        for name in (
            "maximum_fold_censored_notional_fraction",
            "maximum_forced_turnover_fraction",
        ):
            if float(getattr(self, name)) > 1.0:
                raise M03RV6SelectionError(f"{name} must lie in [0,1]")

    def metrics_satisfy_hard_gates(self, row: M03RV6ValidationMetrics) -> bool:
        """Apply economic/risk/data-quality gates; never gate holding duration."""

        if not isinstance(row, M03RV6ValidationMetrics):
            raise M03RV6SelectionError(
                "hard gates require typed M03RV6ValidationMetrics"
            )
        active_risk = M03R_DESIGN.active_risk
        return bool(
            row.net_active_return_20bp > 0.0
            and row.net_active_return_40bp >= 0.0
            and row.annual_tracking_error <= active_risk.annual_tracking_error_ceiling
            and row.active_beta_equivalence_upper_bound
            <= active_risk.absolute_active_market_beta_maximum
            and row.fold_censored_notional_fraction
            <= self.maximum_fold_censored_notional_fraction
            and row.requested_executed_projection_distance
            <= self.maximum_requested_executed_projection_distance
            and row.forced_turnover_fraction <= self.maximum_forced_turnover_fraction
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Content-address identity, gates, ranking, and fail-closed adapter."""

        return {
            "schema": M03R_V6_SELECTION_CONTRACT_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "expected_fold_seed_inventory": [
                asdict(cell) for cell in self.expected_fold_seed_inventory
            ],
            "inference_contract_sha256": self.inference_contract_sha256,
            "common_evaluator_inputs_sha256": self.common_evaluator_inputs_sha256,
            "evaluator_implementation_sha256": self.evaluator_implementation_sha256,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "hard_eligibility_gates": {
                "net_active_return_20bp_strictly_positive": True,
                "net_active_return_40bp_nonnegative": True,
                "annual_tracking_error_ceiling": (
                    M03R_DESIGN.active_risk.annual_tracking_error_ceiling
                ),
                "active_beta_equivalence_upper_bound": (
                    M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
                ),
                "maximum_fold_censored_notional_fraction": (
                    self.maximum_fold_censored_notional_fraction
                ),
                "maximum_requested_executed_projection_distance": (
                    self.maximum_requested_executed_projection_distance
                ),
                "maximum_forced_turnover_fraction": (
                    self.maximum_forced_turnover_fraction
                ),
                "holding_duration_requirement": None,
            },
            "rank_order": [
                "block_bootstrap_lcb95_net_active_return_20bp:descending",
                "information_ratio_20bp:descending",
                "total_portfolio_sharpe_20bp:descending",
                "maximum_drawdown_20bp:ascending",
                "mean_daily_one_way_discretionary_turnover:ascending",
                "discretionary_turnover_cost_20bp:ascending",
                "holding_preference_score:descending-weak-tiebreak",
                "update:ascending",
            ],
            "holding_preference": {
                "promotion_gate": False,
                "floor_trading_sessions": (
                    M03R_V6_HOLDING_PREFERENCE_FLOOR_TRADING_SESSIONS
                ),
                "score_rule": "-max(0,25-RMST60)",
                "survival_and_rmst_are_bound_telemetry": True,
            },
            "chronological_selection_adapter": {
                "schema": M03R_V6_SELECTION_ADAPTER_SCHEMA,
                "available": M03R_V6_SELECTION_ADAPTER_AVAILABLE,
                "blockers": list(M03R_V6_SELECTION_ADAPTER_BLOCKERS),
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    def require_authoritative_adapter(self) -> NoReturn:
        """Fail closed until ledger-backed metrics can be reproduced."""

        raise M03RV6SelectionError(
            "M03R v6 authoritative chronological-ledger selection adapter is "
            "unavailable: " + ", ".join(M03R_V6_SELECTION_ADAPTER_BLOCKERS)
        )


def order_m03r_v6_metrics_for_qualification(
    rows: Sequence[M03RV6ValidationMetrics],
    *,
    contract: M03RV6CheckpointSelectionContract,
) -> tuple[M03RV6ValidationMetrics, ...]:
    """Exercise frozen gates/ranking without minting checkpoint evidence."""

    candidates = tuple(rows)
    if not candidates:
        raise M03RV6SelectionError("qualification ranking requires metrics")
    if not all(isinstance(row, M03RV6ValidationMetrics) for row in candidates):
        raise M03RV6SelectionError("qualification ranking received untyped metrics")
    if len({row.update for row in candidates}) != len(candidates):
        raise M03RV6SelectionError("checkpoint updates must be unique")
    eligible = tuple(
        row for row in candidates if contract.metrics_satisfy_hard_gates(row)
    )
    if not eligible:
        raise M03RV6SelectionError("no v6 checkpoint satisfies the frozen hard gates")
    return tuple(sorted(eligible, key=lambda row: (*row.rank_key, row.receipt_sha256)))


def select_m03r_v6_checkpoint(
    setting_id: str,
    candidates: Sequence[object],
    *,
    contract: M03RV6CheckpointSelectionContract,
) -> NoReturn:
    """Fail closed until authoritative ledger integration can mint candidates."""

    try:
        validate_m03r_v6_artifact_identity(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=setting_id,
        )
    except ValueError as exc:
        raise M03RV6SelectionError(str(exc)) from exc
    if setting_id != contract.setting_id:
        raise M03RV6SelectionError(
            "requested setting_id does not match the v6 selection contract"
        )
    del candidates
    contract.require_authoritative_adapter()


__all__ = [
    "M03R_V6_EXIT_AGE_BIN_COUNT",
    "M03R_V6_HOLDING_PREFERENCE_FLOOR_TRADING_SESSIONS",
    "M03R_V6_SELECTION_ADAPTER_AVAILABLE",
    "M03R_V6_SELECTION_ADAPTER_BLOCKERS",
    "M03R_V6_SELECTION_ADAPTER_SCHEMA",
    "M03R_V6_SELECTION_CONTRACT_SCHEMA",
    "M03R_V6_VALIDATION_METRICS_SCHEMA",
    "M03RV6CheckpointSelectionContract",
    "M03RV6FoldSeed",
    "M03RV6SelectionError",
    "M03RV6ValidationMetrics",
    "order_m03r_v6_metrics_for_qualification",
    "select_m03r_v6_checkpoint",
]
