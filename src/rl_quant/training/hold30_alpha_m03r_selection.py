"""Content-bound, uncertainty-first checkpoint selection for M03R."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    validate_m03r_artifact_identity,
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_SCHEMA = "rl-quant.m03r-checkpoint-selection-evidence-v1"
_SELECTION_CONTRACT_SCHEMA = "rl-quant.m03r-checkpoint-selection-contract-v1"


class M03RSelectionError(ValueError):
    """Checkpoint evidence or a result-moving selection gate is invalid."""


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M03RSelectionError(f"{name} must be a lowercase SHA-256 digest")


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
        raise M03RSelectionError(
            "selection evidence is not canonical-JSON safe"
        ) from exc
    return rendered.encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _optional_nonnegative(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise M03RSelectionError(f"{name} must be finite, nonnegative, or None")


@dataclass(frozen=True, order=True, slots=True)
class M03RFoldSeed:
    """One exact validation cell in the frozen selection inventory."""

    fold_id: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise M03RSelectionError("fold_id must be a non-empty string")
        if self.fold_id != self.fold_id.strip():
            raise M03RSelectionError("fold_id cannot contain surrounding whitespace")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise M03RSelectionError("seed must be a nonnegative integer")


def _require_canonical_inventory(
    name: str, inventory: tuple[M03RFoldSeed, ...]
) -> None:
    if not isinstance(inventory, tuple) or not inventory:
        raise M03RSelectionError(f"{name} must be a non-empty tuple")
    if not all(isinstance(cell, M03RFoldSeed) for cell in inventory):
        raise M03RSelectionError(f"{name} must contain only M03RFoldSeed records")
    if tuple(sorted(inventory)) != inventory:
        raise M03RSelectionError(f"{name} must use canonical fold/seed order")
    if len(set(inventory)) != len(inventory):
        raise M03RSelectionError(f"{name} cannot contain duplicate fold/seed cells")


@dataclass(frozen=True, slots=True)
class M03RCheckpointSelectionContract:
    """Frozen identity, evidence inventory, and result-moving selection gates.

    The inventory and hashes are mandatory.  The numerical holding/projection
    gates default to ``None`` deliberately: a caller cannot make the new
    generation selectable until those result-moving choices are resolved.
    """

    setting_id: str
    expected_fold_seed_inventory: tuple[M03RFoldSeed, ...]
    inference_contract_sha256: str
    source_arrays_sha256: str
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID
    bootstrap_confidence_level: float = 0.95
    minimum_notional_survival_at_20_sessions: float | None = None
    minimum_notional_survival_at_30_sessions: float | None = None
    minimum_restricted_mean_holding_time_through_60_sessions: float | None = None
    maximum_restricted_mean_holding_time_through_60_sessions: float | None = None
    minimum_discretionary_sold_notional: float | None = None
    maximum_fold_censored_notional_fraction: float | None = None
    maximum_requested_executed_projection_distance: float | None = None
    maximum_forced_turnover_fraction: float | None = None

    def __post_init__(self) -> None:
        try:
            validate_m03r_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RSelectionError(str(exc)) from exc
        _require_canonical_inventory(
            "expected_fold_seed_inventory", self.expected_fold_seed_inventory
        )
        _require_digest("inference_contract_sha256", self.inference_contract_sha256)
        _require_digest("source_arrays_sha256", self.source_arrays_sha256)
        if self.bootstrap_confidence_level != 0.95:
            raise M03RSelectionError(
                "M03R checkpoint selection requires exactly 95% bootstrap confidence"
            )
        for name in self._result_gate_names:
            _optional_nonnegative(name, getattr(self, name))
        for name in (
            "minimum_notional_survival_at_20_sessions",
            "minimum_notional_survival_at_30_sessions",
            "maximum_fold_censored_notional_fraction",
            "maximum_forced_turnover_fraction",
        ):
            value = getattr(self, name)
            if value is not None and value > 1.0:
                raise M03RSelectionError(f"{name} must lie in [0,1]")
        lower = self.minimum_restricted_mean_holding_time_through_60_sessions
        upper = self.maximum_restricted_mean_holding_time_through_60_sessions
        if lower is not None and upper is not None and lower > upper:
            raise M03RSelectionError(
                "restricted-mean holding-time minimum cannot exceed maximum"
            )

    @property
    def _result_gate_names(self) -> tuple[str, ...]:
        return (
            "minimum_notional_survival_at_20_sessions",
            "minimum_notional_survival_at_30_sessions",
            "minimum_restricted_mean_holding_time_through_60_sessions",
            "maximum_restricted_mean_holding_time_through_60_sessions",
            "minimum_discretionary_sold_notional",
            "maximum_fold_censored_notional_fraction",
            "maximum_requested_executed_projection_distance",
            "maximum_forced_turnover_fraction",
        )

    @property
    def resolved_for_selection(self) -> bool:
        return all(getattr(self, name) is not None for name in self._result_gate_names)

    def require_resolved(self) -> None:
        if not self.resolved_for_selection:
            missing = tuple(
                name for name in self._result_gate_names if getattr(self, name) is None
            )
            raise M03RSelectionError(
                "M03R checkpoint gates remain unresolved: " + ", ".join(missing)
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": _SELECTION_CONTRACT_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "expected_fold_seed_inventory": [
                asdict(cell) for cell in self.expected_fold_seed_inventory
            ],
            "inference_contract_sha256": self.inference_contract_sha256,
            "source_arrays_sha256": self.source_arrays_sha256,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "result_moving_gates": {
                name: getattr(self, name) for name in self._result_gate_names
            },
        }

    @property
    def receipt_sha256(self) -> str:
        """Content identity for the complete selection contract."""

        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class M03RValidationMetrics:
    """Aggregate continuous inner-validation metrics for one update."""

    update: int
    net_active_return_20bp: float
    net_active_return_40bp: float
    block_bootstrap_lcb95_net_active_return_20bp: float
    annual_tracking_error: float
    active_market_beta: float
    notional_survival_at_20_sessions: float
    notional_survival_at_30_sessions: float
    restricted_mean_holding_time_through_60_sessions: float
    discretionary_sold_notional: float
    fold_censored_notional_fraction: float
    requested_executed_projection_distance: float
    forced_turnover_fraction: float
    information_ratio_20bp: float
    total_portfolio_sharpe_20bp: float
    maximum_drawdown_20bp: float
    turnover_cost_20bp: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
        ):
            raise M03RSelectionError("update must be a positive integer")
        for name in self.__dataclass_fields__:
            if name == "update":
                continue
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise M03RSelectionError(f"{name} must be finite")
        for name in (
            "annual_tracking_error",
            "restricted_mean_holding_time_through_60_sessions",
            "discretionary_sold_notional",
            "requested_executed_projection_distance",
            "maximum_drawdown_20bp",
            "turnover_cost_20bp",
        ):
            if getattr(self, name) < 0.0:
                raise M03RSelectionError(f"{name} cannot be negative")
        for name in (
            "notional_survival_at_20_sessions",
            "notional_survival_at_30_sessions",
            "fold_censored_notional_fraction",
            "forced_turnover_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise M03RSelectionError(f"{name} must lie in [0,1]")

    @property
    def rank_key(self) -> tuple[float, ...]:
        """Rank uncertainty first; point return and ratios are tie breakers."""

        return (
            -self.block_bootstrap_lcb95_net_active_return_20bp,
            -self.net_active_return_20bp,
            -self.information_ratio_20bp,
            -self.total_portfolio_sharpe_20bp,
            self.maximum_drawdown_20bp,
            self.turnover_cost_20bp,
            float(self.update),
        )


@dataclass(frozen=True, slots=True)
class M03RCheckpointCandidate:
    """One immutable, self-verifying checkpoint selection evidence record."""

    protocol_generation: str
    design_id: str
    setting_id: str
    observed_fold_seed_inventory: tuple[M03RFoldSeed, ...]
    inference_contract_sha256: str
    source_arrays_sha256: str
    bootstrap_confidence_level: float
    selection_contract_sha256: str
    checkpoint_bundle_sha256: str
    metrics: M03RValidationMetrics
    receipt_sha256: str

    def __post_init__(self) -> None:
        try:
            validate_m03r_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RSelectionError(str(exc)) from exc
        _require_canonical_inventory(
            "observed_fold_seed_inventory", self.observed_fold_seed_inventory
        )
        for name in (
            "inference_contract_sha256",
            "source_arrays_sha256",
            "selection_contract_sha256",
            "checkpoint_bundle_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.bootstrap_confidence_level != 0.95:
            raise M03RSelectionError(
                "M03R checkpoint evidence requires exactly 95% bootstrap confidence"
            )
        expected = self.recompute_receipt_sha256()
        if self.receipt_sha256 != expected:
            raise M03RSelectionError(
                "M03R checkpoint evidence receipt SHA-256 does not match canonical payload"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": _EVIDENCE_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "observed_fold_seed_inventory": [
                asdict(cell) for cell in self.observed_fold_seed_inventory
            ],
            "inference_contract_sha256": self.inference_contract_sha256,
            "source_arrays_sha256": self.source_arrays_sha256,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "selection_contract_sha256": self.selection_contract_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "metrics": asdict(self.metrics),
        }

    def recompute_receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    def validate_against(self, contract: M03RCheckpointSelectionContract) -> None:
        """Fail on any identity, inventory, source, or contract mismatch."""

        contract.require_resolved()
        mismatches: list[str] = []
        for name in (
            "protocol_generation",
            "design_id",
            "setting_id",
            "inference_contract_sha256",
            "source_arrays_sha256",
            "bootstrap_confidence_level",
        ):
            if getattr(self, name) != getattr(contract, name):
                mismatches.append(name)
        if self.observed_fold_seed_inventory != contract.expected_fold_seed_inventory:
            mismatches.append("fold_seed_inventory")
        if self.selection_contract_sha256 != contract.receipt_sha256:
            mismatches.append("selection_contract_sha256")
        if self.receipt_sha256 != self.recompute_receipt_sha256():
            mismatches.append("receipt_sha256")
        if mismatches:
            raise M03RSelectionError(
                "checkpoint evidence does not match resolved selection contract: "
                + ", ".join(mismatches)
            )

    def eligible(self, contract: M03RCheckpointSelectionContract) -> bool:
        self.validate_against(contract)
        assert contract.minimum_notional_survival_at_20_sessions is not None
        assert contract.minimum_notional_survival_at_30_sessions is not None
        assert (
            contract.minimum_restricted_mean_holding_time_through_60_sessions
            is not None
        )
        assert (
            contract.maximum_restricted_mean_holding_time_through_60_sessions
            is not None
        )
        assert contract.minimum_discretionary_sold_notional is not None
        assert contract.maximum_fold_censored_notional_fraction is not None
        assert contract.maximum_requested_executed_projection_distance is not None
        assert contract.maximum_forced_turnover_fraction is not None
        row = self.metrics
        active_risk = M03R_DESIGN.active_risk
        return bool(
            row.net_active_return_20bp > 0.0
            and row.net_active_return_40bp >= 0.0
            # Deliberately no lower tracking-error eligibility gate.
            and row.annual_tracking_error <= active_risk.annual_tracking_error_ceiling
            and abs(row.active_market_beta - active_risk.active_market_beta_target)
            <= active_risk.absolute_active_market_beta_maximum
            and row.notional_survival_at_20_sessions
            >= contract.minimum_notional_survival_at_20_sessions
            and row.notional_survival_at_30_sessions
            >= contract.minimum_notional_survival_at_30_sessions
            and row.restricted_mean_holding_time_through_60_sessions
            >= contract.minimum_restricted_mean_holding_time_through_60_sessions
            and row.restricted_mean_holding_time_through_60_sessions
            <= contract.maximum_restricted_mean_holding_time_through_60_sessions
            and row.discretionary_sold_notional
            >= contract.minimum_discretionary_sold_notional
            and row.fold_censored_notional_fraction
            <= contract.maximum_fold_censored_notional_fraction
            and row.requested_executed_projection_distance
            <= contract.maximum_requested_executed_projection_distance
            and row.forced_turnover_fraction
            <= contract.maximum_forced_turnover_fraction
        )


def build_m03r_checkpoint_candidate(
    *,
    contract: M03RCheckpointSelectionContract,
    observed_fold_seed_inventory: tuple[M03RFoldSeed, ...],
    checkpoint_bundle_sha256: str,
    metrics: M03RValidationMetrics,
) -> M03RCheckpointCandidate:
    """Build evidence by copying frozen identities and hashing its payload."""

    contract.require_resolved()
    fields: dict[str, Any] = {
        "protocol_generation": contract.protocol_generation,
        "design_id": contract.design_id,
        "setting_id": contract.setting_id,
        "observed_fold_seed_inventory": observed_fold_seed_inventory,
        "inference_contract_sha256": contract.inference_contract_sha256,
        "source_arrays_sha256": contract.source_arrays_sha256,
        "bootstrap_confidence_level": contract.bootstrap_confidence_level,
        "selection_contract_sha256": contract.receipt_sha256,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "metrics": metrics,
    }
    unsigned = M03RCheckpointCandidate.__new__(M03RCheckpointCandidate)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt_sha256 = _payload_sha256(unsigned.canonical_payload())
    return M03RCheckpointCandidate(**fields, receipt_sha256=receipt_sha256)


def select_m03r_checkpoint(
    setting_id: str,
    candidates: Sequence[M03RCheckpointCandidate],
    *,
    contract: M03RCheckpointSelectionContract,
) -> M03RCheckpointCandidate:
    """Select one checkpoint from complete, contract-bound inner evidence."""

    try:
        validate_m03r_artifact_identity(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=setting_id,
        )
    except ValueError as exc:
        raise M03RSelectionError(str(exc)) from exc
    contract.require_resolved()
    if setting_id != contract.setting_id:
        raise M03RSelectionError(
            "requested setting_id does not match the resolved selection contract"
        )
    rows = tuple(candidates)
    if not rows:
        raise M03RSelectionError("checkpoint selection requires candidates")
    if len({row.metrics.update for row in rows}) != len(rows):
        raise M03RSelectionError("checkpoint updates must be unique")
    for row in rows:
        row.validate_against(contract)
    eligible = tuple(row for row in rows if row.eligible(contract))
    if not eligible:
        raise M03RSelectionError("no M03R checkpoint satisfies the frozen gates")
    return min(
        eligible,
        key=lambda row: (
            *row.metrics.rank_key,
            row.checkpoint_bundle_sha256,
            row.receipt_sha256,
        ),
    )


__all__ = [
    "M03RCheckpointCandidate",
    "M03RCheckpointSelectionContract",
    "M03RFoldSeed",
    "M03RSelectionError",
    "M03RValidationMetrics",
    "build_m03r_checkpoint_candidate",
    "select_m03r_checkpoint",
]
