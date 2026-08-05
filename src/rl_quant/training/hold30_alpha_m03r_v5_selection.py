"""V5 content-bound, uncertainty-first checkpoint selection for M03R."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NoReturn

from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    M03R_CANDIDATE_POLICY_RETURNS_SCHEMA,
    M03R_COMMON_EVALUATOR_INPUT_SCHEMA,
    M03RInferencePlan,
    evaluate_m03r_inference,
    validate_m03r_inference_receipt,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    validate_m03r_artifact_identity,
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_SCHEMA = "rl-quant.m03r-v5-checkpoint-selection-evidence-v1"
_SELECTION_CONTRACT_SCHEMA = "rl-quant.m03r-v5-checkpoint-selection-contract-v1"
_VERIFIED_EVALUATOR_SCHEMA = "rl-quant.m03r-v5-verified-inner-evaluator-receipt-v1"
M03R_SELECTION_ADAPTER_SCHEMA = (
    "rl-quant.m03r-v5-chronological-selection-adapter-v1"
)
M03R_SELECTION_ADAPTER_AVAILABLE = False
M03R_SELECTION_ADAPTER_BLOCKERS = (
    "holding_survival_not_reproducible_from_chronological_evaluator_arrays",
    "cause_typed_turnover_not_reproducible_from_chronological_evaluator_arrays",
    "requested_executed_projection_distance_adapter_not_implemented",
)


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
    common_evaluator_inputs_sha256: str
    evaluator_implementation_sha256: str
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
        _require_digest(
            "common_evaluator_inputs_sha256",
            self.common_evaluator_inputs_sha256,
        )
        _require_digest(
            "evaluator_implementation_sha256",
            self.evaluator_implementation_sha256,
        )
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

    def require_result_gates_resolved(self) -> None:
        if not self.resolved_for_selection:
            missing = tuple(
                name for name in self._result_gate_names if getattr(self, name) is None
            )
            raise M03RSelectionError(
                "M03R checkpoint gates remain unresolved: " + ", ".join(missing)
            )

    def require_resolved(self) -> None:
        self.require_result_gates_resolved()
        if not M03R_SELECTION_ADAPTER_AVAILABLE:
            raise M03RSelectionError(
                "M03R chronological selection adapter is unavailable; "
                "selection fails closed until all holding and execution metrics "
                "are reproduced from bound arrays: "
                + ", ".join(M03R_SELECTION_ADAPTER_BLOCKERS)
            )

    def metrics_satisfy_result_gates(self, row: M03RValidationMetrics) -> bool:
        """Evaluate frozen numerical gates without granting evidence status.

        This helper supports deterministic unit tests of the result-moving
        inequalities.  It cannot make a checkpoint selectable; the separate
        chronological adapter gate remains fail closed.
        """

        self.require_result_gates_resolved()
        if not isinstance(row, M03RValidationMetrics):
            raise M03RSelectionError("result gates require M03RValidationMetrics")
        assert self.minimum_notional_survival_at_20_sessions is not None
        assert self.minimum_notional_survival_at_30_sessions is not None
        assert (
            self.minimum_restricted_mean_holding_time_through_60_sessions
            is not None
        )
        assert (
            self.maximum_restricted_mean_holding_time_through_60_sessions
            is not None
        )
        assert self.minimum_discretionary_sold_notional is not None
        assert self.maximum_fold_censored_notional_fraction is not None
        assert self.maximum_requested_executed_projection_distance is not None
        assert self.maximum_forced_turnover_fraction is not None
        active_risk = M03R_DESIGN.active_risk
        return bool(
            row.net_active_return_20bp > 0.0
            and row.net_active_return_40bp >= 0.0
            and row.annual_tracking_error
            <= active_risk.annual_tracking_error_ceiling
            and row.active_beta_equivalence_upper_bound
            <= active_risk.absolute_active_market_beta_maximum
            and row.notional_survival_at_20_sessions
            >= self.minimum_notional_survival_at_20_sessions
            and row.notional_survival_at_30_sessions
            >= self.minimum_notional_survival_at_30_sessions
            and row.restricted_mean_holding_time_through_60_sessions
            >= self.minimum_restricted_mean_holding_time_through_60_sessions
            and row.restricted_mean_holding_time_through_60_sessions
            <= self.maximum_restricted_mean_holding_time_through_60_sessions
            and row.discretionary_sold_notional
            >= self.minimum_discretionary_sold_notional
            and row.fold_censored_notional_fraction
            <= self.maximum_fold_censored_notional_fraction
            and row.requested_executed_projection_distance
            <= self.maximum_requested_executed_projection_distance
            and row.forced_turnover_fraction
            <= self.maximum_forced_turnover_fraction
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": _SELECTION_CONTRACT_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "common_evaluator_inputs_schema": M03R_COMMON_EVALUATOR_INPUT_SCHEMA,
            "expected_fold_seed_inventory": [
                asdict(cell) for cell in self.expected_fold_seed_inventory
            ],
            "inference_contract_sha256": self.inference_contract_sha256,
            "common_evaluator_inputs_sha256": self.common_evaluator_inputs_sha256,
            "evaluator_implementation_sha256": self.evaluator_implementation_sha256,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "chronological_selection_adapter": {
                "schema": M03R_SELECTION_ADAPTER_SCHEMA,
                "available": M03R_SELECTION_ADAPTER_AVAILABLE,
                "blockers": list(M03R_SELECTION_ADAPTER_BLOCKERS),
            },
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
    active_beta_equivalence_upper_bound: float
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
            "active_beta_equivalence_upper_bound",
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
class _M03RVerifiedInnerEvaluatorReceipt:
    """Typed output from an independently executed inner evaluator.

    The common market-side inputs are shared by all checkpoint candidates.
    The policy-return path and this receipt are candidate-specific.  Keeping
    those identities separate permits checkpoints with different return paths
    to compete under one frozen selection contract without accepting a naked,
    caller-authored :class:`M03RValidationMetrics` object.
    """

    protocol_generation: str
    design_id: str
    setting_id: str
    observed_fold_seed_inventory: tuple[M03RFoldSeed, ...]
    inference_contract_sha256: str
    common_evaluator_inputs_sha256: str
    checkpoint_bundle_sha256: str
    candidate_policy_returns_sha256: str
    evaluator_implementation_sha256: str
    bootstrap_confidence_level: float
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
            "common_evaluator_inputs_sha256",
            "checkpoint_bundle_sha256",
            "candidate_policy_returns_sha256",
            "evaluator_implementation_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.bootstrap_confidence_level != 0.95:
            raise M03RSelectionError(
                "M03R verified evaluator receipt requires exactly 95% "
                "bootstrap confidence"
            )
        if not isinstance(self.metrics, M03RValidationMetrics):
            raise M03RSelectionError(
                "verified evaluator metrics must be M03RValidationMetrics"
            )
        if self.receipt_sha256 != self.recompute_receipt_sha256():
            raise M03RSelectionError(
                "verified evaluator receipt SHA-256 does not match canonical payload"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": _VERIFIED_EVALUATOR_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "common_evaluator_inputs_schema": M03R_COMMON_EVALUATOR_INPUT_SCHEMA,
            "observed_fold_seed_inventory": [
                asdict(cell) for cell in self.observed_fold_seed_inventory
            ],
            "inference_contract_sha256": self.inference_contract_sha256,
            "common_evaluator_inputs_sha256": self.common_evaluator_inputs_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "candidate_policy_returns_schema": M03R_CANDIDATE_POLICY_RETURNS_SCHEMA,
            "candidate_policy_returns_sha256": self.candidate_policy_returns_sha256,
            "evaluator_implementation_sha256": self.evaluator_implementation_sha256,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "metrics": asdict(self.metrics),
        }

    def recompute_receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    def validate_against(self, contract: M03RCheckpointSelectionContract) -> None:
        """Verify the evaluator receipt against frozen common inputs."""

        mismatches: list[str] = []
        for name in (
            "protocol_generation",
            "design_id",
            "setting_id",
            "inference_contract_sha256",
            "common_evaluator_inputs_sha256",
            "evaluator_implementation_sha256",
            "bootstrap_confidence_level",
        ):
            if getattr(self, name) != getattr(contract, name):
                mismatches.append(name)
        if self.observed_fold_seed_inventory != contract.expected_fold_seed_inventory:
            mismatches.append("fold_seed_inventory")
        if self.receipt_sha256 != self.recompute_receipt_sha256():
            mismatches.append("candidate_evaluator_receipt_sha256")
        if mismatches:
            raise M03RSelectionError(
                "verified evaluator receipt does not match selection contract: "
                + ", ".join(mismatches)
            )


def validate_m03r_chronological_evaluator_evidence(
    *,
    chronological_evaluator_receipt: Mapping[str, Any],
    score_dates: Any,
    fold_ids: Any,
    policy_net_returns: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
) -> None:
    """Recompute a chronological evaluator receipt from its exact arrays.

    A content hash alone is not independent evidence: callers can hash their
    own invented metrics.  This boundary therefore validates the typed receipt
    and reruns the evaluator over the bound chronological arrays before any
    future selection adapter may consume it.
    """

    if not isinstance(chronological_evaluator_receipt, Mapping):
        raise M03RSelectionError(
            "chronological_evaluator_receipt must be a mapping"
        )
    supplied = dict(chronological_evaluator_receipt)
    try:
        validate_m03r_inference_receipt(supplied)
        recomputed = evaluate_m03r_inference(
            setting_id=str(supplied["setting_id"]),
            score_dates=score_dates,
            fold_ids=fold_ids,
            policy_net_returns=policy_net_returns,
            benchmark_net_returns=benchmark_net_returns,
            risk_free_returns=risk_free_returns,
            market_total_returns=market_total_returns,
            factor_returns=factor_returns,
            plan=plan,
            common_evaluator_inputs_sha256=str(
                supplied["common_evaluator_inputs_sha256"]
            ),
            candidate_policy_returns_sha256=str(
                supplied["candidate_policy_returns_sha256"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RSelectionError(
            "chronological evaluator evidence failed typed-array reproduction"
        ) from exc
    if recomputed != supplied:
        raise M03RSelectionError(
            "chronological evaluator receipt does not reproduce from bound arrays"
        )


def build_m03r_verified_inner_evaluator_receipt(
    *,
    chronological_evaluator_receipt: Mapping[str, Any],
    score_dates: Any,
    fold_ids: Any,
    policy_net_returns: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
) -> NoReturn:
    """Fail closed after proving that the chronological evidence is genuine.

    The current evaluator cannot yet derive survival, censoring, cause-typed
    turnover, and projection-distance selection gates from one authoritative
    execution ledger.  Consequently no public factory may mint a selectable
    receipt from caller-authored aggregate metrics.
    """

    validate_m03r_chronological_evaluator_evidence(
        chronological_evaluator_receipt=chronological_evaluator_receipt,
        score_dates=score_dates,
        fold_ids=fold_ids,
        policy_net_returns=policy_net_returns,
        benchmark_net_returns=benchmark_net_returns,
        risk_free_returns=risk_free_returns,
        market_total_returns=market_total_returns,
        factor_returns=factor_returns,
        plan=plan,
    )
    raise M03RSelectionError(
        "M03R chronological selection adapter is unavailable; verified "
        "holding and execution metrics are not yet reproducible from the "
        "authoritative chronological evaluator arrays"
    )


@dataclass(frozen=True, slots=True)
class M03RCheckpointCandidate:
    """One checkpoint bound to independently verified evaluator evidence."""

    protocol_generation: str
    design_id: str
    setting_id: str
    selection_contract_sha256: str
    checkpoint_bundle_sha256: str
    candidate_policy_returns_sha256: str
    candidate_evaluator_receipt_sha256: str
    verified_evaluator_receipt: _M03RVerifiedInnerEvaluatorReceipt
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
        for name in (
            "selection_contract_sha256",
            "checkpoint_bundle_sha256",
            "candidate_policy_returns_sha256",
            "candidate_evaluator_receipt_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if not isinstance(
            self.verified_evaluator_receipt,
            _M03RVerifiedInnerEvaluatorReceipt,
        ):
            raise M03RSelectionError(
                "checkpoint evidence requires a verified inner-evaluator receipt"
            )
        evaluator = self.verified_evaluator_receipt
        for name in ("protocol_generation", "design_id", "setting_id"):
            if getattr(self, name) != getattr(evaluator, name):
                raise M03RSelectionError(
                    f"checkpoint {name} does not match verified evaluator receipt"
                )
        if (
            self.candidate_policy_returns_sha256
            != evaluator.candidate_policy_returns_sha256
        ):
            raise M03RSelectionError(
                "checkpoint candidate_policy_returns_sha256 does not match "
                "verified evaluator receipt"
            )
        if self.candidate_evaluator_receipt_sha256 != evaluator.receipt_sha256:
            raise M03RSelectionError(
                "checkpoint candidate_evaluator_receipt_sha256 does not match "
                "verified evaluator receipt"
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
            "selection_contract_sha256": self.selection_contract_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "candidate_policy_returns_sha256": self.candidate_policy_returns_sha256,
            "candidate_evaluator_receipt_sha256": (
                self.candidate_evaluator_receipt_sha256
            ),
        }

    def recompute_receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    def validate_against(self, contract: M03RCheckpointSelectionContract) -> None:
        """Fail on any identity, inventory, source, or contract mismatch."""

        contract.require_resolved()
        mismatches: list[str] = []
        for name in ("protocol_generation", "design_id", "setting_id"):
            if getattr(self, name) != getattr(contract, name):
                mismatches.append(name)
        if self.selection_contract_sha256 != contract.receipt_sha256:
            mismatches.append("selection_contract_sha256")
        try:
            self.verified_evaluator_receipt.validate_against(contract)
        except M03RSelectionError as exc:
            mismatches.append(str(exc))
        if self.receipt_sha256 != self.recompute_receipt_sha256():
            mismatches.append("receipt_sha256")
        if mismatches:
            raise M03RSelectionError(
                "checkpoint evidence does not match resolved selection contract: "
                + ", ".join(mismatches)
            )

    def eligible(self, contract: M03RCheckpointSelectionContract) -> bool:
        self.validate_against(contract)
        return contract.metrics_satisfy_result_gates(self.metrics)

    @property
    def metrics(self) -> M03RValidationMetrics:
        """Return only metrics carried by the verified evaluator receipt."""

        return self.verified_evaluator_receipt.metrics

    @property
    def observed_fold_seed_inventory(self) -> tuple[M03RFoldSeed, ...]:
        return self.verified_evaluator_receipt.observed_fold_seed_inventory

    @property
    def inference_contract_sha256(self) -> str:
        return self.verified_evaluator_receipt.inference_contract_sha256

    @property
    def common_evaluator_inputs_sha256(self) -> str:
        return self.verified_evaluator_receipt.common_evaluator_inputs_sha256


def build_m03r_checkpoint_candidate(
    *,
    contract: M03RCheckpointSelectionContract,
    checkpoint_bundle_sha256: str,
    verified_evaluator_receipt: _M03RVerifiedInnerEvaluatorReceipt,
) -> M03RCheckpointCandidate:
    """Build candidate evidence exclusively from verified evaluator output."""

    contract.require_resolved()
    if not isinstance(
        verified_evaluator_receipt,
        _M03RVerifiedInnerEvaluatorReceipt,
    ):
        raise M03RSelectionError(
            "build_m03r_checkpoint_candidate requires a verified evaluator receipt"
        )
    verified_evaluator_receipt.validate_against(contract)
    if checkpoint_bundle_sha256 != verified_evaluator_receipt.checkpoint_bundle_sha256:
        raise M03RSelectionError(
            "checkpoint_bundle_sha256 does not match verified evaluator receipt"
        )
    fields: dict[str, Any] = {
        "protocol_generation": contract.protocol_generation,
        "design_id": contract.design_id,
        "setting_id": contract.setting_id,
        "selection_contract_sha256": contract.receipt_sha256,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "candidate_policy_returns_sha256": (
            verified_evaluator_receipt.candidate_policy_returns_sha256
        ),
        "candidate_evaluator_receipt_sha256": (
            verified_evaluator_receipt.receipt_sha256
        ),
        "verified_evaluator_receipt": verified_evaluator_receipt,
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
    "M03R_SELECTION_ADAPTER_AVAILABLE",
    "M03R_SELECTION_ADAPTER_BLOCKERS",
    "M03R_SELECTION_ADAPTER_SCHEMA",
    "M03RCheckpointCandidate",
    "M03RCheckpointSelectionContract",
    "M03RFoldSeed",
    "M03RSelectionError",
    "M03RValidationMetrics",
    "build_m03r_checkpoint_candidate",
    "build_m03r_verified_inner_evaluator_receipt",
    "select_m03r_checkpoint",
    "validate_m03r_chronological_evaluator_evidence",
]
