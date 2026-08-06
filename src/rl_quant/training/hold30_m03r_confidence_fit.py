"""Deterministic, replayable confidence-calibration fitting for M03R v5/v6.

The governed order is deliberately one-way: train and freeze one checkpoint,
fit this calibrator from that checkpoint's detached inner-validation logits,
freeze the calibrator, then validate or deploy the same checkpoint.  Calibrator
evidence cannot authorize any subsequent policy update.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import date
from typing import Any

import torch
import torch.nn.functional as F

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN as M03R_V5_DESIGN,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN_ID as M03R_V5_DESIGN_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_PRIMARY_BENCHMARK_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_PROTOCOL_GENERATION as M03R_V5_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN as M03R_V6_DESIGN,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN_ID as M03R_V6_DESIGN_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_PROTOCOL_GENERATION as M03R_V6_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationError,
    M03RConfidenceCalibrationManifest,
    bind_m03r_confidence_calibration,
    m03r_confidence_calibration_payload,
    validate_m03r_confidence_calibration_manifest,
)

M03R_CONFIDENCE_FIT_EVIDENCE_SCHEMA = (
    "rl-quant.m03r-confidence-calibration-fit-evidence-v2"
)
M03R_CONFIDENCE_FIT_SOURCE_SCORE_SCHEMA = (
    "rl-quant.m03r-confidence-fit-source-score-array-v1"
)
M03R_CONFIDENCE_FIT_SOURCE_UNIT_RISK_OUTCOME_SCHEMA = (
    "rl-quant.m03r-confidence-fit-standardized-unit-risk-outcome-array-v1"
)
M03R_V6_CONFIDENCE_OUTCOME_EVIDENCE_SCHEMA = (
    "rl-quant.m03r-v6-confidence-outcome-evidence-v1"
)
M03R_V6_CONFIDENCE_POLICY_DAILY_RETURN_ARRAY_SCHEMA = (
    "rl-quant.m03r-v6-confidence-unit-risk-policy-daily-net-simple-returns-v1"
)
M03R_V6_CONFIDENCE_C1_DAILY_RETURN_ARRAY_SCHEMA = (
    "rl-quant.m03r-v6-confidence-c1-daily-net-simple-returns-v1"
)
M03R_V6_CONFIDENCE_OUTCOME_ARRAY_SCHEMA = (
    "rl-quant.m03r-v6-confidence-computed-30-return-active-log-outcomes-v1"
)
M03R_V6_CONFIDENCE_OUTCOME_AGGREGATION_ID = (
    "sum-30-log1p-policy-minus-sum-30-log1p-C1-v1"
)
M03R_V6_CONFIDENCE_OUTCOME_ROW_IDENTITY_SCHEMA = (
    "rl-quant.m03r-v6-confidence-outcome-ordered-row-identity-v1"
)
M03R_CONFIDENCE_FIT_SOURCE_TARGET_SCHEMA = (
    "rl-quant.m03r-confidence-fit-source-target-array-v3"
)
M03R_CONFIDENCE_FIT_ROW_INDEX_SCHEMA = "rl-quant.m03r-confidence-fit-row-index-v1"
M03R_CONFIDENCE_TARGET_CONSTRUCTION_SCHEMA = (
    "rl-quant.m03r-confidence-target-construction-contract-v1"
)
M03R_CONFIDENCE_V6_TARGET_PATH_ID = (
    "frozen-standardized-unit-risk-proposal-before-confidence-sizing-v1"
)
M03R_CONFIDENCE_V5_TARGET_PATH_ID = (
    "immutable-v5-protocol-defined-calibration-outcome-v1"
)
M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID = (
    "freeze-checkpoint-fit-inner-validation-calibrator-freeze-no-policy-updates-v1"
)
M03R_CONFIDENCE_FIT_OPTIMIZER_ID = "cpu-float64-bounded-newton-v1"
M03R_CONFIDENCE_ECE_BINNING_RULE_ID = (
    "ten-equal-width-left-closed-right-open-final-bin-closed-v1"
)
M03R_CONFIDENCE_ECE_BIN_COUNT = 10
M03R_CONFIDENCE_FIT_MAXIMUM_ITERATIONS = 100
M03R_CONFIDENCE_FIT_MAXIMUM_LINE_SEARCH_HALVINGS = 40
M03R_CONFIDENCE_FIT_CONVERGENCE_TOLERANCE = 1e-12
M03R_CONFIDENCE_FIT_L2_REGULARIZATION = 1e-8
M03R_CONFIDENCE_FIT_SLOPE_BOUNDS = (1e-4, 100.0)
M03R_CONFIDENCE_FIT_INTERCEPT_BOUNDS = (-20.0, 20.0)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M03RConfidenceFitError(ValueError):
    """Confidence-fit inputs or replay evidence are invalid."""


@dataclass(frozen=True, slots=True)
class M03RConfidenceTargetConstructionContract:
    """Content-bound semantics of the binary outcome supplied to the fitter."""

    schema: str
    protocol_generation: str
    design_id: str
    target_definition: str
    benchmark_id: str
    post_fill_return_count: int
    proposal_path_id: str
    proposal_risk_standardization: str
    confidence_sizing_relationship: str
    standardized_unit_risk_proposal_required: bool
    final_confidence_sized_policy_path_prohibited: bool
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class M03RV6ConfidenceOutcomeReceipt:
    """Content receipt for the exact economic path used to derive v6 labels."""

    schema: str
    protocol_generation: str
    design_id: str
    proposal_path_manifest_sha256: str
    target_construction_contract_sha256: str
    observation_count: int
    post_fill_return_count: int
    aggregation_id: str
    ordered_row_identity_sha256: str
    policy_daily_net_simple_return_array_sha256: str
    c1_daily_net_simple_return_array_sha256: str
    computed_active_log_return_outcome_array_sha256: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class M03RV6ConfidenceOutcomeEvidence:
    """Typed daily arrays plus their replayable v6 economic-path receipt."""

    receipt: M03RV6ConfidenceOutcomeReceipt
    fold_ids: tuple[str, ...]
    trading_sessions: tuple[str, ...]
    standardized_unit_risk_policy_net_simple_returns: torch.Tensor = dataclass_field(
        repr=False,
        compare=False,
    )
    c1_net_simple_returns: torch.Tensor = dataclass_field(repr=False, compare=False)
    active_log_return_outcomes: torch.Tensor = dataclass_field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class M03RConfidenceCalibrationBinEvidence:
    """One fixed-width ECE bin in canonical index order."""

    bin_index: int
    lower_edge: float
    upper_edge: float
    upper_edge_inclusive: bool
    observation_count: int
    mean_confidence: float
    observed_target_rate: float
    absolute_calibration_gap: float


@dataclass(frozen=True, slots=True)
class M03RConfidenceCalibrationFitEvidence:
    """Content-addressed output of the sole governed fitting route."""

    schema: str
    two_stage_protocol_id: str
    checkpoint_frozen_before_calibration: bool
    post_calibration_policy_updates_permitted: bool
    target_construction_contract: M03RConfidenceTargetConstructionContract
    v6_outcome_receipt: M03RV6ConfidenceOutcomeReceipt | None
    calibration_manifest: M03RConfidenceCalibrationManifest
    source_row_index_sha256: str
    source_fold_array_sha256: str
    source_date_array_sha256: str
    source_standardized_unit_risk_active_log_return_array_sha256: str | None
    calibrated_probability_array_sha256: str
    optimizer_id: str
    optimizer_maximum_iterations: int
    optimizer_maximum_line_search_halvings: int
    optimizer_convergence_tolerance: float
    optimizer_l2_regularization: float
    slope_bounds: tuple[float, float]
    intercept_bounds: tuple[float, float]
    optimizer_iterations: int
    optimizer_converged: bool
    final_binary_log_loss: float
    ece_binning_rule_id: str
    ece_bin_count: int
    ece_bins: tuple[M03RConfidenceCalibrationBinEvidence, ...]
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class _CanonicalCalibrationRows:
    logits: torch.Tensor
    targets: torch.Tensor
    fold_ids: tuple[str, ...]
    trading_sessions: tuple[str, ...]
    row_index_sha256: str
    fold_array_sha256: str
    date_array_sha256: str
    score_array_sha256: str
    standardized_unit_risk_active_log_return_array_sha256: str | None
    target_array_sha256: str


def _canonical_json(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RConfidenceFitError(
            "confidence-fit evidence is not canonical-JSON safe"
        ) from exc
    return rendered.encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M03RConfidenceFitError(f"{name} must be a lowercase SHA-256 digest")


def m03r_confidence_target_construction_payload(
    contract: M03RConfidenceTargetConstructionContract,
) -> dict[str, object]:
    """Return target semantics excluding their claimed digest."""

    return {
        "schema": contract.schema,
        "protocol_generation": contract.protocol_generation,
        "design_id": contract.design_id,
        "target_definition": contract.target_definition,
        "benchmark_id": contract.benchmark_id,
        "post_fill_return_count": contract.post_fill_return_count,
        "proposal_path_id": contract.proposal_path_id,
        "proposal_risk_standardization": contract.proposal_risk_standardization,
        "confidence_sizing_relationship": contract.confidence_sizing_relationship,
        "standardized_unit_risk_proposal_required": (
            contract.standardized_unit_risk_proposal_required
        ),
        "final_confidence_sized_policy_path_prohibited": (
            contract.final_confidence_sized_policy_path_prohibited
        ),
    }


def compute_m03r_confidence_target_construction_sha256(
    contract: M03RConfidenceTargetConstructionContract,
) -> str:
    return _sha256(m03r_confidence_target_construction_payload(contract))


def _target_construction_contract(
    *,
    protocol_generation: str,
    design_id: str,
) -> M03RConfidenceTargetConstructionContract:
    if protocol_generation == M03R_V6_PROTOCOL_GENERATION:
        if design_id != M03R_V6_DESIGN_ID:
            raise M03RConfidenceFitError(
                "v6 confidence fitting requires the immutable v6 design ID"
            )
        unbound = M03RConfidenceTargetConstructionContract(
            schema=M03R_CONFIDENCE_TARGET_CONSTRUCTION_SCHEMA,
            protocol_generation=protocol_generation,
            design_id=design_id,
            target_definition=M03R_V6_DESIGN.model.confidence_target_definition,
            benchmark_id=M03R_PRIMARY_BENCHMARK_ID,
            post_fill_return_count=(
                M03R_V6_DESIGN.temporal.economic_origin_post_fill_return_count
            ),
            proposal_path_id=M03R_CONFIDENCE_V6_TARGET_PATH_ID,
            proposal_risk_standardization=(
                "unit-active-risk-normalized-before-confidence-sizing-v1"
            ),
            confidence_sizing_relationship=(
                "confidence-does-not-enter-target-outcome-path-v1"
            ),
            standardized_unit_risk_proposal_required=True,
            final_confidence_sized_policy_path_prohibited=True,
            contract_sha256="",
        )
    elif protocol_generation == M03R_V5_PROTOCOL_GENERATION:
        if design_id != M03R_V5_DESIGN_ID:
            raise M03RConfidenceFitError(
                "v5 confidence fitting requires the immutable v5 design ID"
            )
        unbound = M03RConfidenceTargetConstructionContract(
            schema=M03R_CONFIDENCE_TARGET_CONSTRUCTION_SCHEMA,
            protocol_generation=protocol_generation,
            design_id=design_id,
            target_definition=M03R_V5_DESIGN.model.confidence_target_definition,
            benchmark_id=M03R_PRIMARY_BENCHMARK_ID,
            post_fill_return_count=30,
            proposal_path_id=M03R_CONFIDENCE_V5_TARGET_PATH_ID,
            proposal_risk_standardization="unspecified-by-immutable-v5-v1",
            confidence_sizing_relationship="unspecified-by-immutable-v5-v1",
            standardized_unit_risk_proposal_required=False,
            final_confidence_sized_policy_path_prohibited=False,
            contract_sha256="",
        )
    else:
        raise M03RConfidenceFitError(
            "confidence-fit protocol generation is unsupported"
        )
    return replace(
        unbound,
        contract_sha256=compute_m03r_confidence_target_construction_sha256(unbound),
    )


def _validate_target_construction_contract(
    contract: M03RConfidenceTargetConstructionContract,
) -> None:
    if not isinstance(contract, M03RConfidenceTargetConstructionContract):
        raise M03RConfidenceFitError("confidence target contract must be typed")
    expected = _target_construction_contract(
        protocol_generation=contract.protocol_generation,
        design_id=contract.design_id,
    )
    if contract != expected:
        raise M03RConfidenceFitError(
            "confidence target construction must match the frozen generation contract"
        )
    _require_digest("target construction contract_sha256", contract.contract_sha256)
    if (
        compute_m03r_confidence_target_construction_sha256(contract)
        != contract.contract_sha256
    ):
        raise M03RConfidenceFitError(
            "confidence target construction payload does not match its digest"
        )


def _float64_hex_values(value: torch.Tensor) -> list[str]:
    return [
        float(item).hex()
        for item in value.detach().to(device="cpu", dtype=torch.float64).tolist()
    ]


def _ordered_row_identity_sha256(
    *,
    fold_ids: tuple[str, ...],
    trading_sessions: tuple[str, ...],
    observation_count: int,
) -> str:
    """Hash the exact source-row order shared by logits and economic paths."""

    if (
        not isinstance(fold_ids, tuple)
        or not isinstance(trading_sessions, tuple)
        or len(fold_ids) != observation_count
        or len(trading_sessions) != observation_count
    ):
        raise M03RConfidenceFitError(
            "v6 confidence outcome fold_ids and trading_sessions must be tuples "
            "aligned one-for-one with economic-path observations"
        )
    rows: list[dict[str, str]] = []
    row_keys: set[tuple[str, str]] = set()
    for index, (fold_id, trading_session) in enumerate(
        zip(fold_ids, trading_sessions, strict=True)
    ):
        if (
            not isinstance(fold_id, str)
            or not fold_id.strip()
            or fold_id != fold_id.strip()
        ):
            raise M03RConfidenceFitError(
                f"fold_ids[{index}] must be a nonempty canonical string"
            )
        if not isinstance(trading_session, str):
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use ISO YYYY-MM-DD"
            )
        try:
            parsed = date.fromisoformat(trading_session)
        except ValueError as exc:
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use ISO YYYY-MM-DD"
            ) from exc
        if parsed.isoformat() != trading_session:
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use canonical ISO YYYY-MM-DD"
            )
        key = (trading_session, fold_id)
        if key in row_keys:
            raise M03RConfidenceFitError(
                "every v6 confidence outcome (trading_session, fold_id) row "
                "must be unique"
            )
        row_keys.add(key)
        rows.append({"trading_session": trading_session, "fold_id": fold_id})
    return _sha256(
        {
            "schema": M03R_V6_CONFIDENCE_OUTCOME_ROW_IDENTITY_SCHEMA,
            "observation_count": observation_count,
            "ordered_rows": rows,
        }
    )


def _canonical_daily_net_simple_returns(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] != 30
        or not value.is_floating_point()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        or not bool((value > -1.0).all())
    ):
        raise M03RConfidenceFitError(
            f"{name} must be detached finite floating [observation,30] net "
            "simple returns strictly greater than -1"
        )
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def m03r_v6_confidence_outcome_receipt_payload(
    receipt: M03RV6ConfidenceOutcomeReceipt,
) -> dict[str, object]:
    """Return the economic-path receipt payload except its claimed digest."""

    return {
        "schema": receipt.schema,
        "protocol_generation": receipt.protocol_generation,
        "design_id": receipt.design_id,
        "proposal_path_manifest_sha256": receipt.proposal_path_manifest_sha256,
        "target_construction_contract_sha256": (
            receipt.target_construction_contract_sha256
        ),
        "observation_count": receipt.observation_count,
        "post_fill_return_count": receipt.post_fill_return_count,
        "aggregation_id": receipt.aggregation_id,
        "ordered_row_identity_sha256": receipt.ordered_row_identity_sha256,
        "policy_daily_net_simple_return_array_sha256": (
            receipt.policy_daily_net_simple_return_array_sha256
        ),
        "c1_daily_net_simple_return_array_sha256": (
            receipt.c1_daily_net_simple_return_array_sha256
        ),
        "computed_active_log_return_outcome_array_sha256": (
            receipt.computed_active_log_return_outcome_array_sha256
        ),
    }


def compute_m03r_v6_confidence_outcome_evidence_sha256(
    receipt: M03RV6ConfidenceOutcomeReceipt,
) -> str:
    return _sha256(m03r_v6_confidence_outcome_receipt_payload(receipt))


def _daily_return_array_sha256(
    *,
    schema: str,
    values: torch.Tensor,
    proposal_path_manifest_sha256: str,
    target_construction_contract_sha256: str,
) -> str:
    return _sha256(
        {
            "schema": schema,
            "proposal_path_manifest_sha256": proposal_path_manifest_sha256,
            "target_construction_contract_sha256": (
                target_construction_contract_sha256
            ),
            "shape": list(values.shape),
            "float64_hex_values_row_major": _float64_hex_values(values.reshape(-1)),
        }
    )


def _active_log_outcome_array_sha256(
    *,
    outcomes: torch.Tensor,
    policy_array_sha256: str,
    c1_array_sha256: str,
    proposal_path_manifest_sha256: str,
    target_construction_contract_sha256: str,
) -> str:
    return _sha256(
        {
            "schema": M03R_V6_CONFIDENCE_OUTCOME_ARRAY_SCHEMA,
            "proposal_path_manifest_sha256": proposal_path_manifest_sha256,
            "target_construction_contract_sha256": (
                target_construction_contract_sha256
            ),
            "policy_daily_net_simple_return_array_sha256": policy_array_sha256,
            "c1_daily_net_simple_return_array_sha256": c1_array_sha256,
            "float64_hex_values": _float64_hex_values(outcomes),
        }
    )


def build_m03r_v6_confidence_outcome_evidence(
    *,
    standardized_unit_risk_policy_net_simple_returns: torch.Tensor,
    c1_net_simple_returns: torch.Tensor,
    fold_ids: tuple[str, ...],
    trading_sessions: tuple[str, ...],
    proposal_path_manifest_sha256: str,
) -> M03RV6ConfidenceOutcomeEvidence:
    """Build v6 outcomes from exact daily paths before confidence sizing."""

    _require_digest("proposal_path_manifest_sha256", proposal_path_manifest_sha256)
    target_contract = _target_construction_contract(
        protocol_generation=M03R_V6_PROTOCOL_GENERATION,
        design_id=M03R_V6_DESIGN_ID,
    )
    policy = _canonical_daily_net_simple_returns(
        "standardized_unit_risk_policy_net_simple_returns",
        standardized_unit_risk_policy_net_simple_returns,
    )
    benchmark = _canonical_daily_net_simple_returns(
        "c1_net_simple_returns",
        c1_net_simple_returns,
    )
    if policy.shape != benchmark.shape:
        raise M03RConfidenceFitError(
            "policy and C1 daily return arrays must have identical [observation,30] shape"
        )
    ordered_row_identity_sha256 = _ordered_row_identity_sha256(
        fold_ids=fold_ids,
        trading_sessions=trading_sessions,
        observation_count=policy.shape[0],
    )
    outcomes = torch.log1p(policy).sum(dim=1) - torch.log1p(benchmark).sum(dim=1)
    policy_sha256 = _daily_return_array_sha256(
        schema=M03R_V6_CONFIDENCE_POLICY_DAILY_RETURN_ARRAY_SCHEMA,
        values=policy,
        proposal_path_manifest_sha256=proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    c1_sha256 = _daily_return_array_sha256(
        schema=M03R_V6_CONFIDENCE_C1_DAILY_RETURN_ARRAY_SCHEMA,
        values=benchmark,
        proposal_path_manifest_sha256=proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    outcome_sha256 = _active_log_outcome_array_sha256(
        outcomes=outcomes,
        policy_array_sha256=policy_sha256,
        c1_array_sha256=c1_sha256,
        proposal_path_manifest_sha256=proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    unbound = M03RV6ConfidenceOutcomeReceipt(
        schema=M03R_V6_CONFIDENCE_OUTCOME_EVIDENCE_SCHEMA,
        protocol_generation=M03R_V6_PROTOCOL_GENERATION,
        design_id=M03R_V6_DESIGN_ID,
        proposal_path_manifest_sha256=proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
        observation_count=policy.shape[0],
        post_fill_return_count=policy.shape[1],
        aggregation_id=M03R_V6_CONFIDENCE_OUTCOME_AGGREGATION_ID,
        ordered_row_identity_sha256=ordered_row_identity_sha256,
        policy_daily_net_simple_return_array_sha256=policy_sha256,
        c1_daily_net_simple_return_array_sha256=c1_sha256,
        computed_active_log_return_outcome_array_sha256=outcome_sha256,
        evidence_sha256="",
    )
    receipt = replace(
        unbound,
        evidence_sha256=compute_m03r_v6_confidence_outcome_evidence_sha256(unbound),
    )
    evidence = M03RV6ConfidenceOutcomeEvidence(
        receipt=receipt,
        fold_ids=fold_ids,
        trading_sessions=trading_sessions,
        standardized_unit_risk_policy_net_simple_returns=policy,
        c1_net_simple_returns=benchmark,
        active_log_return_outcomes=outcomes.contiguous().clone(),
    )
    validate_m03r_v6_confidence_outcome_evidence(evidence)
    return evidence


def _validate_m03r_v6_confidence_outcome_receipt(
    receipt: M03RV6ConfidenceOutcomeReceipt,
) -> M03RConfidenceTargetConstructionContract:
    if not isinstance(receipt, M03RV6ConfidenceOutcomeReceipt):
        raise M03RConfidenceFitError("v6 confidence outcome receipt must be typed")
    target_contract = _target_construction_contract(
        protocol_generation=M03R_V6_PROTOCOL_GENERATION,
        design_id=M03R_V6_DESIGN_ID,
    )
    exact = {
        "schema": M03R_V6_CONFIDENCE_OUTCOME_EVIDENCE_SCHEMA,
        "protocol_generation": M03R_V6_PROTOCOL_GENERATION,
        "design_id": M03R_V6_DESIGN_ID,
        "target_construction_contract_sha256": target_contract.contract_sha256,
        "post_fill_return_count": 30,
        "aggregation_id": M03R_V6_CONFIDENCE_OUTCOME_AGGREGATION_ID,
    }
    for name, expected in exact.items():
        if getattr(receipt, name) != expected:
            raise M03RConfidenceFitError(
                f"v6 confidence outcome receipt {name} drifted"
            )
    _require_digest(
        "proposal_path_manifest_sha256",
        receipt.proposal_path_manifest_sha256,
    )
    for name in (
        "ordered_row_identity_sha256",
        "policy_daily_net_simple_return_array_sha256",
        "c1_daily_net_simple_return_array_sha256",
        "computed_active_log_return_outcome_array_sha256",
        "evidence_sha256",
    ):
        _require_digest(name, getattr(receipt, name))
    if (
        isinstance(receipt.observation_count, bool)
        or not isinstance(receipt.observation_count, int)
        or receipt.observation_count < 2
    ):
        raise M03RConfidenceFitError(
            "v6 confidence outcome observation_count must be at least two"
        )
    if (
        compute_m03r_v6_confidence_outcome_evidence_sha256(receipt)
        != receipt.evidence_sha256
    ):
        raise M03RConfidenceFitError(
            "v6 confidence outcome receipt payload does not match its digest"
        )
    return target_contract


def validate_m03r_v6_confidence_outcome_evidence(
    evidence: M03RV6ConfidenceOutcomeEvidence,
) -> None:
    """Recompute the complete v6 economic path and its content receipt."""

    if not isinstance(evidence, M03RV6ConfidenceOutcomeEvidence):
        raise M03RConfidenceFitError("v6 confidence outcome evidence must be typed")
    receipt = evidence.receipt
    target_contract = _validate_m03r_v6_confidence_outcome_receipt(receipt)
    ordered_row_identity_sha256 = _ordered_row_identity_sha256(
        fold_ids=evidence.fold_ids,
        trading_sessions=evidence.trading_sessions,
        observation_count=receipt.observation_count,
    )
    if ordered_row_identity_sha256 != receipt.ordered_row_identity_sha256:
        raise M03RConfidenceFitError(
            "v6 confidence outcome row identities do not match their content receipt"
        )
    for name, value in (
        (
            "standardized_unit_risk_policy_net_simple_returns",
            evidence.standardized_unit_risk_policy_net_simple_returns,
        ),
        ("c1_net_simple_returns", evidence.c1_net_simple_returns),
    ):
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or not value.is_contiguous()
        ):
            raise M03RConfidenceFitError(
                f"{name} must retain the constructor's canonical CPU float64 layout"
            )
    policy = _canonical_daily_net_simple_returns(
        "standardized_unit_risk_policy_net_simple_returns",
        evidence.standardized_unit_risk_policy_net_simple_returns,
    )
    benchmark = _canonical_daily_net_simple_returns(
        "c1_net_simple_returns",
        evidence.c1_net_simple_returns,
    )
    if policy.shape != benchmark.shape or policy.shape[0] != receipt.observation_count:
        raise M03RConfidenceFitError(
            "v6 confidence outcome receipt dimensions do not match retained arrays"
        )
    expected_outcomes = torch.log1p(policy).sum(dim=1) - torch.log1p(benchmark).sum(
        dim=1
    )
    retained_outcomes = evidence.active_log_return_outcomes
    if (
        retained_outcomes.device.type != "cpu"
        or retained_outcomes.dtype != torch.float64
        or retained_outcomes.ndim != 1
        or retained_outcomes.shape[0] != receipt.observation_count
        or retained_outcomes.requires_grad
        or not bool(torch.isfinite(retained_outcomes).all())
        or not torch.equal(retained_outcomes, expected_outcomes)
    ):
        raise M03RConfidenceFitError(
            "retained v6 active-log-return outcomes do not reproduce from daily paths"
        )
    policy_sha256 = _daily_return_array_sha256(
        schema=M03R_V6_CONFIDENCE_POLICY_DAILY_RETURN_ARRAY_SCHEMA,
        values=policy,
        proposal_path_manifest_sha256=receipt.proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    c1_sha256 = _daily_return_array_sha256(
        schema=M03R_V6_CONFIDENCE_C1_DAILY_RETURN_ARRAY_SCHEMA,
        values=benchmark,
        proposal_path_manifest_sha256=receipt.proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    outcome_sha256 = _active_log_outcome_array_sha256(
        outcomes=expected_outcomes,
        policy_array_sha256=policy_sha256,
        c1_array_sha256=c1_sha256,
        proposal_path_manifest_sha256=receipt.proposal_path_manifest_sha256,
        target_construction_contract_sha256=target_contract.contract_sha256,
    )
    if (
        policy_sha256 != receipt.policy_daily_net_simple_return_array_sha256
        or c1_sha256 != receipt.c1_daily_net_simple_return_array_sha256
        or outcome_sha256 != receipt.computed_active_log_return_outcome_array_sha256
    ):
        raise M03RConfidenceFitError(
            "v6 confidence outcome daily arrays do not match their content receipt"
        )


def _canonicalize_rows(
    raw_logits: torch.Tensor,
    binary_targets: torch.Tensor,
    standardized_unit_risk_30_session_active_log_returns: torch.Tensor | None,
    fold_ids: tuple[str, ...],
    trading_sessions: tuple[str, ...],
    target_construction_contract_sha256: str,
    v6_outcome_evidence_sha256: str | None,
) -> _CanonicalCalibrationRows:
    _require_digest(
        "target_construction_contract_sha256",
        target_construction_contract_sha256,
    )
    if v6_outcome_evidence_sha256 is not None:
        _require_digest(
            "v6_outcome_evidence_sha256",
            v6_outcome_evidence_sha256,
        )
    if (
        not isinstance(raw_logits, torch.Tensor)
        or raw_logits.ndim != 1
        or raw_logits.numel() < 2
        or not raw_logits.is_floating_point()
        or raw_logits.requires_grad
        or not bool(torch.isfinite(raw_logits).all())
    ):
        raise M03RConfidenceFitError(
            "raw_logits must be detached finite floating [observation]"
        )
    if (
        not isinstance(binary_targets, torch.Tensor)
        or binary_targets.ndim != 1
        or binary_targets.numel() != raw_logits.numel()
        or binary_targets.requires_grad
    ):
        raise M03RConfidenceFitError(
            "binary_targets must be detached [observation] aligned to logits"
        )
    targets = binary_targets.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(targets).all()) or not bool(
        ((targets == 0.0) | (targets == 1.0)).all()
    ):
        raise M03RConfidenceFitError("binary_targets must contain only zero or one")
    if float(targets.min()) == float(targets.max()):
        raise M03RConfidenceFitError(
            "confidence calibration requires both binary target classes"
        )
    observations = raw_logits.numel()
    outcomes: torch.Tensor | None = None
    if standardized_unit_risk_30_session_active_log_returns is not None:
        source_outcomes = standardized_unit_risk_30_session_active_log_returns
        if (
            not isinstance(source_outcomes, torch.Tensor)
            or source_outcomes.ndim != 1
            or source_outcomes.numel() != observations
            or not source_outcomes.is_floating_point()
            or source_outcomes.requires_grad
            or not bool(torch.isfinite(source_outcomes).all())
        ):
            raise M03RConfidenceFitError(
                "standardized unit-risk active-log-return outcomes must be "
                "detached finite floating [observation] aligned to logits"
            )
        outcomes = source_outcomes.detach().to(device="cpu", dtype=torch.float64)
        if v6_outcome_evidence_sha256 is None:
            raise M03RConfidenceFitError(
                "v6 outcomes require their economic-path evidence digest"
            )
    elif v6_outcome_evidence_sha256 is not None:
        raise M03RConfidenceFitError(
            "v5 target rows cannot bind a v6 economic-path evidence digest"
        )
    if len(fold_ids) != observations or len(trading_sessions) != observations:
        raise M03RConfidenceFitError(
            "fold_ids and trading_sessions must align one-for-one with logits"
        )
    parsed_dates: list[date] = []
    for index, (fold_id, session) in enumerate(zip(fold_ids, trading_sessions)):
        if (
            not isinstance(fold_id, str)
            or not fold_id.strip()
            or fold_id != fold_id.strip()
        ):
            raise M03RConfidenceFitError(
                f"fold_ids[{index}] must be a nonempty canonical string"
            )
        if not isinstance(session, str):
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use ISO YYYY-MM-DD"
            )
        try:
            parsed = date.fromisoformat(session)
        except ValueError as exc:
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use ISO YYYY-MM-DD"
            ) from exc
        if parsed.isoformat() != session:
            raise M03RConfidenceFitError(
                f"trading_sessions[{index}] must use canonical ISO YYYY-MM-DD"
            )
        parsed_dates.append(parsed)
    row_keys = tuple(zip(trading_sessions, fold_ids))
    if len(set(row_keys)) != observations:
        raise M03RConfidenceFitError(
            "every confidence-fit (trading_session, fold_id) row must be unique"
        )
    order = tuple(
        sorted(
            range(observations),
            key=lambda index: (parsed_dates[index], fold_ids[index]),
        )
    )
    sort_index = torch.tensor(order, dtype=torch.int64, device=raw_logits.device)
    logits = raw_logits.detach().to(dtype=torch.float64)[sort_index].to(device="cpu")
    targets = targets[torch.tensor(order, dtype=torch.int64)]
    if outcomes is not None:
        outcomes = outcomes[torch.tensor(order, dtype=torch.int64)]
    canonical_folds = tuple(fold_ids[position] for position in order)
    canonical_dates = tuple(trading_sessions[position] for position in order)
    if float(logits.std(unbiased=False)) <= 1e-12:
        raise M03RConfidenceFitError(
            "confidence calibration requires nonconstant raw logits"
        )
    row_payload = {
        "schema": M03R_CONFIDENCE_FIT_ROW_INDEX_SCHEMA,
        "rows": [
            {"trading_session": session, "fold_id": fold_id}
            for session, fold_id in zip(canonical_dates, canonical_folds)
        ],
    }
    row_sha256 = _sha256(row_payload)
    fold_sha256 = _sha256(
        {
            "schema": "rl-quant.m03r-confidence-fit-fold-array-v1",
            "row_index_sha256": row_sha256,
            "fold_ids": list(canonical_folds),
        }
    )
    date_sha256 = _sha256(
        {
            "schema": "rl-quant.m03r-confidence-fit-date-array-v1",
            "row_index_sha256": row_sha256,
            "trading_sessions": list(canonical_dates),
        }
    )
    score_sha256 = _sha256(
        {
            "schema": M03R_CONFIDENCE_FIT_SOURCE_SCORE_SCHEMA,
            "row_index_sha256": row_sha256,
            "float64_hex_values": _float64_hex_values(logits),
        }
    )
    outcome_sha256 = (
        _sha256(
            {
                "schema": M03R_CONFIDENCE_FIT_SOURCE_UNIT_RISK_OUTCOME_SCHEMA,
                "row_index_sha256": row_sha256,
                "target_construction_contract_sha256": (
                    target_construction_contract_sha256
                ),
                "v6_outcome_evidence_sha256": v6_outcome_evidence_sha256,
                "float64_hex_values": _float64_hex_values(outcomes),
            }
        )
        if outcomes is not None
        else None
    )
    target_sha256 = _sha256(
        {
            "schema": M03R_CONFIDENCE_FIT_SOURCE_TARGET_SCHEMA,
            "row_index_sha256": row_sha256,
            "target_construction_contract_sha256": (
                target_construction_contract_sha256
            ),
            "source_standardized_unit_risk_active_log_return_array_sha256": (
                outcome_sha256
            ),
            "v6_outcome_evidence_sha256": v6_outcome_evidence_sha256,
            "binary_values": [int(value) for value in targets.tolist()],
        }
    )
    return _CanonicalCalibrationRows(
        logits=logits,
        targets=targets,
        fold_ids=canonical_folds,
        trading_sessions=canonical_dates,
        row_index_sha256=row_sha256,
        fold_array_sha256=fold_sha256,
        date_array_sha256=date_sha256,
        score_array_sha256=score_sha256,
        standardized_unit_risk_active_log_return_array_sha256=outcome_sha256,
        target_array_sha256=target_sha256,
    )


def _regularized_binary_log_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    slope: torch.Tensor,
    intercept: torch.Tensor,
) -> torch.Tensor:
    score = logits * slope + intercept
    binary = F.binary_cross_entropy_with_logits(score, targets, reduction="mean")
    regularization = (
        0.5
        * M03R_CONFIDENCE_FIT_L2_REGULARIZATION
        * (slope.square() + intercept.square())
    )
    return binary + regularization


def _fit_temperature_and_intercept(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, int, bool, float, torch.Tensor]:
    target_rate = float(targets.mean())
    intercept_initial = math.log(target_rate / (1.0 - target_rate))
    slope = torch.tensor(1.0, dtype=torch.float64)
    intercept = torch.tensor(intercept_initial, dtype=torch.float64).clamp(
        *M03R_CONFIDENCE_FIT_INTERCEPT_BOUNDS
    )
    current_loss = _regularized_binary_log_loss(logits, targets, slope, intercept)
    converged = False
    completed_iterations = 0
    for iteration in range(1, M03R_CONFIDENCE_FIT_MAXIMUM_ITERATIONS + 1):
        completed_iterations = iteration
        probabilities = torch.sigmoid(logits * slope + intercept)
        residual = probabilities - targets
        curvature = probabilities * (1.0 - probabilities)
        gradient = torch.stack(
            (
                (residual * logits).mean()
                + M03R_CONFIDENCE_FIT_L2_REGULARIZATION * slope,
                residual.mean() + M03R_CONFIDENCE_FIT_L2_REGULARIZATION * intercept,
            )
        )
        hessian = torch.stack(
            (
                torch.stack(
                    (
                        (curvature * logits.square()).mean()
                        + M03R_CONFIDENCE_FIT_L2_REGULARIZATION,
                        (curvature * logits).mean(),
                    )
                ),
                torch.stack(
                    (
                        (curvature * logits).mean(),
                        curvature.mean() + M03R_CONFIDENCE_FIT_L2_REGULARIZATION,
                    )
                ),
            )
        )
        try:
            newton_step = torch.linalg.solve(hessian, gradient)
        except RuntimeError as exc:
            raise M03RConfidenceFitError(
                "deterministic confidence-fit Hessian is singular"
            ) from exc
        accepted = False
        candidate_slope = slope
        candidate_intercept = intercept
        candidate_loss = current_loss
        for halving in range(M03R_CONFIDENCE_FIT_MAXIMUM_LINE_SEARCH_HALVINGS + 1):
            scale = 0.5**halving
            candidate_slope = (slope - scale * newton_step[0]).clamp(
                *M03R_CONFIDENCE_FIT_SLOPE_BOUNDS
            )
            candidate_intercept = (intercept - scale * newton_step[1]).clamp(
                *M03R_CONFIDENCE_FIT_INTERCEPT_BOUNDS
            )
            candidate_loss = _regularized_binary_log_loss(
                logits,
                targets,
                candidate_slope,
                candidate_intercept,
            )
            if float(candidate_loss) <= float(current_loss) + 1e-15:
                accepted = True
                break
        if not accepted:
            raise M03RConfidenceFitError(
                "deterministic confidence-fit line search did not descend"
            )
        movement = max(
            abs(float(candidate_slope - slope)),
            abs(float(candidate_intercept - intercept)),
        )
        improvement = abs(float(current_loss - candidate_loss))
        slope = candidate_slope
        intercept = candidate_intercept
        current_loss = candidate_loss
        if (
            movement <= M03R_CONFIDENCE_FIT_CONVERGENCE_TOLERANCE
            or improvement <= M03R_CONFIDENCE_FIT_CONVERGENCE_TOLERANCE
        ):
            converged = True
            break
    if not converged:
        raise M03RConfidenceFitError(
            "deterministic confidence calibrator did not converge"
        )
    fitted_scores = logits * slope + intercept
    calibrated = torch.sigmoid(fitted_scores)
    final_binary_log_loss = float(
        F.binary_cross_entropy_with_logits(
            fitted_scores,
            targets,
            reduction="mean",
        )
    )
    return (
        1.0 / float(slope),
        float(intercept),
        completed_iterations,
        converged,
        final_binary_log_loss,
        calibrated,
    )


def _ece_evidence(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, tuple[M03RConfidenceCalibrationBinEvidence, ...]]:
    probability_values = [float(value) for value in probabilities.tolist()]
    target_values = [float(value) for value in targets.tolist()]
    bins: list[M03RConfidenceCalibrationBinEvidence] = []
    weighted_gaps: list[float] = []
    observations = len(probability_values)
    for bin_index in range(M03R_CONFIDENCE_ECE_BIN_COUNT):
        lower = bin_index / M03R_CONFIDENCE_ECE_BIN_COUNT
        upper = (bin_index + 1) / M03R_CONFIDENCE_ECE_BIN_COUNT
        member_indices = [
            index
            for index, probability in enumerate(probability_values)
            if (
                lower <= probability < upper
                or (
                    bin_index == M03R_CONFIDENCE_ECE_BIN_COUNT - 1
                    and probability == 1.0
                )
            )
        ]
        count = len(member_indices)
        if count:
            mean_confidence = (
                math.fsum(probability_values[index] for index in member_indices) / count
            )
            observed_rate = (
                math.fsum(target_values[index] for index in member_indices) / count
            )
        else:
            mean_confidence = 0.0
            observed_rate = 0.0
        gap = abs(mean_confidence - observed_rate) if count else 0.0
        weighted_gaps.append((count / observations) * gap)
        bins.append(
            M03RConfidenceCalibrationBinEvidence(
                bin_index=bin_index,
                lower_edge=lower,
                upper_edge=upper,
                upper_edge_inclusive=(bin_index == M03R_CONFIDENCE_ECE_BIN_COUNT - 1),
                observation_count=count,
                mean_confidence=mean_confidence,
                observed_target_rate=observed_rate,
                absolute_calibration_gap=gap,
            )
        )
    return math.fsum(weighted_gaps), tuple(bins)


def m03r_confidence_calibration_fit_evidence_payload(
    evidence: M03RConfidenceCalibrationFitEvidence,
) -> dict[str, object]:
    """Return the complete canonical evidence payload except its digest."""

    return {
        "schema": evidence.schema,
        "two_stage_protocol_id": evidence.two_stage_protocol_id,
        "checkpoint_frozen_before_calibration": (
            evidence.checkpoint_frozen_before_calibration
        ),
        "post_calibration_policy_updates_permitted": (
            evidence.post_calibration_policy_updates_permitted
        ),
        "target_construction_contract": {
            **m03r_confidence_target_construction_payload(
                evidence.target_construction_contract
            ),
            "contract_sha256": (evidence.target_construction_contract.contract_sha256),
        },
        "v6_outcome_receipt": (
            {
                **m03r_v6_confidence_outcome_receipt_payload(
                    evidence.v6_outcome_receipt
                ),
                "evidence_sha256": evidence.v6_outcome_receipt.evidence_sha256,
            }
            if evidence.v6_outcome_receipt is not None
            else None
        ),
        "calibration_manifest": {
            **m03r_confidence_calibration_payload(evidence.calibration_manifest),
            "manifest_sha256": evidence.calibration_manifest.manifest_sha256,
        },
        "source_row_index_sha256": evidence.source_row_index_sha256,
        "source_fold_array_sha256": evidence.source_fold_array_sha256,
        "source_date_array_sha256": evidence.source_date_array_sha256,
        "source_standardized_unit_risk_active_log_return_array_sha256": (
            evidence.source_standardized_unit_risk_active_log_return_array_sha256
        ),
        "calibrated_probability_array_sha256": (
            evidence.calibrated_probability_array_sha256
        ),
        "optimizer": {
            "id": evidence.optimizer_id,
            "maximum_iterations": evidence.optimizer_maximum_iterations,
            "maximum_line_search_halvings": (
                evidence.optimizer_maximum_line_search_halvings
            ),
            "convergence_tolerance_float64_hex": float(
                evidence.optimizer_convergence_tolerance
            ).hex(),
            "l2_regularization_float64_hex": float(
                evidence.optimizer_l2_regularization
            ).hex(),
            "slope_bounds_float64_hex": [
                float(value).hex() for value in evidence.slope_bounds
            ],
            "intercept_bounds_float64_hex": [
                float(value).hex() for value in evidence.intercept_bounds
            ],
            "iterations": evidence.optimizer_iterations,
            "converged": evidence.optimizer_converged,
            "final_binary_log_loss_float64_hex": float(
                evidence.final_binary_log_loss
            ).hex(),
        },
        "ece": {
            "binning_rule_id": evidence.ece_binning_rule_id,
            "bin_count": evidence.ece_bin_count,
            "bins": [
                {
                    "bin_index": row.bin_index,
                    "lower_edge_float64_hex": float(row.lower_edge).hex(),
                    "upper_edge_float64_hex": float(row.upper_edge).hex(),
                    "upper_edge_inclusive": row.upper_edge_inclusive,
                    "observation_count": row.observation_count,
                    "mean_confidence_float64_hex": float(row.mean_confidence).hex(),
                    "observed_target_rate_float64_hex": float(
                        row.observed_target_rate
                    ).hex(),
                    "absolute_calibration_gap_float64_hex": float(
                        row.absolute_calibration_gap
                    ).hex(),
                }
                for row in evidence.ece_bins
            ],
        },
    }


def compute_m03r_confidence_calibration_fit_evidence_sha256(
    evidence: M03RConfidenceCalibrationFitEvidence,
) -> str:
    return _sha256(m03r_confidence_calibration_fit_evidence_payload(evidence))


def validate_m03r_confidence_calibration_fit_evidence(
    evidence: M03RConfidenceCalibrationFitEvidence,
) -> None:
    """Validate the frozen two-stage protocol and every evidence binding."""

    if not isinstance(evidence, M03RConfidenceCalibrationFitEvidence):
        raise M03RConfidenceFitError("confidence-fit evidence must be typed")
    exact = {
        "schema": M03R_CONFIDENCE_FIT_EVIDENCE_SCHEMA,
        "two_stage_protocol_id": M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID,
        "optimizer_id": M03R_CONFIDENCE_FIT_OPTIMIZER_ID,
        "optimizer_maximum_iterations": M03R_CONFIDENCE_FIT_MAXIMUM_ITERATIONS,
        "optimizer_maximum_line_search_halvings": (
            M03R_CONFIDENCE_FIT_MAXIMUM_LINE_SEARCH_HALVINGS
        ),
        "optimizer_convergence_tolerance": (M03R_CONFIDENCE_FIT_CONVERGENCE_TOLERANCE),
        "optimizer_l2_regularization": M03R_CONFIDENCE_FIT_L2_REGULARIZATION,
        "slope_bounds": M03R_CONFIDENCE_FIT_SLOPE_BOUNDS,
        "intercept_bounds": M03R_CONFIDENCE_FIT_INTERCEPT_BOUNDS,
        "ece_binning_rule_id": M03R_CONFIDENCE_ECE_BINNING_RULE_ID,
        "ece_bin_count": M03R_CONFIDENCE_ECE_BIN_COUNT,
    }
    for field, expected in exact.items():
        if getattr(evidence, field) != expected:
            raise M03RConfidenceFitError(f"confidence-fit evidence {field} drifted")
    if (
        not evidence.checkpoint_frozen_before_calibration
        or evidence.post_calibration_policy_updates_permitted
    ):
        raise M03RConfidenceFitError(
            "confidence fitting requires a frozen checkpoint and forbids later policy updates"
        )
    target_contract = evidence.target_construction_contract
    _validate_target_construction_contract(target_contract)
    source_outcome_sha256 = (
        evidence.source_standardized_unit_risk_active_log_return_array_sha256
    )
    if target_contract.protocol_generation == M03R_V6_PROTOCOL_GENERATION:
        if source_outcome_sha256 is None:
            raise M03RConfidenceFitError(
                "v6 confidence evidence requires its standardized unit-risk "
                "active-log-return outcome array digest"
            )
        _require_digest(
            "source_standardized_unit_risk_active_log_return_array_sha256",
            source_outcome_sha256,
        )
        if evidence.v6_outcome_receipt is None:
            raise M03RConfidenceFitError(
                "v6 confidence fit requires its typed economic-path receipt"
            )
        _validate_m03r_v6_confidence_outcome_receipt(evidence.v6_outcome_receipt)
        if (
            evidence.v6_outcome_receipt.target_construction_contract_sha256
            != target_contract.contract_sha256
        ):
            raise M03RConfidenceFitError(
                "v6 economic-path receipt and target contract disagree"
            )
    elif source_outcome_sha256 is not None or evidence.v6_outcome_receipt is not None:
        raise M03RConfidenceFitError(
            "immutable v5 confidence evidence cannot claim v6 outcome evidence"
        )
    if not evidence.optimizer_converged or not (
        1 <= evidence.optimizer_iterations <= M03R_CONFIDENCE_FIT_MAXIMUM_ITERATIONS
    ):
        raise M03RConfidenceFitError(
            "confidence-fit optimizer lacks converged bounded evidence"
        )
    if not math.isfinite(evidence.final_binary_log_loss):
        raise M03RConfidenceFitError("confidence-fit log loss must be finite")
    for name in (
        "source_row_index_sha256",
        "source_fold_array_sha256",
        "source_date_array_sha256",
        "calibrated_probability_array_sha256",
        "evidence_sha256",
    ):
        _require_digest(name, getattr(evidence, name))
    manifest = evidence.calibration_manifest
    try:
        validate_m03r_confidence_calibration_manifest(
            manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=manifest.setting_id,
            expected_seed=manifest.seed,
            expected_checkpoint_sha256=manifest.checkpoint_sha256,
            expected_model_state_sha256=manifest.model_state_sha256,
            expected_source_score_array_sha256=manifest.source_score_array_sha256,
            expected_source_target_array_sha256=manifest.source_target_array_sha256,
            expected_protocol_generation=manifest.protocol_generation,
            expected_design_id=manifest.design_id,
        )
    except M03RConfidenceCalibrationError as exc:
        raise M03RConfidenceFitError(str(exc)) from exc
    if (
        manifest.protocol_generation != target_contract.protocol_generation
        or manifest.design_id != target_contract.design_id
        or manifest.target_definition != target_contract.target_definition
    ):
        raise M03RConfidenceFitError(
            "confidence manifest and target-construction contract disagree"
        )
    bins = evidence.ece_bins
    if len(bins) != M03R_CONFIDENCE_ECE_BIN_COUNT:
        raise M03RConfidenceFitError("ECE evidence must contain exactly ten bins")
    expected_count = manifest.fit_observation_count
    if sum(row.observation_count for row in bins) != expected_count:
        raise M03RConfidenceFitError("ECE bin counts do not conserve observations")
    for index, row in enumerate(bins):
        expected_lower = index / M03R_CONFIDENCE_ECE_BIN_COUNT
        expected_upper = (index + 1) / M03R_CONFIDENCE_ECE_BIN_COUNT
        if (
            row.bin_index != index
            or row.lower_edge != expected_lower
            or row.upper_edge != expected_upper
            or row.upper_edge_inclusive != (index == M03R_CONFIDENCE_ECE_BIN_COUNT - 1)
            or row.observation_count < 0
        ):
            raise M03RConfidenceFitError("ECE bin geometry or counts drifted")
        numeric = (
            row.mean_confidence,
            row.observed_target_rate,
            row.absolute_calibration_gap,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric
        ):
            raise M03RConfidenceFitError("ECE bin diagnostics must lie in [0,1]")
        expected_gap = (
            abs(row.mean_confidence - row.observed_target_rate)
            if row.observation_count
            else 0.0
        )
        if row.observation_count == 0 and (
            row.mean_confidence != 0.0 or row.observed_target_rate != 0.0
        ):
            raise M03RConfidenceFitError(
                "empty ECE bins must have canonical zero diagnostics"
            )
        if not math.isclose(row.absolute_calibration_gap, expected_gap, abs_tol=1e-15):
            raise M03RConfidenceFitError("ECE bin gap is not reproducible")
    expected_ece = math.fsum(
        (row.observation_count / expected_count) * row.absolute_calibration_gap
        for row in bins
    )
    if not math.isclose(
        manifest.expected_calibration_error,
        expected_ece,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise M03RConfidenceFitError("manifest ECE does not reproduce from bound bins")
    expected_target_rate = math.fsum(
        (row.observation_count / expected_count) * row.observed_target_rate
        for row in bins
    )
    if not math.isclose(
        manifest.observed_target_rate,
        expected_target_rate,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise M03RConfidenceFitError(
            "manifest target rate does not reproduce from bound bins"
        )
    actual = compute_m03r_confidence_calibration_fit_evidence_sha256(evidence)
    if actual != evidence.evidence_sha256:
        raise M03RConfidenceFitError(
            "confidence-fit evidence payload does not match its digest"
        )


def fit_and_bind_m03r_confidence_calibration(
    *,
    setting_id: str,
    seed: int,
    checkpoint_sha256: str,
    model_state_sha256: str,
    raw_logits: torch.Tensor,
    binary_targets: torch.Tensor | None,
    fold_ids: tuple[str, ...],
    trading_sessions: tuple[str, ...],
    checkpoint_frozen_before_calibration: bool,
    protocol_generation: str = M03R_V5_PROTOCOL_GENERATION,
    design_id: str = M03R_V5_DESIGN_ID,
    v6_outcome_evidence: M03RV6ConfidenceOutcomeEvidence | None = None,
) -> M03RConfidenceCalibrationFitEvidence:
    """Fit and bind one calibrator from actual frozen-checkpoint evidence."""

    if checkpoint_frozen_before_calibration is not True:
        raise M03RConfidenceFitError(
            "the policy checkpoint must be frozen before calibration"
        )
    _require_digest("checkpoint_sha256", checkpoint_sha256)
    _require_digest("model_state_sha256", model_state_sha256)
    target_contract = _target_construction_contract(
        protocol_generation=protocol_generation,
        design_id=design_id,
    )
    if protocol_generation == M03R_V6_PROTOCOL_GENERATION:
        if binary_targets is not None:
            raise M03RConfidenceFitError(
                "v6 confidence fitting rejects caller-authored binary_targets"
            )
        if v6_outcome_evidence is None:
            raise M03RConfidenceFitError(
                "v6 confidence fitting requires typed economic-path outcome evidence"
            )
        validate_m03r_v6_confidence_outcome_evidence(v6_outcome_evidence)
        fitter_row_identity_sha256 = _ordered_row_identity_sha256(
            fold_ids=fold_ids,
            trading_sessions=trading_sessions,
            observation_count=raw_logits.numel(),
        )
        if (
            fitter_row_identity_sha256
            != v6_outcome_evidence.receipt.ordered_row_identity_sha256
            or fold_ids != v6_outcome_evidence.fold_ids
            or trading_sessions != v6_outcome_evidence.trading_sessions
        ):
            raise M03RConfidenceFitError(
                "v6 confidence outcome row identities do not match the exact "
                "logit/fitter row order"
            )
        outcomes = v6_outcome_evidence.active_log_return_outcomes.clone()
        derived_binary_targets = (outcomes.detach() > 0.0).to(dtype=torch.float64)
    else:
        if binary_targets is None:
            raise M03RConfidenceFitError(
                "immutable v5 confidence fitting requires binary_targets"
            )
        if v6_outcome_evidence is not None:
            raise M03RConfidenceFitError(
                "immutable v5 cannot accept v6 economic-path outcome evidence"
            )
        outcomes = None
        derived_binary_targets = binary_targets
    rows = _canonicalize_rows(
        raw_logits,
        derived_binary_targets,
        outcomes,
        fold_ids,
        trading_sessions,
        target_contract.contract_sha256,
        (
            v6_outcome_evidence.receipt.evidence_sha256
            if v6_outcome_evidence is not None
            else None
        ),
    )
    (
        temperature,
        intercept,
        iterations,
        converged,
        final_log_loss,
        probabilities,
    ) = _fit_temperature_and_intercept(rows.logits, rows.targets)
    brier_score = float((probabilities - rows.targets).square().mean())
    expected_calibration_error, bins = _ece_evidence(
        probabilities,
        rows.targets,
    )
    probability_sha256 = _sha256(
        {
            "schema": "rl-quant.m03r-calibrated-confidence-array-v1",
            "row_index_sha256": rows.row_index_sha256,
            "float64_hex_values": _float64_hex_values(probabilities),
        }
    )
    manifest = bind_m03r_confidence_calibration(
        setting_id=setting_id,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        source_score_array_sha256=rows.score_array_sha256,
        source_target_array_sha256=rows.target_array_sha256,
        fit_fold_ids=tuple(sorted(set(rows.fold_ids))),
        fit_start_trading_session=min(rows.trading_sessions),
        fit_end_trading_session=max(rows.trading_sessions),
        temperature=temperature,
        intercept=intercept,
        fit_observation_count=rows.logits.numel(),
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        observed_target_rate=float(rows.targets.mean()),
        protocol_generation=protocol_generation,
        design_id=design_id,
    )
    unbound = M03RConfidenceCalibrationFitEvidence(
        schema=M03R_CONFIDENCE_FIT_EVIDENCE_SCHEMA,
        two_stage_protocol_id=M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID,
        checkpoint_frozen_before_calibration=True,
        post_calibration_policy_updates_permitted=False,
        target_construction_contract=target_contract,
        v6_outcome_receipt=(
            v6_outcome_evidence.receipt if v6_outcome_evidence is not None else None
        ),
        calibration_manifest=manifest,
        source_row_index_sha256=rows.row_index_sha256,
        source_fold_array_sha256=rows.fold_array_sha256,
        source_date_array_sha256=rows.date_array_sha256,
        source_standardized_unit_risk_active_log_return_array_sha256=(
            rows.standardized_unit_risk_active_log_return_array_sha256
        ),
        calibrated_probability_array_sha256=probability_sha256,
        optimizer_id=M03R_CONFIDENCE_FIT_OPTIMIZER_ID,
        optimizer_maximum_iterations=M03R_CONFIDENCE_FIT_MAXIMUM_ITERATIONS,
        optimizer_maximum_line_search_halvings=(
            M03R_CONFIDENCE_FIT_MAXIMUM_LINE_SEARCH_HALVINGS
        ),
        optimizer_convergence_tolerance=(M03R_CONFIDENCE_FIT_CONVERGENCE_TOLERANCE),
        optimizer_l2_regularization=M03R_CONFIDENCE_FIT_L2_REGULARIZATION,
        slope_bounds=M03R_CONFIDENCE_FIT_SLOPE_BOUNDS,
        intercept_bounds=M03R_CONFIDENCE_FIT_INTERCEPT_BOUNDS,
        optimizer_iterations=iterations,
        optimizer_converged=converged,
        final_binary_log_loss=final_log_loss,
        ece_binning_rule_id=M03R_CONFIDENCE_ECE_BINNING_RULE_ID,
        ece_bin_count=M03R_CONFIDENCE_ECE_BIN_COUNT,
        ece_bins=bins,
        evidence_sha256="",
    )
    evidence = replace(
        unbound,
        evidence_sha256=compute_m03r_confidence_calibration_fit_evidence_sha256(
            unbound
        ),
    )
    validate_m03r_confidence_calibration_fit_evidence(evidence)
    return evidence


def replay_m03r_confidence_calibration_fit(
    evidence: M03RConfidenceCalibrationFitEvidence,
    *,
    raw_logits: torch.Tensor,
    binary_targets: torch.Tensor | None,
    fold_ids: tuple[str, ...],
    trading_sessions: tuple[str, ...],
    v6_outcome_evidence: M03RV6ConfidenceOutcomeEvidence | None = None,
) -> None:
    """Recompute an evidence receipt exactly from retained fit arrays."""

    validate_m03r_confidence_calibration_fit_evidence(evidence)
    manifest = evidence.calibration_manifest
    replayed = fit_and_bind_m03r_confidence_calibration(
        setting_id=manifest.setting_id,
        seed=manifest.seed,
        checkpoint_sha256=manifest.checkpoint_sha256,
        model_state_sha256=manifest.model_state_sha256,
        raw_logits=raw_logits,
        binary_targets=binary_targets,
        fold_ids=fold_ids,
        trading_sessions=trading_sessions,
        checkpoint_frozen_before_calibration=True,
        protocol_generation=manifest.protocol_generation,
        design_id=manifest.design_id,
        v6_outcome_evidence=v6_outcome_evidence,
    )
    if (
        replayed.evidence_sha256 != evidence.evidence_sha256
        or m03r_confidence_calibration_fit_evidence_payload(replayed)
        != m03r_confidence_calibration_fit_evidence_payload(evidence)
    ):
        raise M03RConfidenceFitError(
            "retained confidence-fit arrays do not replay the bound evidence"
        )


__all__ = [
    "M03R_CONFIDENCE_ECE_BINNING_RULE_ID",
    "M03R_CONFIDENCE_ECE_BIN_COUNT",
    "M03R_CONFIDENCE_FIT_EVIDENCE_SCHEMA",
    "M03R_CONFIDENCE_FIT_OPTIMIZER_ID",
    "M03R_CONFIDENCE_TARGET_CONSTRUCTION_SCHEMA",
    "M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID",
    "M03R_CONFIDENCE_V6_TARGET_PATH_ID",
    "M03R_V6_CONFIDENCE_OUTCOME_AGGREGATION_ID",
    "M03R_V6_CONFIDENCE_OUTCOME_EVIDENCE_SCHEMA",
    "M03R_V6_CONFIDENCE_OUTCOME_ROW_IDENTITY_SCHEMA",
    "M03RConfidenceCalibrationBinEvidence",
    "M03RConfidenceCalibrationFitEvidence",
    "M03RConfidenceFitError",
    "M03RConfidenceTargetConstructionContract",
    "M03RV6ConfidenceOutcomeEvidence",
    "M03RV6ConfidenceOutcomeReceipt",
    "build_m03r_v6_confidence_outcome_evidence",
    "compute_m03r_confidence_calibration_fit_evidence_sha256",
    "compute_m03r_confidence_target_construction_sha256",
    "compute_m03r_v6_confidence_outcome_evidence_sha256",
    "fit_and_bind_m03r_confidence_calibration",
    "m03r_confidence_calibration_fit_evidence_payload",
    "m03r_confidence_target_construction_payload",
    "m03r_v6_confidence_outcome_receipt_payload",
    "replay_m03r_confidence_calibration_fit",
    "validate_m03r_confidence_calibration_fit_evidence",
    "validate_m03r_v6_confidence_outcome_evidence",
]
