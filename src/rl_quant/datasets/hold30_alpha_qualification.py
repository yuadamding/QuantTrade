"""V3 in-memory structural consistency checks for Hold-30 alpha.

This layer recomputes typed tensor identities and the train/inner-validation
chain.  Its caller supplies every tensor and every purported artifact digest,
so it cannot verify files, provider trust, observed-data status, or production
eligibility.  The full evaluation panel also contains outer-period values even
though no outer fold sequence or labels are materialized here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

import torch

from rl_quant.datasets.hold30 import Hold30DatasetError, Hold30DatasetSequence
from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaDataError,
    Hold30AlphaEvaluationPanel,
    Hold30AlphaLabelDomain,
    bind_hold30_alpha_evaluation_panel,
    build_hold30_residual_alpha_labels,
)
from rl_quant.datasets.hold30_folds import materialize_hold30_development_fold
from rl_quant.datasets.hold30_qualification import (
    HOLD30_REQUIRED_EXTERNAL_ARTIFACTS,
    verify_hold30_dataset_against_qualification,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_v3_freeze import (
    bind_hold30_alpha_v3_data_contract,
)
from rl_quant.protocol.hold30_freeze import (
    HOLD30_FOLDS,
    HOLD30_MIN_AXIS_POSITIONS,
    render_hold30_folds,
    sha256_payload,
)
from rl_quant.training.hold30_alpha import (
    Hold30AlphaObjectiveDomainBinding,
    bind_hold30_alpha_objective_inputs,
)

HOLD30_ALPHA_IN_MEMORY_STRUCTURAL_SCHEMA = (
    "rl-quant.hold30-alpha-in-memory-structural-check-v1"
)
HOLD30_ALPHA_IN_MEMORY_LINEAGE_DECLARATION_SCHEMA = (
    "rl-quant.hold30-alpha-in-memory-lineage-declaration-v1"
)
HOLD30_ALPHA_IN_MEMORY_FACTOR_EXPOSURE_SCHEMA = (
    "rl-quant.hold30-alpha-in-memory-factor-exposures-v1"
)

_DIGEST_CHARS = frozenset("0123456789abcdef")


class Hold30AlphaStructuralQualificationError(ValueError):
    """In-memory inputs or their non-production receipt are inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise Hold30AlphaStructuralQualificationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Hold30AlphaStructuralQualificationError(
            f"{name} must be a nonempty stable identifier"
        )
    return value


@dataclass(frozen=True, slots=True)
class Hold30AlphaInMemoryLineageDeclaration:
    """Caller declarations bound structurally, but not verified against files."""

    protocol_generation: str
    source_axis_id: str
    provider_id: str
    provider_snapshot_receipt_sha256: str
    data_snapshot_sha256: str
    raw_market_data_sha256: str
    universe_events_sha256: str
    tradability_events_sha256: str
    corporate_actions_sha256: str
    identifier_events_sha256: str
    declared_source_kind: str = "observed-market-data"
    observed_market_data_declared: bool = True
    synthetic_data: bool = False
    future_selected_universe: bool = False
    point_in_time_universe: bool = True
    schema: str = HOLD30_ALPHA_IN_MEMORY_LINEAGE_DECLARATION_SCHEMA

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaStructuralQualificationError(
                "lineage declaration rejects another protocol generation"
            )
        if self.schema != HOLD30_ALPHA_IN_MEMORY_LINEAGE_DECLARATION_SCHEMA:
            raise Hold30AlphaStructuralQualificationError(
                "lineage declaration schema drifted"
            )
        _require_id("provider_id", self.provider_id)
        for name in (
            "source_axis_id",
            "provider_snapshot_receipt_sha256",
            "data_snapshot_sha256",
            "raw_market_data_sha256",
            "universe_events_sha256",
            "tradability_events_sha256",
            "corporate_actions_sha256",
            "identifier_events_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.declared_source_kind != "observed-market-data"
            or self.observed_market_data_declared is not True
            or self.synthetic_data is not False
            or self.future_selected_universe is not False
            or self.point_in_time_universe is not True
        ):
            raise Hold30AlphaStructuralQualificationError(
                "declared synthetic, substituted, or future-selected data are forbidden"
            )

    @property
    def receipt_id(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Hold30AlphaInMemoryFactorExposureDeclaration:
    """Caller-supplied factor tensors and unverified artifact declarations."""

    protocol_generation: str
    source_axis_id: str
    factor_model_id: str
    factor_names: tuple[str, ...]
    values: torch.Tensor
    valid: torch.Tensor
    known_at_ms: torch.Tensor
    factor_returns_artifact_sha256: str
    factor_plan_sha256: str
    exposure_artifact_sha256: str
    provider_snapshot_receipt_sha256: str
    point_in_time: bool = True
    evaluator_only: bool = True
    policy_feature_access: bool = False
    schema: str = HOLD30_ALPHA_IN_MEMORY_FACTOR_EXPOSURE_SCHEMA

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaStructuralQualificationError(
                "factor exposures reject another protocol generation"
            )
        if self.schema != HOLD30_ALPHA_IN_MEMORY_FACTOR_EXPOSURE_SCHEMA:
            raise Hold30AlphaStructuralQualificationError(
                "factor exposure schema drifted"
            )
        _require_id("factor_model_id", self.factor_model_id)
        if (
            not isinstance(self.factor_names, tuple)
            or not self.factor_names
            or len(set(self.factor_names)) != len(self.factor_names)
            or any(not isinstance(name, str) or not name for name in self.factor_names)
        ):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposure names must be a nonempty unique tuple"
            )
        for name in (
            "source_axis_id",
            "factor_returns_artifact_sha256",
            "factor_plan_sha256",
            "exposure_artifact_sha256",
            "provider_snapshot_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not isinstance(self.values, torch.Tensor)
            or self.values.device.type != "cpu"
            or self.values.dtype != torch.float64
            or self.values.ndim != 4
            or self.values.requires_grad
            or not bool(torch.isfinite(self.values).all())
        ):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposures must be detached finite CPU float64 [position,batch,asset,factor]"
            )
        if (
            not isinstance(self.valid, torch.Tensor)
            or self.valid.device.type != "cpu"
            or self.valid.dtype != torch.bool
            or tuple(self.valid.shape) != tuple(self.values.shape)
        ):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposure validity must be CPU bool with the exposure shape"
            )
        if (
            not isinstance(self.known_at_ms, torch.Tensor)
            or self.known_at_ms.device.type != "cpu"
            or self.known_at_ms.dtype != torch.int64
            or tuple(self.known_at_ms.shape) != tuple(self.values.shape)
            or self.known_at_ms.requires_grad
        ):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposure known-at evidence must be detached CPU int64"
            )
        if self.values.shape[-1] != len(self.factor_names):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposure axis differs from the declared factor names"
            )
        if bool((self.values.masked_select(~self.valid) != 0).any()) or not torch.equal(
            self.known_at_ms.eq(-1), ~self.valid
        ):
            raise Hold30AlphaStructuralQualificationError(
                "invalid factor exposures must be exact zero with known_at_ms=-1"
            )
        if (
            self.point_in_time is not True
            or self.evaluator_only is not True
            or self.policy_feature_access is not False
        ):
            raise Hold30AlphaStructuralQualificationError(
                "factor exposures must remain PIT, evaluator-only, and actor-invisible"
            )

    @property
    def receipt_id(self) -> str:
        return _sha256(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"values", "valid", "known_at_ms"}
            }
            | {
                "values_sha256": _tensor_sha256(self.values),
                "valid_sha256": _tensor_sha256(self.valid),
                "known_at_ms_sha256": _tensor_sha256(self.known_at_ms),
            }
        )


def _axis_dates(sequence: Hold30DatasetSequence) -> tuple[str, ...]:
    timestamps = sequence.decision_timestamps_ms.detach().to(
        device="cpu", dtype=torch.int64
    )
    return tuple(
        datetime.fromtimestamp(int(value) / 1000, tz=UTC)
        .date()
        .isoformat()
        for value in timestamps.tolist()
    )


def _slice_panel(
    panel: Hold30AlphaEvaluationPanel,
    sequence: Hold30DatasetSequence,
    absolute_range: tuple[int, int],
) -> Hold30AlphaEvaluationPanel:
    start, stop = absolute_range
    return replace(
        panel,
        source_axis_id=sequence.axis_id,
        risk_free_returns=panel.risk_free_returns[start : stop - 1],
        risk_free_valid=panel.risk_free_valid[start : stop - 1],
        market_total_returns=panel.market_total_returns[start : stop - 1],
        market_valid=panel.market_valid[start : stop - 1],
        factor_returns=panel.factor_returns[start : stop - 1],
        factor_valid=panel.factor_valid[start : stop - 1],
    )


def _require_variation(name: str, values: torch.Tensor) -> None:
    flat = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() < 2 or float(torch.max(flat) - torch.min(flat)) == 0.0:
        raise Hold30AlphaStructuralQualificationError(
            f"{name} is a prohibited zero/constant substitute"
        )


def _validate_external_series_are_not_substitutes(
    sequence: Hold30DatasetSequence,
    panel: Hold30AlphaEvaluationPanel,
) -> None:
    _require_variation("PIT risk-free returns", panel.risk_free_returns)
    _require_variation("PIT market returns", panel.market_total_returns)
    if torch.equal(
        panel.market_total_returns,
        sequence.c1_benchmark_net_returns.detach().to(device="cpu"),
    ):
        raise Hold30AlphaStructuralQualificationError(
            "PIT market returns cannot be a C1/equal-weight substitute"
        )
    for index, name in enumerate(panel.provenance.factor_names):
        _require_variation(
            f"factor returns {name}",
            panel.factor_returns[..., index],
        )


def _validate_factor_exposures(
    sequence: Hold30DatasetSequence,
    panel: Hold30AlphaEvaluationPanel,
    declaration: Hold30AlphaInMemoryFactorExposureDeclaration,
    lineage: Hold30AlphaInMemoryLineageDeclaration,
) -> None:
    if declaration.source_axis_id != sequence.axis_id:
        raise Hold30AlphaStructuralQualificationError(
            "factor exposure axis differs from the qualified sequence"
        )
    provenance = panel.provenance
    if (
        declaration.factor_model_id != provenance.factor_model_id
        or declaration.factor_names != provenance.factor_names
        or declaration.factor_returns_artifact_sha256
        != provenance.factor_artifact_sha256
        or declaration.factor_plan_sha256 != provenance.factor_plan_sha256
        or declaration.provider_snapshot_receipt_sha256
        != lineage.provider_snapshot_receipt_sha256
    ):
        raise Hold30AlphaStructuralQualificationError(
            "factor exposure declarations differ from evaluation declarations"
        )
    expected_shape = (
        sequence.n_positions,
        sequence.batch_size,
        sequence.num_assets,
        len(provenance.factor_names),
    )
    if tuple(declaration.values.shape) != expected_shape:
        raise Hold30AlphaStructuralQualificationError(
            f"factor exposures must have shape {expected_shape}"
        )
    risky_members = sequence.decision_membership.detach().to(device="cpu").clone()
    risky_members[..., sequence.cash_index] = False
    required = risky_members.unsqueeze(-1).expand(expected_shape)
    if not torch.equal(declaration.valid, required):
        raise Hold30AlphaStructuralQualificationError(
            "factor exposures must cover exactly every PIT active risky member"
        )
    decision_times = sequence.decision_timestamps_ms.detach().to(
        device="cpu", dtype=torch.int64
    ).view(sequence.n_positions, 1, 1, 1)
    if bool((declaration.known_at_ms.masked_select(required) < 0).any()) or bool(
        (
            declaration.known_at_ms.masked_select(required)
            > decision_times.expand(expected_shape).masked_select(required)
        ).any()
    ):
        raise Hold30AlphaStructuralQualificationError(
            "factor exposure evidence is missing or known after the decision"
        )
    for factor_index, factor_name in enumerate(declaration.factor_names):
        values = declaration.values[..., factor_index]
        valid = declaration.valid[..., factor_index]
        selected = values.masked_select(valid)
        _require_variation(f"factor exposures {factor_name}", selected)
        negative_inf = torch.full_like(values, -torch.inf)
        positive_inf = torch.full_like(values, torch.inf)
        row_max = torch.where(valid, values, negative_inf).amax(dim=(1, 2))
        row_min = torch.where(valid, values, positive_inf).amin(dim=(1, 2))
        counts = valid.sum(dim=(1, 2))
        if not bool(((counts > 1) & (row_max > row_min)).any()):
            raise Hold30AlphaStructuralQualificationError(
                f"factor exposures {factor_name} lack cross-sectional variation"
            )


def _validate_lineage_declaration(
    sequence: Hold30DatasetSequence,
    lineage: Hold30AlphaInMemoryLineageDeclaration,
) -> None:
    provenance = sequence.provenance
    expected = {
        "source_axis_id": sequence.axis_id,
        "data_snapshot_sha256": provenance.data_snapshot_sha256,
        "raw_market_data_sha256": provenance.raw_market_data_sha256,
        "universe_events_sha256": provenance.universe_events_sha256,
        "tradability_events_sha256": provenance.tradability_events_sha256,
        "corporate_actions_sha256": provenance.corporate_actions_sha256,
        "identifier_events_sha256": provenance.identifier_events_sha256,
    }
    for name, value in expected.items():
        if getattr(lineage, name) != value:
            raise Hold30AlphaStructuralQualificationError(
                f"lineage declaration does not bind exact {name}"
            )


def _fold_role_chain(
    *,
    fold_index: int,
    role: str,
    sequence: Hold30DatasetSequence,
    panel: Hold30AlphaEvaluationPanel,
    absolute_range: tuple[int, int],
    expected_origins: tuple[int, int],
) -> dict[str, Any]:
    if role not in {"training", "inner-validation"}:
        raise AssertionError("qualification requested an unsupported objective role")
    domain_name = "train" if role == "training" else "validation"
    domain = Hold30AlphaLabelDomain(
        domain_name,
        0,
        sequence.n_positions - 1,
    )
    labels = build_hold30_residual_alpha_labels(sequence, domains=(domain,))
    role_panel = _slice_panel(panel, sequence, absolute_range)
    binding = bind_hold30_alpha_evaluation_panel(sequence, role_panel)
    objective = bind_hold30_alpha_objective_inputs(
        sequence,
        labels,
        role_panel,
        binding,
        Hold30AlphaObjectiveDomainBinding(role=role, domain=domain),
    )
    absolute_origins = tuple(
        int(value) + absolute_range[0]
        for value in objective.score_origin_rows.tolist()
    )
    if absolute_origins != tuple(range(*expected_origins)):
        raise Hold30AlphaStructuralQualificationError(
            f"fold {fold_index} {role} objective rows differ from the frozen fold"
        )
    data_contract = bind_hold30_alpha_v3_data_contract(
        panel=role_panel,
        binding=binding,
        labels=labels,
    )
    return {
        "role": role,
        "absolute_range": list(absolute_range),
        "source_axis_id": sequence.axis_id,
        "evaluation_panel_id": role_panel.panel_id,
        "in_memory_data_binding_receipt_id": binding.receipt_id,
        "residual_labels_id": labels.labels_id,
        "objective_inputs_id": objective.objective_inputs_id,
        "score_origin_rows_sha256": _tensor_sha256(objective.score_origin_rows),
        "score_origin_absolute_range": list(expected_origins),
        "in_memory_v3_data_contract_sha256": sha256_payload(
            data_contract.manifest_payload()
        ),
    }


def qualify_hold30_alpha_in_memory_structure(
    sequence: Hold30DatasetSequence,
    *,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    base_data_qualification_receipt: Mapping[str, Any],
    evaluation_panel: Hold30AlphaEvaluationPanel,
    lineage_declaration: Hold30AlphaInMemoryLineageDeclaration,
    factor_exposure_declaration: Hold30AlphaInMemoryFactorExposureDeclaration,
) -> dict[str, Any]:
    """Recompute typed identities without claiming file-backed qualification."""

    if not isinstance(sequence, Hold30DatasetSequence):
        raise Hold30AlphaStructuralQualificationError(
            "typed Hold30DatasetSequence is required"
        )
    if sequence.n_positions < HOLD30_MIN_AXIS_POSITIONS:
        raise Hold30AlphaStructuralQualificationError(
            f"V3 structural input requires N >= {HOLD30_MIN_AXIS_POSITIONS}"
        )
    if not isinstance(evaluation_panel, Hold30AlphaEvaluationPanel):
        raise Hold30AlphaStructuralQualificationError(
            "typed Hold30AlphaEvaluationPanel is required"
        )
    if not isinstance(
        lineage_declaration,
        Hold30AlphaInMemoryLineageDeclaration,
    ):
        raise Hold30AlphaStructuralQualificationError(
            "typed in-memory lineage declaration is required"
        )
    if not isinstance(
        factor_exposure_declaration,
        Hold30AlphaInMemoryFactorExposureDeclaration,
    ):
        raise Hold30AlphaStructuralQualificationError(
            "typed point-in-time factor exposures are required"
        )
    if not isinstance(external_artifacts, Mapping) or set(
        external_artifacts
    ) != set(HOLD30_REQUIRED_EXTERNAL_ARTIFACTS):
        raise Hold30AlphaStructuralQualificationError(
            "external artifact inventory is not exact"
        )
    try:
        verify_hold30_dataset_against_qualification(
            sequence,
            monthly_rebalance,
            external_artifacts,
            base_data_qualification_receipt,
        )
        full_binding = bind_hold30_alpha_evaluation_panel(
            sequence,
            evaluation_panel,
        )
    except (Hold30DatasetError, Hold30AlphaDataError) as exc:
        raise Hold30AlphaStructuralQualificationError(
            f"base in-memory PIT consistency check failed: {exc}"
        ) from exc
    _validate_lineage_declaration(sequence, lineage_declaration)
    _validate_external_series_are_not_substitutes(sequence, evaluation_panel)
    _validate_factor_exposures(
        sequence,
        evaluation_panel,
        factor_exposure_declaration,
        lineage_declaration,
    )

    axis = _axis_dates(sequence)
    folds = render_hold30_folds(axis)
    if len(folds) != HOLD30_FOLDS:
        raise AssertionError("frozen fold renderer returned the wrong count")
    fold_payload = [asdict(fold) for fold in folds]
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        development = materialize_hold30_development_fold(
            sequence,
            fold,
            monthly_rebalance=monthly_rebalance,
            external_artifacts=external_artifacts,
            data_qualification_receipt=base_data_qualification_receipt,
        )
        fold_rows.append(
            {
                "fold_index": fold.fold_index,
                "fold_sha256": development.fold_sha256,
                "development_fold_receipt_sha256": development.receipt_sha256,
                "training": _fold_role_chain(
                    fold_index=fold.fold_index,
                    role="training",
                    sequence=development.training,
                    panel=evaluation_panel,
                    absolute_range=development.training_absolute_range,
                    expected_origins=fold.training_anchors,
                ),
                "inner_validation": _fold_role_chain(
                    fold_index=fold.fold_index,
                    role="inner-validation",
                    sequence=development.inner_validation,
                    panel=evaluation_panel,
                    absolute_range=development.validation_absolute_range,
                    expected_origins=fold.inner_validation,
                ),
                "outer_sequence_materialized": False,
            }
        )

    declared_artifact_digests = {
        **{path: external_artifacts[path] for path in sorted(external_artifacts)},
        "provider_snapshot_receipt_sha256": (
            lineage_declaration.provider_snapshot_receipt_sha256
        ),
        "risk_free_artifact_sha256": (
            evaluation_panel.provenance.risk_free_artifact_sha256
        ),
        "market_artifact_sha256": (
            evaluation_panel.provenance.market_artifact_sha256
        ),
        "factor_returns_artifact_sha256": (
            evaluation_panel.provenance.factor_artifact_sha256
        ),
        "factor_plan_sha256": evaluation_panel.provenance.factor_plan_sha256,
        "factor_exposure_artifact_sha256": (
            factor_exposure_declaration.exposure_artifact_sha256
        ),
    }
    payload: dict[str, Any] = {
        "schema": HOLD30_ALPHA_IN_MEMORY_STRUCTURAL_SCHEMA,
        "receipt_type": "prelockbox-hold30-alpha-v3-in-memory-structural-check",
        "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
        "passed": True,
        "qualification_scope": "in_memory_structural_consistency_only",
        "structural_consistency_verified": True,
        "real_data_attested": False,
        "caller_declares_synthetic_data": False,
        "caller_declares_future_selected_universe": False,
        "caller_declares_point_in_time_universe": True,
        "data_qualification_complete": False,
        "file_hash_verification": False,
        "provider_trust_verified": False,
        "outer_values_present": True,
        "outer_access_boundary_enforced": False,
        "production_data_eligible": False,
        "production_preflight_acceptable": False,
        "scientific_qualification": False,
        "launch_authorized": False,
        "promotion_authorized": False,
        "policy_feature_access": False,
        "actor_access": False,
        "counts": {
            "positions": sequence.n_positions,
            "assets": sequence.num_assets,
            "factors": len(evaluation_panel.provenance.factor_names),
            "folds": len(fold_rows),
            "role_chains": 2 * len(fold_rows),
        },
        "alpha_horizons": list(HOLD30_ALPHA_HORIZONS),
        "source_axis_id": sequence.axis_id,
        "base_in_memory_qualification_sha256": base_data_qualification_receipt[
            "receipt_sha256"
        ],
        "base_provenance_receipt_id": sequence.provenance.receipt_id,
        "lineage_declaration_receipt_id": lineage_declaration.receipt_id,
        "evaluation_provenance_declaration_receipt_id": (
            evaluation_panel.provenance.receipt_id
        ),
        "full_evaluation_panel_id": evaluation_panel.panel_id,
        "full_in_memory_data_binding_receipt_id": full_binding.receipt_id,
        "factor_exposure_declaration_receipt_id": (
            factor_exposure_declaration.receipt_id
        ),
        "folds_sha256": sha256_payload(fold_payload),
        "fold_chains": fold_rows,
        "declared_artifact_digests": declared_artifact_digests,
    }
    payload["receipt_sha256"] = _sha256(payload)
    verify_hold30_alpha_in_memory_structural_receipt(payload)
    return payload


_RECEIPT_FIELDS = {
    "schema",
    "receipt_type",
    "protocol_generation",
    "passed",
    "qualification_scope",
    "structural_consistency_verified",
    "real_data_attested",
    "caller_declares_synthetic_data",
    "caller_declares_future_selected_universe",
    "caller_declares_point_in_time_universe",
    "data_qualification_complete",
    "file_hash_verification",
    "provider_trust_verified",
    "outer_values_present",
    "outer_access_boundary_enforced",
    "production_data_eligible",
    "production_preflight_acceptable",
    "scientific_qualification",
    "launch_authorized",
    "promotion_authorized",
    "policy_feature_access",
    "actor_access",
    "counts",
    "alpha_horizons",
    "source_axis_id",
    "base_in_memory_qualification_sha256",
    "base_provenance_receipt_id",
    "lineage_declaration_receipt_id",
    "evaluation_provenance_declaration_receipt_id",
    "full_evaluation_panel_id",
    "full_in_memory_data_binding_receipt_id",
    "factor_exposure_declaration_receipt_id",
    "folds_sha256",
    "fold_chains",
    "declared_artifact_digests",
    "receipt_sha256",
}
_FOLD_CHAIN_FIELDS = {
    "fold_index",
    "fold_sha256",
    "development_fold_receipt_sha256",
    "training",
    "inner_validation",
    "outer_sequence_materialized",
}
_ROLE_CHAIN_FIELDS = {
    "role",
    "absolute_range",
    "source_axis_id",
    "evaluation_panel_id",
    "in_memory_data_binding_receipt_id",
    "residual_labels_id",
    "objective_inputs_id",
    "score_origin_rows_sha256",
    "score_origin_absolute_range",
    "in_memory_v3_data_contract_sha256",
}


def _require_half_open_range(name: str, value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise Hold30AlphaStructuralQualificationError(
            f"{name} must be a nonempty nonnegative half-open range"
        )
    return value[0], value[1]


def verify_hold30_alpha_in_memory_structural_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Verify exact receipt shape, authority denial, identities, and self-hash."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural receipt has partial or unknown fields"
        )
    if (
        receipt["schema"] != HOLD30_ALPHA_IN_MEMORY_STRUCTURAL_SCHEMA
        or receipt["receipt_type"]
        != "prelockbox-hold30-alpha-v3-in-memory-structural-check"
        or receipt["protocol_generation"] != HOLD30_ALPHA_PROTOCOL_GENERATION
        or receipt["passed"] is not True
        or receipt["qualification_scope"]
        != "in_memory_structural_consistency_only"
        or receipt["structural_consistency_verified"] is not True
        or receipt["real_data_attested"] is not False
        or receipt["caller_declares_synthetic_data"] is not False
        or receipt["caller_declares_future_selected_universe"] is not False
        or receipt["caller_declares_point_in_time_universe"] is not True
        or receipt["data_qualification_complete"] is not False
        or receipt["file_hash_verification"] is not False
        or receipt["provider_trust_verified"] is not False
        or receipt["outer_values_present"] is not True
        or receipt["outer_access_boundary_enforced"] is not False
        or receipt["production_data_eligible"] is not False
        or receipt["production_preflight_acceptable"] is not False
        or receipt["scientific_qualification"] is not False
        or receipt["launch_authorized"] is not False
        or receipt["promotion_authorized"] is not False
        or receipt["policy_feature_access"] is not False
        or receipt["actor_access"] is not False
        or receipt["alpha_horizons"] != list(HOLD30_ALPHA_HORIZONS)
    ):
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural receipt scope/authority fields are invalid"
        )
    counts = receipt["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {
        "positions",
        "assets",
        "factors",
        "folds",
        "role_chains",
    }:
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural receipt counts are malformed"
        )
    if (
        any(
            isinstance(counts[name], bool) or not isinstance(counts[name], int)
            for name in counts
        )
        or counts["positions"] < HOLD30_MIN_AXIS_POSITIONS
        or counts["assets"] < 301
        or counts["factors"] < 1
        or counts["folds"] != HOLD30_FOLDS
        or counts["role_chains"] != 2 * HOLD30_FOLDS
    ):
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural counts violate the frozen design"
        )
    for name in (
        "source_axis_id",
        "base_in_memory_qualification_sha256",
        "base_provenance_receipt_id",
        "lineage_declaration_receipt_id",
        "evaluation_provenance_declaration_receipt_id",
        "full_evaluation_panel_id",
        "full_in_memory_data_binding_receipt_id",
        "factor_exposure_declaration_receipt_id",
        "folds_sha256",
        "receipt_sha256",
    ):
        _require_digest(name, receipt[name])
    rows = receipt["fold_chains"]
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != HOLD30_FOLDS
    ):
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural receipt needs six fold chains"
        )
    for fold_index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != _FOLD_CHAIN_FIELDS
            or row.get("fold_index") != fold_index
            or row.get("outer_sequence_materialized") is not False
        ):
            raise Hold30AlphaStructuralQualificationError(
                "in-memory fold chain identity is invalid"
            )
        for name in ("fold_sha256", "development_fold_receipt_sha256"):
            _require_digest(name, row.get(name))
        for role_key, role_name in (
            ("training", "training"),
            ("inner_validation", "inner-validation"),
        ):
            role = row.get(role_key)
            if (
                not isinstance(role, Mapping)
                or set(role) != _ROLE_CHAIN_FIELDS
                or role.get("role") != role_name
            ):
                raise Hold30AlphaStructuralQualificationError(
                    "in-memory objective role chain is malformed"
                )
            absolute_range = _require_half_open_range(
                f"fold {fold_index} {role_name} absolute_range",
                role.get("absolute_range"),
            )
            origin_range = _require_half_open_range(
                f"fold {fold_index} {role_name} score origins",
                role.get("score_origin_absolute_range"),
            )
            if not (
                absolute_range[0] <= origin_range[0]
                < origin_range[1] <= absolute_range[1]
            ):
                raise Hold30AlphaStructuralQualificationError(
                    "in-memory objective origins lie outside their role slice"
                )
            for name in (
                "source_axis_id",
                "evaluation_panel_id",
                "in_memory_data_binding_receipt_id",
                "residual_labels_id",
                "objective_inputs_id",
                "score_origin_rows_sha256",
                "in_memory_v3_data_contract_sha256",
            ):
                _require_digest(name, role.get(name))
    declared_digests = receipt["declared_artifact_digests"]
    expected_digest_fields = {
        *HOLD30_REQUIRED_EXTERNAL_ARTIFACTS,
        "provider_snapshot_receipt_sha256",
        "risk_free_artifact_sha256",
        "market_artifact_sha256",
        "factor_returns_artifact_sha256",
        "factor_plan_sha256",
        "factor_exposure_artifact_sha256",
    }
    if (
        not isinstance(declared_digests, Mapping)
        or set(declared_digests) != expected_digest_fields
    ):
        raise Hold30AlphaStructuralQualificationError(
            "declared in-memory artifact digest inventory is incomplete"
        )
    for name, value in declared_digests.items():
        _require_digest(name, value)
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    if _sha256(unsigned) != claimed:
        raise Hold30AlphaStructuralQualificationError(
            "in-memory structural receipt self-hash mismatch"
        )


def verify_hold30_alpha_in_memory_against_structural_receipt(
    sequence: Hold30DatasetSequence,
    *,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    base_data_qualification_receipt: Mapping[str, Any],
    evaluation_panel: Hold30AlphaEvaluationPanel,
    lineage_declaration: Hold30AlphaInMemoryLineageDeclaration,
    factor_exposure_declaration: Hold30AlphaInMemoryFactorExposureDeclaration,
    receipt: Mapping[str, Any],
) -> None:
    """Recompute caller-supplied tensors and require canonical receipt equality."""

    verify_hold30_alpha_in_memory_structural_receipt(receipt)
    recomputed = qualify_hold30_alpha_in_memory_structure(
        sequence,
        monthly_rebalance=monthly_rebalance,
        external_artifacts=external_artifacts,
        base_data_qualification_receipt=base_data_qualification_receipt,
        evaluation_panel=evaluation_panel,
        lineage_declaration=lineage_declaration,
        factor_exposure_declaration=factor_exposure_declaration,
    )
    if _canonical_json(recomputed) != _canonical_json(receipt):
        raise Hold30AlphaStructuralQualificationError(
            "live V3 in-memory inputs differ from the structural receipt"
        )


def require_hold30_alpha_production_data_binding(
    receipt: Mapping[str, Any],
) -> None:
    """Reject this structural receipt at any production-data boundary."""

    verify_hold30_alpha_in_memory_structural_receipt(receipt)
    raise Hold30AlphaStructuralQualificationError(
        "in-memory structural receipts are never production data bindings; "
        "file hashes, provider trust, and an outer-access boundary remain unverified"
    )


__all__ = [
    "HOLD30_ALPHA_IN_MEMORY_FACTOR_EXPOSURE_SCHEMA",
    "HOLD30_ALPHA_IN_MEMORY_LINEAGE_DECLARATION_SCHEMA",
    "HOLD30_ALPHA_IN_MEMORY_STRUCTURAL_SCHEMA",
    "Hold30AlphaInMemoryFactorExposureDeclaration",
    "Hold30AlphaInMemoryLineageDeclaration",
    "Hold30AlphaStructuralQualificationError",
    "qualify_hold30_alpha_in_memory_structure",
    "require_hold30_alpha_production_data_binding",
    "verify_hold30_alpha_in_memory_against_structural_receipt",
    "verify_hold30_alpha_in_memory_structural_receipt",
]
