"""Training-and-prediction-replay-bound outer profitability evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    build_massive_profitability_residual_scores_v1,
    select_massive_profitability_tranches_v1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v4 import (
    MassiveProfitabilityEvaluationPlanV4,
    parse_massive_profitability_evaluation_plan_v4,
)
from rl_quant.evaluation.massive_profitability_evaluation_source_bundle_v5 import (
    MassiveProfitabilityEvaluationRuntimeSourcesV5,
    MassiveProfitabilityEvaluationSourceBundleV5,
    authorize_massive_profitability_evaluation_source_bundle_v5,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v2 import (
    MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256,
    MassiveProfitabilityAuthorizedResidualScoresV2,
    _source_residual_rows,
    _statistics,
    build_massive_profitability_stitched_composite_v2,
    evaluate_massive_profitability_stitched_capacity_v2,
    stitch_massive_profitability_fixed_tranches_v2,
)
from rl_quant.evaluation.massive_profitability_predictions_v3 import (
    MassiveProfitabilityOuterPredictionsV3,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA = (
    "rl-quant.massive-profitability-outer-evidence-v4"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_DATASET = (
    "massive-profitability-outer-evidence-v4"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "phase": "embargoed-phase-plan-v2",
        "tournament": "tournament-plan-v2-and-runtime-replayed-predictions-v3",
        "sources": "authorized-reconciled-evaluation-source-bundle-v5-only",
        "residual_inputs": "package-derived",
        "ledger": "one-stitched-calendar",
        "horizon_weights": (0.25, 0.25, 0.25, 0.25),
        "outer_only": True,
        "profitability_reporting": False,
        "lockbox": False,
        "retuning": False,
        "rl": False,
    }
)


class MassiveProfitabilityOuterEvidenceV4Error(ValueError):
    """The embargoed source-to-P&L evidence differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterEvidenceV4:
    evaluation_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    evaluation_source_bundle_semantic_receipt_sha256: str
    prediction_inventory_sha256: str
    authorized_residual_inventory_sha256: str
    selected_tranche_inventory_sha256: str
    stitched_pnl_receipts: tuple[str, ...]
    stitched_composite_receipts: tuple[str, ...]
    stitched_capacity_receipts: tuple[str, ...]
    fixed_horizon_scaling_receipt_sha256: str
    mean_mv04_net_20bp: float
    mean_mv04_minus_mv02_net_20bp: float
    mean_mv04_minus_shuffle_net_20bp: float
    mv04_net_20bp_lcb95: float
    mv04_minus_mv02_net_20bp_lcb95: float
    mv04_minus_shuffle_net_20bp_lcb95: float
    mean_mv04_net_40bp: float
    mean_mv04_clipped_10m_net_20bp: float
    annualized_mv04_net_return_20bp: float
    annualized_mv04_net_volatility_20bp: float
    mv04_net_sharpe_20bp: float
    mv04_net_sortino_20bp: float
    mv04_maximum_drawdown_20bp: float
    mv04_hit_rate_20bp: float
    mv04_mean_one_way_turnover: float
    mv04_break_even_one_way_cost: float
    positive_mv04_entry_fold_count: int
    calendar_date_count: int
    outer_profitability_gate_passed: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    development_conclusion_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        passed = (
            self.mv04_net_20bp_lcb95 > 0.0
            and self.mv04_minus_mv02_net_20bp_lcb95 > 0.0
            and self.mv04_minus_shuffle_net_20bp_lcb95 > 0.0
            and self.positive_mv04_entry_fold_count >= 3
            and self.mean_mv04_net_40bp >= 0.0
            and self.mean_mv04_clipped_10m_net_20bp > 0.0
        )
        statistics = (
            self.mean_mv04_net_20bp,
            self.mean_mv04_minus_mv02_net_20bp,
            self.mean_mv04_minus_shuffle_net_20bp,
            self.mv04_net_20bp_lcb95,
            self.mv04_minus_mv02_net_20bp_lcb95,
            self.mv04_minus_shuffle_net_20bp_lcb95,
            self.mean_mv04_net_40bp,
            self.mean_mv04_clipped_10m_net_20bp,
            self.annualized_mv04_net_return_20bp,
            self.annualized_mv04_net_volatility_20bp,
            self.mv04_net_sharpe_20bp,
            self.mv04_net_sortino_20bp,
            self.mv04_maximum_drawdown_20bp,
            self.mv04_hit_rate_20bp,
            self.mv04_mean_one_way_turnover,
            self.mv04_break_even_one_way_cost,
        )
        if (
            self.schema != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA
            or len(self.stitched_pnl_receipts) != 4
            or len(self.stitched_composite_receipts) != 4
            or len(self.stitched_capacity_receipts) != 4
            or self.fixed_horizon_scaling_receipt_sha256
            != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
            or any(not math.isfinite(value) for value in statistics)
            or not 0 <= self.positive_mv04_entry_fold_count <= 4
            or self.calendar_date_count <= 0
            or self.outer_profitability_gate_passed != passed
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SHA256
            or not self.development_conclusion_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV4Error(
                "outer evidence V4 identity or authorization differs"
            )
        for value in (
            self.evaluation_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.evaluation_source_bundle_semantic_receipt_sha256,
            self.prediction_inventory_sha256,
            self.authorized_residual_inventory_sha256,
            self.selected_tranche_inventory_sha256,
            *self.stitched_pnl_receipts,
            *self.stitched_composite_receipts,
            *self.stitched_capacity_receipts,
            self.fixed_horizon_scaling_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer evidence V4", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOuterEvidenceV4Error(
                "outer evidence V4 semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_source_bundle_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityOuterEvidenceV4Error(
                "outer evidence V4 committed source differs"
            )


def _authorized_residual_v4(
    *,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV4,
    prediction: MassiveProfitabilityOuterPredictionsV3,
    runtime_sources: MassiveProfitabilityEvaluationRuntimeSourcesV5,
) -> MassiveProfitabilityAuthorizedResidualScoresV2:
    registered = next(
        (
            row
            for row in evaluation_plan.predictions
            if row.fold_index == prediction.fold_index
            and row.setting_id == prediction.setting_id
        ),
        None,
    )
    if (
        registered is None
        or registered.prediction_semantic_receipt_sha256
        != prediction.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "prediction V3 is outside EvaluationPlan V4"
        )
    rows, eligibility = _source_residual_rows(
        prediction=prediction,  # type: ignore[arg-type]
        features=runtime_sources.features,
        feature_accounting=runtime_sources.feature_accounting,
        daily_input_authority=runtime_sources.daily_input_authority,
    )
    research = build_massive_profitability_residual_scores_v1(
        setting_id=prediction.setting_id,
        fold_index=prediction.fold_index,
        evaluation_plan_semantic_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        prediction_semantic_receipt_sha256=prediction.semantic_receipt_sha256,
        rows=rows,
    )
    residual_body = research.unsigned()
    residual_body["outer_evaluation_authorized"] = True
    residual = replace(
        research,
        outer_evaluation_authorized=True,
        semantic_receipt_sha256=semantic_sha256(residual_body),
    )
    residual.validate()
    feature_inventory = semantic_sha256(
        tuple(sorted(row.semantic_receipt_sha256 for row in runtime_sources.features))
    )
    accounting_inventory = semantic_sha256(
        tuple(
            sorted(
                row.semantic_receipt_sha256
                for row in runtime_sources.feature_accounting
            )
        )
    )
    eligibility_inventory = semantic_sha256(
        tuple(row.receipt_sha256 for row in eligibility)
    )
    body = {
        "schema": "rl-quant.massive-profitability-authorized-residual-scores-v2",
        "setting_id": prediction.setting_id,
        "fold_index": prediction.fold_index,
        "residual_scores": asdict(residual),
        "eligibility_dates": tuple(asdict(row) for row in eligibility),
        "evaluation_plan_semantic_receipt_sha256": evaluation_plan.semantic_receipt_sha256,
        "prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
        "feature_inventory_sha256": feature_inventory,
        "feature_accounting_inventory_sha256": accounting_inventory,
        "daily_input_authority_semantic_receipt_sha256": (
            runtime_sources.daily_input_authority.semantic_receipt_sha256
        ),
        "eligibility_inventory_sha256": eligibility_inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityAuthorizedResidualScoresV2(
        setting_id=prediction.setting_id,
        fold_index=prediction.fold_index,
        residual_scores=residual,
        eligibility_dates=eligibility,
        evaluation_plan_semantic_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        prediction_semantic_receipt_sha256=prediction.semantic_receipt_sha256,
        feature_inventory_sha256=feature_inventory,
        feature_accounting_inventory_sha256=accounting_inventory,
        daily_input_authority_semantic_receipt_sha256=(
            runtime_sources.daily_input_authority.semantic_receipt_sha256
        ),
        eligibility_inventory_sha256=eligibility_inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_profitability_outer_evidence_v4(
    *,
    root: str | Path,
    artifact_id: str,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV4,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    evaluation_source_bundle: MassiveProfitabilityEvaluationSourceBundleV5,
    runtime_sources: MassiveProfitabilityEvaluationRuntimeSourcesV5,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV3],
    committed_at_ms: int,
) -> MassiveProfitabilityOuterEvidenceV4:
    """Derive stitched profitability only from the reconciled V5 source bundle."""

    evaluation_plan.validate()
    phase_plan.validate()
    reloaded_plan = parse_massive_profitability_evaluation_plan_v4(
        root=root, loaded_source=evaluation_plan.loaded_source
    )
    authorized_bundle = authorize_massive_profitability_evaluation_source_bundle_v5(
        root=root,
        source_bundle=evaluation_source_bundle,
        runtime_sources=runtime_sources,
    )
    ordered = tuple(sorted(predictions, key=lambda row: (row.fold_index, row.setting_id)))
    if any(
        not isinstance(row, MassiveProfitabilityOuterPredictionsV3)
        or not row.runtime_prediction_replayed
        or not row.outer_prediction_authorized
        for row in ordered
    ):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "OuterEvidence V4 rejects non-V3 or unreplayed predictions"
        )
    expected = tuple(
        (fold_index, setting)
        for fold_index in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    registered = {
        (row.fold_index, row.setting_id): row.prediction_semantic_receipt_sha256
        for row in evaluation_plan.predictions
    }
    if (
        reloaded_plan.semantic_receipt_sha256 != evaluation_plan.semantic_receipt_sha256
        or evaluation_plan.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or authorized_bundle.evaluation_plan_semantic_receipt_sha256
        != evaluation_plan.semantic_receipt_sha256
        or runtime_sources.evaluation_plan.semantic_receipt_sha256
        != evaluation_plan.semantic_receipt_sha256
        or runtime_sources.phase_plan.semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or tuple((row.fold_index, row.setting_id) for row in ordered) != expected
        or any(
            registered.get((row.fold_index, row.setting_id))
            != row.semantic_receipt_sha256
            for row in ordered
        )
    ):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "outer evidence V4 roots or prediction inventory differ"
        )
    residuals = tuple(
        _authorized_residual_v4(
            evaluation_plan=evaluation_plan,
            prediction=prediction,
            runtime_sources=runtime_sources,
        )
        for prediction in ordered
    )
    for fold_index in range(4):
        if len(
            {
                row.eligibility_inventory_sha256
                for row in residuals
                if row.fold_index == fold_index
            }
        ) != 1:
            raise MassiveProfitabilityOuterEvidenceV4Error(
                "outer settings do not share one causal eligibility inventory"
            )
    selected = tuple(
        select_massive_profitability_tranches_v1(
            residual_scores=row.residual_scores,
            target_accounting=runtime_sources.target_accounting,
        )
        for row in residuals
    )
    stitched = tuple(
        stitch_massive_profitability_fixed_tranches_v2(
            selected_by_fold=tuple(
                row for row in selected if row.setting_id == setting
            ),
            target_accounting=runtime_sources.target_accounting,
        )
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    composites = tuple(
        build_massive_profitability_stitched_composite_v2(pnl=row) for row in stitched
    )
    capacity = tuple(
        evaluate_massive_profitability_stitched_capacity_v2(
            selected_by_fold=tuple(
                row for row in selected if row.setting_id == setting
            ),
            target_accounting=runtime_sources.target_accounting,
        )
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    statistics = _statistics(
        evaluation_plan=evaluation_plan,  # type: ignore[arg-type]
        phase_plan=phase_plan,  # type: ignore[arg-type]
        composites=composites,
        capacity=capacity,
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA,
        "evaluation_plan_semantic_receipt_sha256": evaluation_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": evaluation_plan.tournament_plan_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "evaluation_source_bundle_semantic_receipt_sha256": (
            authorized_bundle.semantic_receipt_sha256
        ),
        "prediction_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered)
        ),
        "authorized_residual_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in residuals)
        ),
        "selected_tranche_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in selected)
        ),
        "stitched_pnl_receipts": tuple(row.semantic_receipt_sha256 for row in stitched),
        "stitched_composite_receipts": tuple(
            row.semantic_receipt_sha256 for row in composites
        ),
        "stitched_capacity_receipts": tuple(
            row.semantic_receipt_sha256 for row in capacity
        ),
        "fixed_horizon_scaling_receipt_sha256": (
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        **statistics,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SHA256,
        "development_conclusion_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(not (value.isalnum() or value in "-_") for value in artifact_id):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "outer evidence V4 artifact ID is not path safe"
        )
    relative = f"massive-profitability/outer-evidence-v4/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=authorized_bundle.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-OUTER-EVIDENCE-V4-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_outer_evidence_v4(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_outer_evidence_v4(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterEvidenceV4:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "outer evidence V4 source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "outer evidence V4 source is not canonical JSON"
        )
    for name in (
        "stitched_pnl_receipts",
        "stitched_composite_receipts",
        "stitched_capacity_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityOuterEvidenceV4(**payload, loaded_source=loaded_source)
    result.validate()
    expected = result.semantic_unsigned() | {
        "semantic_receipt_sha256": result.semantic_receipt_sha256
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityOuterEvidenceV4Error(
            "outer evidence V4 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_DATASET",
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SCHEMA",
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V4_SPEC_SHA256",
    "MassiveProfitabilityOuterEvidenceV4",
    "MassiveProfitabilityOuterEvidenceV4Error",
    "materialize_massive_profitability_outer_evidence_v4",
    "parse_massive_profitability_outer_evidence_v4",
]
