"""Source-facing adapter for the embargoed Massive P0 tournament."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v1 import (
    build_massive_profitability_tournament_dataset_v1,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_DATASET,
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA,
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_SPEC_SHA256,
    MassiveProfitabilityTournamentDatasetV2,
    MassiveProfitabilityTournamentPlanV2,
    MassiveProfitabilityTournamentV2Error,
    MassiveProfitabilityTrainingFoldV2,
    parse_massive_profitability_tournament_plan_v2,
)


def adapt_massive_profitability_training_fold_v2(
    fold: MassiveProfitabilityOuterFoldPlanV1,
) -> MassiveProfitabilityTrainingFoldV2:
    fold.validate()
    result = MassiveProfitabilityTrainingFoldV2(**asdict(fold))
    result.validate()
    return result


def materialize_massive_profitability_tournament_plan_v2(
    *,
    root: str | Path,
    artifact_id: str,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    committed_at_ms: int,
) -> MassiveProfitabilityTournamentPlanV2:
    data_gate.validate()
    phase_plan.validate()
    outer_dates = tuple(
        value for fold in phase_plan.outer_folds for value in fold.outer_test_session_dates
    )
    if (
        not data_gate.data_gate_passed
        or not data_gate.development_training_authorized
        or not data_gate.outer_prediction_authorized
        or data_gate.archive_freeze_semantic_receipt_sha256
        != phase_plan.archive_freeze_semantic_receipt_sha256
        or phase_plan.lockbox_session_dates != data_gate.excluded_lockbox_session_dates
        or not set(outer_dates).issubset(data_gate.gated_session_dates)
        or not set(phase_plan.outer_to_lockbox_embargo_session_dates).issubset(
            data_gate.gated_session_dates
        )
        or set(outer_dates) & set(phase_plan.outer_to_lockbox_embargo_session_dates)
    ):
        raise MassiveProfitabilityTournamentV2Error(
            "tournament V2 roots do not share one embargoed non-lockbox experiment"
        )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "fold_receipts": tuple(row.receipt_sha256 for row in phase_plan.outer_folds),
        "embargo_session_dates": phase_plan.outer_to_lockbox_embargo_session_dates,
        "embargo_inventory_sha256": phase_plan.embargo_inventory_sha256,
        "settings": MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
        "development_seeds": MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
        "confirmation_seeds": MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        "specification_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256,
        "development_training_authorized": True,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(not (value.isalnum() or value in "-_") for value in artifact_id):
        raise MassiveProfitabilityTournamentV2Error(
            "tournament V2 artifact ID is not path safe"
        )
    relative = f"massive-profitability/tournament-plan-v2/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_TOURNAMENT_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=data_gate.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-TOURNAMENT-V2-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=loaded
    )


def build_massive_profitability_tournament_dataset_v2(
    *,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
) -> MassiveProfitabilityTournamentDatasetV2:
    base = build_massive_profitability_tournament_dataset_v1(
        data_gate=data_gate,
        phase_plan=phase_plan,  # type: ignore[arg-type]
        features=features,
        targets=targets,
    )
    dates = tuple(row.decision_session_date for row in base.dates)
    embargo = phase_plan.outer_to_lockbox_embargo_session_dates
    if not set(embargo).issubset(dates):
        raise MassiveProfitabilityTournamentV2Error(
            "tournament dataset V2 lacks embargo maturation rows"
        )
    entry = tuple(value for value in dates if value not in set(embargo))
    body = {
        "dates": tuple(row.source_array_sha256 for row in base.dates),
        "data_gate": data_gate.semantic_receipt_sha256,
        "phase_plan": phase_plan.semantic_receipt_sha256,
        "entry_session_dates": entry,
        "maturation_only_session_dates": embargo,
    }
    result = MassiveProfitabilityTournamentDatasetV2(
        dates=base.dates,
        data_gate_semantic_receipt_sha256=data_gate.semantic_receipt_sha256,
        phase_plan_semantic_receipt_sha256=phase_plan.semantic_receipt_sha256,
        entry_session_dates=entry,
        maturation_only_session_dates=embargo,
        dataset_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "adapt_massive_profitability_training_fold_v2",
    "build_massive_profitability_tournament_dataset_v2",
    "materialize_massive_profitability_tournament_plan_v2",
]
