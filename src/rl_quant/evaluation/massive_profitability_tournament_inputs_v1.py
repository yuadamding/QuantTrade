"""Source-authority adapter for the feature-agnostic Massive P0 trainer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
    MassiveProfitabilityPhasePlanV1,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
    MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256,
    MassiveProfitabilityTournamentDatasetV1,
    MassiveProfitabilityTournamentPlanV1,
    MassiveProfitabilityTournamentV1Error,
    MassiveProfitabilityTrainingFoldV1,
    build_massive_profitability_date_tensor_v1,
)

MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SCHEMA = (
    "rl-quant.massive-profitability-tournament-inputs-v1"
)
MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "feature_generation": "massive-profitability-origin-features-v3",
        "target_generation": "massive-profitability-targets-v2",
        "gate_generation": "massive-profitability-data-gate-v2",
        "phase_generation": "massive-profitability-phase-plan-v1",
        "trainer": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA,
        "legacy_generations": "prohibited",
        "lockbox_targets": "excluded",
    }
)


def adapt_massive_profitability_training_fold_v1(
    fold: MassiveProfitabilityOuterFoldPlanV1,
) -> MassiveProfitabilityTrainingFoldV1:
    """Copy a frozen source-facing phase row into the lower training layer."""

    fold.validate()
    result = MassiveProfitabilityTrainingFoldV1(
        fold_index=fold.fold_index,
        fit_session_dates=fold.fit_session_dates,
        inner_purge_session_dates=fold.inner_purge_session_dates,
        inner_validation_session_dates=fold.inner_validation_session_dates,
        outer_purge_session_dates=fold.outer_purge_session_dates,
        outer_test_session_dates=fold.outer_test_session_dates,
        fit_inventory_sha256=fold.fit_inventory_sha256,
        inner_validation_inventory_sha256=fold.inner_validation_inventory_sha256,
        outer_test_inventory_sha256=fold.outer_test_inventory_sha256,
        receipt_sha256=fold.receipt_sha256,
    )
    result.validate()
    return result


def build_massive_profitability_tournament_plan_v1(
    *,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV1,
) -> MassiveProfitabilityTournamentPlanV1:
    """Bind the minimal tournament to one executable non-lockbox data gate."""

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
        or data_gate.outer_test_session_dates != outer_dates
        or data_gate.excluded_lockbox_session_dates != phase_plan.lockbox_session_dates
        or set(data_gate.gated_session_dates) & set(phase_plan.lockbox_session_dates)
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "tournament authorities do not share one qualified non-lockbox experiment"
        )
    body = {
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "fold_receipts": tuple(row.receipt_sha256 for row in phase_plan.outer_folds),
        "settings": MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
        "development_seeds": MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
        "confirmation_seeds": MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        "specification_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
        "input_adapter_source_sha256": (
            MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SOURCE_SHA256
        ),
        "development_training_authorized": True,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA,
    }
    result = MassiveProfitabilityTournamentPlanV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def build_massive_profitability_tournament_dataset_v1(
    *,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
) -> MassiveProfitabilityTournamentDatasetV1:
    """Materialize capacity-matched tensors without admitting lockbox targets."""

    data_gate.validate()
    phase_plan.validate()
    if (
        len(BARS_MIN_V2_FIELDS) != MASSIVE_PROFITABILITY_BARS_DIMENSION_V1
        or len(TAPE_MIN_V2_FIELDS) != MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "model dimensions differ from the frozen source feature inventory"
        )
    if not data_gate.data_gate_passed:
        raise MassiveProfitabilityTournamentV1Error("data gate V2 did not pass")
    ordered_features = tuple(sorted(features, key=lambda row: row.decision_session_date))
    ordered_targets = tuple(sorted(targets, key=lambda row: row.decision_session_date))
    dates = tuple(row.decision_session_date for row in ordered_features)
    if (
        data_gate.archive_freeze_semantic_receipt_sha256
        != phase_plan.archive_freeze_semantic_receipt_sha256
        or dates != data_gate.gated_session_dates
        or dates != tuple(row.decision_session_date for row in ordered_targets)
        or tuple(row.semantic_receipt_sha256 for row in ordered_features)
        != data_gate.feature_receipts
        or tuple(row.semantic_receipt_sha256 for row in ordered_targets)
        != data_gate.target_receipts
        or set(dates) & set(phase_plan.lockbox_session_dates)
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "feature or target inventory differs from the executable gate"
        )
    rows = []
    for feature, target in zip(ordered_features, ordered_targets, strict=True):
        feature.validate()
        target.validate()
        if not feature.source_inputs_data_qualified or not target.source_inputs_data_qualified:
            raise MassiveProfitabilityTournamentV1Error(
                "tournament input is not source-qualified"
            )
        feature_ids = tuple(row.security_id for row in feature.rows)
        target_ids = tuple(row.security_id for row in target.rows)
        if feature_ids != target_ids:
            raise MassiveProfitabilityTournamentV1Error(
                "bars, tape, and targets do not share identical asset support"
            )
        rows.append(
            build_massive_profitability_date_tensor_v1(
                decision_session_date=feature.decision_session_date,
                security_ids=feature_ids,
                bars_values=torch.tensor(
                    [row.bars_values for row in feature.rows], dtype=torch.float32
                ),
                bars_valid=torch.tensor(
                    [row.bars_valid for row in feature.rows], dtype=torch.bool
                ),
                tape_values=torch.tensor(
                    [row.tape_values for row in feature.rows], dtype=torch.float32
                ),
                tape_valid=torch.tensor(
                    [row.tape_valid for row in feature.rows], dtype=torch.bool
                ),
                target_values=torch.tensor(
                    [row.simple_returns for row in target.rows], dtype=torch.float32
                ),
                target_valid=torch.tensor(
                    [row.valid for row in target.rows], dtype=torch.bool
                ),
                feature_semantic_receipt_sha256=feature.semantic_receipt_sha256,
                target_semantic_receipt_sha256=target.semantic_receipt_sha256,
            )
        )
    body = {
        "dates": tuple(row.source_array_sha256 for row in rows),
        "data_gate": data_gate.semantic_receipt_sha256,
        "phase_plan": phase_plan.semantic_receipt_sha256,
    }
    result = MassiveProfitabilityTournamentDatasetV1(
        dates=tuple(rows),
        data_gate_semantic_receipt_sha256=data_gate.semantic_receipt_sha256,
        phase_plan_semantic_receipt_sha256=phase_plan.semantic_receipt_sha256,
        dataset_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_TOURNAMENT_INPUTS_V1_SCHEMA",
    "adapt_massive_profitability_training_fold_v1",
    "build_massive_profitability_tournament_dataset_v1",
    "build_massive_profitability_tournament_plan_v1",
]
