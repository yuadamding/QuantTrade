"""Checkpoint-replayed outer predictions for the embargoed P0 tournament."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityPredictionRowV1,
    _mv00_rows,
    _rows_from_model,
)
from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    MassiveProfitabilityTournamentDatasetV3,
    authorize_massive_profitability_tournament_dataset_v3,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_HORIZONS_V1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MassiveProfitabilityTrainingConfigV1,
    fit_massive_profitability_normalization_v1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentPlanV2,
    MassiveProfitabilityTrainingFoldV2,
    parse_massive_profitability_tournament_plan_v2,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    MassiveProfitabilityModelCheckpointV2,
    parse_massive_profitability_model_checkpoint_v2,
)

MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA = (
    "rl-quant.massive-profitability-predictions-v3"
)
MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET = "massive-profitability-predictions-v3"
MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_PREDICTIONS_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "dataset": "promoted-tournament-dataset-v3",
        "runs": "five-create-only-dataset-bound-checkpoints-v2",
        "inference": "package-replayed-on-cpu",
        "ensemble": "exact-five-seed-output-space-mean",
        "mv00": "package-recomputed-fit-only-normalization",
        "generic_reload": "nonauthorizing",
        "targets": "not-read-by-inference",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveProfitabilityPredictionsV3Error(ValueError):
    """Committed scores differ from replayed dataset-bound checkpoints."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterPredictionsV3:
    setting_id: str
    fold_index: int
    seed_inventory: tuple[int, ...]
    ensemble: bool
    outer_test_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityPredictionRowV1, ...]
    feature_receipts: tuple[str, ...]
    checkpoint_source_receipts: tuple[str, ...]
    trained_run_receipts: tuple[str, ...]
    per_seed_row_inventory_receipts: tuple[str, ...]
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    dataset_v2_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    fold_receipt_sha256: str
    row_inventory_sha256: str
    committed_prediction_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_prediction_replayed: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "seed_inventory": self.seed_inventory,
            "ensemble": self.ensemble,
            "outer_test_session_dates": self.outer_test_session_dates,
            "rows": tuple(asdict(row) for row in self.rows),
            "feature_receipts": self.feature_receipts,
            "checkpoint_source_receipts": self.checkpoint_source_receipts,
            "trained_run_receipts": self.trained_run_receipts,
            "per_seed_row_inventory_receipts": self.per_seed_row_inventory_receipts,
            "dataset_semantic_receipt_sha256": self.dataset_semantic_receipt_sha256,
            "dataset_source_receipt_sha256": self.dataset_source_receipt_sha256,
            "dataset_v2_receipt_sha256": self.dataset_v2_receipt_sha256,
            "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
            "tournament_plan_source_receipt_sha256": self.tournament_plan_source_receipt_sha256,
            "phase_plan_semantic_receipt_sha256": self.phase_plan_semantic_receipt_sha256,
            "fold_receipt_sha256": self.fold_receipt_sha256,
            "row_inventory_sha256": self.row_inventory_sha256,
            "committed_prediction_qualified": self.committed_prediction_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        keys = tuple((row.decision_session_date, row.security_id) for row in self.rows)
        if (
            self.schema != MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA
            or self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or not 0 <= self.fold_index < 4
            or self.outer_test_session_dates
            != tuple(sorted(set(self.outer_test_session_dates)))
            or len(self.outer_test_session_dates) != 126
            or not keys
            or keys != tuple(sorted(set(keys)))
            or tuple(sorted({key[0] for key in keys})) != self.outer_test_session_dates
            or len(self.feature_receipts) != len(self.outer_test_session_dates)
            or not self.committed_prediction_qualified
            or self.runtime_prediction_replayed != self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SHA256
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityPredictionsV3Error(
                "prediction V3 identity or authorization differs"
            )
        expected_seeds = (
            (0,)
            if self.setting_id == "MV00"
            else MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        )
        expected_checkpoints = 0 if self.setting_id == "MV00" else 5
        if (
            self.seed_inventory != expected_seeds
            or self.ensemble != (self.setting_id != "MV00")
            or len(self.checkpoint_source_receipts) != expected_checkpoints
            or len(self.trained_run_receipts) != expected_checkpoints
            or len(self.per_seed_row_inventory_receipts) != len(expected_seeds)
        ):
            raise MassiveProfitabilityPredictionsV3Error(
                "prediction V3 seed or checkpoint inventory differs"
            )
        for row in self.rows:
            row.validate()
        by_date = dict(
            zip(self.outer_test_session_dates, self.feature_receipts, strict=True)
        )
        if any(
            {
                row.feature_semantic_receipt_sha256
                for row in self.rows
                if row.decision_session_date == session_date
            }
            != {by_date[session_date]}
            for session_date in self.outer_test_session_dates
        ):
            raise MassiveProfitabilityPredictionsV3Error(
                "prediction V3 feature support differs"
            )
        for value in (
            *self.feature_receipts,
            *self.checkpoint_source_receipts,
            *self.trained_run_receipts,
            *self.per_seed_row_inventory_receipts,
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.dataset_v2_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.fold_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prediction V3", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.dataset_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityPredictionsV3Error(
                "prediction V3 source transaction differs"
            )


def _average_rows(
    per_seed: Sequence[tuple[MassiveProfitabilityPredictionRowV1, ...]],
) -> tuple[MassiveProfitabilityPredictionRowV1, ...]:
    if not per_seed:
        raise MassiveProfitabilityPredictionsV3Error("prediction replay has no seeds")
    reference = per_seed[0]
    support = tuple((row.decision_session_date, row.security_id) for row in reference)
    if any(
        tuple((row.decision_session_date, row.security_id) for row in rows) != support
        for rows in per_seed[1:]
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "checkpoint prediction support differs across seeds"
        )
    result: list[MassiveProfitabilityPredictionRowV1] = []
    for index, first in enumerate(reference):
        seed_rows = tuple(rows[index] for rows in per_seed)
        averaged = {
            field: tuple(
                sum(getattr(row, field)[horizon] for row in seed_rows) / len(seed_rows)
                for horizon in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1))
            )
            for field in (
                "mean",
                "downside_quantile",
                "median",
                "upside_quantile",
                "scale",
            )
        }
        body = {
            "decision_session_date": first.decision_session_date,
            "security_id": first.security_id,
            **averaged,
            "feature_semantic_receipt_sha256": first.feature_semantic_receipt_sha256,
        }
        result.append(
            MassiveProfitabilityPredictionRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    return tuple(result)


def _replay(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    checkpoints: Sequence[MassiveProfitabilityModelCheckpointV2],
) -> tuple[
    tuple[MassiveProfitabilityPredictionRowV1, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    if dataset.runtime_dataset is None or not dataset.runtime_data_qualified:
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction replay requires promoted Dataset V3 tensors"
        )
    mapping = dataset.runtime_dataset.by_date()
    features = tuple(
        mapping[value].feature_semantic_receipt_sha256
        for value in fold.outer_test_session_dates
    )
    if setting_id == "MV00":
        if checkpoints:
            raise MassiveProfitabilityPredictionsV3Error(
                "MV00 prediction replay accepts no checkpoint"
            )
        normalization = fit_massive_profitability_normalization_v1(
            dataset=dataset.runtime_dataset,  # type: ignore[arg-type]
            fit_session_dates=fold.fit_session_dates,
        )
        rows = _mv00_rows(
            dataset=dataset.runtime_dataset,  # type: ignore[arg-type]
            fold=fold,  # type: ignore[arg-type]
            normalization=normalization,
        )
        inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
        return rows, features, (), (inventory,), (0,)
    ordered = tuple(sorted(checkpoints, key=lambda value: value.run.run_v1.seed))
    if (
        len(ordered) != 5
        or tuple(value.run.run_v1.seed for value in ordered)
        != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 requires the exact five confirmation seeds"
        )
    per_seed: list[tuple[MassiveProfitabilityPredictionRowV1, ...]] = []
    run_receipts: list[str] = []
    inventories: list[str] = []
    frozen_config = MassiveProfitabilityTrainingConfigV1()
    expected_normalization = fit_massive_profitability_normalization_v1(
        dataset=dataset.runtime_dataset,  # type: ignore[arg-type]
        fit_session_dates=fold.fit_session_dates,
        target_scale_floor=frozen_config.target_scale_floor,
    )
    mapping = dataset.runtime_dataset.by_date()
    expected_training_source_receipt = semantic_sha256(
        {
            "fit": tuple(
                mapping[value].source_array_sha256 for value in fold.fit_session_dates
            ),
            "inner_validation": tuple(
                mapping[value].source_array_sha256
                for value in fold.inner_validation_session_dates
            ),
        }
    )
    for checkpoint in ordered:
        checkpoint.validate()
        run = checkpoint.run
        if (
            run.run_v1.setting_id != setting_id
            or run.run_v1.fold_index != fold.fold_index
            or run.dataset_semantic_receipt_sha256 != dataset.semantic_receipt_sha256
            or run.dataset_source_receipt_sha256 != dataset.loaded_source.receipt_sha256
            or run.dataset_v2_receipt_sha256 != dataset.dataset_v2_receipt_sha256
            or run.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
            or run.tournament_plan_source_receipt_sha256
            != tournament_plan.loaded_source.receipt_sha256
            or run.phase_plan_semantic_receipt_sha256
            != tournament_plan.phase_plan_semantic_receipt_sha256
            or run.fold_receipt_sha256 != fold.receipt_sha256
            or run.training_config_receipt_sha256 != frozen_config.receipt_sha256
            or run.run_v1.normalization != expected_normalization
            or run.run_v1.training_source_receipt_sha256
            != expected_training_source_receipt
            or run.run_v1.fit_inventory_sha256 != fold.fit_inventory_sha256
            or run.run_v1.validation_inventory_sha256
            != fold.inner_validation_inventory_sha256
        ):
            raise MassiveProfitabilityPredictionsV3Error(
                "checkpoint is detached from prediction V3 roots"
            )
        rows = _rows_from_model(
            dataset=dataset.runtime_dataset,  # type: ignore[arg-type]
            fold=fold,  # type: ignore[arg-type]
            run=run.run_v1,
            device=torch.device("cpu"),
        )
        per_seed.append(rows)
        run_receipts.append(run.run_receipt_sha256)
        inventories.append(semantic_sha256(tuple(row.receipt_sha256 for row in rows)))
    return (
        _average_rows(per_seed),
        features,
        tuple(run_receipts),
        tuple(inventories),
        MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    )


def publish_massive_profitability_outer_predictions_v3(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    checkpoints: Sequence[MassiveProfitabilityModelCheckpointV2],
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV3:
    promoted = authorize_massive_profitability_tournament_dataset_v3(
        root=root,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
    )
    reloaded_plan = parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=tournament_plan.loaded_source
    )
    fold.validate()
    if (
        reloaded_plan.receipt_sha256 != tournament_plan.receipt_sha256
        or fold.receipt_sha256 != reloaded_plan.fold_receipts[fold.fold_index]
        or set(fold.outer_test_session_dates) & set(reloaded_plan.embargo_session_dates)
        or promoted.data_gate_semantic_receipt_sha256
        != reloaded_plan.data_gate_semantic_receipt_sha256
        or promoted.phase_plan_semantic_receipt_sha256
        != reloaded_plan.phase_plan_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 roots differ from the embargoed tournament"
        )
    reloaded_checkpoints = tuple(
        sorted(
            (
                parse_massive_profitability_model_checkpoint_v2(
                    root=root, loaded_source=value.loaded_source
                )
                for value in checkpoints
            ),
            key=lambda value: value.run.run_v1.seed,
        )
    )
    rows, feature_receipts, run_receipts, per_seed_receipts, seeds = _replay(
        dataset=promoted,
        tournament_plan=reloaded_plan,
        fold=fold,
        setting_id=setting_id,
        checkpoints=reloaded_checkpoints,
    )
    checkpoint_receipts = tuple(
        value.loaded_source.receipt_sha256 for value in reloaded_checkpoints
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SCHEMA,
        "setting_id": setting_id,
        "fold_index": fold.fold_index,
        "seed_inventory": seeds,
        "ensemble": setting_id != "MV00",
        "outer_test_session_dates": fold.outer_test_session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "feature_receipts": feature_receipts,
        "checkpoint_source_receipts": checkpoint_receipts,
        "trained_run_receipts": run_receipts,
        "per_seed_row_inventory_receipts": per_seed_receipts,
        "dataset_semantic_receipt_sha256": promoted.semantic_receipt_sha256,
        "dataset_source_receipt_sha256": promoted.loaded_source.receipt_sha256,
        "dataset_v2_receipt_sha256": promoted.dataset_v2_receipt_sha256,
        "tournament_plan_receipt_sha256": reloaded_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": reloaded_plan.loaded_source.receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "fold_receipt_sha256": fold.receipt_sha256,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "committed_prediction_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 artifact ID is not path safe"
        )
    relative = f"massive-profitability/predictions-v3/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=promoted.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-PREDICTIONS-V3-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_outer_predictions_v3(
        root=root, loaded_source=loaded
    )
    return authorize_massive_profitability_outer_predictions_v3(
        root=root,
        prediction=parsed,
        dataset=promoted,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=reloaded_plan,
        fold=fold,
        checkpoints=reloaded_checkpoints,
    )


def parse_massive_profitability_outer_predictions_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterPredictionsV3:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 source is not canonical JSON"
        )
    raw_rows = payload.pop("rows")
    if not isinstance(raw_rows, list):
        raise MassiveProfitabilityPredictionsV3Error("prediction V3 rows differ")
    rows = tuple(
        MassiveProfitabilityPredictionRowV1(
            **{
                **row,
                "mean": tuple(row["mean"]),
                "downside_quantile": tuple(row["downside_quantile"]),
                "median": tuple(row["median"]),
                "upside_quantile": tuple(row["upside_quantile"]),
                "scale": tuple(row["scale"]),
            }
        )
        for row in raw_rows
    )
    for name in (
        "seed_inventory",
        "outer_test_session_dates",
        "feature_receipts",
        "checkpoint_source_receipts",
        "trained_run_receipts",
        "per_seed_row_inventory_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityOuterPredictionsV3(
        **payload,
        rows=rows,
        loaded_source=loaded_source,
        runtime_prediction_replayed=False,
        outer_prediction_authorized=False,
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "semantic_receipt_sha256": result.semantic_receipt_sha256
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 canonical bytes differ"
        )
    return result


def authorize_massive_profitability_outer_predictions_v3(
    *,
    root: str | Path,
    prediction: MassiveProfitabilityOuterPredictionsV3,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    checkpoints: Sequence[MassiveProfitabilityModelCheckpointV2],
) -> MassiveProfitabilityOuterPredictionsV3:
    parsed = parse_massive_profitability_outer_predictions_v3(
        root=root, loaded_source=prediction.loaded_source
    )
    promoted = authorize_massive_profitability_tournament_dataset_v3(
        root=root,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
    )
    reloaded_plan = parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=tournament_plan.loaded_source
    )
    fold.validate()
    if (
        reloaded_plan.receipt_sha256 != tournament_plan.receipt_sha256
        or fold.receipt_sha256 != reloaded_plan.fold_receipts[fold.fold_index]
        or set(fold.outer_test_session_dates) & set(reloaded_plan.embargo_session_dates)
        or promoted.data_gate_semantic_receipt_sha256
        != reloaded_plan.data_gate_semantic_receipt_sha256
        or promoted.phase_plan_semantic_receipt_sha256
        != reloaded_plan.phase_plan_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 roots differ from the embargoed tournament"
        )
    reloaded_checkpoints = tuple(
        sorted(
            (
                parse_massive_profitability_model_checkpoint_v2(
                    root=root, loaded_source=value.loaded_source
                )
                for value in checkpoints
            ),
            key=lambda value: value.run.run_v1.seed,
        )
    )
    rows, feature_receipts, run_receipts, per_seed_receipts, seeds = _replay(
        dataset=promoted,
        tournament_plan=reloaded_plan,
        fold=fold,
        setting_id=parsed.setting_id,
        checkpoints=reloaded_checkpoints,
    )
    if (
        parsed.semantic_receipt_sha256 != prediction.semantic_receipt_sha256
        or parsed.rows != rows
        or parsed.feature_receipts != feature_receipts
        or parsed.trained_run_receipts != run_receipts
        or parsed.per_seed_row_inventory_receipts != per_seed_receipts
        or parsed.seed_inventory != seeds
        or parsed.checkpoint_source_receipts
        != tuple(value.loaded_source.receipt_sha256 for value in reloaded_checkpoints)
        or parsed.dataset_semantic_receipt_sha256 != promoted.semantic_receipt_sha256
        or parsed.dataset_source_receipt_sha256 != promoted.loaded_source.receipt_sha256
        or parsed.dataset_v2_receipt_sha256 != promoted.dataset_v2_receipt_sha256
        or parsed.tournament_plan_receipt_sha256 != reloaded_plan.receipt_sha256
        or parsed.tournament_plan_source_receipt_sha256
        != reloaded_plan.loaded_source.receipt_sha256
        or parsed.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or parsed.fold_receipt_sha256 != fold.receipt_sha256
        or parsed.outer_test_session_dates != fold.outer_test_session_dates
    ):
        raise MassiveProfitabilityPredictionsV3Error(
            "prediction V3 does not replay from its checkpoints and dataset"
        )
    result = replace(
        parsed,
        runtime_prediction_replayed=True,
        outer_prediction_authorized=True,
    )
    result.validate()
    return result


__all__ = [
    "MassiveProfitabilityOuterPredictionsV3",
    "MassiveProfitabilityPredictionsV3Error",
    "authorize_massive_profitability_outer_predictions_v3",
    "parse_massive_profitability_outer_predictions_v3",
    "publish_massive_profitability_outer_predictions_v3",
]
