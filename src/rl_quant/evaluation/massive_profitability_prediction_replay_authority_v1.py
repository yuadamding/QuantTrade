"""Root-bound prediction replay authority for the Massive P0 tournament."""

from __future__ import annotations

import json
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
from rl_quant.evaluation.massive_profitability_predictions_v3 import (
    MassiveProfitabilityOuterPredictionsV3,
    authorize_massive_profitability_outer_predictions_v3,
    parse_massive_profitability_outer_predictions_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    MassiveProfitabilityTournamentDatasetV3,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    adapt_massive_profitability_training_fold_v2,
)
from rl_quant.evaluation.massive_profitability_training_replay_authority_v1 import (
    MassiveProfitabilityTrainingReplayAuthorityV1,
    authorize_massive_profitability_training_replay_authority_v1,
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
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentPlanV2,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    MassiveProfitabilityModelCheckpointV3,
    load_massive_profitability_prediction_checkpoint_v2_from_v3,
    parse_massive_profitability_model_checkpoint_v3,
)

MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-profitability-prediction-replay-authority-v1"
)
MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET = (
    "massive-profitability-prediction-replay-authority-v1"
)
MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "inventory": "four-folds-four-settings",
        "training": "root-promoted-training-replay-authority-v1",
        "prediction": "package-replayed-from-dataset-and-immutable-checkpoints",
        "ensemble": "exact-five-seed-output-space-mean",
        "mv00": "package-recomputed-fit-only-normalization",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)
_SETTINGS = ("MV00", "MV02", "MV04", "MV04-SHUFFLE")


class MassiveProfitabilityPredictionReplayAuthorityV1Error(ValueError):
    """Committed predictions differ from root-replayed checkpoint inference."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPredictionReplayRowV1:
    fold_index: int
    setting_id: str
    prediction_semantic_receipt_sha256: str
    prediction_source_receipt_sha256: str
    prediction_payload_relative_path: str
    prediction_verified_at_ms: int
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    training_replay_authority_semantic_receipt_sha256: str
    fold_receipt_sha256: str
    prediction_row_inventory_sha256: str
    feature_inventory_sha256: str
    checkpoint_source_receipts: tuple[str, ...]
    trained_run_receipts: tuple[str, ...]
    per_seed_row_inventory_receipts: tuple[str, ...]
    committed_prediction_semantic_receipt_sha256: str
    replayed_prediction_semantic_receipt_sha256: str
    replay_success: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        expected_checkpoints = 0 if self.setting_id == "MV00" else 5
        expected_seed_rows = 1 if self.setting_id == "MV00" else 5
        if (
            not 0 <= self.fold_index < 4
            or self.setting_id not in _SETTINGS
            or not self.prediction_payload_relative_path.startswith(
                "massive-profitability/predictions-v3/"
            )
            or not self.prediction_payload_relative_path.endswith(".json")
            or ".." in self.prediction_payload_relative_path
            or isinstance(self.prediction_verified_at_ms, bool)
            or not isinstance(self.prediction_verified_at_ms, int)
            or self.prediction_verified_at_ms < 0
            or len(self.checkpoint_source_receipts) != expected_checkpoints
            or len(self.trained_run_receipts) != expected_checkpoints
            or len(self.per_seed_row_inventory_receipts) != expected_seed_rows
            or not self.replay_success
            or self.committed_prediction_semantic_receipt_sha256
            != self.replayed_prediction_semantic_receipt_sha256
            or self.prediction_semantic_receipt_sha256
            != self.committed_prediction_semantic_receipt_sha256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay row V1 differs"
            )
        for value in (
            self.prediction_semantic_receipt_sha256,
            self.prediction_source_receipt_sha256,
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.training_replay_authority_semantic_receipt_sha256,
            self.fold_receipt_sha256,
            self.prediction_row_inventory_sha256,
            self.feature_inventory_sha256,
            *self.checkpoint_source_receipts,
            *self.trained_run_receipts,
            *self.per_seed_row_inventory_receipts,
            self.committed_prediction_semantic_receipt_sha256,
            self.replayed_prediction_semantic_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("prediction replay row V1", value)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPredictionReplayAuthorityV1:
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    training_replay_authority_semantic_receipt_sha256: str
    training_replay_authority_source_receipt_sha256: str
    rows: tuple[MassiveProfitabilityPredictionReplayRowV1, ...]
    row_inventory_sha256: str
    committed_prediction_replay_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_predictions_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_semantic_receipt_sha256": self.dataset_semantic_receipt_sha256,
            "dataset_source_receipt_sha256": self.dataset_source_receipt_sha256,
            "data_gate_semantic_receipt_sha256": self.data_gate_semantic_receipt_sha256,
            "phase_plan_semantic_receipt_sha256": self.phase_plan_semantic_receipt_sha256,
            "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
            "tournament_plan_source_receipt_sha256": self.tournament_plan_source_receipt_sha256,
            "training_replay_authority_semantic_receipt_sha256": self.training_replay_authority_semantic_receipt_sha256,
            "training_replay_authority_source_receipt_sha256": self.training_replay_authority_source_receipt_sha256,
            "rows": tuple(asdict(row) for row in self.rows),
            "row_inventory_sha256": self.row_inventory_sha256,
            "committed_prediction_replay_qualified": self.committed_prediction_replay_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        keys = tuple((row.fold_index, row.setting_id) for row in self.rows)
        expected = tuple(
            (fold_index, setting_id)
            for fold_index in range(4)
            for setting_id in _SETTINGS
        )
        for row in self.rows:
            row.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA
            or keys != expected
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not self.committed_prediction_replay_qualified
            or self.runtime_predictions_replayed != self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay authority V1 identity differs"
            )
        for value in (
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.training_replay_authority_semantic_receipt_sha256,
            self.training_replay_authority_source_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prediction replay authority V1", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.dataset_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay authority V1 committed source differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPredictionReplayRuntimeV1:
    authority: MassiveProfitabilityPredictionReplayAuthorityV1
    training_replay_authority: MassiveProfitabilityTrainingReplayAuthorityV1
    predictions: tuple[MassiveProfitabilityOuterPredictionsV3, ...]


def _promoted_v3_checkpoints(
    *,
    root: str | Path,
    training_replay: MassiveProfitabilityTrainingReplayAuthorityV1,
) -> dict[tuple[int, str], tuple[MassiveProfitabilityModelCheckpointV3, ...]]:
    result: dict[tuple[int, str], list[MassiveProfitabilityModelCheckpointV3]] = {}
    for row in training_replay.rows:
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=row.checkpoint_v3_payload_relative_path,
            verified_at_ms=row.checkpoint_v3_verified_at_ms,
        )
        if loaded.receipt_sha256 != row.checkpoint_v3_source_receipt_sha256:
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay checkpoint V3 source differs"
            )
        parsed = parse_massive_profitability_model_checkpoint_v3(
            root=root, loaded_source=loaded
        )
        if (
            parsed.run.semantic_receipt_sha256
            != row.replayed_run_v3_semantic_receipt_sha256
            or parsed.run.run_v2.run_receipt_sha256 != row.run_v2_receipt_sha256
            or parsed.run.checkpoint_v2_source_receipt_sha256
            != row.checkpoint_v2_source_receipt_sha256
        ):
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay checkpoint differs from training replay authority"
            )
        promoted = MassiveProfitabilityModelCheckpointV3(
            run=replace(
                parsed.run,
                runtime_training_replayed=True,
                outer_prediction_authorized=True,
            ),
            loaded_source=parsed.loaded_source,
        )
        promoted.validate()
        result.setdefault((row.fold_index, row.setting_id), []).append(promoted)
    return {
        key: tuple(sorted(values, key=lambda value: value.run.run_v2.run_v1.seed))
        for key, values in result.items()
    }


def _prediction_sources(
    *,
    root: str | Path,
    rows: Sequence[MassiveProfitabilityPredictionReplayRowV1],
) -> tuple[MassiveProfitabilityOuterPredictionsV3, ...]:
    predictions = []
    for row in rows:
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=row.prediction_payload_relative_path,
            verified_at_ms=row.prediction_verified_at_ms,
        )
        if loaded.receipt_sha256 != row.prediction_source_receipt_sha256:
            raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
                "prediction replay source transaction differs"
            )
        predictions.append(
            parse_massive_profitability_outer_predictions_v3(
                root=root, loaded_source=loaded
            )
        )
    return tuple(predictions)


def _replayed_predictions(
    *,
    root: str | Path,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV3],
    training_replay: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> tuple[MassiveProfitabilityOuterPredictionsV3, ...]:
    checkpoint_map = _promoted_v3_checkpoints(
        root=root, training_replay=training_replay
    )
    ordered = tuple(
        sorted(predictions, key=lambda value: (value.fold_index, value.setting_id))
    )
    expected = tuple(
        (fold_index, setting_id) for fold_index in range(4) for setting_id in _SETTINGS
    )
    if tuple((row.fold_index, row.setting_id) for row in ordered) != expected:
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "prediction replay authority requires the exact 16-prediction inventory"
        )
    replayed = []
    for prediction in ordered:
        fold = adapt_massive_profitability_training_fold_v2(
            phase_plan.outer_folds[prediction.fold_index]
        )
        checkpoints_v3 = checkpoint_map.get(
            (prediction.fold_index, prediction.setting_id), ()
        )
        checkpoints_v2 = tuple(
            load_massive_profitability_prediction_checkpoint_v2_from_v3(
                root=root, checkpoint=value
            )
            for value in checkpoints_v3
        )
        replayed.append(
            authorize_massive_profitability_outer_predictions_v3(
                root=root,
                prediction=prediction,
                dataset=dataset,
                data_gate=data_gate,
                phase_plan=phase_plan,
                features=features,
                targets=targets,
                tournament_plan=tournament_plan,
                fold=fold,
                checkpoints=checkpoints_v2,
            )
        )
    return tuple(replayed)


def _row(
    *,
    prediction: MassiveProfitabilityOuterPredictionsV3,
    training_replay: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityPredictionReplayRowV1:
    body = {
        "fold_index": prediction.fold_index,
        "setting_id": prediction.setting_id,
        "prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
        "prediction_source_receipt_sha256": prediction.loaded_source.receipt_sha256,
        "prediction_payload_relative_path": prediction.loaded_source.payload_relative_path,
        "prediction_verified_at_ms": prediction.loaded_source.verified_at_ms,
        "dataset_semantic_receipt_sha256": dataset.semantic_receipt_sha256,
        "dataset_source_receipt_sha256": dataset.loaded_source.receipt_sha256,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan.loaded_source.receipt_sha256,
        "training_replay_authority_semantic_receipt_sha256": training_replay.semantic_receipt_sha256,
        "fold_receipt_sha256": prediction.fold_receipt_sha256,
        "prediction_row_inventory_sha256": prediction.row_inventory_sha256,
        "feature_inventory_sha256": semantic_sha256(prediction.feature_receipts),
        "checkpoint_source_receipts": prediction.checkpoint_source_receipts,
        "trained_run_receipts": prediction.trained_run_receipts,
        "per_seed_row_inventory_receipts": prediction.per_seed_row_inventory_receipts,
        "committed_prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
        "replayed_prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
        "replay_success": True,
    }
    result = MassiveProfitabilityPredictionReplayRowV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _semantic(
    *,
    rows: tuple[MassiveProfitabilityPredictionReplayRowV1, ...],
    training_replay: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA,
        "dataset_semantic_receipt_sha256": dataset.semantic_receipt_sha256,
        "dataset_source_receipt_sha256": dataset.loaded_source.receipt_sha256,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan.loaded_source.receipt_sha256,
        "training_replay_authority_semantic_receipt_sha256": training_replay.semantic_receipt_sha256,
        "training_replay_authority_source_receipt_sha256": training_replay.loaded_source.receipt_sha256,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "committed_prediction_replay_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    body["semantic_receipt_sha256"] = semantic_sha256(body)
    return body


def materialize_massive_profitability_prediction_replay_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    training_replay_authority: MassiveProfitabilityTrainingReplayAuthorityV1,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV3],
    committed_at_ms: int,
) -> MassiveProfitabilityPredictionReplayAuthorityV1:
    """Replay all 16 predictions after root-authorizing all 60 trainings."""

    training_replay = authorize_massive_profitability_training_replay_authority_v1(
        root=root,
        replay_authority=training_replay_authority,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    )
    committed = tuple(
        parse_massive_profitability_outer_predictions_v3(
            root=root, loaded_source=value.loaded_source
        )
        for value in predictions
    )
    replayed = _replayed_predictions(
        root=root,
        predictions=committed,
        training_replay=training_replay,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    )
    rows = tuple(
        _row(
            prediction=value,
            training_replay=training_replay,
            dataset=dataset,
            data_gate=data_gate,
            phase_plan=phase_plan,
            tournament_plan=tournament_plan,
        )
        for value in replayed
    )
    semantic = _semantic(
        rows=rows,
        training_replay=training_replay,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        tournament_plan=tournament_plan,
    )
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "prediction replay authority artifact ID is not path safe"
        )
    relative = (
        f"massive-profitability/prediction-replay-authority-v1/{artifact_id}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=dataset.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-PREDICTION-REPLAY-AUTHORITY-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_prediction_replay_authority_v1(
        root=root, loaded_source=loaded
    )
    if tuple(row.receipt_sha256 for row in parsed.rows) != tuple(
        row.receipt_sha256 for row in rows
    ):
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "committed prediction replay inventory differs"
        )
    result = replace(
        parsed, runtime_predictions_replayed=True, outer_evaluation_authorized=True
    )
    result.validate()
    return result


def parse_massive_profitability_prediction_replay_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityPredictionReplayAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "prediction replay authority V1 is not canonical JSON"
        )
    rows = tuple(
        MassiveProfitabilityPredictionReplayRowV1(
            **{
                **row,
                "checkpoint_source_receipts": tuple(row["checkpoint_source_receipts"]),
                "trained_run_receipts": tuple(row["trained_run_receipts"]),
                "per_seed_row_inventory_receipts": tuple(
                    row["per_seed_row_inventory_receipts"]
                ),
            }
        )
        for row in payload.pop("rows")
    )
    result = MassiveProfitabilityPredictionReplayAuthorityV1(
        **payload,
        rows=rows,
        loaded_source=loaded_source,
        runtime_predictions_replayed=False,
        outer_evaluation_authorized=False,
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "rows": tuple(asdict(row) for row in result.rows),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "prediction replay authority V1 canonical bytes differ"
        )
    return result


def resolve_massive_profitability_prediction_replay_authority_v1(
    *,
    root: str | Path,
    replay_authority: MassiveProfitabilityPredictionReplayAuthorityV1,
    training_replay_authority: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityPredictionReplayRuntimeV1:
    """Reexecute training once, then replay and resolve every prediction source."""

    parsed = parse_massive_profitability_prediction_replay_authority_v1(
        root=root, loaded_source=replay_authority.loaded_source
    )
    training_replay = authorize_massive_profitability_training_replay_authority_v1(
        root=root,
        replay_authority=training_replay_authority,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    )
    committed = _prediction_sources(root=root, rows=parsed.rows)
    replayed = _replayed_predictions(
        root=root,
        predictions=committed,
        training_replay=training_replay,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    )
    rows = tuple(
        _row(
            prediction=value,
            training_replay=training_replay,
            dataset=dataset,
            data_gate=data_gate,
            phase_plan=phase_plan,
            tournament_plan=tournament_plan,
        )
        for value in replayed
    )
    expected_roots = (
        dataset.semantic_receipt_sha256,
        dataset.loaded_source.receipt_sha256,
        data_gate.semantic_receipt_sha256,
        phase_plan.semantic_receipt_sha256,
        tournament_plan.receipt_sha256,
        tournament_plan.loaded_source.receipt_sha256,
        training_replay.semantic_receipt_sha256,
        training_replay.loaded_source.receipt_sha256,
    )
    actual_roots = (
        parsed.dataset_semantic_receipt_sha256,
        parsed.dataset_source_receipt_sha256,
        parsed.data_gate_semantic_receipt_sha256,
        parsed.phase_plan_semantic_receipt_sha256,
        parsed.tournament_plan_receipt_sha256,
        parsed.tournament_plan_source_receipt_sha256,
        parsed.training_replay_authority_semantic_receipt_sha256,
        parsed.training_replay_authority_source_receipt_sha256,
    )
    if (
        parsed.semantic_receipt_sha256 != replay_authority.semantic_receipt_sha256
        or expected_roots != actual_roots
        or tuple(row.receipt_sha256 for row in rows)
        != tuple(row.receipt_sha256 for row in parsed.rows)
    ):
        raise MassiveProfitabilityPredictionReplayAuthorityV1Error(
            "prediction replay authority V1 does not reproduce from its roots"
        )
    authority = replace(
        parsed, runtime_predictions_replayed=True, outer_evaluation_authorized=True
    )
    authority.validate()
    return MassiveProfitabilityPredictionReplayRuntimeV1(
        authority=authority,
        training_replay_authority=training_replay,
        predictions=replayed,
    )


def authorize_massive_profitability_prediction_replay_authority_v1(
    *,
    root: str | Path,
    replay_authority: MassiveProfitabilityPredictionReplayAuthorityV1,
    training_replay_authority: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityPredictionReplayAuthorityV1:
    """Promote a generic authority only through root training and inference replay."""

    return resolve_massive_profitability_prediction_replay_authority_v1(
        root=root,
        replay_authority=replay_authority,
        training_replay_authority=training_replay_authority,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    ).authority


__all__ = [
    "MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_DATASET",
    "MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_PREDICTION_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MassiveProfitabilityPredictionReplayAuthorityV1",
    "MassiveProfitabilityPredictionReplayAuthorityV1Error",
    "MassiveProfitabilityPredictionReplayRowV1",
    "MassiveProfitabilityPredictionReplayRuntimeV1",
    "authorize_massive_profitability_prediction_replay_authority_v1",
    "materialize_massive_profitability_prediction_replay_authority_v1",
    "parse_massive_profitability_prediction_replay_authority_v1",
    "resolve_massive_profitability_prediction_replay_authority_v1",
]
