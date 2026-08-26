"""Dataset-bound trained runs for the embargoed Massive P0 tournament."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256,
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
    MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
    _run_from_payload,
    _run_payload,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256,
)

MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SCHEMA = (
    "rl-quant.massive-profitability-trained-run-v2"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA = (
    "rl-quant.massive-profitability-model-checkpoint-v2"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_DATASET = (
    "massive-profitability-model-checkpoint-v2"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA,
        "encoding": "canonical-json-float32-state",
        "publication": "create-only-source-transaction",
        "pickle": False,
    }
)
MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "dataset": "promoted-tournament-dataset-v3-exact-source-and-semantic",
        "tournament": "create-only-tournament-plan-v2",
        "phase": "embargoed-phase-plan-v2",
        "normalization": "fit-only-bound-in-run-v1",
        "checkpoint": "safe-float32-create-only",
        "prediction": "checkpoint-replay-required",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityTrainedRunV2Error(ValueError):
    """A trained state is detached from its exact source-derived dataset."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTrainedRunV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainedRunV2:
    run_v1: MassiveProfitabilityTrainedRunV1
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    dataset_v2_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    fold_receipt_sha256: str
    training_config_receipt_sha256: str
    normalization_receipt_sha256: str
    model_implementation_source_sha256: str
    training_implementation_source_sha256: str
    training_adapter_source_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    run_receipt_sha256: str
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_v1_receipt_sha256": self.run_v1.run_receipt_sha256,
            "model_state_sha256": self.run_v1.model_state_sha256,
            "setting_id": self.run_v1.setting_id,
            "fold_index": self.run_v1.fold_index,
            "seed": self.run_v1.seed,
            "best_epoch": self.run_v1.best_epoch,
            "validation_rank_ic": self.run_v1.validation_rank_ic,
            "dataset_semantic_receipt_sha256": self.dataset_semantic_receipt_sha256,
            "dataset_source_receipt_sha256": self.dataset_source_receipt_sha256,
            "dataset_v2_receipt_sha256": self.dataset_v2_receipt_sha256,
            "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
            "tournament_plan_source_receipt_sha256": self.tournament_plan_source_receipt_sha256,
            "phase_plan_semantic_receipt_sha256": self.phase_plan_semantic_receipt_sha256,
            "fold_receipt_sha256": self.fold_receipt_sha256,
            "training_config_receipt_sha256": self.training_config_receipt_sha256,
            "normalization_receipt_sha256": self.normalization_receipt_sha256,
            "model_implementation_source_sha256": self.model_implementation_source_sha256,
            "training_implementation_source_sha256": self.training_implementation_source_sha256,
            "training_adapter_source_sha256": self.training_adapter_source_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "outer_prediction_authorized": self.outer_prediction_authorized,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        self.run_v1.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SCHEMA
            or self.run_v1.tournament_plan_receipt_sha256
            != self.tournament_plan_receipt_sha256
            or self.run_v1.training_config_receipt_sha256
            != self.training_config_receipt_sha256
            or self.training_config_receipt_sha256
            != MassiveProfitabilityTrainingConfigV1().receipt_sha256
            or self.run_v1.normalization.receipt_sha256
            != self.normalization_receipt_sha256
            or self.model_implementation_source_sha256
            != MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256
            or self.training_implementation_source_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256
            or self.training_adapter_source_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256
            or self.run_v1.outer_prediction_authorized
            != self.outer_prediction_authorized
            or not self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SOURCE_SHA256
            or self.run_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityTrainedRunV2Error(
                "trained run V2 identity or authorization differs"
            )
        for value in (
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.dataset_v2_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.fold_receipt_sha256,
            self.training_config_receipt_sha256,
            self.normalization_receipt_sha256,
            self.model_implementation_source_sha256,
            self.training_implementation_source_sha256,
            self.training_adapter_source_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.run_receipt_sha256,
        ):
            _digest("trained run V2", value)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityModelCheckpointV2:
    run: MassiveProfitabilityTrainedRunV2
    loaded_source: LoadedMassiveSourceObject
    schema: str = MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA

    def validate(self) -> None:
        self.run.validate()
        self.loaded_source.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.run.dataset_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityTrainedRunV2Error(
                "model checkpoint V2 source transaction differs"
            )


def bind_massive_profitability_trained_run_v2(
    *,
    run_v1: MassiveProfitabilityTrainedRunV1,
    dataset_semantic_receipt_sha256: str,
    dataset_source_receipt_sha256: str,
    dataset_v2_receipt_sha256: str,
    tournament_plan_receipt_sha256: str,
    tournament_plan_source_receipt_sha256: str,
    phase_plan_semantic_receipt_sha256: str,
    fold_receipt_sha256: str,
) -> MassiveProfitabilityTrainedRunV2:
    run_v1.validate()
    body = {
        "schema": MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SCHEMA,
        "run_v1_receipt_sha256": run_v1.run_receipt_sha256,
        "model_state_sha256": run_v1.model_state_sha256,
        "setting_id": run_v1.setting_id,
        "fold_index": run_v1.fold_index,
        "seed": run_v1.seed,
        "best_epoch": run_v1.best_epoch,
        "validation_rank_ic": run_v1.validation_rank_ic,
        "dataset_semantic_receipt_sha256": dataset_semantic_receipt_sha256,
        "dataset_source_receipt_sha256": dataset_source_receipt_sha256,
        "dataset_v2_receipt_sha256": dataset_v2_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan_receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan_source_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan_semantic_receipt_sha256,
        "fold_receipt_sha256": fold_receipt_sha256,
        "training_config_receipt_sha256": run_v1.training_config_receipt_sha256,
        "normalization_receipt_sha256": run_v1.normalization.receipt_sha256,
        "model_implementation_source_sha256": MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256,
        "training_implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
        "training_adapter_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SOURCE_SHA256,
        "outer_prediction_authorized": run_v1.outer_prediction_authorized,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityTrainedRunV2(
        run_v1=run_v1,
        dataset_semantic_receipt_sha256=dataset_semantic_receipt_sha256,
        dataset_source_receipt_sha256=dataset_source_receipt_sha256,
        dataset_v2_receipt_sha256=dataset_v2_receipt_sha256,
        tournament_plan_receipt_sha256=tournament_plan_receipt_sha256,
        tournament_plan_source_receipt_sha256=tournament_plan_source_receipt_sha256,
        phase_plan_semantic_receipt_sha256=phase_plan_semantic_receipt_sha256,
        fold_receipt_sha256=fold_receipt_sha256,
        training_config_receipt_sha256=run_v1.training_config_receipt_sha256,
        normalization_receipt_sha256=run_v1.normalization.receipt_sha256,
        model_implementation_source_sha256=MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256,
        training_implementation_source_sha256=MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256,
        training_adapter_source_sha256=MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TRAINED_RUN_V2_SOURCE_SHA256,
        run_receipt_sha256=semantic_sha256(body),
        outer_prediction_authorized=run_v1.outer_prediction_authorized,
    )
    result.validate()
    return result


def _payload(run: MassiveProfitabilityTrainedRunV2) -> dict[str, object]:
    run.validate()
    return {
        "run_v1": _run_payload(run.run_v1),
        **{
            key: value
            for key, value in run.semantic_unsigned().items()
            if key
            not in {
                "schema",
                "run_v1_receipt_sha256",
                "model_state_sha256",
                "setting_id",
                "fold_index",
                "seed",
                "best_epoch",
                "validation_rank_ic",
            }
        },
        "run_receipt_sha256": run.run_receipt_sha256,
    }


def _from_payload(payload: dict[str, object]) -> MassiveProfitabilityTrainedRunV2:
    raw_run = payload.pop("run_v1")
    if not isinstance(raw_run, dict):
        raise MassiveProfitabilityTrainedRunV2Error("trained run V1 payload differs")
    run_v1 = _run_from_payload(dict(raw_run))
    result = MassiveProfitabilityTrainedRunV2(run_v1=run_v1, **payload)  # type: ignore[arg-type]
    result.validate()
    return result


def publish_massive_profitability_model_checkpoint_v2(
    *,
    root: str | Path,
    artifact_id: str,
    run: MassiveProfitabilityTrainedRunV2,
    committed_at_ms: int,
) -> MassiveProfitabilityModelCheckpointV2:
    run.validate()
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveProfitabilityTrainedRunV2Error(
            "model checkpoint V2 artifact ID is not path safe"
        )
    relative = f"massive-profitability/model-checkpoint-v2/{artifact_id}.json"
    payload = {
        "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA,
        "run": _payload(run),
    }
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=run.dataset_semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-MODEL-CHECKPOINT-V2-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_model_checkpoint_v2(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_model_checkpoint_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityModelCheckpointV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or raw != canonical_json_file_bytes(payload)
        or payload.get("schema") != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA
        or not isinstance(payload.get("run"), dict)
    ):
        raise MassiveProfitabilityTrainedRunV2Error(
            "model checkpoint V2 is not canonical JSON"
        )
    run = _from_payload(dict(payload["run"]))
    result = MassiveProfitabilityModelCheckpointV2(run=run, loaded_source=loaded_source)
    result.validate()
    if (
        canonical_json_file_bytes(
            {
                "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V2_SCHEMA,
                "run": _payload(run),
            }
        )
        != raw
    ):
        raise MassiveProfitabilityTrainedRunV2Error(
            "model checkpoint V2 canonical bytes differ"
        )
    return result


__all__ = [
    "MassiveProfitabilityModelCheckpointV2",
    "MassiveProfitabilityTrainedRunV2",
    "MassiveProfitabilityTrainedRunV2Error",
    "bind_massive_profitability_trained_run_v2",
    "parse_massive_profitability_model_checkpoint_v2",
    "publish_massive_profitability_model_checkpoint_v2",
]
