"""Training-replayed checkpoints for the embargoed Massive P0 tournament."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    MassiveProfitabilityModelCheckpointV2,
    MassiveProfitabilityTrainedRunV2,
    parse_massive_profitability_model_checkpoint_v2,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    _from_payload as _run_v2_from_payload,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    _payload as _run_v2_payload,
)
from rl_quant.training.massive_profitability_training_replay_v3 import (
    MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256,
    MassiveProfitabilityTrainingEpochV3,
    MassiveProfitabilityTrainingRuntimeV3,
)

MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SCHEMA = (
    "rl-quant.massive-profitability-trained-run-v3"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA = (
    "rl-quant.massive-profitability-model-checkpoint-v3"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_DATASET = (
    "massive-profitability-model-checkpoint-v3"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA,
        "encoding": "canonical-json-float32-state-and-complete-epoch-trace",
        "publication": "create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "pickle": False,
    }
)
MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "dataset": "promoted-tournament-dataset-v3",
        "training": "complete-deterministic-one-thread-cpu-replay",
        "trace": "every-epoch-validation-ic-and-model-state",
        "runtime": "python-torch-numpy-platform-and-blas-config-bound",
        "generic_reload": "nonauthorizing",
        "prediction": "prediction-v3-only-after-training-replay",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveProfitabilityTrainedRunV3Error(ValueError):
    """A checkpoint differs from the complete deterministic training replay."""


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainedRunV3:
    run_v2: MassiveProfitabilityTrainedRunV2
    checkpoint_v2_source_receipt_sha256: str
    checkpoint_v2_payload_relative_path: str
    checkpoint_v2_verified_at_ms: int
    training_runtime: MassiveProfitabilityTrainingRuntimeV3
    epoch_trace: tuple[MassiveProfitabilityTrainingEpochV3, ...]
    epoch_trace_receipt_sha256: str
    training_replay_implementation_source_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    committed_training_qualified: bool
    runtime_training_replayed: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_v2_receipt_sha256": self.run_v2.run_receipt_sha256,
            "checkpoint_v2_source_receipt_sha256": self.checkpoint_v2_source_receipt_sha256,
            "checkpoint_v2_payload_relative_path": self.checkpoint_v2_payload_relative_path,
            "checkpoint_v2_verified_at_ms": self.checkpoint_v2_verified_at_ms,
            "model_state_sha256": self.run_v2.run_v1.model_state_sha256,
            "setting_id": self.run_v2.run_v1.setting_id,
            "fold_index": self.run_v2.run_v1.fold_index,
            "seed": self.run_v2.run_v1.seed,
            "dataset_semantic_receipt_sha256": self.run_v2.dataset_semantic_receipt_sha256,
            "dataset_source_receipt_sha256": self.run_v2.dataset_source_receipt_sha256,
            "training_runtime": asdict(self.training_runtime),
            "epoch_trace": tuple(asdict(row) for row in self.epoch_trace),
            "epoch_trace_receipt_sha256": self.epoch_trace_receipt_sha256,
            "training_replay_implementation_source_sha256": self.training_replay_implementation_source_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "committed_training_qualified": self.committed_training_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        self.run_v2.validate()
        self.training_runtime.validate()
        for row in self.epoch_trace:
            row.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SCHEMA
            or not self.epoch_trace
            or tuple(row.epoch for row in self.epoch_trace)
            != tuple(range(len(self.epoch_trace)))
            or len(self.epoch_trace) != self.run_v2.run_v1.completed_epochs
            or self.epoch_trace[self.run_v2.run_v1.best_epoch].validation_rank_ic
            != self.run_v2.run_v1.validation_rank_ic
            or self.epoch_trace[self.run_v2.run_v1.best_epoch].model_state_sha256
            != self.run_v2.run_v1.model_state_sha256
            or self.epoch_trace_receipt_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.epoch_trace))
            or self.training_replay_implementation_source_sha256
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SOURCE_SHA256
            or not self.committed_training_qualified
            or self.runtime_training_replayed != self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityTrainedRunV3Error(
                "trained run V3 identity or replay authorization differs"
            )
        if (
            len(self.checkpoint_v2_source_receipt_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.checkpoint_v2_source_receipt_sha256
            )
        ):
            raise MassiveProfitabilityTrainedRunV3Error(
                "checkpoint V2 source receipt differs"
            )
        if (
            not self.checkpoint_v2_payload_relative_path.startswith(
                "massive-profitability/model-checkpoint-v2/"
            )
            or not self.checkpoint_v2_payload_relative_path.endswith(".json")
            or ".." in self.checkpoint_v2_payload_relative_path
        ):
            raise MassiveProfitabilityTrainedRunV3Error(
                "checkpoint V2 payload path differs"
            )
        if (
            isinstance(self.checkpoint_v2_verified_at_ms, bool)
            or not isinstance(self.checkpoint_v2_verified_at_ms, int)
            or self.checkpoint_v2_verified_at_ms < 0
        ):
            raise MassiveProfitabilityTrainedRunV3Error(
                "checkpoint V2 verification time differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityModelCheckpointV3:
    run: MassiveProfitabilityTrainedRunV3
    loaded_source: LoadedMassiveSourceObject
    schema: str = MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA

    def validate(self) -> None:
        self.run.validate()
        self.loaded_source.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.run.run_v2.dataset_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityTrainedRunV3Error(
                "model checkpoint V3 source transaction differs"
            )


def bind_massive_profitability_trained_run_v3(
    *,
    run_v2: MassiveProfitabilityTrainedRunV2,
    checkpoint_v2_source_receipt_sha256: str,
    checkpoint_v2_payload_relative_path: str,
    checkpoint_v2_verified_at_ms: int,
    training_runtime: MassiveProfitabilityTrainingRuntimeV3,
    epoch_trace: tuple[MassiveProfitabilityTrainingEpochV3, ...],
) -> MassiveProfitabilityTrainedRunV3:
    run_v2.validate()
    training_runtime.validate()
    body = {
        "schema": MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SCHEMA,
        "run_v2_receipt_sha256": run_v2.run_receipt_sha256,
        "checkpoint_v2_source_receipt_sha256": checkpoint_v2_source_receipt_sha256,
        "checkpoint_v2_payload_relative_path": checkpoint_v2_payload_relative_path,
        "checkpoint_v2_verified_at_ms": checkpoint_v2_verified_at_ms,
        "model_state_sha256": run_v2.run_v1.model_state_sha256,
        "setting_id": run_v2.run_v1.setting_id,
        "fold_index": run_v2.run_v1.fold_index,
        "seed": run_v2.run_v1.seed,
        "dataset_semantic_receipt_sha256": run_v2.dataset_semantic_receipt_sha256,
        "dataset_source_receipt_sha256": run_v2.dataset_source_receipt_sha256,
        "training_runtime": asdict(training_runtime),
        "epoch_trace": tuple(asdict(row) for row in epoch_trace),
        "epoch_trace_receipt_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in epoch_trace)
        ),
        "training_replay_implementation_source_sha256": MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SOURCE_SHA256,
        "committed_training_qualified": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityTrainedRunV3(
        run_v2=run_v2,
        checkpoint_v2_source_receipt_sha256=checkpoint_v2_source_receipt_sha256,
        checkpoint_v2_payload_relative_path=checkpoint_v2_payload_relative_path,
        checkpoint_v2_verified_at_ms=checkpoint_v2_verified_at_ms,
        training_runtime=training_runtime,
        epoch_trace=epoch_trace,
        epoch_trace_receipt_sha256=body["epoch_trace_receipt_sha256"],  # type: ignore[arg-type]
        training_replay_implementation_source_sha256=MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_sha256(body),
        committed_training_qualified=True,
        runtime_training_replayed=True,
        outer_prediction_authorized=True,
    )
    result.validate()
    return result


def _payload(run: MassiveProfitabilityTrainedRunV3) -> dict[str, object]:
    run.validate()
    return {
        "run_v2": _run_v2_payload(run.run_v2),
        **run.semantic_unsigned(),
        "semantic_receipt_sha256": run.semantic_receipt_sha256,
    }


def _from_payload(payload: dict[str, object]) -> MassiveProfitabilityTrainedRunV3:
    raw_run = payload.pop("run_v2")
    if not isinstance(raw_run, dict):
        raise MassiveProfitabilityTrainedRunV3Error("trained run V2 payload differs")
    payload.pop("run_v2_receipt_sha256")
    payload.pop("model_state_sha256")
    payload.pop("setting_id")
    payload.pop("fold_index")
    payload.pop("seed")
    payload.pop("dataset_semantic_receipt_sha256")
    payload.pop("dataset_source_receipt_sha256")
    raw_runtime = payload.pop("training_runtime")
    raw_trace = payload.pop("epoch_trace")
    if not isinstance(raw_runtime, dict) or not isinstance(raw_trace, list):
        raise MassiveProfitabilityTrainedRunV3Error("training replay payload differs")
    result = MassiveProfitabilityTrainedRunV3(
        run_v2=_run_v2_from_payload(dict(raw_run)),
        training_runtime=MassiveProfitabilityTrainingRuntimeV3(**raw_runtime),
        epoch_trace=tuple(
            MassiveProfitabilityTrainingEpochV3(
                **{
                    **row,
                    "validation_rank_ic": tuple(row["validation_rank_ic"]),
                }
            )
            for row in raw_trace
        ),
        runtime_training_replayed=False,
        outer_prediction_authorized=False,
        **payload,  # type: ignore[arg-type]
    )
    result.validate()
    return result


def publish_massive_profitability_model_checkpoint_v3(
    *,
    root: str | Path,
    artifact_id: str,
    run: MassiveProfitabilityTrainedRunV3,
    committed_at_ms: int,
) -> MassiveProfitabilityModelCheckpointV3:
    run.validate()
    if not run.runtime_training_replayed or not run.outer_prediction_authorized:
        raise MassiveProfitabilityTrainedRunV3Error(
            "checkpoint V3 publication requires package-replayed training"
        )
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveProfitabilityTrainedRunV3Error(
            "model checkpoint V3 artifact ID is not path safe"
        )
    relative = f"massive-profitability/model-checkpoint-v3/{artifact_id}.json"
    payload = {"schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA, "run": _payload(run)}
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=run.run_v2.dataset_semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-MODEL-CHECKPOINT-V3-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_model_checkpoint_v3(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_model_checkpoint_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityModelCheckpointV3:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or raw != canonical_json_file_bytes(payload)
        or payload.get("schema") != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA
        or not isinstance(payload.get("run"), dict)
    ):
        raise MassiveProfitabilityTrainedRunV3Error(
            "model checkpoint V3 is not canonical JSON"
        )
    run = _from_payload(dict(payload["run"]))
    result = MassiveProfitabilityModelCheckpointV3(run=run, loaded_source=loaded_source)
    result.validate()
    if canonical_json_file_bytes(
        {"schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA, "run": _payload(run)}
    ) != raw:
        raise MassiveProfitabilityTrainedRunV3Error(
            "model checkpoint V3 canonical bytes differ"
        )
    return result


def authorize_massive_profitability_model_checkpoint_v3(
    *,
    root: str | Path,
    checkpoint: MassiveProfitabilityModelCheckpointV3,
    replayed_run: MassiveProfitabilityTrainedRunV3,
) -> MassiveProfitabilityModelCheckpointV3:
    parsed = parse_massive_profitability_model_checkpoint_v3(
        root=root, loaded_source=checkpoint.loaded_source
    )
    replayed_run.validate()
    if (
        not replayed_run.runtime_training_replayed
        or parsed.run.semantic_receipt_sha256 != replayed_run.semantic_receipt_sha256
        or _run_v2_payload(parsed.run.run_v2) != _run_v2_payload(replayed_run.run_v2)
        or parsed.run.training_runtime != replayed_run.training_runtime
        or parsed.run.epoch_trace != replayed_run.epoch_trace
    ):
        raise MassiveProfitabilityTrainedRunV3Error(
            "checkpoint V3 does not reproduce from deterministic training replay"
        )
    result = MassiveProfitabilityModelCheckpointV3(
        run=replace(
            parsed.run,
            runtime_training_replayed=True,
            outer_prediction_authorized=True,
        ),
        loaded_source=parsed.loaded_source,
    )
    result.validate()
    return result


def load_massive_profitability_prediction_checkpoint_v2_from_v3(
    *,
    root: str | Path,
    checkpoint: MassiveProfitabilityModelCheckpointV3,
) -> MassiveProfitabilityModelCheckpointV2:
    """Resolve the exact immutable V2 encoding consumed by Prediction V3."""

    checkpoint.validate()
    if not checkpoint.run.runtime_training_replayed:
        raise MassiveProfitabilityTrainedRunV3Error(
            "Prediction V3 compatibility checkpoint requires training replay"
        )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=checkpoint.run.checkpoint_v2_payload_relative_path,
        verified_at_ms=checkpoint.run.checkpoint_v2_verified_at_ms,
    )
    result = parse_massive_profitability_model_checkpoint_v2(
        root=root, loaded_source=loaded
    )
    if (
        result.loaded_source.receipt_sha256
        != checkpoint.run.checkpoint_v2_source_receipt_sha256
        or result.run.run_receipt_sha256 != checkpoint.run.run_v2.run_receipt_sha256
    ):
        raise MassiveProfitabilityTrainedRunV3Error(
            "Prediction V3 compatibility checkpoint differs from Checkpoint V3"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_DATASET",
    "MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V3_SCHEMA",
    "MASSIVE_PROFITABILITY_TRAINED_RUN_V3_SCHEMA",
    "MassiveProfitabilityModelCheckpointV3",
    "MassiveProfitabilityTrainedRunV3",
    "MassiveProfitabilityTrainedRunV3Error",
    "authorize_massive_profitability_model_checkpoint_v3",
    "bind_massive_profitability_trained_run_v3",
    "load_massive_profitability_prediction_checkpoint_v2_from_v3",
    "parse_massive_profitability_model_checkpoint_v3",
    "publish_massive_profitability_model_checkpoint_v3",
]
