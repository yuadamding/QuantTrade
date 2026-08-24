"""Typed deterministic inference artifacts for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path

from rl_quant.features.massive_pit500_tensor_v0 import (
    MassivePIT500DecisionTensorV0,
    validate_massive_pit500_tensor_v0,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_rolling_features_v0 import (
    MASSIVE_ROLLING_BARS_V0_FIELDS,
    MASSIVE_ROLLING_TAPE_V0_FIELDS,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS,
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_VALIDATION_CHECKPOINT_V0_SCHEMA = "rl-quant.massive-validation-checkpoint-v0"
MASSIVE_VALIDATION_CHECKPOINT_V0_DATASET = "massive-finalized-validation-checkpoint-v0"
MASSIVE_VALIDATION_CHECKPOINT_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_CHECKPOINT_V0_SCHEMA,
        "model": "frozen-linear-distribution-canary",
        "horizons": tuple(
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        ),
    }
)
MASSIVE_VALIDATION_INFERENCE_V0_SCHEMA = "rl-quant.massive-validation-inference-v0"
MASSIVE_VALIDATION_INFERENCE_V0_DATASET = "massive-finalized-validation-inference-v0"
MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "model": "frozen-linear-distribution-canary",
        "missing": "training-mean-imputation-before-normalization",
        "heads": ("mean", "q10", "median", "q90", "scale", "probability-positive"),
        "checkpoint_required": True,
    }
)
MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_INFERENCE_V0_SCHEMA,
        "horizons": tuple(
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        ),
        "outputs": ("mean", "q10", "median", "q90", "scale", "probability_positive"),
    }
)


class MassiveValidationInferenceV0Error(ValueError):
    """Checkpoint or inference output differs from committed bytes."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveValidationInferenceV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveValidationCheckpointV0:
    setting_id: str
    feature_set_id: str
    seed: int
    feature_names: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    normalization_mean: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]
    quantile_offsets: tuple[float, ...]
    predictive_scales: tuple[float, ...]
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_VALIDATION_CHECKPOINT_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VALIDATION_CHECKPOINT_V0_SCHEMA:
            raise MassiveValidationInferenceV0Error("checkpoint schema drifted")
        expected_features = {
            "MV02": ("BARS_V0", MASSIVE_ROLLING_BARS_V0_FIELDS),
            "MV04": (
                "BARS_PLUS_TAPE_V0",
                MASSIVE_ROLLING_BARS_V0_FIELDS + MASSIVE_ROLLING_TAPE_V0_FIELDS,
            ),
        }
        if (
            self.setting_id not in expected_features
            or (
                self.feature_set_id,
                self.feature_names,
            )
            != expected_features[self.setting_id]
        ):
            raise MassiveValidationInferenceV0Error(
                "checkpoint feature contract drifted"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise MassiveValidationInferenceV0Error("checkpoint seed is invalid")
        expected_horizons = tuple(
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        )
        width = len(self.feature_names)
        if (
            self.horizon_ids != expected_horizons
            or len(self.normalization_mean) != width
            or len(self.normalization_scale) != width
            or len(self.weights) != len(expected_horizons)
            or any(len(row) != width for row in self.weights)
            or len(self.biases) != len(expected_horizons)
            or len(self.quantile_offsets) != len(expected_horizons)
            or len(self.predictive_scales) != len(expected_horizons)
        ):
            raise MassiveValidationInferenceV0Error("checkpoint shapes differ")
        scalars = (
            self.normalization_mean
            + self.normalization_scale
            + tuple(value for row in self.weights for value in row)
            + self.biases
            + self.quantile_offsets
            + self.predictive_scales
        )
        if any(not math.isfinite(float(value)) for value in scalars) or any(
            value <= 0
            for value in self.normalization_scale
            + self.quantile_offsets
            + self.predictive_scales
        ):
            raise MassiveValidationInferenceV0Error(
                "checkpoint numerical values differ"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_VALIDATION_CHECKPOINT_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_CHECKPOINT_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationInferenceV0Error("checkpoint source differs")
        _digest("checkpoint receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationInferenceV0Error("checkpoint receipt differs")


def _checkpoint_payload(checkpoint: MassiveValidationCheckpointV0) -> dict[str, object]:
    return {
        key: value
        for key, value in checkpoint.unsigned().items()
        if key != "loaded_source"
    }


def parse_massive_validation_checkpoint_v0(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveValidationCheckpointV0:
    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveValidationInferenceV0Error(
            "checkpoint source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveValidationInferenceV0Error("checkpoint source is not canonical")
    try:
        body = {
            "schema": payload["schema"],
            "setting_id": payload["setting_id"],
            "feature_set_id": payload["feature_set_id"],
            "seed": payload["seed"],
            "feature_names": tuple(payload["feature_names"]),
            "horizon_ids": tuple(payload["horizon_ids"]),
            "normalization_mean": tuple(payload["normalization_mean"]),
            "normalization_scale": tuple(payload["normalization_scale"]),
            "weights": tuple(tuple(row) for row in payload["weights"]),
            "biases": tuple(payload["biases"]),
            "quantile_offsets": tuple(payload["quantile_offsets"]),
            "predictive_scales": tuple(payload["predictive_scales"]),
            "loaded_source": loaded_source,
        }
    except (KeyError, TypeError) as exc:
        raise MassiveValidationInferenceV0Error(
            "checkpoint values are malformed"
        ) from exc
    provisional = MassiveValidationCheckpointV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveValidationCheckpointV0(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    if raw != canonical_json_file_bytes(_checkpoint_payload(result)):
        raise MassiveValidationInferenceV0Error("checkpoint payload differs")
    return result


@dataclass(frozen=True, slots=True)
class MassiveValidationPredictionRowV0:
    security_id: str
    horizon_id: str
    mean: float
    q10: float
    median: float
    q90: float
    scale: float
    probability_positive: float
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or self.horizon_id not in {
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        }:
            raise MassiveValidationInferenceV0Error("prediction identity differs")
        values = (
            self.mean,
            self.q10,
            self.median,
            self.q90,
            self.scale,
            self.probability_positive,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.scale <= 0
            or not 0 <= self.probability_positive <= 1
            or not self.q10 <= self.median <= self.q90
        ):
            raise MassiveValidationInferenceV0Error("prediction distribution differs")
        _digest("prediction receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationInferenceV0Error("prediction receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveValidationInferenceArtifactV0:
    decision_session_date: str
    setting_id: str
    seed: int
    tensor_receipt_sha256: str
    checkpoint_receipt_sha256: str
    inference_spec_receipt_sha256: str
    inference_source_sha256: str
    rows: tuple[MassiveValidationPredictionRowV0, ...]
    prediction_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_VALIDATION_INFERENCE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VALIDATION_INFERENCE_V0_SCHEMA:
            raise MassiveValidationInferenceV0Error("inference schema drifted")
        for name in (
            "tensor_receipt_sha256",
            "checkpoint_receipt_sha256",
            "inference_spec_receipt_sha256",
            "inference_source_sha256",
            "prediction_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.inference_spec_receipt_sha256
            != MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256
            or self.inference_source_sha256
            != MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SHA256
        ):
            raise MassiveValidationInferenceV0Error("inference implementation drifted")
        keys = tuple((row.security_id, row.horizon_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveValidationInferenceV0Error("prediction rows are not canonical")
        for row in self.rows:
            row.validate()
        if self.prediction_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveValidationInferenceV0Error("prediction inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_VALIDATION_INFERENCE_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationInferenceV0Error("inference source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationInferenceV0Error("inference receipt differs")


def _inference_payload(
    artifact: MassiveValidationInferenceArtifactV0,
) -> dict[str, object]:
    return {
        key: value
        for key, value in artifact.unsigned().items()
        if key != "loaded_source"
    }


def materialize_massive_validation_inference_v0(
    *,
    tensor_root: str | Path,
    output_root: str | Path,
    tensor: MassivePIT500DecisionTensorV0,
    checkpoint: MassiveValidationCheckpointV0,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveValidationInferenceArtifactV0:
    validate_massive_pit500_tensor_v0(root=tensor_root, tensor=tensor)
    checkpoint.validate()
    bars_only = checkpoint.feature_set_id == "BARS_V0"
    rows = []
    for index, security_id in enumerate(tensor.security_ids):
        values = tensor.bars_values[index] + (
            () if bars_only else tensor.tape_values[index]
        )
        valid = tensor.bars_valid[index] + (
            () if bars_only else tensor.tape_valid[index]
        )
        normalized = tuple(
            0.0 if not flag else (value - mean) / scale
            for value, flag, mean, scale in zip(
                values,
                valid,
                checkpoint.normalization_mean,
                checkpoint.normalization_scale,
                strict=True,
            )
        )
        for horizon_index, horizon_id in enumerate(checkpoint.horizon_ids):
            mean = checkpoint.biases[horizon_index] + sum(
                weight * value
                for weight, value in zip(
                    checkpoint.weights[horizon_index], normalized, strict=True
                )
            )
            offset = checkpoint.quantile_offsets[horizon_index]
            scale = checkpoint.predictive_scales[horizon_index]
            body = {
                "security_id": security_id,
                "horizon_id": horizon_id,
                "mean": mean,
                "q10": mean - offset,
                "median": mean,
                "q90": mean + offset,
                "scale": scale,
                "probability_positive": 0.5
                * (1.0 + math.erf(mean / (scale * math.sqrt(2.0)))),
            }
            row = MassiveValidationPredictionRowV0(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
            row.validate()
            rows.append(row)
    ordered = tuple(sorted(rows, key=lambda row: (row.security_id, row.horizon_id)))
    relative = f"massive-finalized-v0/decision={tensor.decision_session_date}/inference-{checkpoint.setting_id}-seed{checkpoint.seed}.json"
    placeholder = MassiveValidationInferenceArtifactV0(
        decision_session_date=tensor.decision_session_date,
        setting_id=checkpoint.setting_id,
        seed=checkpoint.seed,
        tensor_receipt_sha256=tensor.receipt_sha256,
        checkpoint_receipt_sha256=checkpoint.receipt_sha256,
        inference_spec_receipt_sha256=MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256,
        inference_source_sha256=MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SHA256,
        rows=ordered,
        prediction_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        loaded_source=tensor.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_inference_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_VALIDATION_INFERENCE_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root, relative_payload_path=relative, verified_at_ms=published_at_ms
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    validate_massive_validation_inference_v0(root=output_root, artifact=result)
    return result


def validate_massive_validation_inference_v0(
    *, root: str | Path, artifact: MassiveValidationInferenceArtifactV0
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveValidationInferenceV0Error("inference source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _inference_payload(artifact)
    ):
        raise MassiveValidationInferenceV0Error("inference bytes differ")


__all__ = [
    "MASSIVE_VALIDATION_CHECKPOINT_V0_DATASET",
    "MASSIVE_VALIDATION_CHECKPOINT_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_INFERENCE_V0_DATASET",
    "MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256",
    "MassiveValidationCheckpointV0",
    "MassiveValidationInferenceArtifactV0",
    "MassiveValidationInferenceV0Error",
    "MassiveValidationPredictionRowV0",
    "materialize_massive_validation_inference_v0",
    "parse_massive_validation_checkpoint_v0",
    "validate_massive_validation_inference_v0",
]
