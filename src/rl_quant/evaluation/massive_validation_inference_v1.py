"""Mask- and staleness-aware inference canaries for finalized validation V1."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_validation_inference_v0 import (
    MassiveValidationPredictionRowV0,
)
from rl_quant.features.massive_pit500_tensor_v1 import (
    MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256,
    MassivePIT500DecisionTensorV1,
    validate_massive_pit500_tensor_v1,
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

MASSIVE_VALIDATION_CHECKPOINT_V1_SCHEMA = "rl-quant.massive-validation-checkpoint-v1"
MASSIVE_VALIDATION_CHECKPOINT_V1_DATASET = "massive-finalized-validation-checkpoint-v1"
MASSIVE_VALIDATION_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_CHECKPOINT_V1_SCHEMA,
        "model": "frozen-linear-distribution-readiness-canary",
        "inputs": "values-plus-validity-masks-plus-source-staleness",
        "horizons": tuple(
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        ),
    }
)
MASSIVE_VALIDATION_INFERENCE_V1_SCHEMA = "rl-quant.massive-validation-inference-v1"
MASSIVE_VALIDATION_INFERENCE_V1_DATASET = "massive-finalized-validation-inference-v1"
MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "tensor_spec": MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256,
        "purpose": "readiness-canary-not-performance-checkpoint",
        "missing": "training-mean-imputation-plus-explicit-validity-input",
        "source_staleness": "explicit-normalized-input",
        "heads": ("mean", "q10", "median", "q90", "scale", "probability-positive"),
    }
)
MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_INFERENCE_V1_SCHEMA,
        "horizons": tuple(
            row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
        ),
        "outputs": ("mean", "q10", "median", "q90", "scale", "probability_positive"),
    }
)


class MassiveValidationInferenceV1Error(ValueError):
    """A V1 checkpoint or inference artifact differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveValidationInferenceV1Error(f"{name} must be a lowercase SHA-256")
    return value


def massive_validation_input_names_v1(setting_id: str) -> tuple[str, ...]:
    if setting_id not in {"MV02", "MV04"}:
        raise MassiveValidationInferenceV1Error("unsupported readiness setting")
    names = tuple(f"bars:value:{name}" for name in MASSIVE_ROLLING_BARS_V0_FIELDS)
    names += tuple(f"bars:valid:{name}" for name in MASSIVE_ROLLING_BARS_V0_FIELDS)
    if setting_id == "MV04":
        names += tuple(f"tape:value:{name}" for name in MASSIVE_ROLLING_TAPE_V0_FIELDS)
        names += tuple(f"tape:valid:{name}" for name in MASSIVE_ROLLING_TAPE_V0_FIELDS)
    return names + ("context:source_staleness_sessions",)


@dataclass(frozen=True, slots=True)
class MassiveValidationCheckpointV1:
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
    schema: str = MASSIVE_VALIDATION_CHECKPOINT_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        expected_set = {
            "MV02": "BARS_MASKS_STALENESS_V1",
            "MV04": "BARS_TAPE_MASKS_STALENESS_V1",
        }
        if (
            self.schema != MASSIVE_VALIDATION_CHECKPOINT_V1_SCHEMA
            or self.setting_id not in expected_set
            or self.feature_set_id != expected_set[self.setting_id]
            or self.feature_names != massive_validation_input_names_v1(self.setting_id)
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise MassiveValidationInferenceV1Error("checkpoint v1 identity differs")
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
            raise MassiveValidationInferenceV1Error("checkpoint v1 shapes differ")
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
            raise MassiveValidationInferenceV1Error("checkpoint v1 values differ")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_VALIDATION_CHECKPOINT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationInferenceV1Error("checkpoint v1 source differs")
        _digest("checkpoint v1 receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationInferenceV1Error("checkpoint v1 receipt differs")


def parse_massive_validation_checkpoint_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveValidationCheckpointV1:
    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
        expected_keys = {
            "schema",
            "setting_id",
            "feature_set_id",
            "seed",
            "feature_names",
            "horizon_ids",
            "normalization_mean",
            "normalization_scale",
            "weights",
            "biases",
            "quantile_offsets",
            "predictive_scales",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise MassiveValidationInferenceV1Error(
                "checkpoint v1 field inventory differs"
            )
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
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MassiveValidationInferenceV1Error(
            "checkpoint v1 bytes are malformed"
        ) from exc
    if raw != canonical_json_file_bytes(payload):
        raise MassiveValidationInferenceV1Error("checkpoint v1 source is not canonical")
    provisional = MassiveValidationCheckpointV1(**body, receipt_sha256="0" * 64)
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    expected_payload = {
        key: value for key, value in result.unsigned().items() if key != "loaded_source"
    }
    if raw != canonical_json_file_bytes(expected_payload):
        raise MassiveValidationInferenceV1Error(
            "checkpoint v1 bytes differ from the parsed checkpoint"
        )
    return result


@dataclass(frozen=True, slots=True)
class MassiveValidationInferenceArtifactV1:
    decision_session_date: str
    setting_id: str
    seed: int
    tensor_receipt_sha256: str
    checkpoint_receipt_sha256: str
    inference_spec_receipt_sha256: str
    inference_source_sha256: str
    input_feature_names: tuple[str, ...]
    rows: tuple[MassiveValidationPredictionRowV0, ...]
    prediction_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_VALIDATION_INFERENCE_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_VALIDATION_INFERENCE_V1_SCHEMA
            or self.input_feature_names
            != massive_validation_input_names_v1(self.setting_id)
        ):
            raise MassiveValidationInferenceV1Error("inference v1 identity differs")
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
            != MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256
            or self.inference_source_sha256
            != MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256
        ):
            raise MassiveValidationInferenceV1Error("inference v1 contract drifted")
        keys = tuple((row.security_id, row.horizon_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveValidationInferenceV1Error("inference v1 rows differ")
        for row in self.rows:
            row.validate()
        if self.prediction_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveValidationInferenceV1Error("inference v1 inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_VALIDATION_INFERENCE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationInferenceV1Error("inference v1 source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationInferenceV1Error("inference v1 receipt differs")


def _inference_payload(
    artifact: MassiveValidationInferenceArtifactV1,
) -> dict[str, object]:
    return {
        key: value
        for key, value in artifact.unsigned().items()
        if key != "loaded_source"
    }


def _raw_inputs(
    *,
    tensor: MassivePIT500DecisionTensorV1,
    checkpoint: MassiveValidationCheckpointV1,
    index: int,
) -> tuple[float, ...]:
    include_tape = checkpoint.setting_id == "MV04"
    values: list[float | None] = [
        value if valid else None
        for value, valid in zip(
            tensor.bars_values[index], tensor.bars_valid[index], strict=True
        )
    ]
    values.extend(float(flag) for flag in tensor.bars_valid[index])
    if include_tape:
        values.extend(
            value if valid else None
            for value, valid in zip(
                tensor.tape_values[index], tensor.tape_valid[index], strict=True
            )
        )
        values.extend(float(flag) for flag in tensor.tape_valid[index])
    values.append(
        tensor.source_staleness_sessions[index]
        if tensor.staleness_valid[index]
        else None
    )
    return tuple(
        mean if value is None else float(value)
        for value, mean in zip(values, checkpoint.normalization_mean, strict=True)
    )


def materialize_massive_validation_inference_v1(
    *,
    tensor_root: str | Path,
    output_root: str | Path,
    tensor: MassivePIT500DecisionTensorV1,
    checkpoint: MassiveValidationCheckpointV1,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveValidationInferenceArtifactV1:
    validate_massive_pit500_tensor_v1(root=tensor_root, tensor=tensor)
    checkpoint.validate()
    rows = []
    for index, security_id in enumerate(tensor.security_ids):
        raw_inputs = _raw_inputs(tensor=tensor, checkpoint=checkpoint, index=index)
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                raw_inputs,
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
    relative = (
        f"massive-finalized-v1/decision={tensor.decision_session_date}/"
        f"inference-v1-{checkpoint.setting_id}-seed{checkpoint.seed}.json"
    )
    placeholder = MassiveValidationInferenceArtifactV1(
        decision_session_date=tensor.decision_session_date,
        setting_id=checkpoint.setting_id,
        seed=checkpoint.seed,
        tensor_receipt_sha256=tensor.receipt_sha256,
        checkpoint_receipt_sha256=checkpoint.receipt_sha256,
        inference_spec_receipt_sha256=MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256,
        inference_source_sha256=MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256,
        input_feature_names=checkpoint.feature_names,
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
        dataset_id=MASSIVE_VALIDATION_INFERENCE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SCHEMA_SHA256,
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
    validate_massive_validation_inference_v1(root=output_root, artifact=result)
    return result


def validate_massive_validation_inference_v1(
    *, root: str | Path, artifact: MassiveValidationInferenceArtifactV1
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveValidationInferenceV1Error(
            "inference v1 source is not JSON"
        ) from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _inference_payload(artifact)
    ):
        raise MassiveValidationInferenceV1Error("inference v1 bytes differ")


__all__ = [
    "MASSIVE_VALIDATION_CHECKPOINT_V1_DATASET",
    "MASSIVE_VALIDATION_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_INFERENCE_V1_DATASET",
    "MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256",
    "MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256",
    "MassiveValidationCheckpointV1",
    "MassiveValidationInferenceArtifactV1",
    "MassiveValidationInferenceV1Error",
    "massive_validation_input_names_v1",
    "materialize_massive_validation_inference_v1",
    "parse_massive_validation_checkpoint_v1",
    "validate_massive_validation_inference_v1",
]
