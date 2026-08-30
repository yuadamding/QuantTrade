"""Fold- and checkpoint-bound calibration for adaptive forecasts.

V1 established the training-only numerical calibration.  V2 retains those
numerics and additionally makes the exact fold, checkpoint, model state,
training window, and fit cutoff part of the immutable artifact identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MassiveAdaptiveForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    build_massive_adaptive_forecast_calibration_v1,
)
from rl_quant.features.massive_adaptive_target_archive_v1 import (
    MassiveAdaptiveTargetArchiveV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)

MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-calibration-v2"
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_DATASET = (
    "massive-adaptive-forecast-calibration-v2"
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA,
        "payload": "canonical-json-calibration-v2",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256 = semantic_sha256(
    {
        "numerics": "training-only-v1",
        "fold_bound": True,
        "checkpoint_bound": True,
        "model_state_bound": True,
        "fit_cutoff_bound": True,
        "validation_refit": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveForecastCalibrationV2Error(ValueError):
    """Calibration provenance or replay differs from its training roots."""


_ISSUER = object()


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastCalibrationV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _checkpoint_source_receipt(checkpoint: object) -> str:
    loaded = getattr(checkpoint, "loaded_source", None)
    receipt = getattr(loaded, "receipt_sha256", None)
    return _digest("checkpoint source receipt", receipt)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastCalibrationV2:
    bucket_ids: tuple[str, ...]
    mean_bias: tuple[float, ...]
    scale_multiplier: tuple[float, ...]
    horizon_error_correlation: tuple[tuple[float, ...], ...]
    observation_counts: tuple[int, ...]
    fold_index: int
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_forecast_archive_receipt_sha256: str
    training_target_archive_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    calibration_fit_stop_session_date: str
    fit_population_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_calibration_replayed: bool
    development_calibration_authorized: bool
    loaded_source: LoadedMassiveSourceObject | None = None
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA
    _issuer: object = field(repr=False, compare=False, default=None)

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "runtime_calibration_replayed",
                "development_calibration_authorized",
                "loaded_source",
                "_issuer",
            }
        }

    def validate(self) -> None:
        count = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
        correlation = np.asarray(self.horizon_error_correlation, dtype=np.float64)
        runtime_present = self.runtime_calibration_replayed
        if (
            self.schema != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA
            or self._issuer is not _ISSUER
            or self.bucket_ids
            != tuple(bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or not self.calibration_fit_stop_session_date
            or len(self.mean_bias) != count
            or len(self.scale_multiplier) != count
            or len(self.observation_counts) != count
            or correlation.shape != (count, count)
            or not np.isfinite(correlation).all()
            or not np.allclose(correlation, correlation.T, atol=1.0e-12, rtol=0.0)
            or float(np.linalg.eigvalsh(correlation).min()) < -1.0e-8
            or any(not math.isfinite(value) for value in self.mean_bias)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.scale_multiplier
            )
            or any(isinstance(value, bool) or value <= 0 for value in self.observation_counts)
            or not isinstance(self.source_data_qualified, bool)
            or self.development_calibration_authorized
            != (runtime_present and self.source_data_qualified and self.loaded_source is not None)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveForecastCalibrationV2Error(
                "adaptive calibration v2 identity or numerical state differs"
            )
        for name in (
            "checkpoint_receipt_sha256",
            "checkpoint_source_receipt_sha256",
            "model_state_receipt_sha256",
            "training_forecast_archive_receipt_sha256",
            "training_target_archive_receipt_sha256",
            "training_window_plan_receipt_sha256",
            "fit_population_receipt_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.loaded_source is not None:
            self.loaded_source.validate()
            if (
                self.loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_DATASET
                or self.loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SOURCE_SCHEMA_SHA256
                or self.loaded_source.receipt.entitlement_receipt_sha256
                != self.fit_population_receipt_sha256
            ):
                raise MassiveAdaptiveForecastCalibrationV2Error(
                    "adaptive calibration v2 source transaction differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_forecast_calibration_v2(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveForecastCalibrationV2:
    """Fit V1 numerics and bind them to one exact fold/checkpoint lineage."""

    checkpoint.validate()
    training_window_plan.validate()
    if (
        training_window_plan.split_role != "training"
        or checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or checkpoint.semantic_receipt_sha256
        != training_forecasts.checkpoint_receipt_sha256
        or checkpoint.model_state_receipt_sha256
        != training_forecasts.model_state_receipt_sha256
        or checkpoint.window_plan_receipt_sha256
        != training_forecasts.window_plan_receipt_sha256
    ):
        raise MassiveAdaptiveForecastCalibrationV2Error(
            "calibration checkpoint, forecast, or training fold differs"
        )
    base = build_massive_adaptive_forecast_calibration_v1(
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )
    stop = max(row.origin_session_date for row in training_window_plan.rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA,
        "bucket_ids": base.bucket_ids,
        "mean_bias": base.mean_bias,
        "scale_multiplier": base.scale_multiplier,
        "horizon_error_correlation": base.horizon_error_correlation,
        "observation_counts": base.observation_counts,
        "fold_index": training_window_plan.fold_index,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": _checkpoint_source_receipt(checkpoint),
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "training_forecast_archive_receipt_sha256": (
            training_forecasts.semantic_receipt_sha256
        ),
        "training_target_archive_receipt_sha256": (
            training_targets.semantic_receipt_sha256
        ),
        "training_window_plan_receipt_sha256": (
            training_window_plan.semantic_receipt_sha256
        ),
        "calibration_fit_stop_session_date": stop,
        "fit_population_receipt_sha256": base.fit_population_receipt_sha256,
        "source_data_qualified": bool(
            base.source_data_qualified
            and checkpoint.development_training_authorized
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256,
    }
    result = MassiveAdaptiveForecastCalibrationV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_calibration_replayed=True,
        development_calibration_authorized=False,
        _issuer=_ISSUER,
    )
    result.validate()
    return result


def _payload(value: MassiveAdaptiveForecastCalibrationV2) -> dict[str, object]:
    return {**value.semantic_unsigned(), "semantic_receipt_sha256": value.semantic_receipt_sha256}


def parse_massive_adaptive_forecast_calibration_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveForecastCalibrationV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveForecastCalibrationV2Error(
            "adaptive calibration v2 payload is not canonical JSON"
        )
    for name in ("bucket_ids", "mean_bias", "scale_multiplier", "observation_counts"):
        payload[name] = tuple(payload[name])
    payload["horizon_error_correlation"] = tuple(
        tuple(row) for row in payload["horizon_error_correlation"]
    )
    result = MassiveAdaptiveForecastCalibrationV2(  # type: ignore[arg-type]
        **payload,
        runtime_calibration_replayed=False,
        development_calibration_authorized=False,
        loaded_source=loaded_source,
        _issuer=_ISSUER,
    )
    result.validate()
    return result


def authorize_massive_adaptive_forecast_calibration_v2(
    *,
    root: str | Path,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveForecastCalibrationV2:
    if calibration.loaded_source is None:
        raise MassiveAdaptiveForecastCalibrationV2Error(
            "adaptive calibration v2 is not attached to a committed source"
        )
    parsed = parse_massive_adaptive_forecast_calibration_v2(
        root=root, loaded_source=calibration.loaded_source
    )
    rebuilt = build_massive_adaptive_forecast_calibration_v2(
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )
    if (
        parsed.semantic_unsigned() != rebuilt.semantic_unsigned()
        or parsed.semantic_receipt_sha256 != rebuilt.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveForecastCalibrationV2Error(
            "adaptive calibration v2 does not replay from training roots"
        )
    result = MassiveAdaptiveForecastCalibrationV2(
        **rebuilt.semantic_unsigned(),  # type: ignore[arg-type]
        semantic_receipt_sha256=rebuilt.semantic_receipt_sha256,
        runtime_calibration_replayed=True,
        development_calibration_authorized=rebuilt.source_data_qualified,
        loaded_source=parsed.loaded_source,
        _issuer=_ISSUER,
    )
    result.validate()
    return result


def materialize_massive_adaptive_forecast_calibration_v2(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    committed_at_ms: int,
) -> MassiveAdaptiveForecastCalibrationV2:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveForecastCalibrationV2Error(
            "adaptive calibration v2 artifact ID is not path safe"
        )
    built = build_massive_adaptive_forecast_calibration_v2(
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )
    relative = f"massive-adaptive/forecast-calibration-v2/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(built))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=built.fit_population_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FORECAST-CALIBRATION-V2-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return authorize_massive_adaptive_forecast_calibration_v2(
        root=root,
        calibration=parse_massive_adaptive_forecast_calibration_v2(
            root=root, loaded_source=loaded
        ),
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )


__all__ = [
    "MassiveAdaptiveForecastCalibrationV2",
    "MassiveAdaptiveForecastCalibrationV2Error",
    "authorize_massive_adaptive_forecast_calibration_v2",
    "build_massive_adaptive_forecast_calibration_v2",
    "materialize_massive_adaptive_forecast_calibration_v2",
    "parse_massive_adaptive_forecast_calibration_v2",
]
