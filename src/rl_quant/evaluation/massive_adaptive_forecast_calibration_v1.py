"""Training-only calibration for adaptive residual-return forecasts.

The artifact is deliberately fitted from the replayed training forecast and
target archives.  Validation forecasts may consume it, but may never fit or
modify it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np

from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MassiveAdaptiveForecastArchiveV1,
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
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)

MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-calibration-v1"
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_DATASET = (
    "massive-adaptive-forecast-calibration-v1"
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA,
        "payload": "canonical-json-calibration-v1",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "fit_role": "training-only",
        "mean": "residual-error-mean-by-bucket",
        "scale": "residual-rmse-over-predicted-scale-by-bucket",
        "correlation": "complete-case-psd-projected-cross-bucket-error-correlation",
        "validation_refit": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveForecastCalibrationV1Error(ValueError):
    """Training calibration does not replay from its forecast/target roots."""


_CALIBRATION_ISSUER = object()


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _psd_correlation(errors: list[np.ndarray], buckets: int) -> np.ndarray:
    complete = np.asarray(
        [row for row in errors if row.shape == (buckets,) and np.isfinite(row).all()],
        dtype=np.float64,
    )
    if complete.shape[0] < 2:
        return np.eye(buckets, dtype=np.float64)
    observed = np.corrcoef(complete, rowvar=False)
    if observed.shape != (buckets, buckets) or not np.isfinite(observed).all():
        return np.eye(buckets, dtype=np.float64)
    observed = (observed + observed.T) * 0.5
    values, vectors = np.linalg.eigh(observed)
    projected = (vectors * np.maximum(values, 1.0e-8)) @ vectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), 1.0e-12))
    projected = projected / np.outer(diagonal, diagonal)
    np.fill_diagonal(projected, 1.0)
    return projected


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastCalibrationV1:
    bucket_ids: tuple[str, ...]
    mean_bias: tuple[float, ...]
    scale_multiplier: tuple[float, ...]
    horizon_error_correlation: tuple[tuple[float, ...], ...]
    observation_counts: tuple[int, ...]
    training_forecast_archive_receipt_sha256: str
    training_target_archive_receipt_sha256: str
    training_window_plan_receipt_sha256: str
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
    specification_sha256: str = MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA
    _issuer: object = field(repr=False, compare=False, default=None)

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "development_calibration_authorized",
                "runtime_calibration_replayed",
                "loaded_source",
                "_issuer",
            }
        }

    def validate(self) -> None:
        buckets = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
        correlation = np.asarray(self.horizon_error_correlation, dtype=np.float64)
        if (
            self.schema != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA
            or self._issuer is not _CALIBRATION_ISSUER
            or self.bucket_ids
            != tuple(bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
            or len(self.mean_bias) != buckets
            or len(self.scale_multiplier) != buckets
            or len(self.observation_counts) != buckets
            or correlation.shape != (buckets, buckets)
            or not np.isfinite(correlation).all()
            or not np.allclose(correlation, correlation.T, atol=1.0e-12, rtol=0.0)
            or float(np.linalg.eigvalsh(correlation).min()) < -1.0e-8
            or any(not math.isfinite(value) for value in self.mean_bias)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.scale_multiplier
            )
            or any(
                isinstance(value, bool) or value <= 0
                for value in self.observation_counts
            )
            or not isinstance(self.source_data_qualified, bool)
            or not isinstance(self.runtime_calibration_replayed, bool)
            or self.development_calibration_authorized
            != (
                self.runtime_calibration_replayed
                and self.source_data_qualified
                and self.loaded_source is not None
            )
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveForecastCalibrationV1Error(
                "adaptive forecast calibration identity or numerical state differs"
            )
        for name in (
            "training_forecast_archive_receipt_sha256",
            "training_target_archive_receipt_sha256",
            "training_window_plan_receipt_sha256",
            "fit_population_receipt_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())
        if self.loaded_source is not None:
            self.loaded_source.validate()
            if (
                self.loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_DATASET
                or self.loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SOURCE_SCHEMA_SHA256
                or self.loaded_source.receipt.entitlement_receipt_sha256
                != self.fit_population_receipt_sha256
            ):
                raise MassiveAdaptiveForecastCalibrationV1Error(
                    "adaptive calibration source transaction differs"
                )


def build_massive_adaptive_forecast_calibration_v1(
    *,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveForecastCalibrationV1:
    """Fit one immutable calibration using training-role observations only."""

    training_forecasts.validate()
    training_targets.validate()
    training_window_plan.validate()
    if (
        training_forecasts.runtime_rows is None
        or not training_forecasts.runtime_forecasts_replayed
        or training_targets.runtime_source_targets is None
        or not training_targets.runtime_roots_replayed
        or training_forecasts.origin_session_dates
        != training_targets.decision_session_dates
        or training_window_plan.split_role != "training"
        or training_forecasts.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or tuple(row.origin_session_date for row in training_window_plan.rows)
        != training_forecasts.origin_session_dates
    ):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "training forecast and target runtimes are not jointly replayed"
        )
    forecast_by_date = {
        row.decision_session_date: row for row in training_forecasts.runtime_rows
    }
    buckets = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    errors_by_bucket: list[list[float]] = [[] for _ in range(buckets)]
    scales_by_bucket: list[list[float]] = [[] for _ in range(buckets)]
    complete_errors: list[np.ndarray] = []
    population: list[tuple[object, ...]] = []
    for source_target in training_targets.runtime_source_targets:
        forecast = forecast_by_date[source_target.decision_session_date]
        target = source_target.targets
        target_by_security = {row.security_id: row for row in target.rows}
        if not set(target.security_ids) <= set(forecast.security_ids):
            raise MassiveAdaptiveForecastCalibrationV1Error(
                "calibration target support exceeds forecast support"
            )
        forecast_index = {
            security_id: index
            for index, security_id in enumerate(forecast.security_ids)
        }
        for security_id in target.security_ids:
            row = target_by_security[security_id]
            index = forecast_index[security_id]
            vector = np.full(buckets, np.nan, dtype=np.float64)
            for bucket in range(buckets):
                if row.training_valid_by_bucket[bucket] and bool(forecast.valid[index]):
                    error = float(row.residual_bucket_returns[bucket]) - float(
                        forecast.residual_mean[index, bucket]
                    )
                    scale = float(forecast.residual_scale[index, bucket])
                    errors_by_bucket[bucket].append(error)
                    scales_by_bucket[bucket].append(scale)
                    vector[bucket] = error
                    population.append(
                        (
                            source_target.decision_session_date,
                            security_id,
                            bucket,
                            row.receipt_sha256,
                            forecast.receipt_sha256,
                        )
                    )
            complete_errors.append(vector)
    if any(not values for values in errors_by_bucket):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "every adaptive bucket requires a training calibration population"
        )
    bias = tuple(float(np.mean(values)) for values in errors_by_bucket)
    multipliers = tuple(
        max(
            1.0e-6,
            float(np.sqrt(np.mean(np.square(errors))))
            / max(float(np.mean(scales)), 1.0e-8),
        )
        for errors, scales in zip(errors_by_bucket, scales_by_bucket, strict=True)
    )
    correlation = _psd_correlation(complete_errors, buckets)
    body = {
        "schema": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA,
        "bucket_ids": tuple(
            bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
        ),
        "mean_bias": bias,
        "scale_multiplier": multipliers,
        "horizon_error_correlation": tuple(
            tuple(float(value) for value in row) for row in correlation
        ),
        "observation_counts": tuple(len(values) for values in errors_by_bucket),
        "training_forecast_archive_receipt_sha256": (
            training_forecasts.semantic_receipt_sha256
        ),
        "training_target_archive_receipt_sha256": (
            training_targets.semantic_receipt_sha256
        ),
        "training_window_plan_receipt_sha256": (
            training_window_plan.semantic_receipt_sha256
        ),
        "fit_population_receipt_sha256": semantic_sha256(tuple(population)),
        "source_data_qualified": bool(
            training_forecasts.committed_source_data_qualified
            and training_targets.committed_source_data_qualified
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveForecastCalibrationV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_calibration_replayed=True,
        development_calibration_authorized=False,
        _issuer=_CALIBRATION_ISSUER,
    )
    result.validate()
    return result


def _artifact_id(value: str) -> str:
    if not value or any(
        not (character.isalnum() or character in "-_") for character in value
    ):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "adaptive calibration artifact ID is not path safe"
        )
    return value


def _payload(value: MassiveAdaptiveForecastCalibrationV1) -> dict[str, object]:
    return {
        **value.semantic_unsigned(),
        "semantic_receipt_sha256": value.semantic_receipt_sha256,
    }


def parse_massive_adaptive_forecast_calibration_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveForecastCalibrationV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "adaptive calibration payload is not canonical JSON"
        )
    for name in (
        "bucket_ids",
        "mean_bias",
        "scale_multiplier",
        "observation_counts",
    ):
        payload[name] = tuple(payload[name])
    payload["horizon_error_correlation"] = tuple(
        tuple(row) for row in payload["horizon_error_correlation"]
    )
    result = MassiveAdaptiveForecastCalibrationV1(  # type: ignore[arg-type]
        **payload,
        runtime_calibration_replayed=False,
        development_calibration_authorized=False,
        loaded_source=loaded_source,
        _issuer=_CALIBRATION_ISSUER,
    )
    result.validate()
    return result


def authorize_massive_adaptive_forecast_calibration_v1(
    *,
    root: str | Path,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveForecastCalibrationV1:
    if calibration.loaded_source is None:
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "adaptive calibration is not attached to a committed source object"
        )
    parsed = parse_massive_adaptive_forecast_calibration_v1(
        root=root, loaded_source=calibration.loaded_source
    )
    rebuilt = build_massive_adaptive_forecast_calibration_v1(
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )
    if (
        parsed.semantic_unsigned() != rebuilt.semantic_unsigned()
        or parsed.semantic_receipt_sha256 != rebuilt.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveForecastCalibrationV1Error(
            "adaptive calibration does not replay from training roots"
        )
    result = MassiveAdaptiveForecastCalibrationV1(
        **rebuilt.semantic_unsigned(),  # type: ignore[arg-type]
        semantic_receipt_sha256=rebuilt.semantic_receipt_sha256,
        runtime_calibration_replayed=True,
        development_calibration_authorized=rebuilt.source_data_qualified,
        loaded_source=parsed.loaded_source,
        _issuer=_CALIBRATION_ISSUER,
    )
    result.validate()
    return result


def materialize_massive_adaptive_forecast_calibration_v1(
    *,
    root: str | Path,
    artifact_id: str,
    training_forecasts: MassiveAdaptiveForecastArchiveV1,
    training_targets: MassiveAdaptiveTargetArchiveV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    committed_at_ms: int,
) -> MassiveAdaptiveForecastCalibrationV1:
    identifier = _artifact_id(artifact_id)
    built = build_massive_adaptive_forecast_calibration_v1(
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )
    relative = f"massive-adaptive/forecast-calibration-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(built))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=built.fit_population_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FORECAST-CALIBRATION-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_forecast_calibration_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_forecast_calibration_v1(
        root=root,
        calibration=generic,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_SCHEMA",
    "MassiveAdaptiveForecastCalibrationV1",
    "MassiveAdaptiveForecastCalibrationV1Error",
    "authorize_massive_adaptive_forecast_calibration_v1",
    "build_massive_adaptive_forecast_calibration_v1",
    "materialize_massive_adaptive_forecast_calibration_v1",
    "parse_massive_adaptive_forecast_calibration_v1",
]
