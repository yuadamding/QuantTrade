"""Typed, content-verified confidence calibration for M03R v5.

The actor emits an unconstrained scalar logit.  This module is the only
authorized route from that logit to the confidence used by the new-active-risk
budget.  Calibration is fit on inner-validation data only and its complete
payload is hashed; a digest-shaped placeholder is not sufficient.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import date

import torch

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    resolve_m03r_v5_setting,
)

M03R_CONFIDENCE_CALIBRATION_SCHEMA = M03R_DESIGN.model.confidence_calibration_schema
M03R_CONFIDENCE_CALIBRATION_METHOD = M03R_DESIGN.model.confidence_calibration_method
M03R_CONFIDENCE_TARGET_DEFINITION = M03R_DESIGN.model.confidence_target_definition
M03R_UNCALIBRATED_CONFIDENCE_SCORE_DEFINITION = "m03r-confidence-head-logit-v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M03RConfidenceCalibrationError(ValueError):
    """A calibrator is unbound, outer-contaminated, or semantically invalid."""


@dataclass(frozen=True, slots=True)
class M03RConfidenceCalibrationManifest:
    """One inner-validation temperature calibrator and its reliability evidence."""

    schema: str
    protocol_generation: str
    design_id: str
    setting_id: str
    seed: int
    checkpoint_sha256: str
    model_state_sha256: str
    source_score_array_sha256: str
    source_target_array_sha256: str
    calibration_method: str
    target_definition: str
    uncalibrated_score_definition: str
    fit_data_role: str
    fit_fold_ids: tuple[str, ...]
    fit_start_trading_session: str
    fit_end_trading_session: str
    outer_data_used: bool
    temperature: float
    intercept: float
    fit_observation_count: int
    brier_score: float
    expected_calibration_error: float
    observed_target_rate: float
    manifest_sha256: str


def m03r_confidence_calibration_payload(
    manifest: M03RConfidenceCalibrationManifest,
) -> dict[str, object]:
    """Return the canonical semantic payload, excluding its claimed digest."""

    return {
        "schema": manifest.schema,
        "protocol_generation": manifest.protocol_generation,
        "design_id": manifest.design_id,
        "setting_id": manifest.setting_id,
        "seed": manifest.seed,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "model_state_sha256": manifest.model_state_sha256,
        "source_score_array_sha256": manifest.source_score_array_sha256,
        "source_target_array_sha256": manifest.source_target_array_sha256,
        "calibration_method": manifest.calibration_method,
        "target_definition": manifest.target_definition,
        "uncalibrated_score_definition": manifest.uncalibrated_score_definition,
        "fit_data_role": manifest.fit_data_role,
        "fit_fold_ids": list(manifest.fit_fold_ids),
        "fit_start_trading_session": manifest.fit_start_trading_session,
        "fit_end_trading_session": manifest.fit_end_trading_session,
        "outer_data_used": manifest.outer_data_used,
        "temperature_float64_hex": float(manifest.temperature).hex(),
        "intercept_float64_hex": float(manifest.intercept).hex(),
        "fit_observation_count": manifest.fit_observation_count,
        "brier_score_float64_hex": float(manifest.brier_score).hex(),
        "expected_calibration_error_float64_hex": float(
            manifest.expected_calibration_error
        ).hex(),
        "observed_target_rate_float64_hex": float(manifest.observed_target_rate).hex(),
    }


def compute_m03r_confidence_calibration_sha256(
    manifest: M03RConfidenceCalibrationManifest,
) -> str:
    payload = m03r_confidence_calibration_payload(manifest)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_date_range(manifest: M03RConfidenceCalibrationManifest) -> None:
    try:
        start = date.fromisoformat(manifest.fit_start_trading_session)
        end = date.fromisoformat(manifest.fit_end_trading_session)
    except ValueError as error:
        raise M03RConfidenceCalibrationError(
            "calibration fit dates must use ISO YYYY-MM-DD"
        ) from error
    if start > end:
        raise M03RConfidenceCalibrationError(
            "calibration fit start cannot follow fit end"
        )


def validate_m03r_confidence_calibration_manifest(
    manifest: M03RConfidenceCalibrationManifest,
    *,
    expected_manifest_sha256: str,
    expected_setting_id: str,
    expected_seed: int,
    expected_checkpoint_sha256: str,
    expected_model_state_sha256: str,
    expected_source_score_array_sha256: str,
    expected_source_target_array_sha256: str,
) -> None:
    """Fail closed unless identity, lineage, evidence, and content all agree."""

    setting = resolve_m03r_v5_setting(expected_setting_id)
    if not setting.use_confidence_scaled_active_risk_budget:
        raise M03RConfidenceCalibrationError(
            "the selected setting does not use a confidence-scaled active-risk budget"
        )
    exact = {
        "schema": M03R_CONFIDENCE_CALIBRATION_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
        "setting_id": expected_setting_id,
        "calibration_method": M03R_CONFIDENCE_CALIBRATION_METHOD,
        "target_definition": M03R_CONFIDENCE_TARGET_DEFINITION,
        "uncalibrated_score_definition": (
            M03R_UNCALIBRATED_CONFIDENCE_SCORE_DEFINITION
        ),
        "fit_data_role": "inner-validation-only",
    }
    for field, expected in exact.items():
        if getattr(manifest, field) != expected:
            raise M03RConfidenceCalibrationError(
                f"confidence calibration {field} must equal {expected!r}"
            )
    if (
        isinstance(expected_seed, bool)
        or not isinstance(expected_seed, int)
        or expected_seed < 0
        or isinstance(manifest.seed, bool)
        or not isinstance(manifest.seed, int)
        or manifest.seed < 0
        or manifest.seed != expected_seed
    ):
        raise M03RConfidenceCalibrationError(
            "confidence calibration seed does not match the checkpoint binding"
        )
    identity_digests = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "model_state_sha256": expected_model_state_sha256,
        "source_score_array_sha256": expected_source_score_array_sha256,
        "source_target_array_sha256": expected_source_target_array_sha256,
    }
    for field, expected in identity_digests.items():
        if not _DIGEST.fullmatch(expected):
            raise M03RConfidenceCalibrationError(
                f"expected {field} must be 64 lowercase hex characters"
            )
        if getattr(manifest, field) != expected:
            raise M03RConfidenceCalibrationError(
                f"confidence calibration {field} does not match the checkpoint binding"
            )
    if manifest.outer_data_used:
        raise M03RConfidenceCalibrationError(
            "outer data cannot fit or select the confidence calibrator"
        )
    if (
        not manifest.fit_fold_ids
        or any(not isinstance(fold, str) or not fold for fold in manifest.fit_fold_ids)
        or len(set(manifest.fit_fold_ids)) != len(manifest.fit_fold_ids)
    ):
        raise M03RConfidenceCalibrationError(
            "fit_fold_ids must be unique non-empty inner-validation fold IDs"
        )
    _validate_date_range(manifest)
    if (
        isinstance(manifest.fit_observation_count, bool)
        or not isinstance(manifest.fit_observation_count, int)
        or manifest.fit_observation_count <= 0
    ):
        raise M03RConfidenceCalibrationError(
            "fit_observation_count must be a positive integer"
        )
    numeric = (
        manifest.temperature,
        manifest.intercept,
        manifest.brier_score,
        manifest.expected_calibration_error,
        manifest.observed_target_rate,
    )
    if any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in numeric
    ):
        raise M03RConfidenceCalibrationError(
            "confidence calibration parameters and diagnostics must be finite"
        )
    if float(manifest.temperature) <= 0:
        raise M03RConfidenceCalibrationError("temperature must be strictly positive")
    for field in (
        "brier_score",
        "expected_calibration_error",
        "observed_target_rate",
    ):
        if not 0.0 <= float(getattr(manifest, field)) <= 1.0:
            raise M03RConfidenceCalibrationError(f"{field} must lie in [0,1]")
    if not _DIGEST.fullmatch(expected_manifest_sha256):
        raise M03RConfidenceCalibrationError(
            "expected confidence calibration SHA-256 must be 64 lowercase hex characters"
        )
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise M03RConfidenceCalibrationError(
            "confidence calibration claimed digest does not match launch binding"
        )
    actual = compute_m03r_confidence_calibration_sha256(manifest)
    if actual != expected_manifest_sha256:
        raise M03RConfidenceCalibrationError(
            "confidence calibration payload does not match its content digest"
        )


def bind_m03r_confidence_calibration(
    *,
    setting_id: str,
    seed: int,
    checkpoint_sha256: str,
    model_state_sha256: str,
    source_score_array_sha256: str,
    source_target_array_sha256: str,
    fit_fold_ids: tuple[str, ...],
    fit_start_trading_session: str,
    fit_end_trading_session: str,
    temperature: float,
    intercept: float,
    fit_observation_count: int,
    brier_score: float,
    expected_calibration_error: float,
    observed_target_rate: float,
) -> M03RConfidenceCalibrationManifest:
    """Build, hash, and validate one complete inner-validation calibrator."""

    unbound = M03RConfidenceCalibrationManifest(
        schema=M03R_CONFIDENCE_CALIBRATION_SCHEMA,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=setting_id,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        source_score_array_sha256=source_score_array_sha256,
        source_target_array_sha256=source_target_array_sha256,
        calibration_method=M03R_CONFIDENCE_CALIBRATION_METHOD,
        target_definition=M03R_CONFIDENCE_TARGET_DEFINITION,
        uncalibrated_score_definition=M03R_UNCALIBRATED_CONFIDENCE_SCORE_DEFINITION,
        fit_data_role="inner-validation-only",
        fit_fold_ids=fit_fold_ids,
        fit_start_trading_session=fit_start_trading_session,
        fit_end_trading_session=fit_end_trading_session,
        outer_data_used=False,
        temperature=temperature,
        intercept=intercept,
        fit_observation_count=fit_observation_count,
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        observed_target_rate=observed_target_rate,
        manifest_sha256="",
    )
    bound = replace(
        unbound,
        manifest_sha256=compute_m03r_confidence_calibration_sha256(unbound),
    )
    validate_m03r_confidence_calibration_manifest(
        bound,
        expected_manifest_sha256=bound.manifest_sha256,
        expected_setting_id=setting_id,
        expected_seed=seed,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_model_state_sha256=model_state_sha256,
        expected_source_score_array_sha256=source_score_array_sha256,
        expected_source_target_array_sha256=source_target_array_sha256,
    )
    return bound


def apply_m03r_confidence_calibration(
    raw_logit: torch.Tensor,
    manifest: M03RConfidenceCalibrationManifest,
    *,
    expected_manifest_sha256: str,
    expected_setting_id: str,
    expected_seed: int,
    expected_checkpoint_sha256: str,
    expected_model_state_sha256: str,
    expected_source_score_array_sha256: str,
    expected_source_target_array_sha256: str,
) -> torch.Tensor:
    """Apply the bound differentiable temperature transform to actor logits."""

    validate_m03r_confidence_calibration_manifest(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_setting_id=expected_setting_id,
        expected_seed=expected_seed,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_state_sha256=expected_model_state_sha256,
        expected_source_score_array_sha256=expected_source_score_array_sha256,
        expected_source_target_array_sha256=expected_source_target_array_sha256,
    )
    if (
        not isinstance(raw_logit, torch.Tensor)
        or not raw_logit.is_floating_point()
        or not bool(torch.isfinite(raw_logit).all())
    ):
        raise M03RConfidenceCalibrationError(
            "raw confidence logits must be finite floating-point tensors"
        )
    return torch.sigmoid(
        raw_logit / float(manifest.temperature) + float(manifest.intercept)
    )


__all__ = [
    "M03R_CONFIDENCE_CALIBRATION_METHOD",
    "M03R_CONFIDENCE_CALIBRATION_SCHEMA",
    "M03R_CONFIDENCE_TARGET_DEFINITION",
    "M03R_UNCALIBRATED_CONFIDENCE_SCORE_DEFINITION",
    "M03RConfidenceCalibrationError",
    "M03RConfidenceCalibrationManifest",
    "apply_m03r_confidence_calibration",
    "bind_m03r_confidence_calibration",
    "compute_m03r_confidence_calibration_sha256",
    "m03r_confidence_calibration_payload",
    "validate_m03r_confidence_calibration_manifest",
]
