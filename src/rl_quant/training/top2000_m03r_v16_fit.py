"""Immutable fit trajectories and adequacy classification for M03R-v16."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_numerical import (
    M03RV16NumericalTrainingError,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
)

M03R_V16_EPOCH_FIT_SCHEMA = "rl-quant.top2000-dev.m03r-v16-epoch-fit-v2"
M03R_V16_TRAINING_ADEQUACY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-adequacy-v3"
)
M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-numerical-training-failure-v1"
)
M03RV16TrainingAdequacyStatus = Literal[
    "adequate",
    "still-improving",
    "collapsed-output",
    "overdispersed-output",
    "optimizer-clipping-dominated",
]


class M03RV16FitError(ValueError):
    """The V16 fit trajectory or adequacy classification drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16NumericalTrainingFailure:
    package_plan_sha256: str
    authorization_receipt_sha256: str
    training_activation_receipt_sha256: str
    worker_plan_sha256: str
    source_tree_root_sha256: str
    rendered_manifest_sha256: str
    pod_template_sha256: str
    launch_authority_receipt_sha256: str
    admitted_job_authority_receipt_sha256: str
    pod_runtime_attestation_receipt_sha256: str
    job_uid: str
    pod_uid: str
    setting_index: int
    setting_id: str
    fold_index: int
    update_index: int
    failure_phase: str
    error_type: str
    error: str
    model_state_sha256: str | None
    optimizer_state_sha256: str | None
    completed_fold_terminal_file_sha256: tuple[str, ...] = ()
    status: Literal["numerically-invalid"] = "numerically-invalid"
    qualification_tail_accessed: bool = False
    outer_qualification_access_started: bool = False
    outer_2026_accessed: bool = False
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA

    def validate(self) -> None:
        for name in (
            "package_plan_sha256",
            "authorization_receipt_sha256",
            "training_activation_receipt_sha256",
            "worker_plan_sha256",
            "source_tree_root_sha256",
            "rendered_manifest_sha256",
            "pod_template_sha256",
            "launch_authority_receipt_sha256",
            "admitted_job_authority_receipt_sha256",
            "pod_runtime_attestation_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for value in self.completed_fold_terminal_file_sha256:
            _digest("completed_fold_terminal_file_sha256", value)
        for name in ("model_state_sha256", "optimizer_state_sha256"):
            value = getattr(self, name)
            if value is not None:
                _digest(name, value)
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(
                -1, M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
            )
            or self.update_index < -1
            or not self.setting_id
            or not self.failure_phase
            or not self.error_type
            or not self.error
            or not self.job_uid
            or not self.pod_uid
            or len(self.completed_fold_terminal_file_sha256)
            > M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
            or self.status != "numerically-invalid"
            or self.qualification_tail_accessed
            or self.outer_qualification_access_started
            or self.outer_2026_accessed
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA
        ):
            raise M03RV16FitError("V16 numerical training failure drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16FitError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class M03RV16TrainingAdequacy:
    setting_index: int
    fold_index: int
    epoch_fit_receipt_sha256: tuple[str, ...]
    final_prediction_to_target_std_ratio: float
    recent_rank_ic_slope: float
    recent_robust_loss_relative_improvement: float
    recent_encoder_clip_fraction: float
    recent_selection_head_clip_fraction: float
    status: M03RV16TrainingAdequacyStatus
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_TRAINING_ADEQUACY_SCHEMA

    def _expected_status(self) -> M03RV16TrainingAdequacyStatus:
        spec = M03R_V16_PREDICTIVE_SPEC
        values = (
            self.final_prediction_to_target_std_ratio,
            self.recent_rank_ic_slope,
            self.recent_robust_loss_relative_improvement,
            self.recent_encoder_clip_fraction,
            self.recent_selection_head_clip_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise M03RV16NumericalTrainingError(
                "V16 nonfinite fit evidence requires a numerical-failure terminal"
            )
        still_improving = (
            self.recent_rank_ic_slope > spec.adequacy_rank_ic_slope_threshold
            or self.recent_robust_loss_relative_improvement
            > spec.adequacy_recent_loss_relative_improvement_threshold
        )
        collapsed = (
            self.final_prediction_to_target_std_ratio
            < spec.adequacy_minimum_prediction_to_target_std_ratio
        )
        overdispersed = (
            self.final_prediction_to_target_std_ratio
            > spec.adequacy_maximum_prediction_to_target_std_ratio
        )
        pervasive_clipping = max(
            self.recent_encoder_clip_fraction,
            self.recent_selection_head_clip_fraction,
        ) > spec.adequacy_maximum_recent_clip_fraction
        if collapsed:
            return "collapsed-output"
        if overdispersed:
            return "overdispersed-output"
        if pervasive_clipping:
            return "optimizer-clipping-dominated"
        if still_improving:
            return "still-improving"
        return "adequate"

    def validate(self) -> None:
        finite = (
            self.final_prediction_to_target_std_ratio,
            self.recent_rank_ic_slope,
            self.recent_robust_loss_relative_improvement,
            self.recent_encoder_clip_fraction,
            self.recent_selection_head_clip_fraction,
        )
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(
                M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
            )
            or len(self.epoch_fit_receipt_sha256)
            != M03R_V16_PREDICTIVE_SPEC.score_training_epochs
            or not all(math.isfinite(value) for value in finite)
            or (
                math.isfinite(self.final_prediction_to_target_std_ratio)
                and self.final_prediction_to_target_std_ratio < 0.0
            )
            or (
                math.isfinite(self.recent_encoder_clip_fraction)
                and not 0.0 <= self.recent_encoder_clip_fraction <= 1.0
            )
            or (
                math.isfinite(self.recent_selection_head_clip_fraction)
                and not 0.0 <= self.recent_selection_head_clip_fraction <= 1.0
            )
            or self.status != self._expected_status()
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_TRAINING_ADEQUACY_SCHEMA
        ):
            raise M03RV16FitError("V16 training adequacy drifted")
        for value in self.epoch_fit_receipt_sha256:
            _digest("epoch_fit_receipt_sha256", value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


def build_m03r_v16_epoch_fit_payload(
    validation: M03RV16InnerValidationReceipt,
    update_rows: Sequence[Sequence[dict[str, Any]]],
    *,
    package_plan_sha256: str,
    worker_plan_sha256: str,
    training_activation_receipt_sha256: str,
    panel_schedule_sha256: str,
    structural_slab_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the compact but complete immutable evidence for one epoch."""

    validation.validate()
    for name, value in (
        ("package_plan_sha256", package_plan_sha256),
        ("worker_plan_sha256", worker_plan_sha256),
        ("training_activation_receipt_sha256", training_activation_receipt_sha256),
        ("panel_schedule_sha256", panel_schedule_sha256),
        ("structural_slab_receipt_sha256", structural_slab_receipt_sha256),
    ):
        _digest(name, value)
    if not update_rows or any(len(row) != 2 for row in update_rows):
        raise M03RV16FitError("V16 epoch update evidence is incomplete")
    flat = tuple(value for row in update_rows for value in row)
    if any(
        value.get("setting_index") != validation.setting_index
        or value.get("fold_index") != validation.fold_index
        for value in flat
    ):
        raise M03RV16FitError("V16 epoch update identity drifted")
    unsigned = {
        "schema": M03R_V16_EPOCH_FIT_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package_plan_sha256,
        "worker_plan_sha256": worker_plan_sha256,
        "training_activation_receipt_sha256": training_activation_receipt_sha256,
        "panel_schedule_sha256": panel_schedule_sha256,
        "structural_slab_receipt_sha256": structural_slab_receipt_sha256,
        "setting_index": validation.setting_index,
        "fold_index": validation.fold_index,
        "epoch_index": validation.epoch_index,
        "completed_score_updates": validation.completed_score_updates,
        "inner_validation": asdict(validation),
        "update_rows": tuple(tuple(value for value in row) for row in update_rows),
        "mean_training_loss": math.fsum(
            float(value["total_loss"]) for value in flat
        )
        / len(flat),
        "mean_encoder_gradient_norm": math.fsum(
            float(value["encoder_gradient_norm_before_clip"]) for value in flat
        )
        / len(flat),
        "mean_selection_head_gradient_norm": math.fsum(
            float(value["selection_head_gradient_norm_before_clip"])
            for value in flat
        )
        / len(flat),
        "encoder_clip_fraction": math.fsum(
            bool(value["encoder_gradient_clipped"]) for value in flat
        )
        / len(flat),
        "selection_head_clip_fraction": math.fsum(
            bool(value["selection_head_gradient_clipped"]) for value in flat
        )
        / len(flat),
        "learning_rate_multiplier": float(flat[-1]["learning_rate_multiplier"]),
        "qualification_tail_accessed": False,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    return {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}


def classify_m03r_v16_training_adequacy(
    validations: tuple[M03RV16InnerValidationReceipt, ...],
    epoch_fit_payloads: tuple[dict[str, Any], ...],
) -> M03RV16TrainingAdequacy:
    """Classify terminal training without selecting an earlier checkpoint."""

    epochs = M03R_V16_PREDICTIVE_SPEC.score_training_epochs
    if len(validations) != epochs or len(epoch_fit_payloads) != epochs:
        raise M03RV16FitError("V16 adequacy requires every frozen epoch")
    for value in validations:
        value.validate()
    final = validations[-1]
    window = M03R_V16_PREDICTIVE_SPEC.adequacy_trend_window_epochs
    if window < 2 or window > epochs:
        raise M03RV16FitError("V16 adequacy trend window drifted")
    recent_validations = validations[-window:]
    ratio = final.selection_prediction_std / max(
        final.selection_target_std, 1.0e-12
    )
    first_loss = recent_validations[0].selection_robust_loss
    loss_improvement = (
        first_loss - final.selection_robust_loss
    ) / max(abs(first_loss), 1.0e-12)
    x_mean = (window - 1) / 2.0
    denominator = math.fsum((index - x_mean) ** 2 for index in range(window))
    ic_mean = math.fsum(
        value.mean_selection_rank_ic for value in recent_validations
    ) / window
    ic_slope = math.fsum(
        (index - x_mean) * (value.mean_selection_rank_ic - ic_mean)
        for index, value in enumerate(recent_validations)
    ) / denominator
    recent = epoch_fit_payloads[-window:]
    provisional = M03RV16TrainingAdequacy(
        setting_index=final.setting_index,
        fold_index=final.fold_index,
        epoch_fit_receipt_sha256=tuple(
            str(value["receipt_sha256"]) for value in epoch_fit_payloads
        ),
        final_prediction_to_target_std_ratio=ratio,
        recent_rank_ic_slope=ic_slope,
        recent_robust_loss_relative_improvement=loss_improvement,
        recent_encoder_clip_fraction=math.fsum(
            float(value["encoder_clip_fraction"]) for value in recent
        )
        / len(recent),
        recent_selection_head_clip_fraction=math.fsum(
            float(value["selection_head_clip_fraction"]) for value in recent
        )
        / len(recent),
        status="adequate",
    )
    result = M03RV16TrainingAdequacy(
        **{
            **asdict(provisional),
            "status": provisional._expected_status(),
        }
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_EPOCH_FIT_SCHEMA",
    "M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA",
    "M03R_V16_TRAINING_ADEQUACY_SCHEMA",
    "M03RV16FitError",
    "M03RV16NumericalTrainingFailure",
    "M03RV16TrainingAdequacy",
    "M03RV16TrainingAdequacyStatus",
    "build_m03r_v16_epoch_fit_payload",
    "classify_m03r_v16_training_adequacy",
]
