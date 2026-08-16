"""Immutable fit trajectories and adequacy classification for M03R-v16."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
)

M03R_V16_EPOCH_FIT_SCHEMA = "rl-quant.top2000-dev.m03r-v16-epoch-fit-v1"
M03R_V16_TRAINING_ADEQUACY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-adequacy-v1"
)
M03RV16TrainingAdequacyStatus = Literal["adequate", "inconclusive-undertrained"]


class M03RV16FitError(ValueError):
    """The V16 fit trajectory or adequacy classification drifted."""


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
    terminal_rank_ic_improvement: float
    terminal_robust_loss_relative_improvement: float
    recent_encoder_clip_fraction: float
    recent_selection_head_clip_fraction: float
    status: M03RV16TrainingAdequacyStatus
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_TRAINING_ADEQUACY_SCHEMA

    def _expected_status(self) -> M03RV16TrainingAdequacyStatus:
        spec = M03R_V16_PREDICTIVE_SPEC
        still_improving = (
            self.terminal_rank_ic_improvement
            > spec.adequacy_terminal_ic_improvement_threshold
            and self.terminal_robust_loss_relative_improvement
            > spec.adequacy_terminal_loss_relative_improvement_threshold
        )
        collapsed = (
            self.final_prediction_to_target_std_ratio
            < spec.adequacy_minimum_prediction_to_target_std_ratio
        )
        pervasive_clipping = max(
            self.recent_encoder_clip_fraction,
            self.recent_selection_head_clip_fraction,
        ) > spec.adequacy_maximum_recent_clip_fraction
        return (
            "inconclusive-undertrained"
            if collapsed or still_improving or pervasive_clipping
            else "adequate"
        )

    def validate(self) -> None:
        finite = (
            self.final_prediction_to_target_std_ratio,
            self.terminal_rank_ic_improvement,
            self.terminal_robust_loss_relative_improvement,
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
            or self.final_prediction_to_target_std_ratio < 0.0
            or not 0.0 <= self.recent_encoder_clip_fraction <= 1.0
            or not 0.0 <= self.recent_selection_head_clip_fraction <= 1.0
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
) -> dict[str, Any]:
    """Build the compact but complete immutable evidence for one epoch."""

    validation.validate()
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
    previous = validations[-2]
    ratio = final.selection_prediction_std / max(
        final.selection_target_std, 1.0e-12
    )
    loss_improvement = (
        previous.selection_robust_loss - final.selection_robust_loss
    ) / max(previous.selection_robust_loss, 1.0e-12)
    recent = epoch_fit_payloads[-2:]
    provisional = M03RV16TrainingAdequacy(
        setting_index=final.setting_index,
        fold_index=final.fold_index,
        epoch_fit_receipt_sha256=tuple(
            str(value["receipt_sha256"]) for value in epoch_fit_payloads
        ),
        final_prediction_to_target_std_ratio=ratio,
        terminal_rank_ic_improvement=(
            final.mean_selection_rank_ic - previous.mean_selection_rank_ic
        ),
        terminal_robust_loss_relative_improvement=loss_improvement,
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
    "M03R_V16_TRAINING_ADEQUACY_SCHEMA",
    "M03RV16FitError",
    "M03RV16TrainingAdequacy",
    "M03RV16TrainingAdequacyStatus",
    "build_m03r_v16_epoch_fit_payload",
    "classify_m03r_v16_training_adequacy",
]
