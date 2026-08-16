"""Training-only diagnostics and fixed terminal checkpoint rule for V16."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from rl_quant.protocol.canonical_artifact import semantic_sha256 as _sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_CHECKPOINT_SELECTION_RULE,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16FoldGeometry,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_numerical import (
    M03RV16NumericalTrainingError,
)
from rl_quant.training.top2000_m03r_v16_objective import m03r_v16_score_loss
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
)

M03R_V16_INNER_VALIDATION_SCHEMA = "rl-quant.top2000-dev.m03r-v16-inner-validation-v3"
M03R_V16_CHECKPOINT_SELECTION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-fixed-terminal-checkpoint-v4"
)
_CHECKPOINT_SELECTION_ISSUER = object()


class M03RV16ValidationError(ValueError):
    """The V16 training-only diagnostic evidence drifted."""


def _digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and bool(
            sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    left = _average_ranks(first.to(torch.float64))
    right = _average_ranks(second.to(torch.float64))
    left -= left.mean()
    right -= right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 1.0e-18:
        return 0.0
    return float((left * right).sum() / denominator)


@dataclass(frozen=True, slots=True)
class M03RV16InnerValidationReceipt:
    setting_index: int
    fold_index: int
    epoch_index: int
    completed_score_updates: int
    origin_count: int
    mean_selection_rank_ic: float
    mean_selection_top_bottom_spread: float
    selection_robust_loss: float
    selection_prediction_std: float
    selection_target_std: float
    model_state_sha256: str
    epoch_checkpoint_file_sha256: str
    batch_receipt_sha256: str
    used_for_checkpoint_selection: bool = False
    qualification_tail_accessed: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_INNER_VALIDATION_SCHEMA

    def validate(self) -> None:
        metrics = (
            self.mean_selection_rank_ic,
            self.mean_selection_top_bottom_spread,
            self.selection_robust_loss,
            self.selection_prediction_std,
            self.selection_target_std,
        )
        spec = M03R_V16_PREDICTIVE_SPEC
        if not all(math.isfinite(value) for value in metrics):
            raise M03RV16NumericalTrainingError(
                "V16 validation metrics are non-finite"
            )
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(spec.chronological_fold_count)
            or self.epoch_index not in range(spec.score_training_epochs)
            or self.completed_score_updates
            != render_m03r_v16_fold_geometries(1001)[
                self.fold_index
            ].training_block_count
            * (self.epoch_index + 1)
            or self.origin_count != spec.inner_validation_origins_per_fold
            or min(
                self.selection_robust_loss,
                self.selection_prediction_std,
                self.selection_target_std,
            )
            < 0.0
            or not all(
                _digest(value)
                for value in (
                    self.model_state_sha256,
                    self.epoch_checkpoint_file_sha256,
                    self.batch_receipt_sha256,
                )
            )
            or self.used_for_checkpoint_selection
            or self.qualification_tail_accessed
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_INNER_VALIDATION_SCHEMA
        ):
            raise M03RV16ValidationError("V16 inner-validation receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16CheckpointSelectionReceipt:
    setting_index: int
    fold_index: int
    selected_epoch_index: int
    selected_model_state_sha256: str
    selected_checkpoint_file_sha256: str
    selected_validation_receipt_sha256: str
    candidate_validation_receipt_sha256: tuple[str, ...]
    observed_epoch_count: int
    stop_authorized: bool
    stop_reason: str
    _issuer: object = field(repr=False)
    validation_metrics_used_for_selection: bool = False
    selection_rule: str = M03R_V16_CHECKPOINT_SELECTION_RULE
    qualification_tail_accessed: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_CHECKPOINT_SELECTION_SCHEMA

    def validate(self) -> None:
        spec = M03R_V16_PREDICTIVE_SPEC
        maximum = spec.score_training_epochs
        if (
            self._issuer is not _CHECKPOINT_SELECTION_ISSUER
            or self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.fold_index not in range(spec.chronological_fold_count)
            or self.observed_epoch_count not in range(1, maximum + 1)
            or self.selected_epoch_index != self.observed_epoch_count - 1
            or len(self.candidate_validation_receipt_sha256)
            != self.observed_epoch_count
            or not all(
                _digest(value)
                for value in (
                    self.selected_model_state_sha256,
                    self.selected_checkpoint_file_sha256,
                    self.selected_validation_receipt_sha256,
                    *self.candidate_validation_receipt_sha256,
                )
            )
            or self.selected_validation_receipt_sha256
            != self.candidate_validation_receipt_sha256[-1]
            or self.stop_authorized != (self.observed_epoch_count == maximum)
            or self.stop_reason
            != ("fixed-terminal-epoch" if self.stop_authorized else "continue")
            or self.validation_metrics_used_for_selection
            or self.selection_rule != M03R_V16_CHECKPOINT_SELECTION_RULE
            or self.qualification_tail_accessed
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_CHECKPOINT_SELECTION_SCHEMA
        ):
            raise M03RV16ValidationError("V16 checkpoint-selection receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        payload = asdict(self)
        payload.pop("_issuer")
        return _sha256(payload)


def evaluate_m03r_v16_inner_validation_batch(
    batch: M03RV16BuiltPredictiveBatch,
    geometry: M03RV16FoldGeometry,
    *,
    epoch_index: int,
    completed_score_updates: int,
    model_state_sha256: str,
    epoch_checkpoint_file_sha256: str,
) -> M03RV16InnerValidationReceipt:
    """Publish diagnostics without allowing them to select an epoch."""

    batch.validate()
    geometry.validate()
    if (
        batch.split != "inner_validation"
        or batch.fold_index != geometry.fold_index
        or batch.split_start_inclusive
        != geometry.inner_validation_origin_start_inclusive
        or batch.split_stop_exclusive != geometry.training_target_stop_exclusive
        or tuple(int(value) for value in batch.origin_indices)
        != tuple(
            range(
                geometry.inner_validation_origin_start_inclusive,
                geometry.inner_validation_origin_stop_exclusive,
            )
        )
        or completed_score_updates != geometry.training_block_count * (epoch_index + 1)
        or batch.policy_state_binding_kind != "model-state-sha256"
        or batch.policy_state_binding_sha256 != model_state_sha256
        or not _digest(model_state_sha256)
        or not _digest(epoch_checkpoint_file_sha256)
    ):
        raise M03RV16ValidationError("V16 validation batch or epoch cursor drifted")
    score_loss = m03r_v16_score_loss(batch.objective)
    selection_ic: list[float] = []
    spread: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for row_index in range(batch.origin_indices.numel()):
        valid = batch.objective.selection_valid[row_index]
        prediction = batch.objective.executable_selection_mean[row_index, valid]
        target = batch.objective.selection_target_economic[row_index, valid]
        selection_ic.append(_spearman(prediction, target))
        order = torch.argsort(prediction, stable=True)
        tail = max(1, order.numel() // 10)
        spread.append(
            float(
                target.index_select(0, order[-tail:]).mean()
                - target.index_select(0, order[:tail]).mean()
            )
        )
        predictions.append(prediction.to(torch.float64))
        targets.append(target.to(torch.float64))
    result = M03RV16InnerValidationReceipt(
        setting_index=batch.objective.setting.setting_index,
        fold_index=geometry.fold_index,
        epoch_index=epoch_index,
        completed_score_updates=completed_score_updates,
        origin_count=len(selection_ic),
        mean_selection_rank_ic=math.fsum(selection_ic) / len(selection_ic),
        mean_selection_top_bottom_spread=math.fsum(spread) / len(spread),
        selection_robust_loss=float(score_loss.selection_robust.detach()),
        selection_prediction_std=float(torch.cat(predictions).std(unbiased=False)),
        selection_target_std=float(torch.cat(targets).std(unbiased=False)),
        model_state_sha256=model_state_sha256,
        epoch_checkpoint_file_sha256=epoch_checkpoint_file_sha256,
        batch_receipt_sha256=batch.receipt_sha256,
    )
    result.validate()
    return result


def select_m03r_v16_score_checkpoint(
    receipts: tuple[M03RV16InnerValidationReceipt, ...],
) -> M03RV16CheckpointSelectionReceipt:
    """Select only the fixed terminal epoch; metrics are diagnostic."""

    spec = M03R_V16_PREDICTIVE_SPEC
    if not receipts or len(receipts) > spec.score_training_epochs:
        raise M03RV16ValidationError("V16 checkpoint candidates are incomplete")
    for epoch_index, receipt in enumerate(receipts):
        receipt.validate()
        if receipt.epoch_index != epoch_index:
            raise M03RV16ValidationError("V16 validation epoch order drifted")
    if len({(row.setting_index, row.fold_index) for row in receipts}) != 1:
        raise M03RV16ValidationError(
            "V16 validation candidates are not one fold-setting"
        )
    selected = receipts[-1]
    stop = len(receipts) == spec.score_training_epochs
    result = M03RV16CheckpointSelectionReceipt(
        setting_index=selected.setting_index,
        fold_index=selected.fold_index,
        selected_epoch_index=selected.epoch_index,
        selected_model_state_sha256=selected.model_state_sha256,
        selected_checkpoint_file_sha256=selected.epoch_checkpoint_file_sha256,
        selected_validation_receipt_sha256=selected.receipt_sha256,
        candidate_validation_receipt_sha256=tuple(
            row.receipt_sha256 for row in receipts
        ),
        observed_epoch_count=len(receipts),
        stop_authorized=stop,
        stop_reason="fixed-terminal-epoch" if stop else "continue",
        _issuer=_CHECKPOINT_SELECTION_ISSUER,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_CHECKPOINT_SELECTION_SCHEMA",
    "M03R_V16_INNER_VALIDATION_SCHEMA",
    "M03RV16CheckpointSelectionReceipt",
    "M03RV16InnerValidationReceipt",
    "M03RV16ValidationError",
    "evaluate_m03r_v16_inner_validation_batch",
    "select_m03r_v16_score_checkpoint",
]
