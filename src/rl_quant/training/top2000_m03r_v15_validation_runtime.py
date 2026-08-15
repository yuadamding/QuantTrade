"""Training-only chronological checkpoint selection for M03R-v15."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import torch

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v15_fold import M03RV15FoldGeometry
from rl_quant.training.top2000_m03r_v15_objective import m03r_v15_predictive_loss
from rl_quant.training.top2000_m03r_v15_policy import (
    Top2000M03RV15PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v15_pretraining_runtime import (
    M03RV15BuiltPredictiveBatch,
    build_m03r_v15_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v15_training_runtime import (
    move_and_bind_m03r_v15_sequence,
)
from rl_quant.training.top2000_m03r_v15_validation_contract import (
    M03R_V15_CHECKPOINT_SELECTION_SCHEMA,
    M03RV15CheckpointSelectionReceipt,
)

M03R_V15_INNER_VALIDATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v15-inner-validation-v1"
)


class M03RV15ValidationRuntimeError(ValueError):
    """The training-only validation or checkpoint selection contract drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M03RV15ValidationRuntimeError(f"{name} is not a lowercase SHA-256")
    return value


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    left = _average_ranks(first.to(torch.float64))
    right = _average_ranks(second.to(torch.float64))
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 1.0e-18:
        return 0.0
    return float((left * right).sum() / denominator)


@dataclass(frozen=True, slots=True)
class M03RV15InnerValidationReceipt:
    setting_index: int
    fold_index: int
    epoch_index: int
    completed_updates: int
    origin_count: int
    mean_action_projected_rank_ic: float
    mean_action_projected_top_bottom_spread: float
    robust_regression_loss: float
    action_projected_prediction_std: float
    target_std: float
    model_state_sha256: str
    batch_receipt_sha256: str
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    schema: str = M03R_V15_INNER_VALIDATION_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_index not in range(2)
            or self.fold_index not in range(6)
            or self.epoch_index not in range(M03R_V15_PREDICTIVE_SPEC.training_epochs)
            or self.completed_updates <= 0
            or self.origin_count
            != M03R_V15_PREDICTIVE_SPEC.inner_validation_origins_per_fold
            or not all(
                math.isfinite(value)
                for value in (
                    self.mean_action_projected_rank_ic,
                    self.mean_action_projected_top_bottom_spread,
                    self.robust_regression_loss,
                    self.action_projected_prediction_std,
                    self.target_std,
                )
            )
            or self.robust_regression_loss < 0.0
            or self.action_projected_prediction_std < 0.0
            or self.target_std < 0.0
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or self.schema != M03R_V15_INNER_VALIDATION_SCHEMA
        ):
            raise M03RV15ValidationRuntimeError("v15 inner-validation receipt drifted")
        _digest("model_state_sha256", self.model_state_sha256)
        _digest("batch_receipt_sha256", self.batch_receipt_sha256)

    @property
    def selection_key(self) -> tuple[float, float, float]:
        self.validate()
        return (
            self.mean_action_projected_rank_ic,
            self.mean_action_projected_top_bottom_spread,
            -self.robust_regression_loss,
        )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def select_m03r_v15_checkpoint(
    receipts: tuple[M03RV15InnerValidationReceipt, ...],
) -> M03RV15CheckpointSelectionReceipt:
    if len(receipts) != M03R_V15_PREDICTIVE_SPEC.training_epochs:
        raise M03RV15ValidationRuntimeError("v15 checkpoint candidates are incomplete")
    for index, receipt in enumerate(receipts):
        receipt.validate()
        if receipt.epoch_index != index:
            raise M03RV15ValidationRuntimeError("v15 validation epoch order drifted")
    if len({(row.setting_index, row.fold_index) for row in receipts}) != 1:
        raise M03RV15ValidationRuntimeError("v15 validation candidates are unpaired")
    # `max` keeps the first candidate on an exact tie, making the tie rule explicit.
    selected = max(receipts, key=lambda row: row.selection_key)
    result = M03RV15CheckpointSelectionReceipt(
        setting_index=selected.setting_index,
        fold_index=selected.fold_index,
        selected_epoch_index=selected.epoch_index,
        selected_model_state_sha256=selected.model_state_sha256,
        selected_validation_receipt_sha256=selected.receipt_sha256,
        candidate_validation_receipt_sha256=tuple(
            row.receipt_sha256 for row in receipts
        ),
    )
    result.validate()
    return result


def run_m03r_v15_inner_validation(
    cache: Top2000VerifiedDevelopmentCache,
    geometry: M03RV15FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    policy: Top2000M03RV15PredictivePolicy,
    *,
    epoch_index: int,
    completed_updates: int,
    device: torch.device,
) -> tuple[M03RV15BuiltPredictiveBatch, M03RV15InnerValidationReceipt]:
    """Evaluate one epoch without reading the outer qualification tail."""

    cache.validate_unmodified()
    geometry.validate()
    risk_source.validate()
    if epoch_index not in range(M03R_V15_PREDICTIVE_SPEC.training_epochs):
        raise M03RV15ValidationRuntimeError("v15 validation epoch cursor drifted")
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=geometry.inner_validation_episode_state_start,
        state_stop_index_exclusive=geometry.inner_validation_episode_state_stop_exclusive,
        max_state_rows=(
            geometry.inner_validation_episode_state_stop_exclusive
            - geometry.inner_validation_episode_state_start
        ),
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v15_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.inner_validation_origin_start_inclusive,
        geometry.inner_validation_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - geometry.inner_validation_episode_state_start
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    bound_sequence = replace(
        sequence,
        decision_state=torch.zeros(
            (*sequence.decision_state.shape[:3], 1),
            dtype=sequence.decision_state.dtype,
            device=sequence.decision_state.device,
        ),
    )
    provider = Top2000M03RV7DecisionStateProvider(inputs)
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            origin_states = provider.replay_origin_states(
                policy.source_policy,
                bound_sequence,
                local_origins,
            )
            batch = build_m03r_v15_batch_from_origin_states(
                policy,
                policy.v15_setting,
                origin_states,
                bound_sequence,
                local_origins,
                sequence_global_state_start=(
                    geometry.inner_validation_episode_state_start
                ),
                split="inner_validation",
                split_start_inclusive=(
                    geometry.inner_validation_origin_start_inclusive
                ),
                split_stop_exclusive=geometry.inner_validation_target_stop_exclusive,
                fold_index=geometry.fold_index,
                source_array_sha256=built.identity.receipt_sha256,
                asset_axis_sha256=cache.action_hash,
                origin_risk_exposures=risk_source.exposures,
            )
            loss = m03r_v15_predictive_loss(batch.objective)
    finally:
        policy.train(was_training)

    date_ic: list[float] = []
    date_spread: list[float] = []
    selected_prediction: list[torch.Tensor] = []
    selected_target: list[torch.Tensor] = []
    for row_index in range(batch.objective.predicted_mean.shape[0]):
        valid = batch.objective.valid[row_index]
        prediction = batch.objective.predicted_mean[row_index, valid]
        target = batch.objective.target_log_return[row_index, valid]
        date_ic.append(_spearman(prediction, target))
        order = torch.argsort(prediction, stable=True)
        tail = max(1, order.numel() // 10)
        date_spread.append(
            float(
                target.index_select(0, order[-tail:]).mean()
                - target.index_select(0, order[:tail]).mean()
            )
        )
        selected_prediction.append(prediction.to(torch.float64))
        selected_target.append(target.to(torch.float64))
    receipt = M03RV15InnerValidationReceipt(
        setting_index=policy.v15_setting.setting_index,
        fold_index=geometry.fold_index,
        epoch_index=epoch_index,
        completed_updates=completed_updates,
        origin_count=len(date_ic),
        mean_action_projected_rank_ic=sum(date_ic) / len(date_ic),
        mean_action_projected_top_bottom_spread=(
            sum(date_spread) / len(date_spread)
        ),
        robust_regression_loss=float(loss.robust_regression),
        action_projected_prediction_std=float(torch.cat(selected_prediction).std()),
        target_std=float(torch.cat(selected_target).std()),
        model_state_sha256=model_state_sha256(policy),
        batch_receipt_sha256=batch.receipt_sha256,
    )
    receipt.validate()
    return batch, receipt


__all__ = [
    "M03R_V15_CHECKPOINT_SELECTION_SCHEMA",
    "M03R_V15_INNER_VALIDATION_SCHEMA",
    "M03RV15CheckpointSelectionReceipt",
    "M03RV15InnerValidationReceipt",
    "M03RV15ValidationRuntimeError",
    "run_m03r_v15_inner_validation",
    "select_m03r_v15_checkpoint",
]
