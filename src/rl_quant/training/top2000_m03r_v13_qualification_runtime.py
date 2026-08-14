"""Exact checkpoint-to-trace fold qualification runtime for M03R-v13."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch

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
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    M03RV9ProjectorManifest,
    M03RV9ProjectorRiskBinding,
    build_m03r_v9_device_risk_state,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v13_checkpoint import M03RV13LoadedCheckpoint
from rl_quant.training.top2000_m03r_v13_fold import M03RV13FoldGeometry
from rl_quant.training.top2000_m03r_v13_policy import (
    Top2000M03RV13PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v13_pretraining_runtime import (
    M03RV13BuiltPredictiveBatch,
    build_m03r_v13_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v13_runtime import (
    M03RV13SimpleSleeveTrace,
    run_m03r_v13_simple_sleeve,
)
from rl_quant.training.top2000_m03r_v13_training_runtime import (
    move_and_bind_m03r_v13_sequence,
)


class M03RV13QualificationRuntimeError(ValueError):
    """The exact loaded checkpoint cannot produce a bound fold trace."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _daily_log_returns(
    cache: Top2000VerifiedDevelopmentCache,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    daily = cache.daily_ohlcv.to(dtype=torch.float64)
    close = daily[..., 3]
    valid = (
        cache.availability[:-1]
        & cache.availability[1:]
        & (close[:-1] > 0.0)
        & (close[1:] > 0.0)
    )
    returns = torch.zeros_like(close)
    returns[1:] = torch.where(
        valid,
        torch.log(close[1:] / close[:-1]),
        torch.zeros_like(close[1:]),
    )
    available = torch.zeros_like(cache.availability)
    available[1:] = valid
    returns[:, 0] = 0.0
    available[:, 0] = False
    receipt = hashlib.sha256(
        json.dumps(
            {
                "schema": "rl-quant.top2000-dev.m03r-v13-past-log-returns-v1",
                "cache_sha256": cache.cache_sha256,
                "asset_axis_sha256": cache.action_hash,
                "return_sha256": _tensor_sha256(returns),
                "availability_sha256": _tensor_sha256(available),
                "current_origin_return_excluded": True,
                "outer_2026_accessed": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return returns, available, receipt


def build_m03r_v13_qualification_risk_state(
    cache: Top2000VerifiedDevelopmentCache,
    geometry: M03RV13FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    risk_binding: M03RV9ProjectorRiskBinding,
    projector: M03RV9ProjectorManifest,
    *,
    device: torch.device,
) -> M03RV9DeviceRiskState:
    """Qualify one V13 fold's exact origin set and transfer it once."""

    cache.validate_unmodified()
    geometry.validate()
    risk_source.validate()
    daily_returns, return_available, receipt = _daily_log_returns(cache)
    return build_m03r_v9_device_risk_state(
        risk_source,
        risk_binding,
        projector,
        daily_log_returns=daily_returns,
        return_available=return_available,
        daily_returns_receipt_sha256=receipt,
        sequence_asset_axis_sha256=cache.action_hash,
        checkpoint_asset_axis_sha256=cache.action_hash,
        origin_state_indices=tuple(
            range(
                geometry.qualification_origin_start_inclusive,
                geometry.qualification_origin_stop_exclusive,
            )
        ),
        device=device,
    )


@dataclass(frozen=True, slots=True)
class M03RV13FoldQualificationResult:
    loaded_checkpoint: M03RV13LoadedCheckpoint
    batch: M03RV13BuiltPredictiveBatch
    trace: M03RV13SimpleSleeveTrace

    def validate(self) -> None:
        self.loaded_checkpoint.validate()
        self.batch.validate()
        self.trace.validate()
        if (
            self.batch.split != "qualification"
            or self.batch.fold_index != self.loaded_checkpoint.fold_index
            or self.batch.objective.setting.setting_index
            != self.loaded_checkpoint.setting_index
            or self.trace.fold_index != self.loaded_checkpoint.fold_index
            or self.trace.setting_index != self.loaded_checkpoint.setting_index
            or self.trace.checkpoint_file_sha256
            != self.loaded_checkpoint.checkpoint_file_sha256
            or self.trace.checkpoint_model_state_sha256
            != self.loaded_checkpoint.model_state_sha256
            or self.trace.qualification_batch_receipt_sha256
            != self.batch.receipt_sha256
        ):
            raise M03RV13QualificationRuntimeError(
                "v13 fold qualification receipt chain drifted"
            )


def run_m03r_v13_fold_qualification(
    cache: Top2000VerifiedDevelopmentCache,
    geometry: M03RV13FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    risk_state: M03RV9DeviceRiskState,
    policy: Top2000M03RV13PredictivePolicy,
    loaded: M03RV13LoadedCheckpoint,
    *,
    device: torch.device,
) -> M03RV13FoldQualificationResult:
    """Replay the exact qualification tail and execute the causal sleeve."""

    cache.validate_unmodified()
    geometry.validate()
    risk_source.validate()
    risk_state.validate()
    loaded.validate()
    if (
        loaded.fold_index != geometry.fold_index
        or loaded.setting_index != policy.v13_setting.setting_index
        or loaded.asset_axis_sha256 != cache.action_hash
        or model_state_sha256(policy) != loaded.model_state_sha256
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or risk_state.asset_axis_sha256 != cache.action_hash
        or tuple(risk_state.origin_state_indices)
        != tuple(
            range(
                geometry.qualification_origin_start_inclusive,
                geometry.qualification_origin_stop_exclusive,
            )
        )
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
    ):
        raise M03RV13QualificationRuntimeError(
            "v13 cache, fold, checkpoint, model, or risk identity drifted"
        )
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=geometry.qualification_episode_state_start,
        state_stop_index_exclusive=(
            geometry.qualification_episode_state_stop_exclusive
        ),
        max_state_rows=(
            geometry.qualification_episode_state_stop_exclusive
            - geometry.qualification_episode_state_start
        ),
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v13_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.qualification_origin_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.int64,
        device=device,
    )
    local_origins = global_origins - geometry.qualification_episode_state_start
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
    origin_states = provider.replay_origin_states(
        policy.source_policy,
        bound_sequence,
        local_origins,
    )
    batch = build_m03r_v13_batch_from_origin_states(
        policy,
        policy.v13_setting,
        origin_states,
        bound_sequence,
        local_origins,
        sequence_global_state_start=geometry.qualification_episode_state_start,
        split="qualification",
        split_start_inclusive=geometry.qualification_origin_start_inclusive,
        split_stop_exclusive=geometry.qualification_target_stop_exclusive,
        fold_index=geometry.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=risk_source.exposures,
    )
    trace = run_m03r_v13_simple_sleeve(
        bound_sequence,
        batch,
        risk_state,
        loaded,
        sequence_global_state_start=geometry.qualification_episode_state_start,
    )
    result = M03RV13FoldQualificationResult(loaded, batch, trace)
    result.validate()
    return result


__all__ = [
    "M03RV13FoldQualificationResult",
    "M03RV13QualificationRuntimeError",
    "build_m03r_v13_qualification_risk_state",
    "run_m03r_v13_fold_qualification",
]
