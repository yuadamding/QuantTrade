"""Point-in-time target construction for TOP2000 M03R-v9 pretraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_HORIZONS,
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
    M03RV9PredictiveSetting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
    M03RV9AlphaPretrainingError,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy

M03R_V9_EXPOSURE_SCHEMA = "rl-quant.top2000-dev.m03r-v9-origin-risk-exposures-v1"


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV9AlphaPretrainingError("identity must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV9OriginRiskExposures:
    """Decision-origin exposure rows aligned to one immutable asset axis."""

    state_start_index: int
    cash_index: int
    projector_exposure_names: tuple[str, ...]
    projector_exposure_families: tuple[str, ...]
    availability_family_names: tuple[str, ...]
    asset_axis_sha256: str
    source_receipt_sha256: str
    exposure_loadings: torch.Tensor  # [state, asset, 1 + projector exposure]
    regression_weights: torch.Tensor  # [state, asset]
    decision_timestamp_ms: torch.Tensor  # [state]
    exposure_available_timestamp_ms: torch.Tensor  # [state, asset, required family]
    tensor_sha256: tuple[str, str, str, str]
    receipt_sha256: str
    schema: str = M03R_V9_EXPOSURE_SCHEMA

    def validate(self) -> None:
        loadings = self.exposure_loadings
        weights = self.regression_weights
        decision_time = self.decision_timestamp_ms
        available_time = self.exposure_available_timestamp_ms
        if (
            isinstance(self.state_start_index, bool)
            or not isinstance(self.state_start_index, int)
            or self.state_start_index < 0
            or isinstance(self.cash_index, bool)
            or not isinstance(self.cash_index, int)
            or not isinstance(loadings, torch.Tensor)
            or loadings.ndim != 3
            or loadings.dtype not in {torch.float32, torch.float64}
            or loadings.requires_grad
            or not bool(torch.isfinite(loadings).all())
            or not 0 <= self.cash_index < loadings.shape[1]
            or not self.projector_exposure_names
            or len(set(self.projector_exposure_names))
            != len(self.projector_exposure_names)
            or len(self.projector_exposure_families)
            != len(self.projector_exposure_names)
            or self.availability_family_names != M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
            or not set(M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES).issubset(
                self.projector_exposure_families
            )
            or any(
                family not in M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
                for family in self.projector_exposure_families
            )
            or loadings.shape[2] != 1 + len(self.projector_exposure_names)
            or not isinstance(weights, torch.Tensor)
            or tuple(weights.shape) != tuple(loadings.shape[:2])
            or weights.dtype != loadings.dtype
            or weights.device != loadings.device
            or weights.requires_grad
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0.0).any())
            or not isinstance(decision_time, torch.Tensor)
            or tuple(decision_time.shape) != (loadings.shape[0],)
            or decision_time.dtype != torch.int64
            or decision_time.requires_grad
            or not isinstance(available_time, torch.Tensor)
            or tuple(available_time.shape)
            != (
                loadings.shape[0],
                loadings.shape[1],
                len(self.availability_family_names),
            )
            or available_time.dtype != torch.int64
            or available_time.requires_grad
            or bool((available_time > decision_time[:, None, None]).any())
            or self.schema != M03R_V9_EXPOSURE_SCHEMA
        ):
            raise M03RV9AlphaPretrainingError("origin risk exposures are malformed")
        risky = torch.ones(loadings.shape[1], device=loadings.device, dtype=torch.bool)
        risky[self.cash_index] = False
        if (
            not torch.equal(
                loadings[:, self.cash_index],
                torch.zeros_like(loadings[:, self.cash_index]),
            )
            or not torch.equal(
                loadings[:, risky, 0], torch.ones_like(loadings[:, risky, 0])
            )
            or not torch.equal(
                weights[:, self.cash_index],
                torch.zeros_like(weights[:, self.cash_index]),
            )
        ):
            raise M03RV9AlphaPretrainingError(
                "intercept/CASH exposure convention drifted"
            )
        family_indices = {
            family: tuple(
                index + 1
                for index, observed in enumerate(self.projector_exposure_families)
                if observed == family
            )
            for family in M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
        }
        sector = loadings[:, risky][:, :, family_indices["sector"]]
        beta = loadings[:, risky][:, :, family_indices["active-beta"]]
        style = loadings[:, risky][:, :, family_indices["style-risk"]]
        if (
            len(family_indices["active-beta"]) != 1
            or bool(((sector != 0.0) & (sector != 1.0)).any())
            or not torch.equal(
                sector.sum(dim=-1),
                torch.ones_like(sector.sum(dim=-1)),
            )
            or not bool((beta != 0.0).any())
            or not bool((style != 0.0).any())
        ):
            raise M03RV9AlphaPretrainingError(
                "sector, active-beta, or style-risk exposure family is not executable"
            )
        observed_hashes = (
            _tensor_sha256(loadings),
            _tensor_sha256(weights),
            _tensor_sha256(decision_time),
            _tensor_sha256(available_time),
        )
        unsigned = {
            "schema": self.schema,
            "state_start_index": self.state_start_index,
            "cash_index": self.cash_index,
            "regression_exposure_names": ("intercept", *self.projector_exposure_names),
            "projector_exposure_names": self.projector_exposure_names,
            "projector_exposure_families": self.projector_exposure_families,
            "availability_family_names": self.availability_family_names,
            "asset_axis_sha256": _digest(self.asset_axis_sha256),
            "source_receipt_sha256": _digest(self.source_receipt_sha256),
            "tensor_sha256": observed_hashes,
            "ridge_lambda": M03R_V9_PREDICTIVE_SPEC.target_ridge_lambda,
        }
        if (
            observed_hashes != self.tensor_sha256
            or _canonical_sha256(unsigned) != self.receipt_sha256
        ):
            raise M03RV9AlphaPretrainingError(
                "origin exposure content or receipt drifted"
            )


def qualify_m03r_v9_origin_risk_exposures(
    *,
    state_start_index: int,
    cash_index: int,
    projector_exposure_names: tuple[str, ...],
    projector_exposure_families: tuple[str, ...],
    availability_family_names: tuple[str, ...] = (
        M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
    ),
    asset_axis_sha256: str,
    source_receipt_sha256: str,
    exposure_loadings: torch.Tensor,
    regression_weights: torch.Tensor,
    decision_timestamp_ms: torch.Tensor,
    exposure_available_timestamp_ms: torch.Tensor,
) -> M03RV9OriginRiskExposures:
    if exposure_loadings.dtype not in {torch.float32, torch.float64}:
        raise M03RV9AlphaPretrainingError(
            "exposure loadings must be float32 or float64"
        )
    loadings = exposure_loadings.detach().to(device="cpu").clone()
    weights = regression_weights.detach().to(device="cpu", dtype=loadings.dtype).clone()
    decision_time = (
        decision_timestamp_ms.detach().to(device="cpu", dtype=torch.int64).clone()
    )
    available_time = (
        exposure_available_timestamp_ms.detach()
        .to(device="cpu", dtype=torch.int64)
        .clone()
    )
    tensor_hashes = (
        _tensor_sha256(loadings),
        _tensor_sha256(weights),
        _tensor_sha256(decision_time),
        _tensor_sha256(available_time),
    )
    unsigned = {
        "schema": M03R_V9_EXPOSURE_SCHEMA,
        "state_start_index": state_start_index,
        "cash_index": cash_index,
        "regression_exposure_names": ("intercept", *projector_exposure_names),
        "projector_exposure_names": projector_exposure_names,
        "projector_exposure_families": projector_exposure_families,
        "availability_family_names": availability_family_names,
        "asset_axis_sha256": _digest(asset_axis_sha256),
        "source_receipt_sha256": _digest(source_receipt_sha256),
        "tensor_sha256": tensor_hashes,
        "ridge_lambda": M03R_V9_PREDICTIVE_SPEC.target_ridge_lambda,
    }
    result = M03RV9OriginRiskExposures(
        state_start_index=state_start_index,
        cash_index=cash_index,
        projector_exposure_names=projector_exposure_names,
        projector_exposure_families=projector_exposure_families,
        availability_family_names=availability_family_names,
        asset_axis_sha256=asset_axis_sha256,
        source_receipt_sha256=source_receipt_sha256,
        exposure_loadings=loadings,
        regression_weights=weights,
        decision_timestamp_ms=decision_time,
        exposure_available_timestamp_ms=available_time,
        tensor_sha256=tensor_hashes,
        receipt_sha256=_canonical_sha256(unsigned),
    )
    result.validate()
    return result


def _factor_residualize(
    target: torch.Tensor,
    valid: torch.Tensor,
    loadings: torch.Tensor,
    regression_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen weighted ridge residual maker to one origin/horizon."""

    selected = torch.nonzero(valid, as_tuple=False).flatten()
    if selected.numel() < 2:
        return torch.zeros_like(target)
    work_dtype = torch.float64
    y = target.index_select(0, selected).to(work_dtype)
    exposure_selected = selected.to(device=loadings.device)
    weight_selected = selected.to(device=regression_weights.device)
    design = loadings.index_select(0, exposure_selected).to(
        device=target.device, dtype=work_dtype
    )
    weights = regression_weights.index_select(0, weight_selected).to(
        device=target.device,
        dtype=work_dtype,
    )
    positive = weights > 0.0
    if int(positive.sum()) < 2:
        return torch.zeros_like(target)
    selected = selected.index_select(
        0, torch.nonzero(positive, as_tuple=False).flatten()
    )
    y = y[positive]
    design = design[positive]
    weights = weights[positive]
    gram = design.T @ (weights.unsqueeze(-1) * design)
    gram = gram + M03R_V9_PREDICTIVE_SPEC.target_ridge_lambda * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    rhs = design.T @ (weights * y)
    coefficients = torch.linalg.solve(gram, rhs)
    residual = y - design @ coefficients
    result = torch.zeros_like(target)
    result[selected] = residual.to(result.dtype)
    return result


def build_m03r_v9_alpha_batch_from_origin_states(
    policy: Top2000M03RV9PredictivePolicy,
    setting: M03RV9PredictiveSetting,
    origin_states: torch.Tensor,
    sequence: Hold30Sequence,
    local_origin_indices: torch.Tensor,
    *,
    sequence_global_state_start: int,
    split: Literal["training", "qualification"],
    split_start_inclusive: int,
    split_stop_exclusive: int,
    fold_index: int,
    source_array_sha256: str,
    asset_axis_sha256: str,
    origin_risk_exposures: M03RV9OriginRiskExposures | None,
) -> M03RV9AlphaPretrainingBatch:
    if (
        not isinstance(policy, Top2000M03RV9PredictivePolicy)
        or policy.setting != setting
        or not isinstance(origin_states, torch.Tensor)
        or origin_states.ndim != 4
        or origin_states.shape[1] != 1
        or origin_states.shape[2] != sequence.num_assets
        or origin_states.shape[3] != policy.token_dim
        or not origin_states.is_floating_point()
        or not bool(torch.isfinite(origin_states).all())
    ):
        raise M03RV9AlphaPretrainingError("V9 policy/state geometry drifted")
    if (
        not isinstance(local_origin_indices, torch.Tensor)
        or local_origin_indices.ndim != 1
        or local_origin_indices.dtype != torch.long
        or local_origin_indices.device != origin_states.device
        or local_origin_indices.numel() == 0
        or local_origin_indices.numel() != origin_states.shape[0]
        or bool((local_origin_indices[1:] <= local_origin_indices[:-1]).any())
    ):
        raise M03RV9AlphaPretrainingError("local origins are not strictly increasing")
    global_origins = local_origin_indices + sequence_global_state_start
    if (
        int(global_origins[0]) < split_start_inclusive
        or int(global_origins[-1]) >= split_stop_exclusive
    ):
        raise M03RV9AlphaPretrainingError("origins leave the declared split")
    if setting.target_mode == "factor-residual":
        if origin_risk_exposures is None:
            raise M03RV9AlphaPretrainingError(
                "factor-residual setting lacks risk exposures"
            )
        origin_risk_exposures.validate()
        if (
            origin_risk_exposures.asset_axis_sha256 != asset_axis_sha256
            or origin_risk_exposures.exposure_loadings.shape[1] != sequence.num_assets
            or origin_risk_exposures.cash_index != sequence.initial_ledger.cash_index
        ):
            raise M03RV9AlphaPretrainingError(
                "risk exposure and sequence asset axes differ"
            )
    elif origin_risk_exposures is not None:
        raise M03RV9AlphaPretrainingError(
            "benchmark-relative setting must not residualize"
        )

    means: list[torch.Tensor] = []
    log_scales: list[torch.Tensor] = []
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        distribution = policy.alpha_distribution(
            origin_states[row_index],
            sequence.decision_available[local_origin],
        )
        means.append(distribution.mean_by_horizon.squeeze(0))
        log_scales.append(distribution.log_scale_by_horizon.squeeze(0))
    prediction = torch.stack(means)
    predicted_log_scale = torch.stack(log_scales)
    targets = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    cash = sequence.initial_ledger.cash_index
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        global_origin = local_origin + sequence_global_state_start
        for horizon_index, horizon in enumerate(M03R_V9_HORIZONS):
            local_first = local_origin + 1
            local_stop = local_first + horizon
            if (
                global_origin + horizon + 1 > split_stop_exclusive
                or local_stop > sequence.asset_returns.shape[0]
            ):
                continue
            stock = torch.log1p(
                sequence.asset_returns[local_first:local_stop, 0].clamp_min(-0.999999)
            ).sum(dim=0)
            benchmark = torch.log1p(
                sequence.benchmark_net_returns[local_first:local_stop, 0].clamp_min(
                    -0.999999
                )
            ).sum()
            row_valid = sequence.decision_available[
                local_first : local_stop + 1, 0
            ].all(dim=0)
            row_valid = row_valid.clone()
            row_valid[cash] = False
            target = (stock - benchmark).detach()
            if origin_risk_exposures is not None:
                exposure_row = global_origin - origin_risk_exposures.state_start_index
                if (
                    not 0
                    <= exposure_row
                    < origin_risk_exposures.exposure_loadings.shape[0]
                ):
                    raise M03RV9AlphaPretrainingError(
                        "origin has no point-in-time exposure row"
                    )
                target = _factor_residualize(
                    target,
                    row_valid,
                    origin_risk_exposures.exposure_loadings[exposure_row],
                    origin_risk_exposures.regression_weights[exposure_row],
                )
            targets[row_index, :, horizon_index] = target
            valid[row_index, :, horizon_index] = row_valid
    batch = M03RV9AlphaPretrainingBatch(
        predicted_mean=prediction,
        predicted_log_scale=predicted_log_scale,
        target_log_return=targets,
        valid=valid,
        origin_indices=global_origins,
        split=split,
        target_mode=setting.target_mode,
        fold_index=fold_index,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        exposure_receipt_sha256=(
            origin_risk_exposures.receipt_sha256
            if origin_risk_exposures is not None
            else None
        ),
    )
    batch.validate()
    return batch


__all__ = [
    "M03R_V9_EXPOSURE_SCHEMA",
    "M03RV9OriginRiskExposures",
    "build_m03r_v9_alpha_batch_from_origin_states",
    "qualify_m03r_v9_origin_risk_exposures",
]
