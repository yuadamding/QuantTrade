"""Causal, target-blind post-hoc diagnostics for completed M03R-v12 models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS,
    M03R_V12_POSTHOC_AUDIT_FOLD_SCHEMA,
    M03R_V12_POSTHOC_AUDIT_HORIZON,
    M03R_V12_POSTHOC_AUDIT_INPUT_SCHEMA,
    M03R_V12_POSTHOC_AUDIT_PANEL_SCHEMA,
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
    M03R_V12_POSTHOC_AUDIT_SPEC,
    M03R_V12_POSTHOC_AUDIT_VARIANTS,
    M03RV12PosthocAuditVariant,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_SETTING_IDS,
)


class M03RV12PosthocInferenceAuditError(ValueError):
    """The v12 post-hoc audit inputs or derived evidence drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV12PosthocInferenceAuditError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


def build_m03r_v12_posthoc_causal_action_mask(
    decision_available_at_origin: torch.Tensor,
    origin_regression_weights: torch.Tensor,
    *,
    cash_index: int = 0,
) -> torch.Tensor:
    """Intersect only decision-origin evidence; never inspect future availability."""

    if (
        not isinstance(decision_available_at_origin, torch.Tensor)
        or decision_available_at_origin.ndim != 2
        or decision_available_at_origin.dtype != torch.bool
        or not isinstance(origin_regression_weights, torch.Tensor)
        or origin_regression_weights.shape != decision_available_at_origin.shape
        or not origin_regression_weights.is_floating_point()
        or not bool(torch.isfinite(origin_regression_weights).all())
        or bool((origin_regression_weights < 0.0).any())
        or not 0 <= cash_index < decision_available_at_origin.shape[1]
    ):
        raise M03RV12PosthocInferenceAuditError(
            "v12 causal action-mask inputs are not aligned"
        )
    qualified_at_origin = origin_regression_weights.to(
        device=decision_available_at_origin.device
    ).gt(0.0)
    result = decision_available_at_origin & qualified_at_origin
    result = result.clone()
    result[:, cash_index] = False
    if bool((result.sum(dim=1) < 2).any()):
        raise M03RV12PosthocInferenceAuditError(
            "v12 causal action mask has insufficient origin-known support"
        )
    return result


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditInputs:
    setting_index: int
    setting_id: str
    fold_index: int
    horizon_sessions: int
    checkpoint_file_sha256: str
    checkpoint_model_state_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    action_mask_source_sha256: str
    post_fill_return_source_sha256: str
    origin_indices: torch.Tensor
    raw_economic_mean: torch.Tensor
    raw_rank_score: torch.Tensor
    economic_mean: torch.Tensor
    rank_score: torch.Tensor
    selected_scale: torch.Tensor
    target_log_return: torch.Tensor
    label_valid: torch.Tensor
    causal_action_mask: torch.Tensor
    fill_execution_mask: torch.Tensor
    post_fill_asset_returns: torch.Tensor
    benchmark_target_weights: torch.Tensor
    asset_weight_caps: torch.Tensor
    array_sha256: tuple[str, ...]
    outer_2026_accessed: bool = False
    outcomes_used_to_construct_actions: bool = False
    protocol_sha256: str = M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V12_POSTHOC_AUDIT_INPUT_SCHEMA

    @property
    def arrays(self) -> tuple[torch.Tensor, ...]:
        return (
            self.origin_indices,
            self.raw_economic_mean,
            self.raw_rank_score,
            self.economic_mean,
            self.rank_score,
            self.selected_scale,
            self.target_log_return,
            self.label_valid,
            self.causal_action_mask,
            self.fill_execution_mask,
            self.post_fill_asset_returns,
            self.benchmark_target_weights,
            self.asset_weight_caps,
        )

    def validate(self) -> None:
        dates, assets = self.economic_mean.shape
        aligned_float = (
            self.raw_economic_mean,
            self.raw_rank_score,
            self.rank_score,
            self.selected_scale,
            self.target_log_return,
            self.post_fill_asset_returns,
            self.benchmark_target_weights,
            self.asset_weight_caps,
        )
        if (
            self.setting_index not in range(3)
            or self.setting_id != M03R_V12_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.horizon_sessions != M03R_V12_POSTHOC_AUDIT_HORIZON
            or self.origin_indices.ndim != 1
            or self.origin_indices.dtype != torch.int64
            or self.origin_indices.shape != (dates,)
            or dates < 2
            or assets < 3
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
            or self.economic_mean.ndim != 2
            or not self.economic_mean.is_floating_point()
            or not bool(torch.isfinite(self.economic_mean).all())
            or any(
                value.shape != (dates, assets)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in aligned_float
            )
            or self.label_valid.shape != (dates, assets)
            or self.label_valid.dtype != torch.bool
            or self.causal_action_mask.shape != (dates, assets)
            or self.causal_action_mask.dtype != torch.bool
            or self.fill_execution_mask.shape != (dates, assets)
            or self.fill_execution_mask.dtype != torch.bool
            or bool(self.label_valid[:, 0].any())
            or bool(self.causal_action_mask[:, 0].any())
            or bool(self.fill_execution_mask[:, 0].any())
            or bool((self.label_valid & ~self.causal_action_mask).any())
            or bool((self.label_valid.sum(dim=1) < 2).any())
            or bool((self.causal_action_mask.sum(dim=1) < 2).any())
            or bool((self.selected_scale <= 0.0).any())
            or bool((self.benchmark_target_weights < 0.0).any())
            or not bool(
                torch.allclose(
                    self.benchmark_target_weights.sum(dim=1),
                    torch.ones(dates, dtype=self.benchmark_target_weights.dtype),
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or bool((self.asset_weight_caps < self.benchmark_target_weights).any())
            or self.array_sha256
            != tuple(_tensor_sha256(value) for value in self.arrays)
            or self.outer_2026_accessed
            or self.outcomes_used_to_construct_actions
            or self.protocol_sha256 != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V12_POSTHOC_AUDIT_INPUT_SCHEMA
        ):
            raise M03RV12PosthocInferenceAuditError(
                "v12 post-hoc audit inputs drifted"
            )
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_model_state_sha256", self.checkpoint_model_state_sha256),
            ("source_array_sha256", self.source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("action_mask_source_sha256", self.action_mask_source_sha256),
            ("post_fill_return_source_sha256", self.post_fill_return_source_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "setting_index": self.setting_index,
                "setting_id": self.setting_id,
                "fold_index": self.fold_index,
                "horizon_sessions": self.horizon_sessions,
                "checkpoint_file_sha256": self.checkpoint_file_sha256,
                "checkpoint_model_state_sha256": self.checkpoint_model_state_sha256,
                "source_array_sha256": self.source_array_sha256,
                "asset_axis_sha256": self.asset_axis_sha256,
                "action_mask_source_sha256": self.action_mask_source_sha256,
                "post_fill_return_source_sha256": (
                    self.post_fill_return_source_sha256
                ),
                "array_sha256": self.array_sha256,
                "outer_2026_accessed": self.outer_2026_accessed,
                "outcomes_used_to_construct_actions": (
                    self.outcomes_used_to_construct_actions
                ),
            }
        )


def build_m03r_v12_posthoc_audit_inputs(
    *,
    setting_index: int,
    fold_index: int,
    checkpoint_file_sha256: str,
    checkpoint_model_state_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
    action_mask_source_sha256: str,
    post_fill_return_source_sha256: str,
    origin_indices: torch.Tensor,
    raw_economic_mean: torch.Tensor,
    raw_rank_score: torch.Tensor,
    economic_mean: torch.Tensor,
    rank_score: torch.Tensor,
    selected_scale: torch.Tensor,
    target_log_return: torch.Tensor,
    label_valid: torch.Tensor,
    causal_action_mask: torch.Tensor,
    fill_execution_mask: torch.Tensor,
    post_fill_asset_returns: torch.Tensor,
    benchmark_target_weights: torch.Tensor,
    asset_weight_caps: torch.Tensor,
) -> M03RV12PosthocAuditInputs:
    arrays = tuple(
        value.detach().to(device="cpu").clone()
        for value in (
            origin_indices.to(torch.int64),
            raw_economic_mean,
            raw_rank_score,
            economic_mean,
            rank_score,
            selected_scale,
            target_log_return,
            label_valid,
            causal_action_mask,
            fill_execution_mask,
            post_fill_asset_returns,
            benchmark_target_weights,
            asset_weight_caps,
        )
    )
    result = M03RV12PosthocAuditInputs(
        setting_index=setting_index,
        setting_id=M03R_V12_SETTING_IDS[setting_index],
        fold_index=fold_index,
        horizon_sessions=M03R_V12_POSTHOC_AUDIT_HORIZON,
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_model_state_sha256=checkpoint_model_state_sha256,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        action_mask_source_sha256=action_mask_source_sha256,
        post_fill_return_source_sha256=post_fill_return_source_sha256,
        origin_indices=arrays[0],
        raw_economic_mean=arrays[1],
        raw_rank_score=arrays[2],
        economic_mean=arrays[3],
        rank_score=arrays[4],
        selected_scale=arrays[5],
        target_log_return=arrays[6],
        label_valid=arrays[7],
        causal_action_mask=arrays[8],
        fill_execution_mask=arrays[9],
        post_fill_asset_returns=arrays[10],
        benchmark_target_weights=arrays[11],
        asset_weight_caps=arrays[12],
        array_sha256=tuple(_tensor_sha256(value) for value in arrays),
    )
    result.validate()
    return result


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    ordered = value.index_select(0, order)
    _unique, counts = torch.unique_consecutive(ordered, return_counts=True)
    stops = counts.cumsum(0)
    starts = stops - counts
    average = 0.5 * (starts + stops - 1).to(torch.float64)
    ranks = torch.repeat_interleave(average, counts)
    result = torch.empty_like(ranks)
    result[order] = ranks
    return result


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    first = _average_ranks(left)
    second = _average_ranks(right)
    first -= first.mean()
    second -= second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return (
        0.0
        if float(denominator) == 0.0
        else float((first * second).sum() / denominator)
    )


def _transform_score(
    score: torch.Tensor,
    mask: torch.Tensor,
    variant: M03RV12PosthocAuditVariant,
    *,
    setting_index: int,
    fold_index: int,
) -> torch.Tensor:
    variant.validate()
    if variant.signal_transform == "original":
        return score.clone()
    if variant.signal_transform == "zero":
        return torch.zeros_like(score)
    if variant.signal_transform == "sign-flipped":
        return -score
    output = score.clone()
    for date in range(score.shape[0]):
        selected = torch.nonzero(mask[date], as_tuple=False).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            M03R_V12_POSTHOC_AUDIT_SPEC.shuffle_seed
            + setting_index * 10_000
            + fold_index * 100
            + date
            + (0 if variant.score_channel == "economic-mean" else 1_000_000)
        )
        shuffled = selected.index_select(
            0, torch.randperm(selected.numel(), generator=generator)
        )
        output[date, selected] = score[date, shuffled]
    return output


def _capped_proportional(
    strength: torch.Tensor,
    capacity: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    allocation = torch.zeros_like(strength)
    remaining = float(mass)
    active = (strength > 0.0) & (capacity > 0.0)
    for _ in range(strength.numel()):
        if remaining <= 1.0e-14 or not bool(active.any()):
            break
        weights = torch.where(active, strength, torch.zeros_like(strength))
        total = float(weights.sum())
        if total <= 0.0:
            break
        proposal = remaining * weights / total
        room = (capacity - allocation).clamp_min(0.0)
        increment = torch.minimum(proposal, room)
        used = float(increment.sum())
        allocation += increment
        remaining -= used
        active &= room - increment > 1.0e-14
        if used <= 1.0e-14:
            break
    return allocation


def _target_weights(
    benchmark: torch.Tensor,
    caps: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
    maximum_mass: float,
) -> tuple[torch.Tensor, float]:
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() < 2 or bool(score.index_select(0, selected).eq(0.0).all()):
        return benchmark.clone(), 0.0
    ranked = _average_ranks(score.index_select(0, selected))
    midpoint = 0.5 * (ranked.min() + ranked.max())
    centered = ranked - midpoint
    positive = torch.zeros_like(score, dtype=torch.float64)
    negative = torch.zeros_like(score, dtype=torch.float64)
    positive[selected] = centered.clamp_min(0.0)
    negative[selected] = (-centered).clamp_min(0.0)
    buy_capacity = (caps - benchmark).clamp_min(0.0).to(torch.float64)
    sell_capacity = benchmark.clamp_min(0.0).to(torch.float64)
    buy_capacity[~mask] = 0.0
    sell_capacity[~mask] = 0.0
    mass = min(
        maximum_mass,
        float(buy_capacity[positive > 0.0].sum()),
        float(sell_capacity[negative > 0.0].sum()),
    )
    if mass <= 1.0e-14:
        return benchmark.clone(), 0.0
    buys = _capped_proportional(positive, buy_capacity, mass)
    sells = _capped_proportional(negative, sell_capacity, mass)
    common = min(float(buys.sum()), float(sells.sum()))
    if common <= 1.0e-14:
        return benchmark.clone(), 0.0
    buys *= common / buys.sum()
    sells *= common / sells.sum()
    result = benchmark.to(torch.float64) + buys - sells
    if (
        bool((result < -2.0e-12).any())
        or bool((result - caps.to(torch.float64) > 2.0e-12).any())
        or not math.isclose(
            float(result.sum()), float(benchmark.sum()), abs_tol=2.0e-10
        )
    ):
        raise M03RV12PosthocInferenceAuditError(
            "v12 audit target-weight construction violated its constraints"
        )
    return result, common


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditFoldEvidence:
    setting_index: int
    setting_id: str
    fold_index: int
    variant_id: str
    input_receipt_sha256: str
    checkpoint_file_sha256: str
    origin_indices: torch.Tensor
    date_spearman_ic: torch.Tensor
    date_top_bottom_spread: torch.Tensor
    policy_gross_returns: torch.Tensor
    benchmark_gross_returns: torch.Tensor
    policy_one_way_turnover: torch.Tensor
    benchmark_one_way_turnover: torch.Tensor
    active_mass: torch.Tensor
    signal_projection_retention: torch.Tensor
    target_weight_trace: torch.Tensor
    net_active_return_by_cost: tuple[torch.Tensor, ...]
    mean_score_cross_sectional_std: float
    selected_scale_quantiles: tuple[float, ...]
    no_action_fraction: float
    cap_hit_fraction: float
    array_sha256: tuple[str, ...]
    receipt_sha256: str
    outcomes_used_to_construct_actions: bool = False
    action_mask_uses_future_availability: bool = False
    chronology_action_count_equals_return_count: bool = True
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V12_POSTHOC_AUDIT_FOLD_SCHEMA

    @property
    def arrays(self) -> tuple[torch.Tensor, ...]:
        return (
            self.origin_indices,
            self.date_spearman_ic,
            self.date_top_bottom_spread,
            self.policy_gross_returns,
            self.benchmark_gross_returns,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.active_mass,
            self.signal_projection_retention,
            self.target_weight_trace,
            *self.net_active_return_by_cost,
        )

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "variant_id": self.variant_id,
            "input_receipt_sha256": self.input_receipt_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "mean_score_cross_sectional_std": self.mean_score_cross_sectional_std,
            "selected_scale_quantiles": self.selected_scale_quantiles,
            "no_action_fraction": self.no_action_fraction,
            "cap_hit_fraction": self.cap_hit_fraction,
            "array_sha256": self.array_sha256,
            "outcomes_used_to_construct_actions": (
                self.outcomes_used_to_construct_actions
            ),
            "action_mask_uses_future_availability": (
                self.action_mask_uses_future_availability
            ),
            "chronology_action_count_equals_return_count": (
                self.chronology_action_count_equals_return_count
            ),
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "outer_2026_accessed": self.outer_2026_accessed,
            "development_only": True,
            "posthoc_exploratory": True,
            "reportable": False,
            "promotion_eligible": False,
            "economic_generation_may_be_minted": False,
        }

    def validate(self) -> None:
        variant = next(
            (
                row
                for row in M03R_V12_POSTHOC_AUDIT_VARIANTS
                if row.variant_id == self.variant_id
            ),
            None,
        )
        dates = self.origin_indices.numel()
        one_dimensional = self.arrays[1:9] + self.net_active_return_by_cost
        if (
            variant is None
            or self.setting_index not in range(3)
            or self.setting_id != M03R_V12_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.origin_indices.ndim != 1
            or self.origin_indices.dtype != torch.int64
            or dates < 2
            or any(
                value.ndim != 1
                or value.shape != (dates,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in one_dimensional
            )
            or self.target_weight_trace.ndim != 2
            or self.target_weight_trace.shape[0] != dates
            or not bool(torch.isfinite(self.target_weight_trace).all())
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.active_mass < 0.0).any())
            or bool((self.signal_projection_retention < 0.0).any())
            or len(self.net_active_return_by_cost)
            != len(M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS)
            or not math.isfinite(self.mean_score_cross_sectional_std)
            or self.mean_score_cross_sectional_std < 0.0
            or len(self.selected_scale_quantiles) != 5
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.selected_scale_quantiles
            )
            or not 0.0 <= self.no_action_fraction <= 1.0
            or not 0.0 <= self.cap_hit_fraction <= 1.0
            or self.array_sha256
            != tuple(_tensor_sha256(value) for value in self.arrays)
            or self.outcomes_used_to_construct_actions
            or self.action_mask_uses_future_availability
            or not self.chronology_action_count_equals_return_count
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V12_POSTHOC_AUDIT_FOLD_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12PosthocInferenceAuditError(
                "v12 post-hoc fold evidence drifted"
            )
        _digest("input_receipt_sha256", self.input_receipt_sha256)
        _digest("checkpoint_file_sha256", self.checkpoint_file_sha256)


def build_m03r_v12_posthoc_audit_fold_evidence(
    inputs: M03RV12PosthocAuditInputs,
    variant: M03RV12PosthocAuditVariant,
) -> M03RV12PosthocAuditFoldEvidence:
    """Replay one target-blind score variant with one action per earned return."""

    inputs.validate()
    variant.validate()
    score = (
        inputs.economic_mean
        if variant.score_channel == "economic-mean"
        else inputs.rank_score
    ).to(torch.float64)
    transformed = _transform_score(
        score,
        inputs.causal_action_mask,
        variant,
        setting_index=inputs.setting_index,
        fold_index=inputs.fold_index,
    )
    raw_score = (
        inputs.raw_economic_mean
        if variant.score_channel == "economic-mean"
        else inputs.raw_rank_score
    ).to(torch.float64)
    projection_retention = []
    for date in range(score.shape[0]):
        mask = inputs.causal_action_mask[date]
        raw_norm = torch.linalg.vector_norm(raw_score[date, mask])
        projected_norm = torch.linalg.vector_norm(score[date, mask])
        projection_retention.append(
            1.0
            if float(raw_norm) <= 1.0e-14 and float(projected_norm) <= 1.0e-14
            else float(projected_norm / raw_norm.clamp_min(1.0e-14))
        )
    dates, _assets = score.shape
    date_ic: list[float] = []
    date_spread: list[float] = []
    date_std: list[float] = []
    for date in range(dates):
        valid = inputs.label_valid[date]
        observed_score = transformed[date, valid]
        target = inputs.target_log_return[date, valid].to(torch.float64)
        score_std = float(observed_score.std(unbiased=False))
        date_std.append(score_std)
        if score_std == 0.0:
            date_ic.append(0.0)
            date_spread.append(0.0)
        else:
            date_ic.append(_spearman(observed_score, target))
            order = torch.argsort(observed_score, stable=True)
            tail = max(
                1, order.numel() // M03R_V12_POSTHOC_AUDIT_SPEC.quantile_count
            )
            date_spread.append(
                float(
                    target.index_select(0, order[-tail:]).mean()
                    - target.index_select(0, order[:tail]).mean()
                )
            )

    current_policy = inputs.benchmark_target_weights[0].to(torch.float64).clone()
    current_benchmark = current_policy.clone()
    policy_gross: list[torch.Tensor] = []
    benchmark_gross: list[torch.Tensor] = []
    policy_turnover: list[torch.Tensor] = []
    benchmark_turnover: list[torch.Tensor] = []
    active_mass: list[float] = []
    targets: list[torch.Tensor] = []
    for date in range(dates):
        benchmark = inputs.benchmark_target_weights[date].to(torch.float64)
        target, mass = _target_weights(
            benchmark,
            inputs.asset_weight_caps[date],
            transformed[date],
            inputs.causal_action_mask[date] & inputs.fill_execution_mask[date],
            variant.maximum_active_one_way_mass,
        )
        policy_turnover.append(0.5 * (target - current_policy).abs().sum())
        benchmark_turnover.append(0.5 * (benchmark - current_benchmark).abs().sum())
        returns = inputs.post_fill_asset_returns[date].to(torch.float64)
        policy_return = (target * returns).sum()
        benchmark_return = (benchmark * returns).sum()
        if float(policy_return) <= -1.0 or float(benchmark_return) <= -1.0:
            raise M03RV12PosthocInferenceAuditError(
                "v12 post-hoc chronology lost all value"
            )
        policy_gross.append(policy_return)
        benchmark_gross.append(benchmark_return)
        current_policy = target * (1.0 + returns) / (1.0 + policy_return)
        current_benchmark = benchmark * (1.0 + returns) / (1.0 + benchmark_return)
        active_mass.append(mass)
        targets.append(target)

    policy_gross_tensor = torch.stack(policy_gross)
    benchmark_gross_tensor = torch.stack(benchmark_gross)
    policy_turnover_tensor = torch.stack(policy_turnover)
    benchmark_turnover_tensor = torch.stack(benchmark_turnover)
    gross_active = policy_gross_tensor - benchmark_gross_tensor
    incremental_turnover = policy_turnover_tensor - benchmark_turnover_tensor
    net_by_cost = tuple(
        gross_active - (basis_points / 10_000.0) * incremental_turnover
        for basis_points in M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS
    )
    active_mass_tensor = torch.tensor(active_mass, dtype=torch.float64)
    projection_retention_tensor = torch.tensor(
        projection_retention, dtype=torch.float64
    )
    cap = variant.maximum_active_one_way_mass
    selected_scale = inputs.selected_scale[inputs.causal_action_mask].to(torch.float64)
    scale_quantiles = tuple(
        float(value)
        for value in torch.quantile(
            selected_scale,
            torch.tensor((0.0, 0.25, 0.50, 0.75, 1.0), dtype=torch.float64),
        )
    )
    arrays = tuple(
        _cpu64(value)
        for value in (
            inputs.origin_indices,
            torch.tensor(date_ic, dtype=torch.float64),
            torch.tensor(date_spread, dtype=torch.float64),
            policy_gross_tensor,
            benchmark_gross_tensor,
            policy_turnover_tensor,
            benchmark_turnover_tensor,
            active_mass_tensor,
            projection_retention_tensor,
            torch.stack(targets),
            *net_by_cost,
        )
    )
    # Preserve the integer chronology axis after the common detached copy.
    arrays = (inputs.origin_indices.detach().to(torch.int64).clone(), *arrays[1:])
    provisional = M03RV12PosthocAuditFoldEvidence(
        setting_index=inputs.setting_index,
        setting_id=inputs.setting_id,
        fold_index=inputs.fold_index,
        variant_id=variant.variant_id,
        input_receipt_sha256=inputs.receipt_sha256,
        checkpoint_file_sha256=inputs.checkpoint_file_sha256,
        origin_indices=arrays[0],
        date_spearman_ic=arrays[1],
        date_top_bottom_spread=arrays[2],
        policy_gross_returns=arrays[3],
        benchmark_gross_returns=arrays[4],
        policy_one_way_turnover=arrays[5],
        benchmark_one_way_turnover=arrays[6],
        active_mass=arrays[7],
        signal_projection_retention=arrays[8],
        target_weight_trace=arrays[9],
        net_active_return_by_cost=tuple(arrays[10:]),
        mean_score_cross_sectional_std=sum(date_std) / len(date_std),
        selected_scale_quantiles=scale_quantiles,
        no_action_fraction=float((active_mass_tensor <= 1.0e-14).float().mean()),
        cap_hit_fraction=float((active_mass_tensor >= cap - 1.0e-12).float().mean()),
        array_sha256=tuple(_tensor_sha256(value) for value in arrays),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def _bootstrap_indices(
    fold_lengths: tuple[int, ...],
    *,
    block_sessions: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        M03R_V12_POSTHOC_AUDIT_SPEC.bootstrap_seed + block_sessions
    )
    total = sum(fold_lengths)
    draws = torch.empty(
        (M03R_V12_POSTHOC_AUDIT_SPEC.bootstrap_replicates, total),
        dtype=torch.long,
    )
    offset = 0
    for replicate in range(M03R_V12_POSTHOC_AUDIT_SPEC.bootstrap_replicates):
        write = 0
        offset = 0
        for length in fold_lengths:
            selected: list[int] = []
            while len(selected) < length:
                start = int(torch.randint(length, (1,), generator=generator))
                selected.extend(
                    offset + (start + step) % length
                    for step in range(block_sessions)
                )
            draws[replicate, write : write + length] = torch.tensor(
                selected[:length], dtype=torch.long
            )
            write += length
            offset += length
    return draws


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditPanelReport:
    setting_index: int
    setting_id: str
    variant_id: str
    fold_receipt_sha256: tuple[str, ...]
    mean_date_spearman_ic: float
    positive_mean_ic_fold_count: int
    mean_top_bottom_spread: float
    positive_spread_fold_count: int
    annualized_gross_active_return: float
    annualized_net_active_return_by_cost: tuple[float, ...]
    gross_active_lcb_by_block: tuple[float, ...]
    net_10bp_lcb_by_block: tuple[float, ...]
    spread_lcb_by_block: tuple[float, ...]
    aggregate_break_even_one_way_cost_basis_points: float | None
    favorable_cost_dominance: bool
    mean_policy_one_way_turnover: float
    mean_incremental_one_way_turnover: float
    mean_active_mass: float
    mean_no_action_fraction: float
    median_signal_projection_retention: float
    receipt_sha256: str
    economic_generation_may_be_minted: bool = False
    outer_2026_accessed: bool = False
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V12_POSTHOC_AUDIT_PANEL_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "variant_id": self.variant_id,
            "fold_receipt_sha256": self.fold_receipt_sha256,
            "mean_date_spearman_ic": self.mean_date_spearman_ic,
            "positive_mean_ic_fold_count": self.positive_mean_ic_fold_count,
            "mean_top_bottom_spread": self.mean_top_bottom_spread,
            "positive_spread_fold_count": self.positive_spread_fold_count,
            "annualized_gross_active_return": self.annualized_gross_active_return,
            "annualized_net_active_return_by_cost": (
                self.annualized_net_active_return_by_cost
            ),
            "gross_active_lcb_by_block": self.gross_active_lcb_by_block,
            "net_10bp_lcb_by_block": self.net_10bp_lcb_by_block,
            "spread_lcb_by_block": self.spread_lcb_by_block,
            "aggregate_break_even_one_way_cost_basis_points": (
                self.aggregate_break_even_one_way_cost_basis_points
            ),
            "favorable_cost_dominance": self.favorable_cost_dominance,
            "mean_policy_one_way_turnover": self.mean_policy_one_way_turnover,
            "mean_incremental_one_way_turnover": (
                self.mean_incremental_one_way_turnover
            ),
            "mean_active_mass": self.mean_active_mass,
            "mean_no_action_fraction": self.mean_no_action_fraction,
            "median_signal_projection_retention": (
                self.median_signal_projection_retention
            ),
            "economic_generation_may_be_minted": (
                self.economic_generation_may_be_minted
            ),
            "outer_2026_accessed": self.outer_2026_accessed,
            "development_only": True,
            "posthoc_exploratory": True,
            "reportable": self.reportable,
            "promotion_eligible": self.promotion_eligible,
        }

    def validate(self) -> None:
        variant = next(
            (
                row
                for row in M03R_V12_POSTHOC_AUDIT_VARIANTS
                if row.variant_id == self.variant_id
            ),
            None,
        )
        finite_values = (
            self.mean_date_spearman_ic,
            self.mean_top_bottom_spread,
            self.annualized_gross_active_return,
            *self.annualized_net_active_return_by_cost,
            *self.gross_active_lcb_by_block,
            *self.net_10bp_lcb_by_block,
            *self.spread_lcb_by_block,
            self.mean_policy_one_way_turnover,
            self.mean_incremental_one_way_turnover,
            self.mean_active_mass,
            self.mean_no_action_fraction,
            self.median_signal_projection_retention,
        )
        if (
            variant is None
            or self.setting_index not in range(3)
            or self.setting_id != M03R_V12_SETTING_IDS[self.setting_index]
            or len(self.fold_receipt_sha256) != 6
            or len(set(self.fold_receipt_sha256)) != 6
            or not all(math.isfinite(value) for value in finite_values)
            or self.positive_mean_ic_fold_count not in range(7)
            or self.positive_spread_fold_count not in range(7)
            or len(self.annualized_net_active_return_by_cost)
            != len(M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS)
            or len(self.gross_active_lcb_by_block) != 3
            or len(self.net_10bp_lcb_by_block) != 3
            or len(self.spread_lcb_by_block) != 3
            or (
                self.aggregate_break_even_one_way_cost_basis_points is not None
                and (
                    not math.isfinite(
                        self.aggregate_break_even_one_way_cost_basis_points
                    )
                    or self.aggregate_break_even_one_way_cost_basis_points <= 0.0
                )
            )
            or not 0.0 <= self.mean_no_action_fraction <= 1.0
            or self.economic_generation_may_be_minted
            or self.outer_2026_accessed
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V12_POSTHOC_AUDIT_PANEL_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12PosthocInferenceAuditError(
                "v12 post-hoc panel report drifted"
            )
        for value in self.fold_receipt_sha256:
            _digest("fold_receipt_sha256", value)


def build_m03r_v12_posthoc_audit_panel_report(
    folds: tuple[M03RV12PosthocAuditFoldEvidence, ...],
) -> M03RV12PosthocAuditPanelReport:
    """Aggregate six folds with fold-bounded common moving-block draws."""

    if len(folds) != 6:
        raise M03RV12PosthocInferenceAuditError(
            "v12 post-hoc panel requires exactly six folds"
        )
    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    for row in ordered:
        row.validate()
    first = ordered[0]
    if (
        tuple(row.fold_index for row in ordered) != tuple(range(6))
        or any(row.setting_index != first.setting_index for row in ordered)
        or any(row.variant_id != first.variant_id for row in ordered)
    ):
        raise M03RV12PosthocInferenceAuditError(
            "v12 post-hoc panel identities are not one complete family"
        )
    fold_lengths = tuple(row.origin_indices.numel() for row in ordered)
    gross = torch.cat(
        tuple(
            row.policy_gross_returns - row.benchmark_gross_returns
            for row in ordered
        )
    )
    net_by_cost = tuple(
        torch.cat(tuple(row.net_active_return_by_cost[index] for row in ordered))
        for index in range(len(M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS))
    )
    spread = torch.cat(tuple(row.date_top_bottom_spread for row in ordered))
    blocks = M03R_V12_POSTHOC_AUDIT_SPEC.bootstrap_blocks
    gross_lcb: list[float] = []
    net_lcb: list[float] = []
    spread_lcb: list[float] = []
    net_10bp = net_by_cost[M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS.index(10)]
    for block in blocks:
        draws = _bootstrap_indices(fold_lengths, block_sessions=block)
        gross_lcb.append(
            float(torch.quantile(gross[draws].mean(dim=1), 0.025)) * 252.0
        )
        net_lcb.append(
            float(torch.quantile(net_10bp[draws].mean(dim=1), 0.025)) * 252.0
        )
        spread_lcb.append(
            float(torch.quantile(spread[draws].mean(dim=1), 0.025))
        )
    policy_turnover = torch.cat(
        tuple(row.policy_one_way_turnover for row in ordered)
    )
    benchmark_turnover = torch.cat(
        tuple(row.benchmark_one_way_turnover for row in ordered)
    )
    incremental_turnover = policy_turnover - benchmark_turnover
    gross_sum = float(gross.sum())
    turnover_sum = float(incremental_turnover.sum())
    favorable = gross_sum > 0.0 and turnover_sum <= 0.0
    break_even = (
        10_000.0 * gross_sum / turnover_sum
        if gross_sum > 0.0 and turnover_sum > 0.0
        else None
    )
    fold_mean_ic = tuple(float(row.date_spearman_ic.mean()) for row in ordered)
    fold_mean_spread = tuple(
        float(row.date_top_bottom_spread.mean()) for row in ordered
    )
    retention = torch.cat(
        tuple(row.signal_projection_retention for row in ordered)
    )
    provisional = M03RV12PosthocAuditPanelReport(
        setting_index=first.setting_index,
        setting_id=first.setting_id,
        variant_id=first.variant_id,
        fold_receipt_sha256=tuple(row.receipt_sha256 for row in ordered),
        mean_date_spearman_ic=float(
            torch.cat(tuple(row.date_spearman_ic for row in ordered)).mean()
        ),
        positive_mean_ic_fold_count=sum(value > 0.0 for value in fold_mean_ic),
        mean_top_bottom_spread=float(spread.mean()),
        positive_spread_fold_count=sum(value > 0.0 for value in fold_mean_spread),
        annualized_gross_active_return=252.0 * float(gross.mean()),
        annualized_net_active_return_by_cost=tuple(
            252.0 * float(value.mean()) for value in net_by_cost
        ),
        gross_active_lcb_by_block=tuple(gross_lcb),
        net_10bp_lcb_by_block=tuple(net_lcb),
        spread_lcb_by_block=tuple(spread_lcb),
        aggregate_break_even_one_way_cost_basis_points=break_even,
        favorable_cost_dominance=favorable,
        mean_policy_one_way_turnover=float(policy_turnover.mean()),
        mean_incremental_one_way_turnover=float(incremental_turnover.mean()),
        mean_active_mass=float(
            torch.cat(tuple(row.active_mass for row in ordered)).mean()
        ),
        mean_no_action_fraction=sum(row.no_action_fraction for row in ordered) / 6.0,
        median_signal_projection_retention=float(retention.median()),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


__all__ = [
    "M03RV12PosthocAuditFoldEvidence",
    "M03RV12PosthocAuditInputs",
    "M03RV12PosthocAuditPanelReport",
    "M03RV12PosthocInferenceAuditError",
    "build_m03r_v12_posthoc_audit_fold_evidence",
    "build_m03r_v12_posthoc_audit_inputs",
    "build_m03r_v12_posthoc_audit_panel_report",
    "build_m03r_v12_posthoc_causal_action_mask",
]
