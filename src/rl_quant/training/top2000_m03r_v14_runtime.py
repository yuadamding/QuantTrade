"""Causal fixed-rank qualification sleeve for M03R-v14.

The sleeve is deliberately non-learned.  It tests whether the checkpoint's
single direct h3 score orders future factor-residual returns after applying
only the decision-origin action operator.  An action is filled before, and is
therefore exposed to, exactly one following return transition.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PREDICTIVE_SPEC,
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SELECTED_HORIZON_SESSIONS,
    M03R_V14_SETTING_IDS,
    M03R_V14_SIMPLE_SLEEVE_RULE,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    project_m03r_v9_active_book,
)
from rl_quant.training.top2000_m03r_v14_checkpoint import M03RV14LoadedCheckpoint
from rl_quant.training.top2000_m03r_v14_pretraining_runtime import (
    M03RV14BuiltPredictiveBatch,
)

M03R_V14_SIMPLE_SLEEVE_TRACE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-causal-fixed-rank-sleeve-v1"
)


class M03RV14RuntimeError(ValueError):
    """The v14 causal qualification chronology or evidence drifted."""


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
        raise M03RV14RuntimeError(f"{name} must be a lowercase SHA-256")
    return value


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    sorted_value = value.index_select(0, order)
    ranks = torch.empty_like(sorted_value, dtype=torch.float64)
    start = 0
    while start < sorted_value.numel():
        stop = start + 1
        while stop < sorted_value.numel() and bool(
            sorted_value[stop] == sorted_value[start]
        ):
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    result = torch.empty_like(ranks)
    result[order] = ranks
    return result


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    first = _average_ranks(prediction.to(torch.float64))
    second = _average_ranks(target.to(torch.float64))
    first -= first.mean()
    second -= second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 0.0:
        return 0.0
    return float((first * second).sum() / denominator)


def _scale_quantiles_tensor(scale: torch.Tensor) -> torch.Tensor:
    """Return frozen scale quantiles without crossing the tensor's device."""

    probabilities = torch.tensor(
        (0.0, 0.25, 0.50, 0.75, 1.0),
        dtype=scale.dtype,
        device=scale.device,
    )
    return torch.quantile(scale, probabilities)


def _capped_proportional(
    strength: torch.Tensor,
    capacity: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    allocation = torch.zeros_like(strength)
    remaining = mass
    active = (strength > 0.0) & (capacity > 0.0)
    for _ in range(strength.numel()):
        if remaining <= 1.0e-14 or not bool(active.any()):
            break
        weights = torch.where(active, strength, torch.zeros_like(strength))
        total = float(weights.sum())
        if total <= 0.0:
            break
        room = (capacity - allocation).clamp_min(0.0)
        increment = torch.minimum(remaining * weights / total, room)
        used = float(increment.sum())
        allocation += increment
        remaining -= used
        active &= room - increment > 1.0e-14
        if used <= 1.0e-14:
            break
    return allocation


def _fixed_rank_target(
    benchmark: torch.Tensor,
    caps: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Build a benchmark-relative rank sleeve with predeclared small mass."""

    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() < 2:
        return benchmark.clone(), 0.0
    observed = score.index_select(0, selected).to(torch.float64)
    if float(observed.max() - observed.min()) <= 0.0:
        return benchmark.clone(), 0.0
    ranks = _average_ranks(observed)
    centered = ranks - 0.5 * (float(ranks.min()) + float(ranks.max()))
    positive = torch.zeros_like(score, dtype=torch.float64)
    negative = torch.zeros_like(score, dtype=torch.float64)
    positive[selected] = centered.clamp_min(0.0)
    negative[selected] = (-centered).clamp_min(0.0)
    buy_capacity = (caps - benchmark).clamp_min(0.0).to(torch.float64)
    sell_capacity = benchmark.clamp_min(0.0).to(torch.float64)
    buy_capacity[~mask] = 0.0
    sell_capacity[~mask] = 0.0
    maximum = M03R_V14_PREDICTIVE_SPEC.simple_sleeve_maximum_active_one_way_mass
    mass = min(
        maximum,
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
    requested = benchmark.to(torch.float64) + buys - sells
    # ``requested`` is intentionally accumulated in float64.  Compare its
    # simplex mass with the same float64 view of the anchor rather than with a
    # float32 reduction of ``benchmark``.  Large equal-weight books can differ
    # by several 1e-8 solely because the two reductions use different dtypes;
    # that is not a portfolio-constraint violation.
    benchmark_mass = float(benchmark.to(torch.float64).sum())
    if (
        not bool(torch.isfinite(requested).all())
        or bool((requested < -2.0e-12).any())
        or bool((requested - caps.to(torch.float64) > 2.0e-12).any())
        or not math.isclose(
            float(requested.sum()), benchmark_mass, abs_tol=2.0e-10
        )
    ):
        raise M03RV14RuntimeError("v14 rank target violated its weight constraints")
    return requested, common


@dataclass(frozen=True, slots=True)
class M03RV14SimpleSleeveTrace:
    setting_index: int
    setting_id: str
    fold_index: int
    checkpoint_file_sha256: str
    checkpoint_model_state_sha256: str
    qualification_batch_receipt_sha256: str
    qualification_source_array_sha256: str
    asset_axis_sha256: str
    risk_state_sha256: str
    risk_manifest_sha256: str
    action_operator_receipt_sha256: tuple[str, ...]
    origin_indices: torch.Tensor
    date_spearman_ic: torch.Tensor
    date_top_bottom_spread: torch.Tensor
    policy_gross_returns: torch.Tensor
    benchmark_gross_returns: torch.Tensor
    policy_one_way_turnover: torch.Tensor
    benchmark_one_way_turnover: torch.Tensor
    active_mass: torch.Tensor
    signal_projection_retention: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    requested_weight_trace: torch.Tensor
    executed_weight_trace: torch.Tensor
    net_active_return_by_cost: tuple[torch.Tensor, ...]
    selected_scale_quantiles: tuple[float, ...]
    no_action_fraction: float
    active_mass_cap_fraction: float
    array_sha256: tuple[str, ...]
    trace_sha256: str
    simple_sleeve_rule: str = M03R_V14_SIMPLE_SLEEVE_RULE
    selected_horizon_sessions: int = M03R_V14_SELECTED_HORIZON_SESSIONS
    action_mask_uses_future_availability: bool = False
    chronology_action_count_equals_return_count: bool = True
    learned_hazard_enabled: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_SIMPLE_SLEEVE_TRACE_SCHEMA

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
            self.requested_to_executed_retention,
            self.requested_weight_trace,
            self.executed_weight_trace,
            *self.net_active_return_by_cost,
        )

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_model_state_sha256": self.checkpoint_model_state_sha256,
            "qualification_batch_receipt_sha256": (
                self.qualification_batch_receipt_sha256
            ),
            "qualification_source_array_sha256": (
                self.qualification_source_array_sha256
            ),
            "asset_axis_sha256": self.asset_axis_sha256,
            "risk_state_sha256": self.risk_state_sha256,
            "risk_manifest_sha256": self.risk_manifest_sha256,
            "action_operator_receipt_sha256": (
                self.action_operator_receipt_sha256
            ),
            "selected_scale_quantiles": self.selected_scale_quantiles,
            "no_action_fraction": self.no_action_fraction,
            "active_mass_cap_fraction": self.active_mass_cap_fraction,
            "array_sha256": self.array_sha256,
            "simple_sleeve_rule": self.simple_sleeve_rule,
            "selected_horizon_sessions": self.selected_horizon_sessions,
            "action_mask_uses_future_availability": (
                self.action_mask_uses_future_availability
            ),
            "chronology_action_count_equals_return_count": (
                self.chronology_action_count_equals_return_count
            ),
            "learned_hazard_enabled": self.learned_hazard_enabled,
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "outer_2026_accessed": self.outer_2026_accessed,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        dates = self.origin_indices.numel()
        one_dimensional = self.arrays[1:10] + self.net_active_return_by_cost
        if (
            self.setting_index not in range(len(M03R_V14_SETTING_IDS))
            or self.setting_id != M03R_V14_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.origin_indices.ndim != 1
            or self.origin_indices.dtype != torch.int64
            or dates != M03R_V14_PREDICTIVE_SPEC.qualification_origins_per_fold
            or bool((self.origin_indices[1:] != self.origin_indices[:-1] + 1).any())
            or any(
                value.ndim != 1
                or value.shape != (dates,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in one_dimensional
            )
            or self.requested_weight_trace.ndim != 2
            or self.requested_weight_trace.shape[0] != dates
            or self.executed_weight_trace.shape != self.requested_weight_trace.shape
            or not bool(torch.isfinite(self.requested_weight_trace).all())
            or not bool(torch.isfinite(self.executed_weight_trace).all())
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.active_mass < 0.0).any())
            or bool((self.signal_projection_retention < 0.0).any())
            or bool((self.requested_to_executed_retention < 0.0).any())
            or len(self.net_active_return_by_cost)
            != len(M03R_V14_PREDICTIVE_SPEC.evaluation_cost_basis_points)
            or len(self.action_operator_receipt_sha256) != dates
            or len(self.selected_scale_quantiles) != 5
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.selected_scale_quantiles
            )
            or not 0.0 <= self.no_action_fraction <= 1.0
            or not 0.0 <= self.active_mass_cap_fraction <= 1.0
            or self.array_sha256 != tuple(_tensor_sha256(row) for row in self.arrays)
            or self.simple_sleeve_rule != M03R_V14_SIMPLE_SLEEVE_RULE
            or self.selected_horizon_sessions
            != M03R_V14_SELECTED_HORIZON_SESSIONS
            or self.action_mask_uses_future_availability
            or not self.chronology_action_count_equals_return_count
            or self.learned_hazard_enabled
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_SIMPLE_SLEEVE_TRACE_SCHEMA
            or self.trace_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV14RuntimeError("v14 simple-sleeve trace drifted")
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_model_state_sha256", self.checkpoint_model_state_sha256),
            ("qualification_batch_receipt_sha256", self.qualification_batch_receipt_sha256),
            ("qualification_source_array_sha256", self.qualification_source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("risk_state_sha256", self.risk_state_sha256),
            ("risk_manifest_sha256", self.risk_manifest_sha256),
            *(("action_operator_receipt_sha256", row) for row in self.action_operator_receipt_sha256),
        ):
            _digest(name, value)


def run_m03r_v14_simple_sleeve(
    sequence: Hold30Sequence,
    batch: M03RV14BuiltPredictiveBatch,
    risk_state: M03RV9DeviceRiskState,
    loaded: M03RV14LoadedCheckpoint,
    *,
    sequence_global_state_start: int,
) -> M03RV14SimpleSleeveTrace:
    """Execute one action then one post-fill return for every h3 origin."""

    batch.validate()
    loaded.validate()
    risk_state.validate()
    objective = batch.objective
    dates = objective.predicted_mean.shape[0]
    local_origins = batch.origin_indices - sequence_global_state_start
    if (
        batch.split != "qualification"
        or batch.fold_index != loaded.fold_index
        or objective.setting.setting_index != loaded.setting_index
        or batch.asset_axis_sha256 != loaded.asset_axis_sha256
        or dates != M03R_V14_PREDICTIVE_SPEC.qualification_origins_per_fold
        or bool((batch.origin_indices[1:] != batch.origin_indices[:-1] + 1).any())
        or int(local_origins.min()) < 0
        or int(local_origins.max()) + 1 >= sequence.asset_returns.shape[0]
        or sequence.axis_id != loaded.asset_axis_sha256
        or tuple(risk_state.origin_state_indices)
        != tuple(int(value) for value in batch.origin_indices)
        or risk_state.asset_axis_sha256 != loaded.asset_axis_sha256
        or risk_state.source_exposure_receipt_sha256
        != batch.exposure_receipt_sha256
    ):
        raise M03RV14RuntimeError(
            "v14 checkpoint, batch, sequence, or risk chronology drifted"
        )
    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=sequence.axis_id,
        checkpoint_asset_axis_sha256=loaded.asset_axis_sha256,
        expected_manifest_sha256=risk_state.manifest_sha256,
    )

    first_fill = int(local_origins[0]) + 1
    current_policy = sequence.benchmark_weights[first_fill].clone()
    current_benchmark = current_policy.clone()
    date_ic: list[float] = []
    date_spread: list[float] = []
    policy_gross: list[torch.Tensor] = []
    benchmark_gross: list[torch.Tensor] = []
    policy_turnover: list[torch.Tensor] = []
    benchmark_turnover: list[torch.Tensor] = []
    active_mass: list[float] = []
    signal_retention: list[torch.Tensor] = []
    risk_retention: list[torch.Tensor] = []
    requested_trace: list[torch.Tensor] = []
    executed_trace: list[torch.Tensor] = []

    for row_index, local_origin_tensor in enumerate(local_origins):
        local_origin = int(local_origin_tensor)
        global_origin = int(batch.origin_indices[row_index])
        operator = batch.action_residual_operators[row_index]
        target_operator = batch.target_residual_operators[row_index]
        executable_score = objective.predicted_mean[row_index]
        raw_mean = batch.raw_predicted_mean[row_index]
        valid = objective.valid[row_index]
        observed_mean = executable_score[valid]
        observed_target = objective.target_log_return[row_index, valid]
        if observed_mean.numel() < 2:
            raise M03RV14RuntimeError("v14 qualification date has no diagnostic support")
        date_ic.append(_spearman(observed_mean, observed_target))
        order = torch.argsort(observed_mean, stable=True)
        tail = max(1, order.numel() // 10)
        date_spread.append(
            float(
                observed_target.index_select(0, order[-tail:]).mean()
                - observed_target.index_select(0, order[:tail]).mean()
            )
        )
        fill = local_origin + 1
        benchmark = sequence.benchmark_weights[fill]
        fill_mask = sequence.fill_membership[fill] & sequence.fill_availability[fill]
        trade_mask = fill_mask & operator.qualified_asset_mask.to(
            device=fill_mask.device
        ).unsqueeze(0)
        requested, mass = _fixed_rank_target(
            benchmark.squeeze(0),
            sequence.risk_asset_caps[fill].squeeze(0),
            executable_score,
            trade_mask.squeeze(0),
        )
        projection = project_m03r_v9_active_book(
            requested.to(dtype=benchmark.dtype).unsqueeze(0),
            benchmark,
            fill_mask,
            sequence.risk_asset_caps[fill],
            sequence.risk_gross_max[fill],
            risk_state,
            origin_state_index=global_origin,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=loaded.asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        executed = projection.projected_weights
        policy_turnover.append(0.5 * (executed - current_policy).abs().sum())
        benchmark_turnover.append(0.5 * (benchmark - current_benchmark).abs().sum())
        returns = sequence.asset_returns[fill]
        policy_return = (executed * returns).sum()
        benchmark_return = (benchmark * returns).sum()
        if float(policy_return) <= -1.0 or float(benchmark_return) <= -1.0:
            raise M03RV14RuntimeError("v14 qualification chronology lost all value")
        policy_gross.append(policy_return)
        benchmark_gross.append(benchmark_return)
        current_policy = executed * (1.0 + returns) / (1.0 + policy_return)
        current_benchmark = benchmark * (1.0 + returns) / (1.0 + benchmark_return)
        raw_norm = torch.linalg.vector_norm(
            raw_mean[trade_mask.squeeze(0)].to(torch.float64)
        )
        residual_norm = torch.linalg.vector_norm(
            executable_score[trade_mask.squeeze(0)].to(torch.float64)
        )
        signal_retention.append(
            torch.ones((), device=executable_score.device, dtype=torch.float64)
            if float(raw_norm) <= 1.0e-14 and float(residual_norm) <= 1.0e-14
            else residual_norm / raw_norm.clamp_min(1.0e-14)
        )
        risk_retention.append(projection.requested_to_executed_retention.squeeze(0))
        active_mass.append(mass)
        requested_trace.append(requested)
        executed_trace.append(executed.squeeze(0))
        # Ensure diagnostics may use future-label validity, while execution
        # remains bound to the distinct origin-only operator.
        if target_operator.receipt_sha256 == operator.receipt_sha256 and not torch.equal(
            target_operator.qualified_asset_mask,
            operator.qualified_asset_mask,
        ):
            raise M03RV14RuntimeError("v14 residual operator identity collision")

    policy_gross_tensor = torch.stack(policy_gross)
    benchmark_gross_tensor = torch.stack(benchmark_gross)
    policy_turnover_tensor = torch.stack(policy_turnover)
    benchmark_turnover_tensor = torch.stack(benchmark_turnover)
    gross_active = policy_gross_tensor - benchmark_gross_tensor
    incremental_turnover = policy_turnover_tensor - benchmark_turnover_tensor
    net_by_cost = tuple(
        gross_active - (basis_points / 10_000.0) * incremental_turnover
        for basis_points in M03R_V14_PREDICTIVE_SPEC.evaluation_cost_basis_points
    )
    scale = torch.exp(objective.predicted_log_scale)[
        torch.stack(
            tuple(operator.qualified_asset_mask for operator in batch.action_residual_operators)
        )
    ].to(torch.float64)
    if scale.numel() == 0:
        raise M03RV14RuntimeError("v14 qualification has no selected scale values")
    scale_quantiles = tuple(
        float(value)
        for value in _scale_quantiles_tensor(scale)
    )
    arrays = tuple(
        row.detach().to(device="cpu", dtype=torch.float64).clone()
        for row in (
            batch.origin_indices,
            torch.tensor(date_ic, dtype=torch.float64),
            torch.tensor(date_spread, dtype=torch.float64),
            policy_gross_tensor,
            benchmark_gross_tensor,
            policy_turnover_tensor,
            benchmark_turnover_tensor,
            torch.tensor(active_mass, dtype=torch.float64),
            torch.stack(signal_retention),
            torch.stack(risk_retention),
            torch.stack(requested_trace),
            torch.stack(executed_trace),
            *net_by_cost,
        )
    )
    arrays = (batch.origin_indices.detach().to(device="cpu", dtype=torch.int64), *arrays[1:])
    cap = M03R_V14_PREDICTIVE_SPEC.simple_sleeve_maximum_active_one_way_mass
    active_mass_tensor = arrays[7]
    provisional = M03RV14SimpleSleeveTrace(
        setting_index=loaded.setting_index,
        setting_id=loaded.setting_id,
        fold_index=loaded.fold_index,
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        checkpoint_model_state_sha256=loaded.model_state_sha256,
        qualification_batch_receipt_sha256=batch.receipt_sha256,
        qualification_source_array_sha256=batch.source_array_sha256,
        asset_axis_sha256=batch.asset_axis_sha256,
        risk_state_sha256=risk_state.state_sha256,
        risk_manifest_sha256=risk_state.manifest_sha256,
        action_operator_receipt_sha256=tuple(
            row.receipt_sha256 for row in batch.action_residual_operators
        ),
        origin_indices=arrays[0],
        date_spearman_ic=arrays[1],
        date_top_bottom_spread=arrays[2],
        policy_gross_returns=arrays[3],
        benchmark_gross_returns=arrays[4],
        policy_one_way_turnover=arrays[5],
        benchmark_one_way_turnover=arrays[6],
        active_mass=arrays[7],
        signal_projection_retention=arrays[8],
        requested_to_executed_retention=arrays[9],
        requested_weight_trace=arrays[10],
        executed_weight_trace=arrays[11],
        net_active_return_by_cost=tuple(arrays[12:]),
        selected_scale_quantiles=scale_quantiles,
        no_action_fraction=float((active_mass_tensor <= 1.0e-14).float().mean()),
        active_mass_cap_fraction=float(
            (active_mass_tensor >= cap - 1.0e-12).float().mean()
        ),
        array_sha256=tuple(_tensor_sha256(row) for row in arrays),
        trace_sha256="0" * 64,
    )
    result = replace(
        provisional,
        trace_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V14_SIMPLE_SLEEVE_TRACE_SCHEMA",
    "M03RV14RuntimeError",
    "M03RV14SimpleSleeveTrace",
    "run_m03r_v14_simple_sleeve",
]
