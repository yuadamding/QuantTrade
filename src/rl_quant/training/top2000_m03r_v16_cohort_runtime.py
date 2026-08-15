"""Horizon-matched staggered-cohort economics for M03R-v16.

Every qualification decision creates one small rank cohort.  Fixed-horizon
cohorts earn exactly 21 or 30 post-fill returns; the Hold-30-primary cohort is
released with the frozen age-clock hazard.  The final decision receives its
entire declared return path and all remaining active risk is liquidated and
charged at the terminal boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_COHORT_SLEEVE_RULE,
    M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
    M03RV16PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    project_m03r_v9_active_book,
)

M03R_V16_COHORT_TRACE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-horizon-matched-cohort-trace-v1"
)


class M03RV16CohortRuntimeError(ValueError):
    """The V16 cohort chronology, risk projection, or accounting drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical(tuple(tensor.shape)))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise M03RV16CohortRuntimeError(f"{name} must be a lowercase SHA-256")
    return value


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty_like(sorted_values, dtype=torch.float64)
    start = 0
    while start < sorted_values.numel():
        stop = start + 1
        while stop < sorted_values.numel() and bool(
            sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    result = torch.empty_like(ranks)
    result[order] = ranks
    return result


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


def _new_rank_cohort(
    anchor: torch.Tensor,
    caps: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
    *,
    cash_index: int,
    one_way_mass: float,
) -> torch.Tensor:
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    result = torch.zeros_like(anchor, dtype=torch.float64)
    if selected.numel() < 2 or one_way_mass <= 0.0:
        return result
    observed = score.index_select(0, selected).to(torch.float64)
    if float(observed.max() - observed.min()) <= 0.0:
        return result
    ranks = _average_ranks(observed)
    centered = ranks - 0.5 * (float(ranks.min()) + float(ranks.max()))
    positive = torch.zeros_like(result)
    negative = torch.zeros_like(result)
    positive[selected] = centered.clamp_min(0.0)
    negative[selected] = (-centered).clamp_min(0.0)
    buy_capacity = (caps.to(torch.float64) - anchor.to(torch.float64)).clamp_min(0.0)
    sell_capacity = anchor.to(torch.float64).clamp_min(0.0)
    buy_capacity[~mask] = 0.0
    sell_capacity[~mask] = 0.0
    buy_capacity[cash_index] = 0.0
    sell_capacity[cash_index] = 0.0
    mass = min(
        one_way_mass,
        float(buy_capacity[positive > 0.0].sum()),
        float(sell_capacity[negative > 0.0].sum()),
    )
    if mass <= 1.0e-14:
        return result
    buys = _capped_proportional(positive, buy_capacity, mass)
    sells = _capped_proportional(negative, sell_capacity, mass)
    common = min(float(buys.sum()), float(sells.sum()))
    if common <= 1.0e-14:
        return result
    result = buys * (common / buys.sum()) - sells * (common / sells.sum())
    result[cash_index] = -result.sum()
    if not math.isclose(float(result.sum()), 0.0, abs_tol=2.0e-12):
        raise M03RV16CohortRuntimeError("V16 cohort is not self-financing")
    return result


def _survival_lifetime() -> float:
    survival = 1.0
    result = 0.0
    for hazard in M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60:
        result += survival
        survival *= 1.0 - hazard
    return result


@dataclass(slots=True)
class _LiveCohort:
    active_weights: torch.Tensor
    age: int = 0


@dataclass(frozen=True, slots=True)
class M03RV16CohortTrace:
    setting_index: int
    setting_id: str
    fold_index: int
    checkpoint_file_sha256: str
    checkpoint_model_state_sha256: str
    qualification_batch_receipt_sha256: str
    asset_axis_sha256: str
    risk_manifest_sha256: str
    risk_state_sha256: str
    decision_origin_indices: torch.Tensor
    execution_origin_indices: torch.Tensor
    policy_gross_returns: torch.Tensor
    benchmark_gross_returns: torch.Tensor
    policy_one_way_turnover: torch.Tensor
    benchmark_one_way_turnover: torch.Tensor
    active_one_way_mass: torch.Tensor
    cohort_entry_one_way_mass: torch.Tensor
    cohort_release_one_way_mass: torch.Tensor
    weighted_mean_cohort_age: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    absolute_policy_cost_by_cost: tuple[torch.Tensor, ...]
    incremental_active_cost_by_cost: tuple[torch.Tensor, ...]
    net_active_return_by_cost: tuple[torch.Tensor, ...]
    terminal_liquidation_one_way_turnover: float
    terminal_preliquidation_active_one_way_mass: float
    array_sha256: tuple[str, ...]
    trace_sha256: str
    cohort_rule: str = M03R_V16_COHORT_SLEEVE_RULE
    action_mask_uses_future_availability: bool = False
    final_decision_receives_full_horizon: bool = True
    terminal_active_risk_liquidated: bool = True
    learned_hazard_enabled: bool = False
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_COHORT_TRACE_SCHEMA

    @property
    def arrays(self) -> tuple[torch.Tensor, ...]:
        return (
            self.decision_origin_indices,
            self.execution_origin_indices,
            self.policy_gross_returns,
            self.benchmark_gross_returns,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.active_one_way_mass,
            self.cohort_entry_one_way_mass,
            self.cohort_release_one_way_mass,
            self.weighted_mean_cohort_age,
            self.requested_to_executed_retention,
            *self.absolute_policy_cost_by_cost,
            *self.incremental_active_cost_by_cost,
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
            "asset_axis_sha256": self.asset_axis_sha256,
            "risk_manifest_sha256": self.risk_manifest_sha256,
            "risk_state_sha256": self.risk_state_sha256,
            "terminal_liquidation_one_way_turnover": (
                self.terminal_liquidation_one_way_turnover
            ),
            "terminal_preliquidation_active_one_way_mass": (
                self.terminal_preliquidation_active_one_way_mass
            ),
            "array_sha256": self.array_sha256,
            "cohort_rule": self.cohort_rule,
            "action_mask_uses_future_availability": (
                self.action_mask_uses_future_availability
            ),
            "final_decision_receives_full_horizon": (
                self.final_decision_receives_full_horizon
            ),
            "terminal_active_risk_liquidated": (self.terminal_active_risk_liquidated),
            "learned_hazard_enabled": self.learned_hazard_enabled,
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "reinforcement_learning_updates": self.reinforcement_learning_updates,
            "outer_2026_accessed": self.outer_2026_accessed,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        spec = M03R_V16_PREDICTIVE_SPEC
        decisions = spec.qualification_origins_per_fold
        steps = decisions + spec.cohort_no_new_decision_tail_sessions
        one_dimensional = self.arrays[2:]
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.setting_id != M03R_V16_SETTINGS[self.setting_index].setting_id
            or self.fold_index not in range(spec.chronological_fold_count)
            or self.decision_origin_indices.shape != (decisions,)
            or self.execution_origin_indices.shape != (steps,)
            or self.decision_origin_indices.dtype != torch.int64
            or self.execution_origin_indices.dtype != torch.int64
            or not torch.equal(
                self.execution_origin_indices[:decisions],
                self.decision_origin_indices,
            )
            or bool(
                (
                    self.execution_origin_indices[1:]
                    != self.execution_origin_indices[:-1] + 1
                ).any()
            )
            or any(
                value.shape != (steps,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in one_dimensional
            )
            or len(self.absolute_policy_cost_by_cost)
            != len(spec.evaluation_cost_basis_points)
            or len(self.incremental_active_cost_by_cost)
            != len(spec.evaluation_cost_basis_points)
            or len(self.net_active_return_by_cost)
            != len(spec.evaluation_cost_basis_points)
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.active_one_way_mass < 0.0).any())
            or bool((self.cohort_entry_one_way_mass < 0.0).any())
            or bool((self.cohort_release_one_way_mass < 0.0).any())
            or bool((self.requested_to_executed_retention < 0.0).any())
            or not math.isfinite(self.terminal_liquidation_one_way_turnover)
            or self.terminal_liquidation_one_way_turnover < 0.0
            or not math.isfinite(self.terminal_preliquidation_active_one_way_mass)
            or self.terminal_preliquidation_active_one_way_mass < 0.0
            or self.array_sha256 != tuple(_tensor_sha256(row) for row in self.arrays)
            or self.cohort_rule != M03R_V16_COHORT_SLEEVE_RULE
            or self.action_mask_uses_future_availability
            or not self.final_decision_receives_full_horizon
            or not self.terminal_active_risk_liquidated
            or self.learned_hazard_enabled
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_COHORT_TRACE_SCHEMA
            or self.trace_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV16CohortRuntimeError("V16 cohort trace drifted")
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_model_state_sha256", self.checkpoint_model_state_sha256),
            (
                "qualification_batch_receipt_sha256",
                self.qualification_batch_receipt_sha256,
            ),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("risk_manifest_sha256", self.risk_manifest_sha256),
            ("risk_state_sha256", self.risk_state_sha256),
        ):
            _digest(name, value)

    @property
    def gross_active_returns(self) -> torch.Tensor:
        self.validate()
        return self.policy_gross_returns - self.benchmark_gross_returns

    def aggregate_break_even_one_way_cost_basis_points(self) -> float:
        self.validate()
        gross = float(self.gross_active_returns.sum())
        incremental_turnover = float(
            (self.policy_one_way_turnover - self.benchmark_one_way_turnover).sum()
        )
        if gross <= 0.0:
            return 0.0
        if incremental_turnover <= 0.0:
            return math.inf
        return 10_000.0 * gross / incremental_turnover


def _drift(
    weights: torch.Tensor, returns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gross = 1.0 + torch.dot(weights, returns)
    if not bool(torch.isfinite(gross)) or float(gross) <= 0.0:
        raise M03RV16CohortRuntimeError("V16 cohort book has invalid gross return")
    return weights * (1.0 + returns) / gross, gross - 1.0


def run_m03r_v16_horizon_matched_cohort_sleeve(
    setting: M03RV16PredictiveSetting,
    *,
    fold_index: int,
    checkpoint_file_sha256: str,
    checkpoint_model_state_sha256: str,
    qualification_batch_receipt_sha256: str,
    asset_axis_sha256: str,
    decision_origin_indices: torch.Tensor,
    executable_selection_scores: torch.Tensor,
    selection_valid: torch.Tensor,
    post_fill_asset_returns: torch.Tensor,
    benchmark_weights: torch.Tensor,
    fill_available: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    risk_state: M03RV9DeviceRiskState,
) -> M03RV16CohortTrace:
    """Run one closed cohort path from a reloaded predictive checkpoint."""

    setting.__post_init__()
    risk_state.validate()
    spec = M03R_V16_PREDICTIVE_SPEC
    decisions = spec.qualification_origins_per_fold
    steps = decisions + spec.cohort_no_new_decision_tail_sessions
    if (
        fold_index not in range(spec.chronological_fold_count)
        or decision_origin_indices.shape != (decisions,)
        or decision_origin_indices.dtype != torch.int64
        or executable_selection_scores.ndim != 2
        or executable_selection_scores.shape[0] != decisions
        or selection_valid.shape != executable_selection_scores.shape
        or selection_valid.dtype != torch.bool
    ):
        raise M03RV16CohortRuntimeError("V16 cohort decision inputs drifted")
    assets = executable_selection_scores.shape[1]
    execution_origins = torch.arange(
        int(decision_origin_indices[0]),
        int(decision_origin_indices[0]) + steps,
        dtype=torch.int64,
        device=decision_origin_indices.device,
    )
    expected_shape = (steps, assets)
    if (
        not torch.equal(execution_origins[:decisions], decision_origin_indices)
        or tuple(risk_state.origin_state_indices)
        != tuple(int(value) for value in execution_origins)
        or risk_state.asset_count != assets
        or risk_state.asset_axis_sha256 != asset_axis_sha256
        or any(
            not isinstance(value, torch.Tensor)
            or value.shape != expected_shape
            or value.device != executable_selection_scores.device
            or not bool(torch.isfinite(value).all())
            for value in (
                post_fill_asset_returns,
                benchmark_weights,
                risk_asset_caps,
            )
        )
        or fill_available.shape != expected_shape
        or fill_available.dtype != torch.bool
        or fill_available.device != executable_selection_scores.device
        or risk_gross_max.shape != (steps,)
        or risk_gross_max.device != executable_selection_scores.device
        or not bool(torch.isfinite(risk_gross_max).all())
        or bool((post_fill_asset_returns <= -1.0).any())
    ):
        raise M03RV16CohortRuntimeError("V16 cohort execution tensors drifted")
    for name, value in (
        ("checkpoint_file_sha256", checkpoint_file_sha256),
        ("checkpoint_model_state_sha256", checkpoint_model_state_sha256),
        ("qualification_batch_receipt_sha256", qualification_batch_receipt_sha256),
        ("asset_axis_sha256", asset_axis_sha256),
    ):
        _digest(name, value)

    if setting.setting_index == 0:
        entry_mass = spec.cohort_total_active_one_way_mass / 21.0
        fixed_horizon: int | None = 21
    elif setting.setting_index == 1:
        entry_mass = spec.cohort_total_active_one_way_mass / 30.0
        fixed_horizon = 30
    else:
        entry_mass = spec.cohort_total_active_one_way_mass / _survival_lifetime()
        fixed_horizon = None

    cash = risk_state.cash_index
    current_policy = benchmark_weights[0].clone()
    current_benchmark = current_policy.clone()
    cohorts: list[_LiveCohort] = []
    policy_returns: list[torch.Tensor] = []
    benchmark_returns: list[torch.Tensor] = []
    policy_turnover: list[torch.Tensor] = []
    benchmark_turnover: list[torch.Tensor] = []
    active_mass_rows: list[torch.Tensor] = []
    entry_rows: list[torch.Tensor] = []
    release_rows: list[torch.Tensor] = []
    mean_age_rows: list[torch.Tensor] = []
    retention_rows: list[torch.Tensor] = []

    for step in range(steps):
        benchmark = benchmark_weights[step]
        existing = torch.zeros(assets, dtype=torch.float64, device=benchmark.device)
        for cohort in cohorts:
            existing += cohort.active_weights
        anchor = benchmark.to(torch.float64) + existing
        entry = torch.zeros_like(existing)
        if step < decisions:
            mask = selection_valid[step] & fill_available[step]
            mask = mask.clone()
            mask[cash] = False
            entry = _new_rank_cohort(
                anchor,
                risk_asset_caps[step],
                executable_selection_scores[step],
                mask,
                cash_index=cash,
                one_way_mass=entry_mass,
            )
            if float(entry.abs().sum()) > 0.0:
                cohorts.append(_LiveCohort(entry))
        requested = (anchor + entry).to(benchmark.dtype).unsqueeze(0)
        projection = project_m03r_v9_active_book(
            requested,
            benchmark.unsqueeze(0),
            fill_available[step].unsqueeze(0),
            risk_asset_caps[step].unsqueeze(0),
            risk_gross_max[step].reshape(1),
            risk_state,
            origin_state_index=int(execution_origins[step]),
            sequence_asset_axis_sha256=asset_axis_sha256,
            checkpoint_asset_axis_sha256=asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        executed = projection.projected_weights.squeeze(0)
        policy_turnover.append(0.5 * (executed - current_policy).abs().sum())
        benchmark_turnover.append(0.5 * (benchmark - current_benchmark).abs().sum())
        current_policy, policy_return = _drift(executed, post_fill_asset_returns[step])
        current_benchmark, benchmark_return = _drift(
            benchmark, post_fill_asset_returns[step]
        )
        policy_returns.append(policy_return)
        benchmark_returns.append(benchmark_return)
        active_mass_rows.append(
            0.5 * (executed - benchmark).to(torch.float64).abs().sum()
        )
        entry_rows.append(0.5 * entry.abs().sum())
        retention_rows.append(projection.requested_to_executed_retention.squeeze(0))

        released = 0.0
        retained: list[_LiveCohort] = []
        for cohort in cohorts:
            cohort.age += 1
            before = cohort.active_weights
            if fixed_horizon is not None:
                multiplier = 0.0 if cohort.age >= fixed_horizon else 1.0
            else:
                hazard_index = (
                    min(
                        cohort.age,
                        len(M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60),
                    )
                    - 1
                )
                multiplier = (
                    1.0 - M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60[hazard_index]
                )
            cohort.active_weights = before * multiplier
            released += 0.5 * float((before - cohort.active_weights).abs().sum())
            if float(cohort.active_weights.abs().sum()) > 1.0e-14:
                retained.append(cohort)
        cohorts = retained
        release_rows.append(executed.new_tensor(released))
        masses = [0.5 * float(value.active_weights.abs().sum()) for value in cohorts]
        total_mass = math.fsum(masses)
        mean_age = (
            math.fsum(
                mass * cohort.age for mass, cohort in zip(masses, cohorts, strict=True)
            )
            / total_mass
            if total_mass > 0.0
            else 0.0
        )
        mean_age_rows.append(executed.new_tensor(mean_age))

    terminal_active = 0.5 * float((current_policy - current_benchmark).abs().sum())
    terminal_turnover = terminal_active
    policy_turnover[-1] = policy_turnover[-1] + policy_turnover[-1].new_tensor(
        terminal_turnover
    )
    policy_return_tensor = torch.stack(policy_returns)
    benchmark_return_tensor = torch.stack(benchmark_returns)
    policy_turnover_tensor = torch.stack(policy_turnover)
    benchmark_turnover_tensor = torch.stack(benchmark_turnover)
    gross_active = policy_return_tensor - benchmark_return_tensor
    absolute_costs = tuple(
        policy_turnover_tensor * (cost / 10_000.0)
        for cost in spec.evaluation_cost_basis_points
    )
    incremental_costs = tuple(
        (policy_turnover_tensor - benchmark_turnover_tensor) * (cost / 10_000.0)
        for cost in spec.evaluation_cost_basis_points
    )
    net_active = tuple(gross_active - value for value in incremental_costs)
    provisional = M03RV16CohortTrace(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        fold_index=fold_index,
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_model_state_sha256=checkpoint_model_state_sha256,
        qualification_batch_receipt_sha256=qualification_batch_receipt_sha256,
        asset_axis_sha256=asset_axis_sha256,
        risk_manifest_sha256=risk_state.manifest_sha256,
        risk_state_sha256=risk_state.state_sha256,
        decision_origin_indices=decision_origin_indices.detach().clone(),
        execution_origin_indices=execution_origins,
        policy_gross_returns=policy_return_tensor,
        benchmark_gross_returns=benchmark_return_tensor,
        policy_one_way_turnover=policy_turnover_tensor,
        benchmark_one_way_turnover=benchmark_turnover_tensor,
        active_one_way_mass=torch.stack(active_mass_rows),
        cohort_entry_one_way_mass=torch.stack(entry_rows),
        cohort_release_one_way_mass=torch.stack(release_rows),
        weighted_mean_cohort_age=torch.stack(mean_age_rows),
        requested_to_executed_retention=torch.stack(retention_rows),
        absolute_policy_cost_by_cost=absolute_costs,
        incremental_active_cost_by_cost=incremental_costs,
        net_active_return_by_cost=net_active,
        terminal_liquidation_one_way_turnover=terminal_turnover,
        terminal_preliquidation_active_one_way_mass=terminal_active,
        array_sha256=(),
        trace_sha256="0" * 64,
    )
    arrays = provisional.arrays
    result = replace(
        provisional,
        array_sha256=tuple(_tensor_sha256(row) for row in arrays),
    )
    result = replace(result, trace_sha256=_sha256(result.unsigned_payload()))
    result.validate()
    return result


__all__ = [
    "M03R_V16_COHORT_TRACE_SCHEMA",
    "M03RV16CohortRuntimeError",
    "M03RV16CohortTrace",
    "run_m03r_v16_horizon_matched_cohort_sleeve",
]
