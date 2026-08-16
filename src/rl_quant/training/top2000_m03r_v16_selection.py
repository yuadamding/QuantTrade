"""Fold-preserving inference and the primary-R2 stop gate for M03R-v16."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTING_IDS,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256 as _sha256
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
)
from rl_quant.training.top2000_m03r_v16_qualification_runtime import (
    M03RV16FoldQualificationResult,
)

M03R_V16_BOOTSTRAP_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v16-bootstrap-plan-v2"
M03R_V16_QUALIFICATION_SCHEMA = "rl-quant.top2000-dev.m03r-v16-qualification-v2"
M03R_V16_PANEL_DECISION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-panel-decision-v2"
)
M03R_V16_PANEL_DECISION_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-panel-decision-file-v2"
)
_MAX_PANEL_DECISION_BYTES = 1024**2
M03R_V16_BOOTSTRAP_BLOCK_SESSIONS = (
    M03R_V16_PREDICTIVE_SPEC.bootstrap_primary_block_sessions,
    *M03R_V16_PREDICTIVE_SPEC.bootstrap_sensitivity_block_sessions,
)


class M03RV16SelectionError(ValueError):
    """The V16 chronology, inference, or primary-hypothesis gate drifted."""


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
        raise M03RV16SelectionError(f"{name} must be a lowercase SHA-256")
    return value


def _draw_indices(
    fold_lengths: tuple[int, ...],
    *,
    block_sessions: int,
    stream: int,
) -> torch.Tensor:
    if not fold_lengths or len(set(fold_lengths)) != 1:
        raise M03RV16SelectionError(
            "V16 hierarchical bootstrap requires equal nonempty fold lengths"
        )
    fold_length = fold_lengths[0]
    if block_sessions <= 0 or block_sessions > fold_length:
        raise M03RV16SelectionError("V16 bootstrap block exceeds its source fold")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        M03R_V16_PREDICTIVE_SPEC.bootstrap_seed
        + block_sessions * 1_000_003
        + stream * 10_000_019
    )
    replicates = M03R_V16_PREDICTIVE_SPEC.bootstrap_replicates
    fold_count = len(fold_lengths)
    source_folds = torch.randint(
        fold_count,
        (replicates, fold_count),
        generator=generator,
        dtype=torch.int64,
    )
    blocks = (fold_length + block_sessions - 1) // block_sessions
    starts = torch.randint(
        fold_length - block_sessions + 1,
        (replicates, fold_count, blocks),
        generator=generator,
        dtype=torch.int64,
    )
    offsets = torch.arange(block_sessions, dtype=torch.int64)
    local = (starts.unsqueeze(-1) + offsets).reshape(
        replicates, fold_count, -1
    )[:, :, :fold_length]
    selected = local + source_folds.unsqueeze(-1) * fold_length
    return selected.reshape(replicates, fold_count * fold_length)


@dataclass(frozen=True, slots=True)
class M03RV16BootstrapPlan:
    decision_chronology_sha256: str
    execution_chronology_sha256: str
    decision_fold_lengths: tuple[int, ...]
    execution_fold_lengths: tuple[int, ...]
    diagnostic_draw_sha256_by_block: tuple[str, str, str]
    economic_draw_sha256_by_block: tuple[str, str, str]
    block_sessions: tuple[int, int, int] = M03R_V16_BOOTSTRAP_BLOCK_SESSIONS
    replicates: int = M03R_V16_PREDICTIVE_SPEC.bootstrap_replicates
    bootstrap_seed: int = M03R_V16_PREDICTIVE_SPEC.bootstrap_seed
    fold_boundaries_preserved: bool = True
    fold_clusters_resampled: bool = True
    within_fold_blocks_nonwrapping: bool = True
    circular_blocks_used: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_BOOTSTRAP_PLAN_SCHEMA

    def validate(self) -> None:
        decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
        executions = (
            decisions
            + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
        )
        expected_decision = (decisions,) * M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
        expected_execution = (executions,) * M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
        if (
            self.decision_fold_lengths != expected_decision
            or self.execution_fold_lengths != expected_execution
            or self.block_sessions != M03R_V16_BOOTSTRAP_BLOCK_SESSIONS
            or self.replicates != M03R_V16_PREDICTIVE_SPEC.bootstrap_replicates
            or self.bootstrap_seed != M03R_V16_PREDICTIVE_SPEC.bootstrap_seed
            or self.diagnostic_draw_sha256_by_block
            != tuple(
                _tensor_sha256(
                    _draw_indices(
                        self.decision_fold_lengths,
                        block_sessions=block,
                        stream=1,
                    )
                )
                for block in self.block_sessions
            )
            or self.economic_draw_sha256_by_block
            != tuple(
                _tensor_sha256(
                    _draw_indices(
                        self.execution_fold_lengths,
                        block_sessions=block,
                        stream=2,
                    )
                )
                for block in self.block_sessions
            )
            or not self.fold_boundaries_preserved
            or not self.fold_clusters_resampled
            or not self.within_fold_blocks_nonwrapping
            or self.circular_blocks_used
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_BOOTSTRAP_PLAN_SCHEMA
        ):
            raise M03RV16SelectionError("V16 bootstrap plan drifted")
        _digest("decision_chronology_sha256", self.decision_chronology_sha256)
        _digest("execution_chronology_sha256", self.execution_chronology_sha256)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _ordered_contiguous(rows: tuple[torch.Tensor, ...], *, expected: int) -> list[int]:
    chronology: list[int] = []
    previous_stop: int | None = None
    for row in rows:
        if (
            not isinstance(row, torch.Tensor)
            or row.ndim != 1
            or row.dtype != torch.int64
            or row.numel() != expected
            or bool((row[1:] != row[:-1] + 1).any())
            or (previous_stop is not None and int(row[0]) <= previous_stop)
        ):
            raise M03RV16SelectionError(
                "V16 fold chronology is not ordered and disjoint"
            )
        chronology.extend(int(value) for value in row)
        previous_stop = int(row[-1])
    return chronology


def build_m03r_v16_bootstrap_plan(
    decision_origins_by_fold: tuple[torch.Tensor, ...],
    execution_origins_by_fold: tuple[torch.Tensor, ...],
) -> M03RV16BootstrapPlan:
    folds = M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
    if len(decision_origins_by_fold) != folds or len(execution_origins_by_fold) != folds:
        raise M03RV16SelectionError("V16 bootstrap requires five complete folds")
    decisions = _ordered_contiguous(
        decision_origins_by_fold,
        expected=M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold,
    )
    executions = _ordered_contiguous(
        execution_origins_by_fold,
        expected=(
            M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
            + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
        ),
    )
    decision_lengths = tuple(int(row.numel()) for row in decision_origins_by_fold)
    execution_lengths = tuple(int(row.numel()) for row in execution_origins_by_fold)
    diagnostic = tuple(
        _tensor_sha256(
            _draw_indices(decision_lengths, block_sessions=block, stream=1)
        )
        for block in M03R_V16_BOOTSTRAP_BLOCK_SESSIONS
    )
    economic = tuple(
        _tensor_sha256(
            _draw_indices(execution_lengths, block_sessions=block, stream=2)
        )
        for block in M03R_V16_BOOTSTRAP_BLOCK_SESSIONS
    )
    result = M03RV16BootstrapPlan(
        decision_chronology_sha256=_sha256(decisions),
        execution_chronology_sha256=_sha256(executions),
        decision_fold_lengths=decision_lengths,
        execution_fold_lengths=execution_lengths,
        diagnostic_draw_sha256_by_block=(diagnostic[0], diagnostic[1], diagnostic[2]),
        economic_draw_sha256_by_block=(economic[0], economic[1], economic[2]),
    )
    result.validate()
    return result


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


def _predictive_diagnostics(
    score: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    date_ic: list[float] = []
    date_spread: list[float] = []
    for row in range(score.shape[0]):
        observed_score = score[row, valid[row]]
        observed_target = target[row, valid[row]]
        if observed_score.numel() < 2:
            raise M03RV16SelectionError("V16 diagnostic date has no support")
        date_ic.append(_spearman(observed_score, observed_target))
        order = torch.argsort(observed_score, stable=True)
        tail = max(1, order.numel() // 10)
        date_spread.append(
            float(
                observed_target.index_select(0, order[-tail:]).mean()
                - observed_target.index_select(0, order[:tail]).mean()
            )
        )
    return (
        torch.tensor(date_ic, dtype=torch.float64),
        torch.tensor(date_spread, dtype=torch.float64),
    )


@dataclass(frozen=True, slots=True)
class M03RV16ReconciledFoldEvidence:
    """Minimal immutable fold evidence used by the independent aggregator."""

    trace: M03RV16CohortTrace
    executable_selection_mean: torch.Tensor
    selection_target_economic: torch.Tensor
    selection_valid: torch.Tensor
    terminal_checkpoint_authority_sha256: str
    qualified_score_authority_sha256: str
    panel_schedule_sha256: str

    def validate(self) -> None:
        self.trace.validate()
        score = self.executable_selection_mean
        target = self.selection_target_economic
        valid = self.selection_valid
        decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
        if (
            score.ndim != 2
            or target.shape != score.shape
            or valid.shape != score.shape
            or valid.dtype != torch.bool
            or score.shape[0] != decisions
            or not score.is_floating_point()
            or not target.is_floating_point()
            or not bool(torch.isfinite(score).all())
            or not bool(torch.isfinite(target).all())
            or self.trace.terminal_checkpoint_authority_sha256
            != self.terminal_checkpoint_authority_sha256
            or self.trace.qualified_score_authority_sha256
            != self.qualified_score_authority_sha256
            or self.trace.panel_schedule_sha256 != self.panel_schedule_sha256
            or self.trace.diagnostic_valid_sha256 != _tensor_sha256(valid)
        ):
            raise M03RV16SelectionError("V16 reconciled fold evidence drifted")
        for name, value in (
            (
                "terminal_checkpoint_authority_sha256",
                self.terminal_checkpoint_authority_sha256,
            ),
            (
                "qualified_score_authority_sha256",
                self.qualified_score_authority_sha256,
            ),
            ("panel_schedule_sha256", self.panel_schedule_sha256),
        ):
            _digest(name, value)


def reconcile_m03r_v16_fold_result(
    result: M03RV16FoldQualificationResult,
) -> M03RV16ReconciledFoldEvidence:
    """Reduce an in-memory qualification authority to auditable CPU evidence."""

    result.validate()
    objective = result.score_authority.batch.objective
    trace = result.trace
    cpu_trace = replace(
        trace,
        decision_origin_indices=trace.decision_origin_indices.detach().cpu(),
        execution_origin_indices=trace.execution_origin_indices.detach().cpu(),
        policy_gross_returns=trace.policy_gross_returns.detach().cpu(),
        benchmark_gross_returns=trace.benchmark_gross_returns.detach().cpu(),
        policy_one_way_turnover=trace.policy_one_way_turnover.detach().cpu(),
        benchmark_one_way_turnover=trace.benchmark_one_way_turnover.detach().cpu(),
        active_one_way_mass=trace.active_one_way_mass.detach().cpu(),
        cohort_entry_one_way_mass=(
            trace.cohort_entry_one_way_mass.detach().cpu()
        ),
        signal_cohort_mass_reduction_after_execution=(
            trace.signal_cohort_mass_reduction_after_execution.detach().cpu()
        ),
        weighted_mean_cohort_age=trace.weighted_mean_cohort_age.detach().cpu(),
        requested_to_executed_retention=(
            trace.requested_to_executed_retention.detach().cpu()
        ),
        risk_repair_active_one_way_mass=(
            trace.risk_repair_active_one_way_mass.detach().cpu()
        ),
        prior_risk_repair_unwind_one_way_mass=(
            trace.prior_risk_repair_unwind_one_way_mass.detach().cpu()
        ),
        risk_projection_request_to_execution_one_way_distance=(
            trace.risk_projection_request_to_execution_one_way_distance.detach().cpu()
        ),
        absolute_policy_cost_by_cost=tuple(
            row.detach().cpu() for row in trace.absolute_policy_cost_by_cost
        ),
        benchmark_cost_by_cost=tuple(
            row.detach().cpu() for row in trace.benchmark_cost_by_cost
        ),
        incremental_active_cost_by_cost=tuple(
            row.detach().cpu() for row in trace.incremental_active_cost_by_cost
        ),
        net_policy_return_by_cost=tuple(
            row.detach().cpu() for row in trace.net_policy_return_by_cost
        ),
        net_benchmark_return_by_cost=tuple(
            row.detach().cpu() for row in trace.net_benchmark_return_by_cost
        ),
        net_active_return_by_cost=tuple(
            row.detach().cpu() for row in trace.net_active_return_by_cost
        ),
    )
    cpu_trace.validate()
    evidence = M03RV16ReconciledFoldEvidence(
        trace=cpu_trace,
        executable_selection_mean=(
            objective.executable_selection_mean.detach().cpu()
        ),
        selection_target_economic=(
            objective.selection_target_economic.detach().cpu()
        ),
        selection_valid=objective.selection_valid.detach().cpu(),
        terminal_checkpoint_authority_sha256=(
            result.terminal_checkpoint_authority.receipt_sha256
        ),
        qualified_score_authority_sha256=result.score_authority.receipt_sha256,
        panel_schedule_sha256=(
            result.terminal_checkpoint_authority.panel_schedule.receipt_sha256
        ),
    )
    evidence.validate()
    return evidence


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveQualification:
    setting_index: int
    setting_id: str
    fold_trace_sha256: tuple[str, ...]
    terminal_checkpoint_authority_sha256: tuple[str, ...]
    qualified_score_authority_sha256: tuple[str, ...]
    panel_schedule_sha256: str
    bootstrap_plan_sha256: str
    mean_projected_rank_ic: float
    positive_mean_ic_fold_count: int
    positive_spread_fold_count: int
    annualized_gross_active_return: float
    annualized_net_active_return_10bp: float
    gross_active_lcb_by_block: tuple[float, float, float]
    net_10bp_active_lcb_by_block: tuple[float, float, float]
    spread_lcb_by_block: tuple[float, float, float]
    break_even_category: Literal[
        "finite-positive",
        "favorable-cost-dominance",
        "no-positive-break-even",
    ]
    break_even_one_way_cost_basis_points: float | None
    absolute_policy_break_even_one_way_cost_basis_points: float
    median_risk_projection_retention: float
    minimum_fold_median_risk_projection_retention: float
    median_weighted_cohort_age: float
    gates_passed: bool
    primary_hypothesis_passed: bool
    three_seed_confirmation_may_be_minted: bool
    economic_generation_may_be_minted: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_QUALIFICATION_SCHEMA

    def _expected_gates(self) -> bool:
        spec = M03R_V16_PREDICTIVE_SPEC
        primary_block = 0
        return (
            self.mean_projected_rank_ic >= spec.minimum_mean_spearman_rank_ic
            and self.positive_mean_ic_fold_count
            >= spec.minimum_positive_mean_ic_fold_count
            and self.positive_spread_fold_count
            >= spec.minimum_positive_spread_fold_count
            and self.gross_active_lcb_by_block[primary_block]
            > spec.minimum_gross_active_return_lcb
            and self.net_10bp_active_lcb_by_block[primary_block]
            > spec.minimum_net_10bp_active_return_lcb
            and self.spread_lcb_by_block[primary_block] > spec.minimum_spread_lcb
            and (
                self.break_even_category == "favorable-cost-dominance"
                or (
                    self.break_even_one_way_cost_basis_points is not None
                    and self.break_even_one_way_cost_basis_points
                    >= spec.minimum_break_even_one_way_cost_basis_points
                )
            )
        )

    def validate(self) -> None:
        fold_count = M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
        finite = (
            self.mean_projected_rank_ic,
            self.annualized_gross_active_return,
            self.annualized_net_active_return_10bp,
            *self.gross_active_lcb_by_block,
            *self.net_10bp_active_lcb_by_block,
            *self.spread_lcb_by_block,
            self.absolute_policy_break_even_one_way_cost_basis_points,
            self.median_risk_projection_retention,
            self.minimum_fold_median_risk_projection_retention,
            self.median_weighted_cohort_age,
        )
        primary = self.setting_index == M03R_V16_PREDICTIVE_SPEC.primary_setting_index
        if (
            self.setting_index not in range(len(M03R_V16_SETTING_IDS))
            or self.setting_id != M03R_V16_SETTING_IDS[self.setting_index]
            or len(self.fold_trace_sha256) != fold_count
            or len(self.terminal_checkpoint_authority_sha256) != fold_count
            or len(self.qualified_score_authority_sha256) != fold_count
            or any(not math.isfinite(value) for value in finite)
            or any(
                value not in range(fold_count + 1)
                for value in (
                    self.positive_mean_ic_fold_count,
                    self.positive_spread_fold_count,
                )
            )
            or (
                self.break_even_category == "finite-positive"
                and (
                    self.break_even_one_way_cost_basis_points is None
                    or self.break_even_one_way_cost_basis_points <= 0.0
                    or not math.isfinite(self.break_even_one_way_cost_basis_points)
                )
            )
            or (
                self.break_even_category != "finite-positive"
                and self.break_even_one_way_cost_basis_points is not None
            )
            or self.gates_passed != self._expected_gates()
            or self.primary_hypothesis_passed != (primary and self.gates_passed)
            or self.three_seed_confirmation_may_be_minted
            != self.primary_hypothesis_passed
            or self.economic_generation_may_be_minted
            or self.reinforcement_learning_authorized
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_QUALIFICATION_SCHEMA
        ):
            raise M03RV16SelectionError("V16 predictive qualification drifted")
        for name, values in (
            ("fold_trace_sha256", self.fold_trace_sha256),
            (
                "terminal_checkpoint_authority_sha256",
                self.terminal_checkpoint_authority_sha256,
            ),
            (
                "qualified_score_authority_sha256",
                self.qualified_score_authority_sha256,
            ),
            ("panel_schedule_sha256", (self.panel_schedule_sha256,)),
            ("bootstrap_plan_sha256", (self.bootstrap_plan_sha256,)),
        ):
            for value in values:
                _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _lcb_by_block(
    value: torch.Tensor,
    *,
    fold_lengths: tuple[int, ...],
    stream: int,
    annualize: bool,
) -> tuple[float, float, float]:
    estimates: list[float] = []
    for block in M03R_V16_BOOTSTRAP_BLOCK_SESSIONS:
        draws = _draw_indices(
            fold_lengths,
            block_sessions=block,
            stream=stream,
        )
        distribution = value[draws].mean(dim=1)
        estimate = float(torch.quantile(distribution, 0.025))
        estimates.append(252.0 * estimate if annualize else estimate)
    return (estimates[0], estimates[1], estimates[2])


def qualify_m03r_v16_reconciled_evidence(
    evidence: tuple[M03RV16ReconciledFoldEvidence, ...],
    bootstrap: M03RV16BootstrapPlan,
) -> M03RV16PredictiveQualification:
    fold_count = M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
    if len(evidence) != fold_count:
        raise M03RV16SelectionError("V16 qualification requires five folds")
    ordered = tuple(
        sorted(evidence, key=lambda row: row.trace.fold_index)
    )
    for row in ordered:
        row.validate()
    bootstrap.validate()
    setting = ordered[0].trace.setting_index
    schedule = ordered[0].panel_schedule_sha256
    decisions = tuple(row.trace.decision_origin_indices for row in ordered)
    executions = tuple(row.trace.execution_origin_indices for row in ordered)
    if (
        tuple(row.trace.fold_index for row in ordered) != tuple(range(fold_count))
        or any(row.trace.setting_index != setting for row in ordered)
        or any(row.panel_schedule_sha256 != schedule for row in ordered)
        or bootstrap.decision_chronology_sha256
        != _sha256([int(value) for row in decisions for value in row])
        or bootstrap.execution_chronology_sha256
        != _sha256([int(value) for row in executions for value in row])
    ):
        raise M03RV16SelectionError("V16 fold authority or chronology drifted")

    diagnostic = tuple(
        _predictive_diagnostics(
            row.executable_selection_mean,
            row.selection_target_economic,
            row.selection_valid,
        )
        for row in ordered
    )
    date_ic = torch.cat(tuple(row[0] for row in diagnostic))
    spread = torch.cat(tuple(row[1] for row in diagnostic))
    gross = torch.cat(
        tuple(row.trace.gross_active_returns.to(torch.float64) for row in ordered)
    )
    policy_turnover = torch.cat(
        tuple(row.trace.policy_one_way_turnover.to(torch.float64) for row in ordered)
    )
    benchmark_turnover = torch.cat(
        tuple(
            row.trace.benchmark_one_way_turnover.to(torch.float64) for row in ordered
        )
    )
    incremental_turnover = policy_turnover - benchmark_turnover
    net10 = gross - 0.001 * incremental_turnover

    gross_sum = float(gross.sum())
    turnover_sum = float(incremental_turnover.sum())
    break_even: float | None = None
    if gross_sum > 0.0 and turnover_sum > 0.0:
        category: Literal[
            "finite-positive", "favorable-cost-dominance", "no-positive-break-even"
        ] = "finite-positive"
        break_even = 10_000.0 * gross_sum / turnover_sum
    elif gross_sum > 0.0:
        category = "favorable-cost-dominance"
    else:
        category = "no-positive-break-even"
    policy_gross_sum = float(
        torch.cat(
            tuple(row.trace.policy_gross_returns.to(torch.float64) for row in ordered)
        ).sum()
    )
    absolute_turnover_sum = float(policy_turnover.sum())
    absolute_break_even = (
        10_000.0 * policy_gross_sum / absolute_turnover_sum
        if policy_gross_sum > 0.0 and absolute_turnover_sum > 0.0
        else 0.0
    )
    fold_mean_ic = tuple(float(row[0].mean()) for row in diagnostic)
    fold_mean_spread = tuple(float(row[1].mean()) for row in diagnostic)
    fold_retention = tuple(
        float(row.trace.requested_to_executed_retention.median()) for row in ordered
    )
    gross_lcb = _lcb_by_block(
        gross,
        fold_lengths=bootstrap.execution_fold_lengths,
        stream=2,
        annualize=True,
    )
    net_lcb = _lcb_by_block(
        net10,
        fold_lengths=bootstrap.execution_fold_lengths,
        stream=2,
        annualize=True,
    )
    spread_lcb = _lcb_by_block(
        spread,
        fold_lengths=bootstrap.decision_fold_lengths,
        stream=1,
        annualize=False,
    )
    provisional = M03RV16PredictiveQualification(
        setting_index=setting,
        setting_id=M03R_V16_SETTING_IDS[setting],
        fold_trace_sha256=tuple(row.trace.trace_sha256 for row in ordered),
        terminal_checkpoint_authority_sha256=tuple(
            row.terminal_checkpoint_authority_sha256 for row in ordered
        ),
        qualified_score_authority_sha256=tuple(
            row.qualified_score_authority_sha256 for row in ordered
        ),
        panel_schedule_sha256=schedule,
        bootstrap_plan_sha256=bootstrap.receipt_sha256,
        mean_projected_rank_ic=float(date_ic.mean()),
        positive_mean_ic_fold_count=sum(value > 0.0 for value in fold_mean_ic),
        positive_spread_fold_count=sum(value > 0.0 for value in fold_mean_spread),
        annualized_gross_active_return=252.0 * float(gross.mean()),
        annualized_net_active_return_10bp=252.0 * float(net10.mean()),
        gross_active_lcb_by_block=gross_lcb,
        net_10bp_active_lcb_by_block=net_lcb,
        spread_lcb_by_block=spread_lcb,
        break_even_category=category,
        break_even_one_way_cost_basis_points=break_even,
        absolute_policy_break_even_one_way_cost_basis_points=absolute_break_even,
        median_risk_projection_retention=float(
            torch.cat(
                tuple(row.trace.requested_to_executed_retention for row in ordered)
            ).median()
        ),
        minimum_fold_median_risk_projection_retention=min(fold_retention),
        median_weighted_cohort_age=float(
            torch.cat(tuple(row.trace.weighted_mean_cohort_age for row in ordered)).median()
        ),
        gates_passed=False,
        primary_hypothesis_passed=False,
        three_seed_confirmation_may_be_minted=False,
    )
    gates = provisional._expected_gates()
    primary_pass = (
        setting == M03R_V16_PREDICTIVE_SPEC.primary_setting_index and gates
    )
    result = replace(
        provisional,
        gates_passed=gates,
        primary_hypothesis_passed=primary_pass,
        three_seed_confirmation_may_be_minted=primary_pass,
    )
    result.validate()
    return result


def qualify_m03r_v16_predictive_candidate(
    results: tuple[M03RV16FoldQualificationResult, ...],
    bootstrap: M03RV16BootstrapPlan,
) -> M03RV16PredictiveQualification:
    """Qualify live fold results through the same CPU evidence path as aggregation."""

    return qualify_m03r_v16_reconciled_evidence(
        tuple(reconcile_m03r_v16_fold_result(row) for row in results),
        bootstrap,
    )


@dataclass(frozen=True, slots=True)
class M03RV16PanelDecision:
    setting_qualification_sha256: tuple[str, str, str]
    bootstrap_plan_sha256: str
    panel_schedule_sha256: str
    primary_setting_index: int
    primary_setting_id: str
    primary_hypothesis_passed: bool
    primary_training_adequacy: Literal[
        "adequate", "inconclusive-undertrained"
    ]
    primary_training_adequacy_receipt_sha256: tuple[str, ...]
    next_research_action: Literal[
        "three-seed-predictive-confirmation",
        "ordered-five-minute-representation",
        "longer-training-protocol",
    ]
    daily_target_or_loss_tuning_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_PANEL_DECISION_SCHEMA

    def validate(
        self,
        qualifications: tuple[M03RV16PredictiveQualification, ...],
        bootstrap: M03RV16BootstrapPlan,
    ) -> None:
        for row in qualifications:
            row.validate()
        bootstrap.validate()
        primary_index = M03R_V16_PREDICTIVE_SPEC.primary_setting_index
        primary_passed = qualifications[primary_index].primary_hypothesis_passed
        expected_action = (
            "longer-training-protocol"
            if self.primary_training_adequacy == "inconclusive-undertrained"
            else (
                "three-seed-predictive-confirmation"
                if primary_passed
                else "ordered-five-minute-representation"
            )
        )
        if (
            len(qualifications) != len(M03R_V16_SETTING_IDS)
            or tuple(row.setting_index for row in qualifications) != (0, 1, 2)
            or tuple(row.setting_id for row in qualifications) != M03R_V16_SETTING_IDS
            or any(
                row.bootstrap_plan_sha256 != bootstrap.receipt_sha256
                for row in qualifications
            )
            or len({row.panel_schedule_sha256 for row in qualifications}) != 1
            or self.setting_qualification_sha256
            != tuple(row.receipt_sha256 for row in qualifications)
            or self.bootstrap_plan_sha256 != bootstrap.receipt_sha256
            or self.panel_schedule_sha256 != qualifications[0].panel_schedule_sha256
            or self.primary_setting_index != primary_index
            or self.primary_setting_id != M03R_V16_SETTING_IDS[primary_index]
            or self.primary_hypothesis_passed != primary_passed
            or self.primary_training_adequacy
            not in {"adequate", "inconclusive-undertrained"}
            or len(self.primary_training_adequacy_receipt_sha256)
            != M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
            or self.next_research_action != expected_action
            or self.daily_target_or_loss_tuning_authorized
            or self.economic_generation_may_be_minted
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_PANEL_DECISION_SCHEMA
        ):
            raise M03RV16SelectionError("V16 panel decision drifted")
        for value in self.setting_qualification_sha256:
            _digest("setting_qualification_sha256", value)
        for value in self.primary_training_adequacy_receipt_sha256:
            _digest("primary_training_adequacy_receipt_sha256", value)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def build_m03r_v16_panel_decision(
    qualifications: tuple[M03RV16PredictiveQualification, ...],
    bootstrap: M03RV16BootstrapPlan,
    *,
    primary_training_adequacy: Literal[
        "adequate", "inconclusive-undertrained"
    ],
    primary_training_adequacy_receipt_sha256: tuple[str, ...],
) -> M03RV16PanelDecision:
    ordered = tuple(sorted(qualifications, key=lambda row: row.setting_index))
    if len(ordered) != len(M03R_V16_SETTING_IDS):
        raise M03RV16SelectionError("V16 panel decision requires three settings")
    primary_index = M03R_V16_PREDICTIVE_SPEC.primary_setting_index
    primary_passed = ordered[primary_index].primary_hypothesis_passed
    qualification_receipts = (
        ordered[0].receipt_sha256,
        ordered[1].receipt_sha256,
        ordered[2].receipt_sha256,
    )
    result = M03RV16PanelDecision(
        setting_qualification_sha256=qualification_receipts,
        bootstrap_plan_sha256=bootstrap.receipt_sha256,
        panel_schedule_sha256=ordered[0].panel_schedule_sha256,
        primary_setting_index=primary_index,
        primary_setting_id=M03R_V16_SETTING_IDS[primary_index],
        primary_hypothesis_passed=primary_passed,
        primary_training_adequacy=primary_training_adequacy,
        primary_training_adequacy_receipt_sha256=(
            primary_training_adequacy_receipt_sha256
        ),
        next_research_action=(
            "longer-training-protocol"
            if primary_training_adequacy == "inconclusive-undertrained"
            else (
                "three-seed-predictive-confirmation"
                if primary_passed
                else "ordered-five-minute-representation"
            )
        ),
    )
    result.validate(ordered, bootstrap)
    return result


def write_m03r_v16_panel_decision(
    path: str | Path,
    decision: M03RV16PanelDecision,
    qualifications: tuple[M03RV16PredictiveQualification, ...],
    bootstrap: M03RV16BootstrapPlan,
) -> str:
    ordered = tuple(sorted(qualifications, key=lambda row: row.setting_index))
    decision.validate(ordered, bootstrap)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV16SelectionError("V16 panel decision target already exists")
    payload = json.dumps(
        {
            "schema": M03R_V16_PANEL_DECISION_FILE_SCHEMA,
            "decision": asdict(decision),
            "receipt_sha256": decision.receipt_sha256,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def load_m03r_v16_panel_decision(
    path: str | Path,
    *,
    expected_file_sha256: str,
    qualifications: tuple[M03RV16PredictiveQualification, ...],
    bootstrap: M03RV16BootstrapPlan,
) -> M03RV16PanelDecision:
    _digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16SelectionError("V16 panel decision is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PANEL_DECISION_BYTES
        ):
            raise M03RV16SelectionError("V16 panel decision type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16SelectionError("V16 panel decision changed while read")
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16SelectionError("V16 panel decision file hash drifted")
    try:
        payload = json.loads(raw)
        row = dict(payload["decision"])
        row["setting_qualification_sha256"] = tuple(
            row["setting_qualification_sha256"]
        )
        row["primary_training_adequacy_receipt_sha256"] = tuple(
            row["primary_training_adequacy_receipt_sha256"]
        )
        decision = M03RV16PanelDecision(**row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV16SelectionError("V16 panel decision is malformed") from exc
    ordered = tuple(sorted(qualifications, key=lambda value: value.setting_index))
    if (
        payload.get("schema") != M03R_V16_PANEL_DECISION_FILE_SCHEMA
        or payload.get("receipt_sha256") != decision.receipt_sha256
    ):
        raise M03RV16SelectionError("V16 panel decision receipt drifted")
    decision.validate(ordered, bootstrap)
    return decision


__all__ = [
    "M03R_V16_BOOTSTRAP_BLOCK_SESSIONS",
    "M03R_V16_BOOTSTRAP_PLAN_SCHEMA",
    "M03R_V16_QUALIFICATION_SCHEMA",
    "M03R_V16_PANEL_DECISION_FILE_SCHEMA",
    "M03R_V16_PANEL_DECISION_SCHEMA",
    "M03RV16BootstrapPlan",
    "M03RV16PanelDecision",
    "M03RV16PredictiveQualification",
    "M03RV16ReconciledFoldEvidence",
    "M03RV16SelectionError",
    "build_m03r_v16_bootstrap_plan",
    "build_m03r_v16_panel_decision",
    "load_m03r_v16_panel_decision",
    "qualify_m03r_v16_predictive_candidate",
    "qualify_m03r_v16_reconciled_evidence",
    "reconcile_m03r_v16_fold_result",
    "write_m03r_v16_panel_decision",
]
