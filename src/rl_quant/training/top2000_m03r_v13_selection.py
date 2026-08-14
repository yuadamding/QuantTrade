"""Joint six-fold block inference and fail-closed stop gate for M03R-v13."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_PREDICTIVE_SPEC,
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v13_runtime import M03RV13SimpleSleeveTrace

M03R_V13_BOOTSTRAP_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v13-bootstrap-plan-v1"
M03R_V13_QUALIFICATION_SCHEMA = "rl-quant.top2000-dev.m03r-v13-qualification-v1"


class M03RV13SelectionError(ValueError):
    """The v13 chronology, bootstrap, or predictive stop gate drifted."""


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
        raise M03RV13SelectionError(f"{name} must be a lowercase SHA-256")
    return value


def _draw_indices(
    fold_lengths: tuple[int, ...],
    *,
    block_sessions: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        M03R_V13_PREDICTIVE_SPEC.bootstrap_seed + block_sessions * 1_000_003
    )
    rows: list[torch.Tensor] = []
    offset = 0
    block_offset = torch.arange(block_sessions, dtype=torch.int64)
    for fold_length in fold_lengths:
        blocks = (fold_length + block_sessions - 1) // block_sessions
        starts = torch.randint(
            fold_length,
            (M03R_V13_PREDICTIVE_SPEC.bootstrap_replicates, blocks),
            generator=generator,
            dtype=torch.int64,
        )
        local = (starts.unsqueeze(-1) + block_offset) % fold_length
        rows.append(
            local.reshape(M03R_V13_PREDICTIVE_SPEC.bootstrap_replicates, -1)[
                :, :fold_length
            ]
            + offset
        )
        offset += fold_length
    return torch.cat(rows, dim=1)


@dataclass(frozen=True, slots=True)
class M03RV13BootstrapPlan:
    chronology_sha256: str
    fold_lengths: tuple[int, ...]
    draw_sha256_by_block: tuple[str, str, str]
    block_sessions: tuple[int, int, int] = (10, 21, 30)
    replicates: int = M03R_V13_PREDICTIVE_SPEC.bootstrap_replicates
    bootstrap_seed: int = M03R_V13_PREDICTIVE_SPEC.bootstrap_seed
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256
    schema: str = M03R_V13_BOOTSTRAP_PLAN_SCHEMA

    def validate(self) -> None:
        if (
            self.fold_lengths != (63, 63, 63, 63, 63, 63)
            or self.block_sessions != (10, 21, 30)
            or self.replicates != M03R_V13_PREDICTIVE_SPEC.bootstrap_replicates
            or self.bootstrap_seed != M03R_V13_PREDICTIVE_SPEC.bootstrap_seed
            or self.draw_sha256_by_block
            != tuple(
                _tensor_sha256(
                    _draw_indices(self.fold_lengths, block_sessions=block)
                )
                for block in self.block_sessions
            )
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
            or self.schema != M03R_V13_BOOTSTRAP_PLAN_SCHEMA
        ):
            raise M03RV13SelectionError("v13 bootstrap plan drifted")
        _digest("chronology_sha256", self.chronology_sha256)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def build_m03r_v13_bootstrap_plan(
    origin_indices_by_fold: tuple[torch.Tensor, ...],
) -> M03RV13BootstrapPlan:
    if len(origin_indices_by_fold) != 6:
        raise M03RV13SelectionError("v13 bootstrap requires six folds")
    chronology: list[int] = []
    previous_stop: int | None = None
    for row in origin_indices_by_fold:
        if (
            not isinstance(row, torch.Tensor)
            or row.ndim != 1
            or row.dtype != torch.int64
            or row.numel() != 63
            or bool((row[1:] != row[:-1] + 1).any())
            or (previous_stop is not None and int(row[0]) <= previous_stop)
        ):
            raise M03RV13SelectionError("v13 fold chronology is not ordered/disjoint")
        chronology.extend(int(value) for value in row)
        previous_stop = int(row[-1])
    lengths = tuple(int(row.numel()) for row in origin_indices_by_fold)
    draw_rows = tuple(
        _tensor_sha256(_draw_indices(lengths, block_sessions=block))
        for block in (10, 21, 30)
    )
    plan = M03RV13BootstrapPlan(
        chronology_sha256=_sha256(chronology),
        fold_lengths=lengths,
        draw_sha256_by_block=(draw_rows[0], draw_rows[1], draw_rows[2]),
    )
    plan.validate()
    return plan


@dataclass(frozen=True, slots=True)
class M03RV13PredictiveQualification:
    setting_index: int
    setting_id: str
    fold_trace_sha256: tuple[str, ...]
    bootstrap_plan_sha256: str
    mean_rank_ic: float
    positive_mean_ic_fold_count: int
    positive_median_ic_fold_count: int
    positive_date_fraction_fold_count: int
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
    median_signal_projection_retention: float
    minimum_fold_median_signal_projection_retention: float
    median_risk_projection_retention: float
    minimum_fold_median_risk_projection_retention: float
    passed: bool
    economic_generation_may_be_minted: bool
    economic_panel_authorized: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256
    schema: str = M03R_V13_QUALIFICATION_SCHEMA

    def _expected_pass(self) -> bool:
        spec = M03R_V13_PREDICTIVE_SPEC
        primary = 1
        return (
            self.mean_rank_ic >= spec.minimum_mean_spearman_rank_ic
            and self.positive_mean_ic_fold_count
            >= spec.minimum_positive_mean_ic_fold_count
            and self.positive_median_ic_fold_count
            >= spec.minimum_positive_median_ic_fold_count
            and self.positive_date_fraction_fold_count
            >= spec.minimum_positive_date_fraction_fold_count
            and self.positive_spread_fold_count
            >= spec.minimum_positive_spread_fold_count
            and self.gross_active_lcb_by_block[primary]
            > spec.minimum_gross_active_return_lcb
            and self.net_10bp_active_lcb_by_block[primary]
            > spec.minimum_net_10bp_active_return_lcb
            and self.spread_lcb_by_block[primary] > spec.minimum_spread_lcb
            and (
                self.break_even_category == "favorable-cost-dominance"
                or (
                    self.break_even_one_way_cost_basis_points is not None
                    and self.break_even_one_way_cost_basis_points
                    >= spec.minimum_break_even_one_way_cost_basis_points
                )
            )
            and self.median_signal_projection_retention
            >= spec.minimum_median_signal_projection_retention
            and self.minimum_fold_median_signal_projection_retention
            >= spec.minimum_fold_median_signal_projection_retention
            and self.median_risk_projection_retention
            >= spec.minimum_median_risk_projection_retention
            and self.minimum_fold_median_risk_projection_retention
            >= spec.minimum_fold_median_risk_projection_retention
        )

    def validate(self) -> None:
        finite = (
            self.mean_rank_ic,
            self.annualized_gross_active_return,
            self.annualized_net_active_return_10bp,
            *self.gross_active_lcb_by_block,
            *self.net_10bp_active_lcb_by_block,
            *self.spread_lcb_by_block,
            self.median_signal_projection_retention,
            self.minimum_fold_median_signal_projection_retention,
            self.median_risk_projection_retention,
            self.minimum_fold_median_risk_projection_retention,
        )
        if (
            self.setting_index not in range(len(M03R_V13_SETTING_IDS))
            or self.setting_id != M03R_V13_SETTING_IDS[self.setting_index]
            or len(self.fold_trace_sha256) != 6
            or len(set(self.fold_trace_sha256)) != 6
            or any(not math.isfinite(value) for value in finite)
            or any(
                value not in range(7)
                for value in (
                    self.positive_mean_ic_fold_count,
                    self.positive_median_ic_fold_count,
                    self.positive_date_fraction_fold_count,
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
            or self.passed != self._expected_pass()
            or self.economic_generation_may_be_minted != self.passed
            or self.economic_panel_authorized
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
            or self.schema != M03R_V13_QUALIFICATION_SCHEMA
        ):
            raise M03RV13SelectionError("v13 predictive qualification drifted")
        for value in (*self.fold_trace_sha256, self.bootstrap_plan_sha256):
            _digest("qualification_sha256", value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def qualify_m03r_v13_predictive_candidate(
    traces: tuple[M03RV13SimpleSleeveTrace, ...],
    bootstrap: M03RV13BootstrapPlan,
) -> M03RV13PredictiveQualification:
    if len(traces) != 6:
        raise M03RV13SelectionError("v13 qualification requires six fold traces")
    ordered = tuple(sorted(traces, key=lambda row: row.fold_index))
    for row in ordered:
        row.validate()
    bootstrap.validate()
    setting = ordered[0].setting_index
    chronology = [int(value) for row in ordered for value in row.origin_indices]
    if (
        tuple(row.fold_index for row in ordered) != tuple(range(6))
        or any(row.setting_index != setting for row in ordered)
        or bootstrap.fold_lengths
        != tuple(int(row.origin_indices.numel()) for row in ordered)
        or bootstrap.chronology_sha256 != _sha256(chronology)
    ):
        raise M03RV13SelectionError("v13 fold identities or chronology drifted")
    gross = torch.cat(
        tuple(
            (row.policy_gross_returns - row.benchmark_gross_returns).to(torch.float64)
            for row in ordered
        )
    )
    turnover = torch.cat(
        tuple(
            (row.policy_one_way_turnover - row.benchmark_one_way_turnover).to(
                torch.float64
            )
            for row in ordered
        )
    )
    net10 = gross - 0.001 * turnover
    spread = torch.cat(
        tuple(row.date_top_bottom_spread.to(torch.float64) for row in ordered)
    )

    def lcb_by_block(value: torch.Tensor, *, annualize: bool) -> tuple[float, float, float]:
        result: list[float] = []
        for block in bootstrap.block_sessions:
            draws = _draw_indices(bootstrap.fold_lengths, block_sessions=block)
            distribution = value[draws].mean(dim=1)
            estimate = float(torch.quantile(distribution, 0.025))
            result.append(252.0 * estimate if annualize else estimate)
        return (result[0], result[1], result[2])

    gross_sum = float(gross.sum())
    turnover_sum = float(turnover.sum())
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
    mean_ic = tuple(float(row.date_spearman_ic.mean()) for row in ordered)
    median_ic = tuple(float(row.date_spearman_ic.median()) for row in ordered)
    positive_date = tuple(
        float((row.date_spearman_ic > 0.0).to(torch.float64).mean())
        for row in ordered
    )
    signal_fold = tuple(
        float(row.signal_projection_retention.median()) for row in ordered
    )
    risk_fold = tuple(
        float(row.requested_to_executed_retention.median()) for row in ordered
    )
    gross_lcb = lcb_by_block(gross, annualize=True)
    net_lcb = lcb_by_block(net10, annualize=True)
    spread_lcb = lcb_by_block(spread, annualize=False)
    provisional = M03RV13PredictiveQualification(
        setting_index=setting,
        setting_id=M03R_V13_SETTING_IDS[setting],
        fold_trace_sha256=tuple(row.trace_sha256 for row in ordered),
        bootstrap_plan_sha256=bootstrap.receipt_sha256,
        mean_rank_ic=sum(mean_ic) / 6.0,
        positive_mean_ic_fold_count=sum(value > 0.0 for value in mean_ic),
        positive_median_ic_fold_count=sum(value > 0.0 for value in median_ic),
        positive_date_fraction_fold_count=sum(
            value > M03R_V13_PREDICTIVE_SPEC.minimum_positive_ic_date_fraction
            for value in positive_date
        ),
        positive_spread_fold_count=sum(
            float(row.date_top_bottom_spread.mean()) > 0.0 for row in ordered
        ),
        annualized_gross_active_return=252.0 * float(gross.mean()),
        annualized_net_active_return_10bp=252.0 * float(net10.mean()),
        gross_active_lcb_by_block=gross_lcb,
        net_10bp_active_lcb_by_block=net_lcb,
        spread_lcb_by_block=spread_lcb,
        break_even_category=category,
        break_even_one_way_cost_basis_points=break_even,
        median_signal_projection_retention=float(
            torch.cat(
                tuple(row.signal_projection_retention for row in ordered)
            ).median()
        ),
        minimum_fold_median_signal_projection_retention=min(signal_fold),
        median_risk_projection_retention=float(
            torch.cat(
                tuple(row.requested_to_executed_retention for row in ordered)
            ).median()
        ),
        minimum_fold_median_risk_projection_retention=min(risk_fold),
        passed=False,
        economic_generation_may_be_minted=False,
    )
    passed = provisional._expected_pass()
    result = M03RV13PredictiveQualification(
        **{
            **asdict(provisional),
            "passed": passed,
            "economic_generation_may_be_minted": passed,
        }
    )
    result.validate()
    return result


__all__ = [
    "M03R_V13_BOOTSTRAP_PLAN_SCHEMA",
    "M03R_V13_QUALIFICATION_SCHEMA",
    "M03RV13BootstrapPlan",
    "M03RV13PredictiveQualification",
    "M03RV13SelectionError",
    "build_m03r_v13_bootstrap_plan",
    "qualify_m03r_v13_predictive_candidate",
]
