"""Predictive and simple-sleeve advancement gates for M03R-v9."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_HORIZONS,
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_SETTING_IDS,
    M03RV9HorizonBinding,
    resolve_m03r_v9_setting,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaFoldEvidence,
    M03RV9AlphaPretrainingError,
)

M03R_V9_SIMPLE_SLEEVE_SCHEMA = "rl-quant.top2000-dev.m03r-v9-simple-sleeve-fold-v1"
M03R_V9_PREDICTIVE_QUALIFICATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-predictive-qualification-v1"
)


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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV9AlphaPretrainingError("simple-sleeve identity is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV9SimpleSleeveFoldEvidence:
    setting_id: str
    fold_index: int
    selected_horizon_sessions: int
    observation_count: int
    annualized_gross_active_return: float
    annualized_net_active_return_10bp: float
    net_active_return_10bp_lcb: float
    break_even_one_way_cost_basis_points: float | None
    policy_turnover_mean: float
    benchmark_turnover_mean: float
    median_signal_null_retention: float
    median_requested_to_executed_retention: float
    minimum_requested_to_executed_retention: float
    requested_projected_books_distinct: bool
    array_sha256: tuple[str, ...]
    source_receipt_sha256: str
    horizon_binding_sha256: str
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    lcb_rule: str = "normal-95-daily-ddof1-development-v1"
    schema: str = M03R_V9_SIMPLE_SLEEVE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.setting_id not in M03R_V9_SETTING_IDS
            or not 0 <= self.fold_index < 6
            or self.selected_horizon_sessions not in {21, 30}
            or self.observation_count < 2
            or not all(
                math.isfinite(value)
                for value in (
                    self.annualized_gross_active_return,
                    self.annualized_net_active_return_10bp,
                    self.net_active_return_10bp_lcb,
                    self.policy_turnover_mean,
                    self.benchmark_turnover_mean,
                    self.median_signal_null_retention,
                    self.median_requested_to_executed_retention,
                    self.minimum_requested_to_executed_retention,
                )
            )
            or (
                self.break_even_one_way_cost_basis_points is not None
                and not math.isfinite(self.break_even_one_way_cost_basis_points)
            )
            or self.policy_turnover_mean < 0.0
            or self.benchmark_turnover_mean < 0.0
            or not 0.0 <= self.median_signal_null_retention <= 1.0 + 1.0e-8
            or not 0.0 <= self.median_requested_to_executed_retention <= 1.0 + 1.0e-8
            or not 0.0
            <= self.minimum_requested_to_executed_retention
            <= self.median_requested_to_executed_retention + 1.0e-8
            or not isinstance(self.requested_projected_books_distinct, bool)
            or len(self.array_sha256) != 8
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.lcb_rule != "normal-95-daily-ddof1-development-v1"
            or self.schema != M03R_V9_SIMPLE_SLEEVE_SCHEMA
        ):
            raise M03RV9AlphaPretrainingError("simple-sleeve fold evidence is invalid")
        for value in (
            *self.array_sha256,
            self.source_receipt_sha256,
            self.horizon_binding_sha256,
        ):
            _digest(value)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def build_m03r_v9_simple_sleeve_fold_evidence(
    *,
    setting_id: str,
    fold_index: int,
    horizon_binding: M03RV9HorizonBinding,
    policy_gross_returns: torch.Tensor,
    benchmark_gross_returns: torch.Tensor,
    policy_one_way_turnover: torch.Tensor,
    benchmark_one_way_turnover: torch.Tensor,
    requested_weight_trace: torch.Tensor,
    projected_weight_trace: torch.Tensor,
    signal_null_retention: torch.Tensor,
    requested_to_executed_retention: torch.Tensor,
    source_receipt_sha256: str,
) -> M03RV9SimpleSleeveFoldEvidence:
    horizon_binding.__post_init__()
    arrays = (
        policy_gross_returns,
        benchmark_gross_returns,
        policy_one_way_turnover,
        benchmark_one_way_turnover,
    )
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype not in {torch.float32, torch.float64}
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        for value in arrays
    ):
        raise M03RV9AlphaPretrainingError(
            "simple-sleeve arrays must be detached finite vectors"
        )
    length = policy_gross_returns.numel()
    if length < 2 or any(value.numel() != length for value in arrays):
        raise M03RV9AlphaPretrainingError(
            "simple-sleeve vectors do not share a usable chronology"
        )
    if bool((policy_one_way_turnover < 0.0).any()) or bool(
        (benchmark_one_way_turnover < 0.0).any()
    ):
        raise M03RV9AlphaPretrainingError("turnover cannot be negative")
    if (
        not isinstance(requested_weight_trace, torch.Tensor)
        or requested_weight_trace.ndim != 2
        or requested_weight_trace.shape[0] != length
        or requested_weight_trace.dtype not in {torch.float32, torch.float64}
        or requested_weight_trace.requires_grad
        or not bool(torch.isfinite(requested_weight_trace).all())
        or not isinstance(projected_weight_trace, torch.Tensor)
        or projected_weight_trace.shape != requested_weight_trace.shape
        or projected_weight_trace.dtype != requested_weight_trace.dtype
        or projected_weight_trace.requires_grad
        or not bool(torch.isfinite(projected_weight_trace).all())
        or not isinstance(signal_null_retention, torch.Tensor)
        or tuple(signal_null_retention.shape) != (length,)
        or signal_null_retention.dtype not in {torch.float32, torch.float64}
        or signal_null_retention.requires_grad
        or not bool(torch.isfinite(signal_null_retention).all())
        or not isinstance(requested_to_executed_retention, torch.Tensor)
        or tuple(requested_to_executed_retention.shape) != (length,)
        or requested_to_executed_retention.dtype != signal_null_retention.dtype
        or requested_to_executed_retention.requires_grad
        or not bool(torch.isfinite(requested_to_executed_retention).all())
        or bool((signal_null_retention < 0.0).any())
        or bool((signal_null_retention > 1.0 + 1.0e-8).any())
        or bool((requested_to_executed_retention < 0.0).any())
        or bool((requested_to_executed_retention > 1.0 + 1.0e-8).any())
    ):
        raise M03RV9AlphaPretrainingError("simple-sleeve stage traces are malformed")
    gross = policy_gross_returns.to(torch.float64) - benchmark_gross_returns.to(
        torch.float64
    )
    incremental_turnover = policy_one_way_turnover.to(
        torch.float64
    ) - benchmark_one_way_turnover.to(torch.float64)
    net10 = gross - 10.0e-4 * incremental_turnover
    annualized_gross = float(gross.mean() * 252.0)
    annualized_net10 = float(net10.mean() * 252.0)
    standard_error = net10.std(unbiased=True) / math.sqrt(length)
    lcb = float(252.0 * (net10.mean() - 1.959963984540054 * standard_error))
    mean_incremental_turnover = float(incremental_turnover.mean())
    break_even = (
        10_000.0 * float(gross.mean()) / mean_incremental_turnover
        if mean_incremental_turnover > 0.0
        else None
    )
    result = M03RV9SimpleSleeveFoldEvidence(
        setting_id=setting_id,
        fold_index=fold_index,
        selected_horizon_sessions=horizon_binding.economic_execution_horizon,
        observation_count=length,
        annualized_gross_active_return=annualized_gross,
        annualized_net_active_return_10bp=annualized_net10,
        net_active_return_10bp_lcb=lcb,
        break_even_one_way_cost_basis_points=break_even,
        policy_turnover_mean=float(policy_one_way_turnover.to(torch.float64).mean()),
        benchmark_turnover_mean=float(
            benchmark_one_way_turnover.to(torch.float64).mean()
        ),
        median_signal_null_retention=float(
            signal_null_retention.to(torch.float64).median()
        ),
        median_requested_to_executed_retention=float(
            requested_to_executed_retention.to(torch.float64).median()
        ),
        minimum_requested_to_executed_retention=float(
            requested_to_executed_retention.to(torch.float64).min()
        ),
        requested_projected_books_distinct=not torch.equal(
            requested_weight_trace,
            projected_weight_trace,
        ),
        array_sha256=tuple(
            _tensor_sha256(value)
            for value in (
                *arrays,
                requested_weight_trace,
                projected_weight_trace,
                signal_null_retention,
                requested_to_executed_retention,
            )
        ),
        source_receipt_sha256=_digest(source_receipt_sha256),
        horizon_binding_sha256=horizon_binding.receipt_sha256,
    )
    result.__post_init__()
    return result


@dataclass(frozen=True, slots=True)
class M03RV9PredictiveQualification:
    setting_id: str
    selected_horizon_sessions: int
    horizon_binding_sha256: str
    fold_alpha_receipt_sha256: tuple[str, ...]
    fold_sleeve_receipt_sha256: tuple[str, ...]
    mean_rank_ic: float
    positive_rank_ic_fold_count: int
    mean_top_bottom_spread: float
    positive_spread_fold_count: int
    mean_simple_sleeve_gross_active_return: float
    mean_simple_sleeve_net_active_return_10bp: float
    mean_simple_sleeve_net_active_return_10bp_lcb: float
    gross_positive_fold_count: int
    mean_break_even_one_way_cost_basis_points: float | None
    passed: bool
    economic_generation_may_be_minted: bool
    economic_panel_authorized: bool
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_PREDICTIVE_QUALIFICATION_SCHEMA

    def __post_init__(self) -> None:
        expected_pass = (
            self.mean_rank_ic >= M03R_V9_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic
            and self.positive_rank_ic_fold_count
            >= M03R_V9_PREDICTIVE_SPEC.minimum_positive_rank_ic_fold_count
            and self.mean_top_bottom_spread
            > M03R_V9_PREDICTIVE_SPEC.minimum_mean_top_bottom_spread
            and self.positive_spread_fold_count
            >= M03R_V9_PREDICTIVE_SPEC.minimum_positive_spread_fold_count
            and self.mean_simple_sleeve_gross_active_return
            > M03R_V9_PREDICTIVE_SPEC.minimum_simple_sleeve_gross_active_return
            and self.mean_simple_sleeve_net_active_return_10bp
            > M03R_V9_PREDICTIVE_SPEC.minimum_simple_sleeve_net_active_return_10bp
            and self.gross_positive_fold_count
            >= M03R_V9_PREDICTIVE_SPEC.minimum_gross_positive_fold_count
            and self.mean_break_even_one_way_cost_basis_points is not None
            and self.mean_break_even_one_way_cost_basis_points
            >= M03R_V9_PREDICTIVE_SPEC.minimum_break_even_one_way_cost_basis_points
        )
        if (
            self.setting_id not in M03R_V9_SETTING_IDS
            or self.selected_horizon_sessions not in {21, 30}
            or len(self.fold_alpha_receipt_sha256) != 6
            or len(self.fold_sleeve_receipt_sha256) != 6
            or len(set(self.fold_alpha_receipt_sha256)) != 6
            or len(set(self.fold_sleeve_receipt_sha256)) != 6
            or not all(
                math.isfinite(value)
                for value in (
                    self.mean_rank_ic,
                    self.mean_top_bottom_spread,
                    self.mean_simple_sleeve_gross_active_return,
                    self.mean_simple_sleeve_net_active_return_10bp,
                    self.mean_simple_sleeve_net_active_return_10bp_lcb,
                )
            )
            or not 0 <= self.positive_rank_ic_fold_count <= 6
            or not 0 <= self.positive_spread_fold_count <= 6
            or not 0 <= self.gross_positive_fold_count <= 6
            or (
                self.mean_break_even_one_way_cost_basis_points is not None
                and not math.isfinite(self.mean_break_even_one_way_cost_basis_points)
            )
            or self.passed != expected_pass
            or self.economic_generation_may_be_minted != expected_pass
            or self.economic_panel_authorized
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.schema != M03R_V9_PREDICTIVE_QUALIFICATION_SCHEMA
        ):
            raise M03RV9AlphaPretrainingError(
                "predictive qualification receipt is malformed"
            )
        for value in (
            self.horizon_binding_sha256,
            *self.fold_alpha_receipt_sha256,
            *self.fold_sleeve_receipt_sha256,
        ):
            _digest(value)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def qualify_m03r_v9_predictive_candidate(
    *,
    setting_id: str,
    horizon_binding: M03RV9HorizonBinding,
    alpha_folds: tuple[M03RV9AlphaFoldEvidence, ...],
    sleeve_folds: tuple[M03RV9SimpleSleeveFoldEvidence, ...],
) -> M03RV9PredictiveQualification:
    horizon_binding.__post_init__()
    if setting_id not in M03R_V9_SETTING_IDS:
        raise M03RV9AlphaPretrainingError("unknown V9 setting")
    if (
        len(alpha_folds) != 6
        or len(sleeve_folds) != 6
        or tuple(sorted(row.fold_index for row in alpha_folds)) != tuple(range(6))
        or tuple(sorted(row.fold_index for row in sleeve_folds)) != tuple(range(6))
    ):
        raise M03RV9AlphaPretrainingError(
            "qualification requires exactly six paired folds"
        )
    alpha = tuple(sorted(alpha_folds, key=lambda row: row.fold_index))
    sleeve = tuple(sorted(sleeve_folds, key=lambda row: row.fold_index))
    setting = resolve_m03r_v9_setting(setting_id)
    horizon = horizon_binding.qualification_horizon
    horizon_index = M03R_V9_HORIZONS.index(horizon)
    if any(row.target_mode != setting.target_mode for row in alpha) or any(
        row.setting_id != setting_id
        or row.selected_horizon_sessions != horizon
        or row.horizon_binding_sha256 != horizon_binding.receipt_sha256
        for row in sleeve
    ):
        raise M03RV9AlphaPretrainingError("simple-sleeve horizon or setting drifted")
    rank_values = [row.mean_spearman_rank_ic[horizon_index] for row in alpha]
    spread_values = [row.mean_top_bottom_decile_spread[horizon_index] for row in alpha]
    gross_values = [row.annualized_gross_active_return for row in sleeve]
    net_values = [row.annualized_net_active_return_10bp for row in sleeve]
    lcb_values = [row.net_active_return_10bp_lcb for row in sleeve]
    break_even_values = [
        row.break_even_one_way_cost_basis_points
        for row in sleeve
        if row.break_even_one_way_cost_basis_points is not None
    ]
    mean_rank = sum(rank_values) / 6.0
    positive_rank = sum(value > 0.0 for value in rank_values)
    mean_spread = sum(spread_values) / 6.0
    positive_spread = sum(value > 0.0 for value in spread_values)
    mean_gross = sum(gross_values) / 6.0
    mean_net = sum(net_values) / 6.0
    gross_positive = sum(value > 0.0 for value in gross_values)
    mean_break_even = (
        sum(break_even_values) / len(break_even_values)
        if len(break_even_values) == 6
        else None
    )
    passed = (
        mean_rank >= M03R_V9_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic
        and positive_rank >= M03R_V9_PREDICTIVE_SPEC.minimum_positive_rank_ic_fold_count
        and mean_spread > M03R_V9_PREDICTIVE_SPEC.minimum_mean_top_bottom_spread
        and positive_spread
        >= M03R_V9_PREDICTIVE_SPEC.minimum_positive_spread_fold_count
        and mean_gross
        > M03R_V9_PREDICTIVE_SPEC.minimum_simple_sleeve_gross_active_return
        and mean_net
        > M03R_V9_PREDICTIVE_SPEC.minimum_simple_sleeve_net_active_return_10bp
        and gross_positive >= M03R_V9_PREDICTIVE_SPEC.minimum_gross_positive_fold_count
        and mean_break_even is not None
        and mean_break_even
        >= M03R_V9_PREDICTIVE_SPEC.minimum_break_even_one_way_cost_basis_points
    )
    return M03RV9PredictiveQualification(
        setting_id=setting_id,
        selected_horizon_sessions=horizon,
        horizon_binding_sha256=horizon_binding.receipt_sha256,
        fold_alpha_receipt_sha256=tuple(row.receipt_sha256 for row in alpha),
        fold_sleeve_receipt_sha256=tuple(row.receipt_sha256 for row in sleeve),
        mean_rank_ic=mean_rank,
        positive_rank_ic_fold_count=positive_rank,
        mean_top_bottom_spread=mean_spread,
        positive_spread_fold_count=positive_spread,
        mean_simple_sleeve_gross_active_return=mean_gross,
        mean_simple_sleeve_net_active_return_10bp=mean_net,
        mean_simple_sleeve_net_active_return_10bp_lcb=sum(lcb_values) / 6.0,
        gross_positive_fold_count=gross_positive,
        mean_break_even_one_way_cost_basis_points=mean_break_even,
        passed=passed,
        economic_generation_may_be_minted=passed,
        # Passing this predictive receipt permits design/freeze of a distinct
        # economic generation; it never authorizes an economic Job directly.
        economic_panel_authorized=False,
    )


def select_m03r_v9_horizon(
    candidates: tuple[M03RV9PredictiveQualification, ...],
) -> M03RV9PredictiveQualification:
    for row in candidates:
        row.__post_init__()
    eligible = [row for row in candidates if row.passed]
    if not eligible:
        raise M03RV9AlphaPretrainingError(
            "no setting-horizon pair passed the predictive gate"
        )
    setting_ids = {row.setting_id for row in eligible}
    if len(setting_ids) != 1:
        raise M03RV9AlphaPretrainingError(
            "horizon selection cannot mix predictive settings"
        )
    return max(
        eligible,
        key=lambda row: (
            row.mean_simple_sleeve_net_active_return_10bp_lcb,
            row.selected_horizon_sessions == 30,
        ),
    )


__all__ = [
    "M03R_V9_PREDICTIVE_QUALIFICATION_SCHEMA",
    "M03R_V9_SIMPLE_SLEEVE_SCHEMA",
    "M03RV9PredictiveQualification",
    "M03RV9SimpleSleeveFoldEvidence",
    "build_m03r_v9_simple_sleeve_fold_evidence",
    "qualify_m03r_v9_predictive_candidate",
    "select_m03r_v9_horizon",
]
