"""Joint block inference and corrected predictive qualification for M03R-v11."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from torch import Tensor

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_BOOTSTRAP_RULE,
    M03R_V11_PREDICTIVE_SPEC,
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTING_IDS,
    resolve_m03r_v11_setting,
)

M03R_V11_BOOTSTRAP_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v11-bootstrap-plan-v1"
M03R_V11_FOLD_EVIDENCE_SCHEMA = "rl-quant.top2000-dev.m03r-v11-fold-evidence-v1"
M03R_V11_QUALIFICATION_SCHEMA = "rl-quant.top2000-dev.m03r-v11-qualification-v1"


class M03RV11SelectionError(ValueError):
    """The v11 chronology, inference, or predictive gate drifted."""


def _torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise M03RV11SelectionError(
            "PyTorch is required for v11 statistical tensor operations"
        ) from exc
    return torch


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


def _tensor_sha256(value: Tensor) -> str:
    torch = _torch_module()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11SelectionError(f"{name} must be a lowercase SHA-256")
    return value


def _draw_indices(
    fold_lengths: tuple[int, ...],
    *,
    replicates: int,
    block_sessions: int,
    seed: int,
) -> Tensor:
    torch = _torch_module()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + block_sessions * 1_000_003)
    rows: list[Tensor] = []
    offset = 0
    block_offset = torch.arange(block_sessions, dtype=torch.long)
    for fold_length in fold_lengths:
        blocks = (fold_length + block_sessions - 1) // block_sessions
        starts = torch.randint(
            fold_length,
            (replicates, blocks),
            generator=generator,
            dtype=torch.long,
        )
        local = (starts.unsqueeze(-1) + block_offset) % fold_length
        rows.append(local.reshape(replicates, -1)[:, :fold_length] + offset)
        offset += fold_length
    return cast("Tensor", torch.cat(rows, dim=1))


@dataclass(frozen=True, slots=True)
class M03RV11BootstrapPlan:
    chronology_sha256: str
    fold_lengths: tuple[int, ...]
    bootstrap_seed: int
    draw_sha256_by_block: tuple[str, str, str]
    block_sessions: tuple[int, int, int] = (10, 21, 30)
    replicates: int = 10_000
    rule: str = M03R_V11_BOOTSTRAP_RULE
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_BOOTSTRAP_PLAN_SCHEMA

    def validate(self) -> None:
        if (
            len(self.fold_lengths) != 6
            or any(value <= 0 for value in self.fold_lengths)
            or isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or self.bootstrap_seed < 0
            or self.block_sessions != (10, 21, 30)
            or self.replicates != M03R_V11_PREDICTIVE_SPEC.bootstrap_replicates
            or self.rule != M03R_V11_BOOTSTRAP_RULE
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_BOOTSTRAP_PLAN_SCHEMA
        ):
            raise M03RV11SelectionError("v11 bootstrap plan drifted")
        _digest("chronology_sha256", self.chronology_sha256)
        expected = tuple(
            _tensor_sha256(
                _draw_indices(
                    self.fold_lengths,
                    replicates=self.replicates,
                    block_sessions=block,
                    seed=self.bootstrap_seed,
                )
            )
            for block in self.block_sessions
        )
        if self.draw_sha256_by_block != expected:
            raise M03RV11SelectionError("v11 bootstrap draws drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def build_m03r_v11_bootstrap_plan(
    score_session_index_by_fold: tuple[Tensor, ...],
    *,
    bootstrap_seed: int,
) -> M03RV11BootstrapPlan:
    torch = _torch_module()
    if len(score_session_index_by_fold) != 6:
        raise M03RV11SelectionError("v11 bootstrap requires six folds")
    chronology_rows: list[int] = []
    lengths: list[int] = []
    previous_stop: int | None = None
    for row in score_session_index_by_fold:
        if (
            not isinstance(row, torch.Tensor)
            or row.ndim != 1
            or row.dtype != torch.int64
            or row.numel() == 0
            or bool((row[1:] <= row[:-1]).any())
            or (previous_stop is not None and int(row[0]) <= previous_stop)
        ):
            raise M03RV11SelectionError("v11 fold chronology is not disjoint ordered")
        chronology_rows.extend(int(value) for value in row)
        lengths.append(int(row.numel()))
        previous_stop = int(row[-1])
    chronology_sha256 = _sha256(chronology_rows)
    fold_lengths = tuple(lengths)
    draw_rows = tuple(
        _tensor_sha256(
            _draw_indices(
                fold_lengths,
                replicates=M03R_V11_PREDICTIVE_SPEC.bootstrap_replicates,
                block_sessions=block,
                seed=bootstrap_seed,
            )
        )
        for block in (10, 21, 30)
    )
    draws = (draw_rows[0], draw_rows[1], draw_rows[2])
    result = M03RV11BootstrapPlan(
        chronology_sha256=chronology_sha256,
        fold_lengths=fold_lengths,
        bootstrap_seed=bootstrap_seed,
        draw_sha256_by_block=draws,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV11FoldEvidence:
    setting_index: int
    setting_id: str
    fold_index: int
    horizon_sessions: int
    score_session_index: Tensor
    gross_active_return: Tensor
    policy_one_way_turnover: Tensor
    benchmark_one_way_turnover: Tensor
    top_bottom_spread: Tensor
    requested_to_executed_retention: Tensor
    mean_spearman_rank_ic: float
    median_spearman_rank_ic: float
    positive_ic_date_fraction: float
    mean_prediction_cross_sectional_std: float
    mean_target_cross_sectional_std: float
    checkpoint_file_sha256: str
    episode_schedule_sha256: str
    residual_operator_root_sha256: str
    array_sha256: tuple[str, str, str, str, str, str]
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_FOLD_EVIDENCE_SCHEMA

    def validate(self) -> None:
        torch = _torch_module()
        setting = resolve_m03r_v11_setting(self.setting_index)
        arrays = (
            self.score_session_index,
            self.gross_active_return,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.top_bottom_spread,
            self.requested_to_executed_retention,
        )
        if (
            self.setting_id != setting.setting_id
            or self.fold_index not in range(6)
            or self.horizon_sessions not in {21, 30}
            or self.score_session_index.ndim != 1
            or self.score_session_index.dtype != torch.int64
            or self.score_session_index.numel() == 0
            or bool(
                (self.score_session_index[1:] <= self.score_session_index[:-1]).any()
            )
            or any(
                value.ndim != 1
                or value.shape != self.score_session_index.shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in arrays[1:]
            )
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.requested_to_executed_retention < 0.0).any())
            or any(
                not math.isfinite(value)
                for value in (
                    self.mean_spearman_rank_ic,
                    self.median_spearman_rank_ic,
                    self.positive_ic_date_fraction,
                    self.mean_prediction_cross_sectional_std,
                    self.mean_target_cross_sectional_std,
                )
            )
            or not 0.0 <= self.positive_ic_date_fraction <= 1.0
            or self.mean_prediction_cross_sectional_std < 0.0
            or self.mean_target_cross_sectional_std <= 0.0
            or self.array_sha256 != tuple(_tensor_sha256(value) for value in arrays)
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_FOLD_EVIDENCE_SCHEMA
        ):
            raise M03RV11SelectionError("v11 fold evidence drifted")
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("episode_schedule_sha256", self.episode_schedule_sha256),
            ("residual_operator_root_sha256", self.residual_operator_root_sha256),
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
                "mean_spearman_rank_ic": self.mean_spearman_rank_ic,
                "median_spearman_rank_ic": self.median_spearman_rank_ic,
                "positive_ic_date_fraction": self.positive_ic_date_fraction,
                "mean_prediction_cross_sectional_std": (
                    self.mean_prediction_cross_sectional_std
                ),
                "mean_target_cross_sectional_std": (
                    self.mean_target_cross_sectional_std
                ),
                "checkpoint_file_sha256": self.checkpoint_file_sha256,
                "episode_schedule_sha256": self.episode_schedule_sha256,
                "residual_operator_root_sha256": (self.residual_operator_root_sha256),
                "array_sha256": self.array_sha256,
            }
        )


def build_m03r_v11_fold_evidence(
    *,
    setting_index: int,
    fold_index: int,
    horizon_sessions: int,
    score_session_index: Tensor,
    gross_active_return: Tensor,
    policy_one_way_turnover: Tensor,
    benchmark_one_way_turnover: Tensor,
    top_bottom_spread: Tensor,
    requested_to_executed_retention: Tensor,
    mean_spearman_rank_ic: float,
    median_spearman_rank_ic: float,
    positive_ic_date_fraction: float,
    mean_prediction_cross_sectional_std: float,
    mean_target_cross_sectional_std: float,
    checkpoint_file_sha256: str,
    episode_schedule_sha256: str,
    residual_operator_root_sha256: str,
) -> M03RV11FoldEvidence:
    setting = resolve_m03r_v11_setting(setting_index)
    arrays = (
        score_session_index,
        gross_active_return,
        policy_one_way_turnover,
        benchmark_one_way_turnover,
        top_bottom_spread,
        requested_to_executed_retention,
    )
    array_rows = tuple(_tensor_sha256(value) for value in arrays)
    result = M03RV11FoldEvidence(
        setting_index=setting_index,
        setting_id=setting.setting_id,
        fold_index=fold_index,
        horizon_sessions=horizon_sessions,
        score_session_index=score_session_index,
        gross_active_return=gross_active_return,
        policy_one_way_turnover=policy_one_way_turnover,
        benchmark_one_way_turnover=benchmark_one_way_turnover,
        top_bottom_spread=top_bottom_spread,
        requested_to_executed_retention=requested_to_executed_retention,
        mean_spearman_rank_ic=mean_spearman_rank_ic,
        median_spearman_rank_ic=median_spearman_rank_ic,
        positive_ic_date_fraction=positive_ic_date_fraction,
        mean_prediction_cross_sectional_std=mean_prediction_cross_sectional_std,
        mean_target_cross_sectional_std=mean_target_cross_sectional_std,
        checkpoint_file_sha256=checkpoint_file_sha256,
        episode_schedule_sha256=episode_schedule_sha256,
        residual_operator_root_sha256=residual_operator_root_sha256,
        array_sha256=(
            array_rows[0],
            array_rows[1],
            array_rows[2],
            array_rows[3],
            array_rows[4],
            array_rows[5],
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV11PredictiveQualification:
    setting_index: int
    setting_id: str
    horizon_sessions: int
    fold_receipt_sha256: tuple[str, ...]
    bootstrap_plan_sha256: str
    mean_rank_ic: float
    positive_mean_ic_fold_count: int
    positive_median_ic_fold_count: int
    positive_date_fraction_fold_count: int
    positive_spread_fold_count: int
    annualized_gross_active_return: float
    annualized_net_active_return_10bp: float
    gross_active_return_lcb: float
    net_active_return_10bp_lcb: float
    top_bottom_spread_lcb: float
    prediction_dispersion_gate_passed: bool
    prediction_target_dispersion_ratio_gate_passed: bool
    break_even_category: Literal[
        "finite-positive",
        "favorable-cost-dominance",
        "no-positive-break-even",
    ]
    break_even_one_way_cost_basis_points: float | None
    median_projection_retention: float
    minimum_fold_median_projection_retention: float
    passed: bool
    economic_generation_may_be_minted: bool
    economic_panel_authorized: bool = False
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_QUALIFICATION_SCHEMA

    def validate(self) -> None:
        expected_pass = (
            self.mean_rank_ic >= M03R_V11_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic
            and self.positive_mean_ic_fold_count >= 4
            and self.positive_median_ic_fold_count >= 4
            and self.positive_date_fraction_fold_count >= 4
            and self.positive_spread_fold_count >= 4
            and self.prediction_dispersion_gate_passed
            and self.prediction_target_dispersion_ratio_gate_passed
            and self.gross_active_return_lcb > 0.0
            and self.net_active_return_10bp_lcb > 0.0
            and self.top_bottom_spread_lcb > 0.0
            and (
                self.break_even_category == "favorable-cost-dominance"
                or (
                    self.break_even_one_way_cost_basis_points is not None
                    and self.break_even_one_way_cost_basis_points >= 10.0
                )
            )
            and self.median_projection_retention >= 0.50
            and self.minimum_fold_median_projection_retention >= 0.20
        )
        if (
            isinstance(self.setting_index, bool)
            or self.setting_index not in range(len(M03R_V11_SETTING_IDS))
            or self.setting_id != M03R_V11_SETTING_IDS[self.setting_index]
            or self.horizon_sessions not in {21, 30}
            or len(self.fold_receipt_sha256) != 6
            or len(set(self.fold_receipt_sha256)) != 6
            or any(
                not math.isfinite(value)
                for value in (
                    self.mean_rank_ic,
                    self.annualized_gross_active_return,
                    self.annualized_net_active_return_10bp,
                    self.gross_active_return_lcb,
                    self.net_active_return_10bp_lcb,
                    self.top_bottom_spread_lcb,
                    self.median_projection_retention,
                    self.minimum_fold_median_projection_retention,
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
            or any(
                not 0 <= value <= 6
                for value in (
                    self.positive_mean_ic_fold_count,
                    self.positive_median_ic_fold_count,
                    self.positive_date_fraction_fold_count,
                    self.positive_spread_fold_count,
                )
            )
            or self.passed != expected_pass
            or self.economic_generation_may_be_minted != expected_pass
            or self.economic_panel_authorized
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_QUALIFICATION_SCHEMA
        ):
            raise M03RV11SelectionError("v11 predictive qualification drifted")
        for value in (*self.fold_receipt_sha256, self.bootstrap_plan_sha256):
            _digest("qualification_sha256", value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def qualify_m03r_v11_predictive_candidate(
    folds: tuple[M03RV11FoldEvidence, ...],
    bootstrap: M03RV11BootstrapPlan,
) -> M03RV11PredictiveQualification:
    torch = _torch_module()
    if len(folds) != 6:
        raise M03RV11SelectionError("v11 qualification requires six folds")
    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    for row in ordered:
        row.validate()
    bootstrap.validate()
    setting_index = ordered[0].setting_index
    horizon = ordered[0].horizon_sessions
    schedule_sha = ordered[0].episode_schedule_sha256
    if (
        tuple(row.fold_index for row in ordered) != tuple(range(6))
        or any(
            row.setting_index != setting_index
            or row.horizon_sessions != horizon
            or row.episode_schedule_sha256 != schedule_sha
            for row in ordered
        )
        or bootstrap.fold_lengths
        != tuple(int(row.score_session_index.numel()) for row in ordered)
        or bootstrap.chronology_sha256
        != _sha256([int(value) for row in ordered for value in row.score_session_index])
    ):
        raise M03RV11SelectionError("v11 fold identities or common chronology drifted")
    gross = torch.cat([row.gross_active_return.to(torch.float64) for row in ordered])
    incremental_turnover = torch.cat(
        [
            (row.policy_one_way_turnover - row.benchmark_one_way_turnover).to(
                torch.float64
            )
            for row in ordered
        ]
    )
    spread = torch.cat([row.top_bottom_spread.to(torch.float64) for row in ordered])
    retention = torch.cat(
        [row.requested_to_executed_retention.to(torch.float64) for row in ordered]
    )
    net10 = gross - 0.001 * incremental_turnover
    primary_draws = _draw_indices(
        bootstrap.fold_lengths,
        replicates=bootstrap.replicates,
        block_sessions=bootstrap.block_sessions[1],
        seed=bootstrap.bootstrap_seed,
    )

    def lcb(value: Tensor) -> float:
        draws = value.index_select(0, primary_draws.flatten()).reshape(
            primary_draws.shape
        )
        return float(torch.quantile(draws.mean(dim=1), 0.025))

    gross_sum = float(gross.sum())
    incremental_turnover_sum = float(incremental_turnover.sum())
    break_even: float | None = None
    if gross_sum > 0.0 and incremental_turnover_sum > 0.0:
        category: Literal[
            "finite-positive", "favorable-cost-dominance", "no-positive-break-even"
        ] = "finite-positive"
        break_even = 10_000.0 * gross_sum / incremental_turnover_sum
    elif gross_sum > 0.0:
        category = "favorable-cost-dominance"
    else:
        category = "no-positive-break-even"
    rank_values = tuple(row.mean_spearman_rank_ic for row in ordered)
    median_values = tuple(row.median_spearman_rank_ic for row in ordered)
    positive_fraction_values = tuple(row.positive_ic_date_fraction for row in ordered)
    prediction_dispersion_ok = all(
        row.mean_prediction_cross_sectional_std
        >= M03R_V11_PREDICTIVE_SPEC.minimum_prediction_cross_sectional_std
        for row in ordered
    )
    ratio_min, ratio_max = (
        M03R_V11_PREDICTIVE_SPEC.prediction_target_dispersion_ratio_range
    )
    dispersion_ratio_ok = all(
        ratio_min
        <= row.mean_prediction_cross_sectional_std / row.mean_target_cross_sectional_std
        <= ratio_max
        for row in ordered
    )
    mean_rank = sum(rank_values) / 6.0
    positive_mean_count = sum(value > 0.0 for value in rank_values)
    positive_median_count = sum(value > 0.0 for value in median_values)
    positive_fraction_count = sum(value > 0.50 for value in positive_fraction_values)
    positive_spread_count = sum(
        float(row.top_bottom_spread.mean()) > 0.0 for row in ordered
    )
    fold_retention = tuple(
        float(row.requested_to_executed_retention.to(torch.float64).median())
        for row in ordered
    )
    gross_lcb = 252.0 * lcb(gross)
    net_lcb = 252.0 * lcb(net10)
    spread_lcb = lcb(spread)
    median_retention = float(retention.median())
    minimum_fold_retention = min(fold_retention)
    passed = (
        mean_rank >= 0.020
        and positive_mean_count >= 4
        and positive_median_count >= 4
        and positive_fraction_count >= 4
        and positive_spread_count >= 4
        and prediction_dispersion_ok
        and dispersion_ratio_ok
        and gross_lcb > 0.0
        and net_lcb > 0.0
        and spread_lcb > 0.0
        and (
            category == "favorable-cost-dominance"
            or (break_even is not None and break_even >= 10.0)
        )
        and median_retention >= 0.50
        and minimum_fold_retention >= 0.20
    )
    result = M03RV11PredictiveQualification(
        setting_index=setting_index,
        setting_id=M03R_V11_SETTING_IDS[setting_index],
        horizon_sessions=horizon,
        fold_receipt_sha256=tuple(row.receipt_sha256 for row in ordered),
        bootstrap_plan_sha256=bootstrap.receipt_sha256,
        mean_rank_ic=mean_rank,
        positive_mean_ic_fold_count=positive_mean_count,
        positive_median_ic_fold_count=positive_median_count,
        positive_date_fraction_fold_count=positive_fraction_count,
        positive_spread_fold_count=positive_spread_count,
        annualized_gross_active_return=252.0 * float(gross.mean()),
        annualized_net_active_return_10bp=252.0 * float(net10.mean()),
        gross_active_return_lcb=gross_lcb,
        net_active_return_10bp_lcb=net_lcb,
        top_bottom_spread_lcb=spread_lcb,
        prediction_dispersion_gate_passed=prediction_dispersion_ok,
        prediction_target_dispersion_ratio_gate_passed=dispersion_ratio_ok,
        break_even_category=category,
        break_even_one_way_cost_basis_points=break_even,
        median_projection_retention=median_retention,
        minimum_fold_median_projection_retention=minimum_fold_retention,
        passed=passed,
        economic_generation_may_be_minted=passed,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V11_BOOTSTRAP_PLAN_SCHEMA",
    "M03R_V11_FOLD_EVIDENCE_SCHEMA",
    "M03R_V11_QUALIFICATION_SCHEMA",
    "M03RV11BootstrapPlan",
    "M03RV11FoldEvidence",
    "M03RV11PredictiveQualification",
    "M03RV11SelectionError",
    "build_m03r_v11_bootstrap_plan",
    "build_m03r_v11_fold_evidence",
    "qualify_m03r_v11_predictive_candidate",
]
