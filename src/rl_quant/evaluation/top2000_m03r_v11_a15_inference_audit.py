"""Diagnostics and block inference for the M03R-v11 a15 post-hoc audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS,
    M03R_V11_A15_AUDIT_COST_BASIS_POINTS,
    M03R_V11_A15_AUDIT_FOLD_SCHEMA,
    M03R_V11_A15_AUDIT_PANEL_SCHEMA,
    M03R_V11_A15_AUDIT_QUANTILE_COUNTS,
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
    M03R_V11_A15_INFERENCE_AUDIT_SPEC,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_runtime import (
    M03RV11A15AuditReplayTrace,
)


class M03RV11A15InferenceAuditError(ValueError):
    """The exact audit inputs, diagnostics, or inference drifted."""


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
        raise M03RV11A15InferenceAuditError(f"{name} must be a lowercase SHA-256")
    return value


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    row = value.detach().to(device="cpu", dtype=torch.float64)
    order = torch.argsort(row, stable=True)
    sorted_row = row.index_select(0, order)
    ranks = torch.empty_like(sorted_row)
    start = 0
    while start < sorted_row.numel():
        stop = start + 1
        while stop < sorted_row.numel() and bool(sorted_row[stop] == sorted_row[start]):
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
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


def _ece_rows(
    probability: torch.Tensor,
    outcome: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    probability = probability[valid].to(torch.float64).clamp(0.0, 1.0)
    outcome = outcome[valid].to(torch.float64)
    if probability.numel() == 0:
        raise M03RV11A15InferenceAuditError("audit calibration has no valid rows")
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    bins = M03R_V11_A15_INFERENCE_AUDIT_SPEC.calibration_bins
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = probability.ge(lower) & (
            probability.le(upper) if index == bins - 1 else probability.lt(upper)
        )
        count = int(mask.sum())
        mean_probability = 0.0 if count == 0 else float(probability[mask].mean())
        mean_outcome = 0.0 if count == 0 else float(outcome[mask].mean())
        gap = abs(mean_probability - mean_outcome)
        ece += count / probability.numel() * gap
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "mean_outcome": mean_outcome,
                "absolute_gap": gap,
            }
        )
    return ece, tuple(rows)


def _quantile_returns(
    signal: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    quantiles: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for date in range(signal.shape[0]):
        mask = valid[date]
        count = int(mask.sum())
        if count < quantiles:
            raise M03RV11A15InferenceAuditError(
                "audit quantile row has insufficient valid assets"
            )
        order = torch.argsort(signal[date, mask], stable=True)
        realized = target[date, mask].index_select(0, order)
        bins = torch.tensor_split(realized, quantiles)
        if any(value.numel() == 0 for value in bins):
            raise M03RV11A15InferenceAuditError("audit quantile row is empty")
        rows.append(torch.stack([value.mean() for value in bins]))
    return torch.stack(rows).to(device="cpu", dtype=torch.float64)


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditFoldEvidence:
    setting_index: int
    setting_id: str
    fold_index: int
    horizon_sessions: int
    variant_id: str
    replay_trace_sha256: str
    target_source_array_sha256: str
    score_session_index: torch.Tensor
    gross_active_return: torch.Tensor
    incremental_one_way_turnover: torch.Tensor
    quantile_target_return: tuple[torch.Tensor, torch.Tensor]
    score_to_requested_delta_spearman: torch.Tensor
    valid_asset_count: torch.Tensor
    selected_scale_quantiles: tuple[float, ...]
    brier_probability_beats_10bp: float
    ece_probability_beats_10bp: float
    calibration_bins: tuple[dict[str, float | int], ...]
    action_cap_hit_fraction: float
    mean_buy_gate: float
    mean_sell_gate: float
    annualized_carry_active_return: float
    annualized_anchor_repair_active_return: float
    annualized_alpha_signal_active_return: float
    array_sha256: tuple[str, ...]
    receipt_sha256: str
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_FOLD_SCHEMA

    @property
    def arrays(self) -> tuple[torch.Tensor, ...]:
        return (
            self.score_session_index,
            self.gross_active_return,
            self.incremental_one_way_turnover,
            self.quantile_target_return[0],
            self.quantile_target_return[1],
            self.score_to_requested_delta_spearman,
            self.valid_asset_count,
        )

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "horizon_sessions": self.horizon_sessions,
            "variant_id": self.variant_id,
            "replay_trace_sha256": self.replay_trace_sha256,
            "target_source_array_sha256": self.target_source_array_sha256,
            "selected_scale_quantiles": self.selected_scale_quantiles,
            "brier_probability_beats_10bp": self.brier_probability_beats_10bp,
            "ece_probability_beats_10bp": self.ece_probability_beats_10bp,
            "calibration_bins": self.calibration_bins,
            "action_cap_hit_fraction": self.action_cap_hit_fraction,
            "mean_buy_gate": self.mean_buy_gate,
            "mean_sell_gate": self.mean_sell_gate,
            "annualized_carry_active_return": self.annualized_carry_active_return,
            "annualized_anchor_repair_active_return": (
                self.annualized_anchor_repair_active_return
            ),
            "annualized_alpha_signal_active_return": (
                self.annualized_alpha_signal_active_return
            ),
            "array_sha256": self.array_sha256,
            "posthoc_exploratory": True,
            "economic_generation_may_be_minted": False,
            "outer_2026_accessed": False,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        rows = self.gross_active_return.numel()
        if (
            self.setting_index not in (0, 1)
            or self.fold_index not in range(6)
            or self.horizon_sessions not in (21, 30)
            or self.score_session_index.ndim != 1
            or self.score_session_index.dtype != torch.int64
            or rows < 2
            or self.score_session_index.numel() != rows
            or any(
                value.ndim != 1
                or value.numel() != rows
                or not bool(torch.isfinite(value).all())
                for value in (
                    self.gross_active_return,
                    self.incremental_one_way_turnover,
                    self.score_to_requested_delta_spearman,
                    self.valid_asset_count,
                )
            )
            or self.quantile_target_return[0].shape != (rows, 10)
            or self.quantile_target_return[1].shape != (rows, 20)
            or any(
                not bool(torch.isfinite(value).all())
                for value in self.quantile_target_return
            )
            or bool((self.valid_asset_count < 20).any())
            or len(self.selected_scale_quantiles) != 5
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.selected_scale_quantiles
            )
            or not 0.0 <= self.brier_probability_beats_10bp <= 1.0
            or not 0.0 <= self.ece_probability_beats_10bp <= 1.0
            or len(self.calibration_bins)
            != M03R_V11_A15_INFERENCE_AUDIT_SPEC.calibration_bins
            or not 0.0 <= self.action_cap_hit_fraction <= 1.0
            or not 0.0 <= self.mean_buy_gate <= 1.0
            or not 0.0 <= self.mean_sell_gate <= 1.0
            or any(
                not math.isfinite(value)
                for value in (
                    self.annualized_carry_active_return,
                    self.annualized_anchor_repair_active_return,
                    self.annualized_alpha_signal_active_return,
                )
            )
            or self.array_sha256
            != tuple(_tensor_sha256(value) for value in self.arrays)
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_FOLD_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditError("a15 audit fold evidence drifted")
        _digest("replay_trace_sha256", self.replay_trace_sha256)
        _digest("target_source_array_sha256", self.target_source_array_sha256)


def build_m03r_v11_a15_audit_fold_evidence(
    trace: M03RV11A15AuditReplayTrace,
    *,
    score_session_index: torch.Tensor,
    target_log_return: torch.Tensor,
    valid: torch.Tensor,
    target_source_array_sha256: str,
) -> M03RV11A15AuditFoldEvidence:
    """Build exploratory diagnostics without feeding outcomes back into actions."""

    trace.validate()
    _digest("target_source_array_sha256", target_source_array_sha256)
    rows, assets = trace.feasible_signal_trace.shape
    if (
        score_session_index.shape != (rows,)
        or score_session_index.dtype != torch.int64
        or target_log_return.shape != (rows, assets)
        or valid.shape != (rows, assets)
        or valid.dtype != torch.bool
        or not bool(torch.isfinite(target_log_return[valid]).all())
        or bool(valid[:, 0].any())
    ):
        raise M03RV11A15InferenceAuditError("a15 audit target axes drifted")
    signal = trace.feasible_signal_trace
    requested_delta = trace.requested_weight_trace - trace.anchor_weight_trace
    score_action = []
    valid_counts = []
    for date in range(rows):
        mask = valid[date]
        valid_counts.append(int(mask.sum()))
        score_action.append(_spearman(signal[date, mask], requested_delta[date, mask]))
    quantile_rows = tuple(
        _quantile_returns(signal, target_log_return, valid, quantiles)
        for quantiles in M03R_V11_A15_AUDIT_QUANTILE_COUNTS
    )
    probability = trace.entry_probability_trace
    outcome = target_log_return > 0.001
    brier = float(
        (probability[valid].to(torch.float64) - outcome[valid].to(torch.float64))
        .square()
        .mean()
    )
    ece, calibration_bins = _ece_rows(probability, outcome, valid)
    selected_scale = trace.selected_scale_trace[valid].to(torch.float64)
    scale_quantiles = tuple(
        float(value)
        for value in torch.quantile(
            selected_scale,
            torch.tensor((0.0, 0.25, 0.50, 0.75, 1.0), dtype=torch.float64),
        )
    )
    allowed = trace.allowed_incremental_turnover
    cap_active = allowed > 1.0e-12
    cap_hit = cap_active & trace.requested_incremental_turnover.ge(allowed - 2.0e-7)
    gross = trace.policy_gross_returns - trace.benchmark_gross_returns
    incremental_turnover = (
        trace.policy_one_way_turnover - trace.benchmark_one_way_turnover
    )
    arrays = tuple(
        value.detach().to(device="cpu").clone()
        for value in (
            score_session_index,
            gross,
            incremental_turnover,
            quantile_rows[0],
            quantile_rows[1],
            torch.tensor(score_action, dtype=torch.float64),
            torch.tensor(valid_counts, dtype=torch.float64),
        )
    )
    provisional = M03RV11A15AuditFoldEvidence(
        setting_index=trace.setting_index,
        setting_id=trace.setting_id,
        fold_index=trace.fold_index,
        horizon_sessions=trace.selected_horizon_sessions,
        variant_id=trace.variant_id,
        replay_trace_sha256=trace.trace_sha256,
        target_source_array_sha256=target_source_array_sha256,
        score_session_index=arrays[0],
        gross_active_return=arrays[1],
        incremental_one_way_turnover=arrays[2],
        quantile_target_return=(arrays[3], arrays[4]),
        score_to_requested_delta_spearman=arrays[5],
        valid_asset_count=arrays[6],
        selected_scale_quantiles=scale_quantiles,
        brier_probability_beats_10bp=brier,
        ece_probability_beats_10bp=ece,
        calibration_bins=calibration_bins,
        action_cap_hit_fraction=(
            0.0
            if not bool(cap_active.any())
            else float(cap_hit[cap_active].float().mean())
        ),
        mean_buy_gate=float(trace.buy_gate_trace[valid].mean()),
        mean_sell_gate=float(trace.sell_gate_trace[valid].mean()),
        annualized_carry_active_return=252.0 * float(trace.carry_active_return.mean()),
        annualized_anchor_repair_active_return=(
            252.0 * float(trace.anchor_repair_active_return.mean())
        ),
        annualized_alpha_signal_active_return=(
            252.0 * float(trace.alpha_signal_active_return.mean())
        ),
        array_sha256=tuple(_tensor_sha256(value) for value in arrays),
        receipt_sha256="0" * 64,
    )
    result = M03RV11A15AuditFoldEvidence(
        **{
            **asdict(provisional),
            "receipt_sha256": _sha256(provisional.unsigned_payload()),
        }
    )
    result.validate()
    return result


def _draw_indices(
    fold_lengths: tuple[int, ...],
    *,
    block_sessions: int,
) -> torch.Tensor:
    spec = M03R_V11_A15_INFERENCE_AUDIT_SPEC
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.bootstrap_seed + block_sessions)
    total = sum(fold_lengths)
    output = torch.empty((spec.bootstrap_replicates, total), dtype=torch.long)
    offsets = []
    cursor = 0
    for length in fold_lengths:
        offsets.append(cursor)
        cursor += length
    for replicate in range(spec.bootstrap_replicates):
        write = 0
        for offset, length in zip(offsets, fold_lengths, strict=True):
            selected: list[int] = []
            while len(selected) < length:
                start = int(torch.randint(length, (1,), generator=generator))
                selected.extend(
                    offset + ((start + step) % length) for step in range(block_sessions)
                )
            output[replicate, write : write + length] = torch.tensor(
                selected[:length], dtype=torch.long
            )
            write += length
    return output


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditPanelReport:
    setting_index: int
    setting_id: str
    horizon_sessions: int
    variant_id: str
    fold_receipt_sha256: tuple[str, ...]
    annualized_gross_active_return: float
    annualized_net_active_return_by_cost: tuple[tuple[int, float], ...]
    annualized_lcb_by_block_and_cost: tuple[
        tuple[int, tuple[tuple[int, float], ...]], ...
    ]
    top_bottom_lcb_by_block_and_quantiles: tuple[
        tuple[int, tuple[tuple[int, float], ...]], ...
    ]
    aggregate_break_even_one_way_cost_basis_points: float | None
    break_even_category: str
    mean_action_cap_hit_fraction: float
    mean_score_to_action_spearman: float
    mean_brier_probability_beats_10bp: float
    mean_ece_probability_beats_10bp: float
    annualized_carry_active_return: float
    annualized_anchor_repair_active_return: float
    annualized_alpha_signal_active_return: float
    receipt_sha256: str
    training_performed: bool = False
    checkpoint_selection_performed: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_PANEL_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        metrics = (
            self.annualized_gross_active_return,
            self.mean_action_cap_hit_fraction,
            self.mean_score_to_action_spearman,
            self.mean_brier_probability_beats_10bp,
            self.mean_ece_probability_beats_10bp,
            self.annualized_carry_active_return,
            self.annualized_anchor_repair_active_return,
            self.annualized_alpha_signal_active_return,
        )
        if (
            self.setting_index not in (0, 1)
            or self.horizon_sessions not in (21, 30)
            or len(self.fold_receipt_sha256) != 6
            or len(set(self.fold_receipt_sha256)) != 6
            or any(not math.isfinite(value) for value in metrics)
            or not 0.0 <= self.mean_action_cap_hit_fraction <= 1.0
            or not 0.0 <= self.mean_brier_probability_beats_10bp <= 1.0
            or not 0.0 <= self.mean_ece_probability_beats_10bp <= 1.0
            or tuple(row[0] for row in self.annualized_net_active_return_by_cost)
            != M03R_V11_A15_AUDIT_COST_BASIS_POINTS
            or tuple(row[0] for row in self.annualized_lcb_by_block_and_cost)
            != M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS
            or tuple(row[0] for row in self.top_bottom_lcb_by_block_and_quantiles)
            != M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS
            or self.break_even_category
            not in (
                "finite-positive",
                "favorable-cost-dominance",
                "no-positive-break-even",
            )
            or self.training_performed
            or self.checkpoint_selection_performed
            or self.economic_generation_may_be_minted
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_PANEL_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditError("a15 audit panel report drifted")
        for value in self.fold_receipt_sha256:
            _digest("fold_receipt_sha256", value)


def build_m03r_v11_a15_audit_panel_report(
    folds: tuple[M03RV11A15AuditFoldEvidence, ...],
) -> M03RV11A15AuditPanelReport:
    """Aggregate six folds with common, fold-bounded moving-block draws."""

    if len(folds) != 6:
        raise M03RV11A15InferenceAuditError("a15 audit panel requires six folds")
    ordered = tuple(sorted(folds, key=lambda value: value.fold_index))
    for fold in ordered:
        fold.validate()
    identity = (
        ordered[0].setting_index,
        ordered[0].setting_id,
        ordered[0].horizon_sessions,
        ordered[0].variant_id,
    )
    if tuple(row.fold_index for row in ordered) != tuple(range(6)) or any(
        (
            row.setting_index,
            row.setting_id,
            row.horizon_sessions,
            row.variant_id,
        )
        != identity
        for row in ordered
    ):
        raise M03RV11A15InferenceAuditError("a15 audit fold identities drifted")
    gross = torch.cat([row.gross_active_return for row in ordered]).to(torch.float64)
    turnover = torch.cat([row.incremental_one_way_turnover for row in ordered]).to(
        torch.float64
    )
    fold_lengths = tuple(row.gross_active_return.numel() for row in ordered)
    net_by_cost = tuple(
        (
            cost,
            252.0 * float((gross - cost / 10_000.0 * turnover).mean()),
        )
        for cost in M03R_V11_A15_AUDIT_COST_BASIS_POINTS
    )
    lcb_rows = []
    spread_rows = []
    for block in M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS:
        draws = _draw_indices(fold_lengths, block_sessions=block)

        def lcb(value: torch.Tensor) -> float:
            sampled = value.index_select(0, draws.flatten()).reshape(draws.shape)
            return float(torch.quantile(sampled.mean(dim=1), 0.025))

        lcb_rows.append(
            (
                block,
                tuple(
                    (
                        cost,
                        252.0 * lcb(gross - cost / 10_000.0 * turnover),
                    )
                    for cost in M03R_V11_A15_AUDIT_COST_BASIS_POINTS
                ),
            )
        )
        quantile_lcbs = []
        for quantile_index, quantile_count in enumerate(
            M03R_V11_A15_AUDIT_QUANTILE_COUNTS
        ):
            daily = torch.cat(
                [
                    row.quantile_target_return[quantile_index][:, -1]
                    - row.quantile_target_return[quantile_index][:, 0]
                    for row in ordered
                ]
            )
            quantile_lcbs.append((quantile_count, lcb(daily)))
        spread_rows.append((block, tuple(quantile_lcbs)))
    gross_sum = float(gross.sum())
    turnover_sum = float(turnover.sum())
    break_even = None
    if gross_sum > 0.0 and turnover_sum > 0.0:
        break_even_category = "finite-positive"
        break_even = 10_000.0 * gross_sum / turnover_sum
    elif gross_sum > 0.0:
        break_even_category = "favorable-cost-dominance"
    else:
        break_even_category = "no-positive-break-even"
    provisional = M03RV11A15AuditPanelReport(
        setting_index=identity[0],
        setting_id=identity[1],
        horizon_sessions=identity[2],
        variant_id=identity[3],
        fold_receipt_sha256=tuple(row.receipt_sha256 for row in ordered),
        annualized_gross_active_return=252.0 * float(gross.mean()),
        annualized_net_active_return_by_cost=net_by_cost,
        annualized_lcb_by_block_and_cost=tuple(lcb_rows),
        top_bottom_lcb_by_block_and_quantiles=tuple(spread_rows),
        aggregate_break_even_one_way_cost_basis_points=break_even,
        break_even_category=break_even_category,
        mean_action_cap_hit_fraction=sum(row.action_cap_hit_fraction for row in ordered)
        / 6.0,
        mean_score_to_action_spearman=float(
            torch.cat([row.score_to_requested_delta_spearman for row in ordered]).mean()
        ),
        mean_brier_probability_beats_10bp=sum(
            row.brier_probability_beats_10bp for row in ordered
        )
        / 6.0,
        mean_ece_probability_beats_10bp=sum(
            row.ece_probability_beats_10bp for row in ordered
        )
        / 6.0,
        annualized_carry_active_return=sum(
            row.annualized_carry_active_return for row in ordered
        )
        / 6.0,
        annualized_anchor_repair_active_return=sum(
            row.annualized_anchor_repair_active_return for row in ordered
        )
        / 6.0,
        annualized_alpha_signal_active_return=sum(
            row.annualized_alpha_signal_active_return for row in ordered
        )
        / 6.0,
        receipt_sha256="0" * 64,
    )
    result = M03RV11A15AuditPanelReport(
        **{
            **asdict(provisional),
            "receipt_sha256": _sha256(provisional.unsigned_payload()),
        }
    )
    result.validate()
    return result


__all__ = [
    "M03R_V11_A15_AUDIT_FOLD_SCHEMA",
    "M03R_V11_A15_AUDIT_PANEL_SCHEMA",
    "M03RV11A15AuditFoldEvidence",
    "M03RV11A15AuditPanelReport",
    "M03RV11A15InferenceAuditError",
    "build_m03r_v11_a15_audit_fold_evidence",
    "build_m03r_v11_a15_audit_panel_report",
]
