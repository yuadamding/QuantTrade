"""V10 sleeve lineage, predictive gates, and horizon selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_HORIZONS,
    M03R_V10_PREDICTIVE_SPEC,
    M03R_V10_PROTOCOL_SHA256,
    M03R_V10_SETTING_IDS,
    resolve_m03r_v10_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
    M03RV9HorizonBinding,
)
from rl_quant.training.top2000_m03r_v10_diagnostics import (
    M03RV10FoldDiagnostics,
)
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    M03RV9AlphaHeadIdentity,
)
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState
from rl_quant.training.top2000_m03r_v9_runtime import (
    M03RV9SimpleSleeveTrace,
    run_m03r_v9_simple_sleeve,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    M03RV9SimpleSleeveFoldEvidence,
    build_m03r_v9_simple_sleeve_fold_evidence,
)
from rl_quant.training.hold30_runtime import Hold30Sequence

M03R_V10_SLEEVE_TRACE_SCHEMA = "rl-quant.top2000-dev.m03r-v10-sleeve-trace-v1"
M03R_V10_SLEEVE_FOLD_SCHEMA = "rl-quant.top2000-dev.m03r-v10-sleeve-fold-v1"
M03R_V10_QUALIFICATION_SCHEMA = "rl-quant.top2000-dev.m03r-v10-qualification-v1"


class M03RV10SelectionError(ValueError):
    """The v10 imported sleeve, fold evidence, or gate drifted."""


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


def _digest(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV10SelectionError("v10 selection identity is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV10ImportedSleeveTrace:
    setting_index: int
    setting_id: str
    setting_receipt_sha256: str
    imported_trace: M03RV9SimpleSleeveTrace
    imported_trace_sha256: str
    imported_runtime_protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    imported_runtime_setting_id: str = M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_SLEEVE_TRACE_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v10_setting(self.setting_index)
        self.imported_trace.validate()
        if (
            self.setting_id != setting.setting_id
            or self.setting_receipt_sha256 != setting.receipt_sha256
            or self.imported_trace.setting_id != self.imported_runtime_setting_id
            or self.imported_trace_sha256 != self.imported_trace.trace_sha256
            or self.imported_runtime_protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.imported_runtime_setting_id
            != M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_SLEEVE_TRACE_SCHEMA
        ):
            raise M03RV10SelectionError("v10 imported sleeve trace drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "setting_index": self.setting_index,
                "setting_id": self.setting_id,
                "setting_receipt_sha256": self.setting_receipt_sha256,
                "imported_trace_sha256": self.imported_trace_sha256,
                "imported_runtime_protocol_sha256": (
                    self.imported_runtime_protocol_sha256
                ),
                "imported_runtime_setting_id": self.imported_runtime_setting_id,
                "fold_index": self.imported_trace.fold_index,
                "horizon_binding_sha256": (self.imported_trace.horizon_binding_sha256),
                "risk_state_sha256": self.imported_trace.risk_state_sha256,
                "source_receipt_sha256": self.imported_trace.source_receipt_sha256,
                "array_sha256": self.imported_trace.array_sha256,
                "economic_optimizer_updates": 0,
            }
        )


def run_m03r_v10_simple_sleeve(
    sequence: Hold30Sequence,
    alpha_distributions: tuple[M03RV9AlphaDistribution, ...],
    risk_state: M03RV9DeviceRiskState,
    horizon_binding: M03RV9HorizonBinding,
    alpha_head_identity: M03RV9AlphaHeadIdentity,
    *,
    setting_index: int,
    fold_index: int,
    state_start_index: int,
    checkpoint_asset_axis_sha256: str,
    source_receipt_sha256: str,
    benchmark_gross_returns: torch.Tensor,
    benchmark_one_way_turnover: torch.Tensor,
) -> M03RV10ImportedSleeveTrace:
    setting = resolve_m03r_v10_setting(setting_index)
    imported = run_m03r_v9_simple_sleeve(
        sequence,
        alpha_distributions,
        risk_state,
        horizon_binding,
        alpha_head_identity,
        setting_id=M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID,
        fold_index=fold_index,
        state_start_index=state_start_index,
        checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
        source_receipt_sha256=source_receipt_sha256,
        benchmark_gross_returns=benchmark_gross_returns,
        benchmark_one_way_turnover=benchmark_one_way_turnover,
    )
    result = M03RV10ImportedSleeveTrace(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        setting_receipt_sha256=setting.receipt_sha256,
        imported_trace=imported,
        imported_trace_sha256=imported.trace_sha256,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV10SleeveFoldEvidence:
    setting_index: int
    setting_id: str
    imported_evidence: M03RV9SimpleSleeveFoldEvidence
    imported_evidence_sha256: str
    v10_trace_receipt_sha256: str
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_SLEEVE_FOLD_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v10_setting(self.setting_index)
        self.imported_evidence.__post_init__()
        if (
            self.setting_id != setting.setting_id
            or self.imported_evidence.setting_id
            != M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.imported_evidence_sha256 != self.imported_evidence.receipt_sha256
            or self.imported_evidence.source_receipt_sha256
            != self.v10_trace_receipt_sha256
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_SLEEVE_FOLD_SCHEMA
        ):
            raise M03RV10SelectionError("v10 sleeve fold evidence drifted")
        _digest(self.v10_trace_receipt_sha256)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def build_m03r_v10_sleeve_fold_evidence(
    trace: M03RV10ImportedSleeveTrace,
    horizon_binding: M03RV9HorizonBinding,
) -> M03RV10SleeveFoldEvidence:
    trace.validate()
    base = trace.imported_trace
    imported = build_m03r_v9_simple_sleeve_fold_evidence(
        setting_id=M03R_V10_IMPORTED_ARCHITECTURE_SETTING_ID,
        fold_index=base.fold_index,
        horizon_binding=horizon_binding,
        policy_gross_returns=base.policy_gross_returns,
        benchmark_gross_returns=base.benchmark_gross_returns,
        policy_one_way_turnover=base.policy_one_way_turnover,
        benchmark_one_way_turnover=base.benchmark_one_way_turnover,
        requested_weight_trace=base.requested_weight_trace,
        projected_weight_trace=base.projected_weight_trace,
        signal_null_retention=base.signal_null_retention,
        requested_to_executed_retention=base.requested_to_executed_retention,
        source_receipt_sha256=trace.receipt_sha256,
    )
    result = M03RV10SleeveFoldEvidence(
        setting_index=trace.setting_index,
        setting_id=trace.setting_id,
        imported_evidence=imported,
        imported_evidence_sha256=imported.receipt_sha256,
        v10_trace_receipt_sha256=trace.receipt_sha256,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV10PredictiveQualification:
    setting_index: int
    setting_id: str
    selected_horizon_sessions: int
    horizon_binding_sha256: str
    fold_diagnostic_receipt_sha256: tuple[str, ...]
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
    economic_panel_authorized: bool = False
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_QUALIFICATION_SCHEMA

    def validate(self) -> None:
        expected_pass = (
            self.mean_rank_ic >= M03R_V10_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic
            and self.positive_rank_ic_fold_count
            >= M03R_V10_PREDICTIVE_SPEC.minimum_positive_rank_ic_fold_count
            and self.mean_top_bottom_spread
            > M03R_V10_PREDICTIVE_SPEC.minimum_mean_top_bottom_spread
            and self.positive_spread_fold_count
            >= M03R_V10_PREDICTIVE_SPEC.minimum_positive_spread_fold_count
            and self.mean_simple_sleeve_gross_active_return
            > M03R_V10_PREDICTIVE_SPEC.minimum_simple_sleeve_gross_active_return
            and self.mean_simple_sleeve_net_active_return_10bp
            > M03R_V10_PREDICTIVE_SPEC.minimum_simple_sleeve_net_active_return_10bp
            and self.gross_positive_fold_count
            >= M03R_V10_PREDICTIVE_SPEC.minimum_gross_positive_fold_count
            and self.mean_break_even_one_way_cost_basis_points is not None
            and self.mean_break_even_one_way_cost_basis_points
            >= M03R_V10_PREDICTIVE_SPEC.minimum_break_even_one_way_cost_basis_points
        )
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or not 0 <= self.setting_index < len(M03R_V10_SETTING_IDS)
            or self.setting_id != M03R_V10_SETTING_IDS[self.setting_index]
            or self.selected_horizon_sessions not in {21, 30}
            or len(self.fold_diagnostic_receipt_sha256) != 6
            or len(self.fold_sleeve_receipt_sha256) != 6
            or len(set(self.fold_diagnostic_receipt_sha256)) != 6
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
            or self.passed != expected_pass
            or self.economic_generation_may_be_minted != expected_pass
            or self.economic_panel_authorized
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_QUALIFICATION_SCHEMA
        ):
            raise M03RV10SelectionError("v10 qualification receipt drifted")
        for value in (
            self.horizon_binding_sha256,
            *self.fold_diagnostic_receipt_sha256,
            *self.fold_sleeve_receipt_sha256,
        ):
            _digest(value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def qualify_m03r_v10_predictive_candidate(
    *,
    setting_index: int,
    horizon_binding: M03RV9HorizonBinding,
    diagnostics: tuple[M03RV10FoldDiagnostics, ...],
    sleeve_folds: tuple[M03RV10SleeveFoldEvidence, ...],
) -> M03RV10PredictiveQualification:
    setting = resolve_m03r_v10_setting(setting_index)
    horizon_binding.__post_init__()
    if (
        len(diagnostics) != 6
        or len(sleeve_folds) != 6
        or tuple(sorted(row.fold_index for row in diagnostics)) != tuple(range(6))
        or tuple(sorted(row.imported_evidence.fold_index for row in sleeve_folds))
        != tuple(range(6))
    ):
        raise M03RV10SelectionError("v10 qualification requires six paired folds")
    diagnostic = tuple(sorted(diagnostics, key=lambda row: row.fold_index))
    sleeve = tuple(
        sorted(sleeve_folds, key=lambda row: row.imported_evidence.fold_index)
    )
    for diagnostic_row in diagnostic:
        diagnostic_row.validate()
    for sleeve_row in sleeve:
        sleeve_row.validate()
    horizon = horizon_binding.qualification_horizon
    horizon_index = M03R_V10_HORIZONS.index(horizon)
    if any(
        row.setting_id != setting.setting_id
        or row.setting_receipt_sha256 != setting.receipt_sha256
        for row in diagnostic
    ) or any(
        row.setting_id != setting.setting_id
        or row.imported_evidence.selected_horizon_sessions != horizon
        or row.imported_evidence.horizon_binding_sha256
        != horizon_binding.receipt_sha256
        for row in sleeve
    ):
        raise M03RV10SelectionError("v10 setting or horizon evidence drifted")
    rank_values = [row.mean_spearman_rank_ic[horizon_index] for row in diagnostic]
    spread_values = [
        row.mean_top_bottom_decile_spread[horizon_index] for row in diagnostic
    ]
    gross_values = [
        row.imported_evidence.annualized_gross_active_return for row in sleeve
    ]
    net_values = [
        row.imported_evidence.annualized_net_active_return_10bp for row in sleeve
    ]
    lcb_values = [row.imported_evidence.net_active_return_10bp_lcb for row in sleeve]
    break_even_values = [
        row.imported_evidence.break_even_one_way_cost_basis_points for row in sleeve
    ]
    mean_rank = sum(rank_values) / 6.0
    positive_rank = sum(value > 0.0 for value in rank_values)
    mean_spread = sum(spread_values) / 6.0
    positive_spread = sum(value > 0.0 for value in spread_values)
    mean_gross = sum(gross_values) / 6.0
    mean_net = sum(net_values) / 6.0
    gross_positive = sum(value > 0.0 for value in gross_values)
    mean_break_even = (
        sum(value for value in break_even_values if value is not None) / 6.0
        if all(value is not None for value in break_even_values)
        else None
    )
    passed = (
        mean_rank >= M03R_V10_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic
        and positive_rank
        >= M03R_V10_PREDICTIVE_SPEC.minimum_positive_rank_ic_fold_count
        and mean_spread > M03R_V10_PREDICTIVE_SPEC.minimum_mean_top_bottom_spread
        and positive_spread
        >= M03R_V10_PREDICTIVE_SPEC.minimum_positive_spread_fold_count
        and mean_gross
        > M03R_V10_PREDICTIVE_SPEC.minimum_simple_sleeve_gross_active_return
        and mean_net
        > M03R_V10_PREDICTIVE_SPEC.minimum_simple_sleeve_net_active_return_10bp
        and gross_positive >= M03R_V10_PREDICTIVE_SPEC.minimum_gross_positive_fold_count
        and mean_break_even is not None
        and mean_break_even
        >= M03R_V10_PREDICTIVE_SPEC.minimum_break_even_one_way_cost_basis_points
    )
    result = M03RV10PredictiveQualification(
        setting_index=setting_index,
        setting_id=setting.setting_id,
        selected_horizon_sessions=horizon,
        horizon_binding_sha256=horizon_binding.receipt_sha256,
        fold_diagnostic_receipt_sha256=tuple(row.receipt_sha256 for row in diagnostic),
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
    )
    result.validate()
    return result


def select_m03r_v10_horizon(
    candidates: tuple[M03RV10PredictiveQualification, ...],
) -> M03RV10PredictiveQualification:
    for row in candidates:
        row.validate()
    eligible = [row for row in candidates if row.passed]
    if not eligible:
        raise M03RV10SelectionError("no v10 setting-horizon pair passed")
    if len({row.setting_id for row in eligible}) != 1:
        raise M03RV10SelectionError("v10 horizon selection cannot mix settings")
    return max(
        eligible,
        key=lambda row: (
            row.mean_simple_sleeve_net_active_return_10bp_lcb,
            row.selected_horizon_sessions == 30,
        ),
    )


__all__ = [
    "M03R_V10_QUALIFICATION_SCHEMA",
    "M03R_V10_SLEEVE_FOLD_SCHEMA",
    "M03R_V10_SLEEVE_TRACE_SCHEMA",
    "M03RV10ImportedSleeveTrace",
    "M03RV10PredictiveQualification",
    "M03RV10SelectionError",
    "M03RV10SleeveFoldEvidence",
    "build_m03r_v10_sleeve_fold_evidence",
    "qualify_m03r_v10_predictive_candidate",
    "run_m03r_v10_simple_sleeve",
    "select_m03r_v10_horizon",
]
