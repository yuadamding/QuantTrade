from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTINGS
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v10_diagnostics import (
    M03RV10FoldDiagnostics,
)
from rl_quant.training.top2000_m03r_v10_selection import (
    M03RV10SelectionError,
    M03RV10SleeveFoldEvidence,
    qualify_m03r_v10_predictive_candidate,
    select_m03r_v10_horizon,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    build_m03r_v9_simple_sleeve_fold_evidence,
)


def _diagnostic(fold: int, setting: int = 1) -> M03RV10FoldDiagnostics:
    row = M03R_V10_SETTINGS[setting]
    result = M03RV10FoldDiagnostics(
        setting_index=setting,
        setting_id=row.setting_id,
        setting_receipt_sha256=row.receipt_sha256,
        fold_index=fold,
        mean_spearman_rank_ic=(0.0, 0.03, 0.03, 0.0),
        population_std_spearman_rank_ic=(0.0, 0.01, 0.01, 0.0),
        median_spearman_rank_ic=(0.0, 0.03, 0.03, 0.0),
        positive_ic_date_fraction=(0.0, 0.75, 0.75, 0.0),
        mean_top_bottom_decile_spread=(0.0, 0.001, 0.001, 0.0),
        mean_prediction_cross_sectional_std=(0.01, 0.02, 0.02, 0.03),
        mean_target_cross_sectional_std=(0.01, 0.02, 0.02, 0.03),
        mean_predicted_scale=(0.01, 0.02, 0.02, 0.03),
        valid_date_counts=(63, 63, 63, 63),
        source_array_sha256=f"{fold + 1:x}" * 64,
        asset_axis_sha256="a" * 64,
        exposure_receipt_sha256="b" * 64,
    )
    result.validate()
    return result


def _sleeve(
    fold: int, horizon: int = 30, setting: int = 1
) -> M03RV10SleeveFoldEvidence:
    requested = torch.full((40, 5), 0.20, dtype=torch.float64)
    projected = requested.clone()
    projected[:, 1] -= 0.01
    projected[:, 2] += 0.01
    trace_receipt = f"{fold + 7:x}" * 64
    imported = build_m03r_v9_simple_sleeve_fold_evidence(
        setting_id="V9-P0-factor-residual-ranked",
        fold_index=fold,
        horizon_binding=M03RV9HorizonBinding(horizon, horizon, horizon),
        policy_gross_returns=torch.full((40,), 0.001, dtype=torch.float64),
        benchmark_gross_returns=torch.zeros(40, dtype=torch.float64),
        policy_one_way_turnover=torch.full((40,), 0.01, dtype=torch.float64),
        benchmark_one_way_turnover=torch.zeros(40, dtype=torch.float64),
        requested_weight_trace=requested,
        projected_weight_trace=projected,
        signal_null_retention=torch.full((40,), 0.75, dtype=torch.float64),
        requested_to_executed_retention=torch.full((40,), 0.80, dtype=torch.float64),
        source_receipt_sha256=trace_receipt,
    )
    result = M03RV10SleeveFoldEvidence(
        setting_index=setting,
        setting_id=M03R_V10_SETTINGS[setting].setting_id,
        imported_evidence=imported,
        imported_evidence_sha256=imported.receipt_sha256,
        v10_trace_receipt_sha256=trace_receipt,
    )
    result.validate()
    return result


def test_v10_six_fold_gate_passes_without_authorizing_economic_job() -> None:
    result = qualify_m03r_v10_predictive_candidate(
        setting_index=1,
        horizon_binding=M03RV9HorizonBinding(30, 30, 30),
        diagnostics=tuple(_diagnostic(fold) for fold in range(6)),
        sleeve_folds=tuple(_sleeve(fold) for fold in range(6)),
    )
    assert result.passed
    assert result.economic_generation_may_be_minted
    assert not result.economic_panel_authorized
    assert result.mean_rank_ic == pytest.approx(0.03)
    assert result.positive_rank_ic_fold_count == 6
    assert result.mean_break_even_one_way_cost_basis_points is not None
    assert result.mean_break_even_one_way_cost_basis_points >= 10.0


def test_v10_gate_rejects_weak_ic_and_horizon_or_setting_mismatch() -> None:
    binding = M03RV9HorizonBinding(30, 30, 30)
    weak = tuple(
        replace(
            _diagnostic(fold),
            mean_spearman_rank_ic=(0.0, 0.01, 0.01, 0.0),
        )
        for fold in range(6)
    )
    failed = qualify_m03r_v10_predictive_candidate(
        setting_index=1,
        horizon_binding=binding,
        diagnostics=weak,
        sleeve_folds=tuple(_sleeve(fold) for fold in range(6)),
    )
    assert not failed.passed
    with pytest.raises(M03RV10SelectionError, match="setting or horizon"):
        qualify_m03r_v10_predictive_candidate(
            setting_index=1,
            horizon_binding=binding,
            diagnostics=tuple(_diagnostic(fold) for fold in range(6)),
            sleeve_folds=tuple(_sleeve(fold, horizon=21) for fold in range(6)),
        )


def test_v10_horizon_selection_uses_lcb_and_ties_to_30() -> None:
    rows = []
    for horizon in (21, 30):
        rows.append(
            qualify_m03r_v10_predictive_candidate(
                setting_index=1,
                horizon_binding=M03RV9HorizonBinding(horizon, horizon, horizon),
                diagnostics=tuple(_diagnostic(fold) for fold in range(6)),
                sleeve_folds=tuple(_sleeve(fold, horizon=horizon) for fold in range(6)),
            )
        )
    assert select_m03r_v10_horizon(tuple(rows)).selected_horizon_sessions == 30
    forged = replace(rows[0], setting_id=M03R_V10_SETTINGS[2].setting_id)
    with pytest.raises(M03RV10SelectionError):
        select_m03r_v10_horizon((forged,))
