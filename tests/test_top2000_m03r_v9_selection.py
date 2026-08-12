from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaFoldEvidence,
    M03RV9AlphaPretrainingError,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    build_m03r_v9_simple_sleeve_fold_evidence,
    qualify_m03r_v9_predictive_candidate,
    select_m03r_v9_horizon,
)


def _alpha(fold: int, *, horizon: int = 30) -> M03RV9AlphaFoldEvidence:
    values = [0.0, 0.03, 0.03, 0.0]
    values[1 if horizon == 21 else 2] = 0.025
    return M03RV9AlphaFoldEvidence(
        fold_index=fold,
        target_mode="factor-residual",
        mean_spearman_rank_ic=tuple(values),  # type: ignore[arg-type]
        mean_top_bottom_decile_spread=(0.0, 0.001, 0.001, 0.0),
        valid_date_counts=(20, 20, 20, 20),
        source_array_sha256=f"{fold + 1:x}" * 64,
        asset_axis_sha256="a" * 64,
        exposure_receipt_sha256="b" * 64,
    )


def _sleeve(fold: int, horizon: int = 30) -> object:
    requested = torch.full((40, 5), 0.20, dtype=torch.float64)
    projected = requested.clone()
    projected[:, 1] -= 0.01
    projected[:, 2] += 0.01
    return build_m03r_v9_simple_sleeve_fold_evidence(
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
        source_receipt_sha256="c" * 64,
    )


def test_complete_predictive_and_tradeability_gate_passes() -> None:
    binding = M03RV9HorizonBinding(30, 30, 30)
    result = qualify_m03r_v9_predictive_candidate(
        setting_id="V9-P0-factor-residual-ranked",
        horizon_binding=binding,
        alpha_folds=tuple(_alpha(fold) for fold in range(6)),
        sleeve_folds=tuple(_sleeve(fold) for fold in range(6)),  # type: ignore[arg-type]
    )
    assert result.passed and result.economic_generation_may_be_minted
    assert not result.economic_panel_authorized
    assert result.mean_rank_ic >= 0.02
    assert result.positive_rank_ic_fold_count == 6
    assert result.mean_simple_sleeve_gross_active_return > 0.0
    assert result.mean_simple_sleeve_net_active_return_10bp > 0.0
    assert result.mean_break_even_one_way_cost_basis_points is not None
    assert result.mean_break_even_one_way_cost_basis_points >= 10.0


def test_failed_ic_or_horizon_mismatch_blocks_economic_panel() -> None:
    binding = M03RV9HorizonBinding(30, 30, 30)
    weak = tuple(
        replace(_alpha(fold), mean_spearman_rank_ic=(0.0, 0.01, 0.01, 0.0))
        for fold in range(6)
    )
    result = qualify_m03r_v9_predictive_candidate(
        setting_id="V9-P0-factor-residual-ranked",
        horizon_binding=binding,
        alpha_folds=weak,
        sleeve_folds=tuple(_sleeve(fold) for fold in range(6)),  # type: ignore[arg-type]
    )
    assert not result.passed and not result.economic_panel_authorized
    with pytest.raises(M03RV9AlphaPretrainingError, match="horizon or setting"):
        qualify_m03r_v9_predictive_candidate(
            setting_id="V9-P0-factor-residual-ranked",
            horizon_binding=binding,
            alpha_folds=tuple(_alpha(fold) for fold in range(6)),
            sleeve_folds=tuple(_sleeve(fold, 21) for fold in range(6)),  # type: ignore[arg-type]
        )


def test_horizon_selection_uses_10bp_lcb_and_ties_to_30() -> None:
    binding_21 = M03RV9HorizonBinding(21, 21, 21)
    binding_30 = M03RV9HorizonBinding(30, 30, 30)
    row21 = qualify_m03r_v9_predictive_candidate(
        setting_id="V9-P0-factor-residual-ranked",
        horizon_binding=binding_21,
        alpha_folds=tuple(_alpha(fold, horizon=21) for fold in range(6)),
        sleeve_folds=tuple(_sleeve(fold, 21) for fold in range(6)),  # type: ignore[arg-type]
    )
    row30 = qualify_m03r_v9_predictive_candidate(
        setting_id="V9-P0-factor-residual-ranked",
        horizon_binding=binding_30,
        alpha_folds=tuple(_alpha(fold, horizon=30) for fold in range(6)),
        sleeve_folds=tuple(_sleeve(fold, 30) for fold in range(6)),  # type: ignore[arg-type]
    )
    assert select_m03r_v9_horizon((row21, row30)).selected_horizon_sessions == 30


def test_sleeve_reconciles_10bp_cost_from_gross_and_incremental_turnover() -> None:
    row = _sleeve(0)
    assert row.annualized_gross_active_return == pytest.approx(0.252)  # type: ignore[attr-defined]
    assert row.annualized_net_active_return_10bp == pytest.approx(0.24948)  # type: ignore[attr-defined]


def test_rehashed_forged_pass_flag_is_rejected_by_selector() -> None:
    binding = M03RV9HorizonBinding(30, 30, 30)
    failed = qualify_m03r_v9_predictive_candidate(
        setting_id="V9-P0-factor-residual-ranked",
        horizon_binding=binding,
        alpha_folds=tuple(
            replace(_alpha(fold), mean_spearman_rank_ic=(0.0, 0.0, 0.0, 0.0))
            for fold in range(6)
        ),
        sleeve_folds=tuple(_sleeve(fold) for fold in range(6)),  # type: ignore[arg-type]
    )
    with pytest.raises(M03RV9AlphaPretrainingError, match="malformed"):
        select_m03r_v9_horizon(
            (
                replace(
                    failed,
                    passed=True,
                    economic_generation_may_be_minted=True,
                ),
            )
        )
