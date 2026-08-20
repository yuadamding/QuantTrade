from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.evaluation.alpha_panel import (
    AlphaCrossSection,
    AlphaDiscoveryGate,
    AlphaPanelEvaluationError,
    FoldBootstrapConfig,
    evaluate_alpha_panel,
    fold_cluster_block_bootstrap_lcb,
    rank_information_coefficient,
    tail_spread,
)


def test_rank_ic_and_tail_spread_recover_planted_order() -> None:
    target = tuple(float(value) for value in range(20))
    assert rank_information_coefficient(target, target) == pytest.approx(1.0)
    assert rank_information_coefficient(tuple(reversed(target)), target) == pytest.approx(-1.0)
    assert tail_spread(target, target) == pytest.approx(16.0)


def test_rank_ic_rejects_constant_null_cross_section() -> None:
    with pytest.raises(AlphaPanelEvaluationError, match="undefined"):
        rank_information_coefficient((1.0, 1.0, 1.0), (0.0, 1.0, 2.0))


def _planted_panel() -> tuple[AlphaCrossSection, ...]:
    rows: list[AlphaCrossSection] = []
    session = 0
    for fold in range(5):
        for offset in range(3):
            target = tuple(
                (asset - 9.5) / 100.0 + 0.0001 * (fold + offset)
                for asset in range(20)
            )
            rows.append(
                AlphaCrossSection(
                    session_index=session,
                    fold_index=fold,
                    model_score=target,
                    baseline_score=tuple(reversed(target)),
                    target=target,
                    valid=(True,) * 20,
                )
            )
            session += 1
    return tuple(rows)


def test_paired_panel_requires_model_improvement_on_common_support() -> None:
    summary = evaluate_alpha_panel(
        _planted_panel(),
        FoldBootstrapConfig(replicates=200, block_sessions=3, seed=19),
    )

    assert summary.mean_model_ic == pytest.approx(1.0)
    assert summary.mean_baseline_ic == pytest.approx(-1.0)
    assert summary.model_minus_baseline_ic_lcb95 > 0.0
    assert summary.model_tail_spread_lcb95 > 0.0
    assert summary.positive_model_ic_fold_count == 5


def test_discovery_gate_blocks_year_or_sector_concentration() -> None:
    summary = evaluate_alpha_panel(
        _planted_panel(),
        FoldBootstrapConfig(replicates=200, block_sessions=3, seed=19),
    )
    gate = AlphaDiscoveryGate(
        summary=summary,
        signal_only_gross_return_lcb95=0.001,
        maximum_year_contribution_fraction=0.40,
        maximum_sector_contribution_fraction=0.30,
    )
    assert gate.passed
    assert not replace(gate, maximum_year_contribution_fraction=0.51).passed
    assert not replace(gate, maximum_sector_contribution_fraction=0.36).passed
    with pytest.raises(AlphaPanelEvaluationError, match="thresholds"):
        replace(gate, minimum_mean_ic=0.0).validate()


def test_fold_bootstrap_resamples_outer_folds_as_clusters() -> None:
    config = FoldBootstrapConfig(
        replicates=500,
        block_sessions=4,
        seed=23,
        lower_probability=0.025,
    )
    positive = fold_cluster_block_bootstrap_lcb(
        ((0.1,) * 4, (0.2,) * 4, (0.3,) * 4, (0.4,) * 4, (0.5,) * 4),
        config,
    )
    mixed = fold_cluster_block_bootstrap_lcb(
        ((-1.0,) * 4, (1.0,) * 4, (1.0,) * 4, (1.0,) * 4, (1.0,) * 4),
        config,
    )

    assert positive > 0.0
    assert mixed < 1.0


def test_panel_rejects_mixed_or_reordered_sessions() -> None:
    rows = list(_planted_panel())
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(AlphaPanelEvaluationError, match="sorted and unique"):
        evaluate_alpha_panel(
            rows,
            FoldBootstrapConfig(replicates=100, block_sessions=2),
        )


def test_invalid_assets_are_removed_from_all_paired_metrics() -> None:
    row = AlphaCrossSection(
        session_index=0,
        fold_index=0,
        model_score=(0.0, 1.0, 2.0, 1_000.0),
        baseline_score=(2.0, 1.0, 0.0, -1_000.0),
        target=(0.0, 1.0, 2.0, -1_000.0),
        valid=(True, True, True, False),
    )

    assert row.model_ic == pytest.approx(1.0)
    assert row.baseline_ic == pytest.approx(-1.0)
