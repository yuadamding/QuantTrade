from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import M03R_V13_SETTING_IDS
from rl_quant.training.top2000_m03r_v13_selection import (
    build_m03r_v13_bootstrap_plan,
    qualify_m03r_v13_predictive_candidate,
)


class _Trace(SimpleNamespace):
    def validate(self) -> None:
        return None


def _traces(*, gross: float) -> tuple[_Trace, ...]:
    rows: list[_Trace] = []
    for fold in range(6):
        origins = torch.arange(469 + 93 * fold, 532 + 93 * fold, dtype=torch.int64)
        rows.append(
            _Trace(
                setting_index=0,
                fold_index=fold,
                origin_indices=origins,
                policy_gross_returns=torch.full((63,), gross, dtype=torch.float64),
                benchmark_gross_returns=torch.zeros(63, dtype=torch.float64),
                policy_one_way_turnover=torch.full(
                    (63,), 0.0010, dtype=torch.float64
                ),
                benchmark_one_way_turnover=torch.full(
                    (63,), 0.0005, dtype=torch.float64
                ),
                date_top_bottom_spread=torch.full(
                    (63,), 0.0010, dtype=torch.float64
                ),
                date_spearman_ic=torch.full((63,), 0.03, dtype=torch.float64),
                signal_projection_retention=torch.ones(63, dtype=torch.float64),
                requested_to_executed_retention=torch.ones(
                    63, dtype=torch.float64
                ),
                trace_sha256=f"{fold + 1:064x}",
            )
        )
    return tuple(rows)


def test_v13_joint_gate_passes_only_complete_positive_evidence() -> None:
    traces = _traces(gross=0.001)
    plan = build_m03r_v13_bootstrap_plan(
        tuple(row.origin_indices for row in traces)
    )
    result = qualify_m03r_v13_predictive_candidate(traces, plan)  # type: ignore[arg-type]
    assert result.setting_id == M03R_V13_SETTING_IDS[0]
    assert result.passed is True
    assert result.economic_generation_may_be_minted is True
    assert result.gross_active_lcb_by_block[1] > 0.0
    assert result.net_10bp_active_lcb_by_block[1] > 0.0
    assert result.break_even_one_way_cost_basis_points is not None
    assert result.break_even_one_way_cost_basis_points >= 10.0


def test_v13_joint_gate_stops_negative_gross_evidence() -> None:
    traces = _traces(gross=-0.001)
    plan = build_m03r_v13_bootstrap_plan(
        tuple(row.origin_indices for row in traces)
    )
    result = qualify_m03r_v13_predictive_candidate(traces, plan)  # type: ignore[arg-type]
    assert result.passed is False
    assert result.economic_generation_may_be_minted is False
    assert result.break_even_category == "no-positive-break-even"
    assert result.break_even_one_way_cost_basis_points is None
