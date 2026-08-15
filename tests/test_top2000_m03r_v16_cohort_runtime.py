from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    run_m03r_v16_horizon_matched_cohort_sleeve,
)


class _Risk(SimpleNamespace):
    def validate(self) -> None:
        return None


def _inputs() -> dict[str, Any]:
    decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
    steps = decisions + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
    assets = 6
    origins = torch.arange(100, 100 + decisions, dtype=torch.int64)
    scores = torch.linspace(-1.0, 1.0, assets).expand(decisions, -1).clone()
    scores[:, 0] = 0.0
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[:, 0] = False
    returns = torch.zeros((steps, assets), dtype=torch.float32)
    # Only the final earned return distinguishes the strongest and weakest
    # assets.  A chronology defect that drops the last cohort loses this edge.
    returns[-1, 1] = -0.10
    returns[-1, -1] = 0.10
    benchmark = torch.zeros((steps, assets), dtype=torch.float32)
    benchmark[:, 0] = 0.95
    benchmark[:, 1:] = 0.01
    available = torch.ones((steps, assets), dtype=torch.bool)
    caps = torch.ones((steps, assets), dtype=torch.float32)
    caps[:, 1:] = 0.02
    gross = torch.ones(steps, dtype=torch.float32)
    execution_origins = tuple(range(100, 100 + steps))
    risk = _Risk(
        origin_state_indices=execution_origins,
        asset_count=assets,
        cash_index=0,
        asset_axis_sha256="a" * 64,
        manifest_sha256="b" * 64,
        state_sha256="c" * 64,
    )
    return {
        "fold_index": 0,
        "checkpoint_file_sha256": "d" * 64,
        "checkpoint_model_state_sha256": "e" * 64,
        "qualification_batch_receipt_sha256": "f" * 64,
        "asset_axis_sha256": "a" * 64,
        "decision_origin_indices": origins,
        "executable_selection_scores": scores,
        "selection_valid": valid,
        "post_fill_asset_returns": returns,
        "benchmark_weights": benchmark,
        "fill_available": available,
        "risk_asset_caps": caps,
        "risk_gross_max": gross,
        "risk_state": risk,
    }


def _identity_projection(
    requested_weights: torch.Tensor,
    _benchmark_weights: torch.Tensor,
    _trade_mask: torch.Tensor,
    _risk_asset_caps: torch.Tensor,
    _risk_gross_max: torch.Tensor,
    _risk_state: Any,
    **_kwargs: Any,
) -> Any:
    return SimpleNamespace(
        projected_weights=requested_weights,
        requested_to_executed_retention=torch.ones(
            requested_weights.shape[0],
            dtype=requested_weights.dtype,
            device=requested_weights.device,
        ),
    )


@pytest.mark.parametrize("setting_index", (0, 1, 2))
def test_v16_cohort_path_is_closed_and_costs_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    setting_index: int,
) -> None:
    import rl_quant.training.top2000_m03r_v16_cohort_runtime as runtime

    monkeypatch.setattr(runtime, "project_m03r_v9_active_book", _identity_projection)
    trace = run_m03r_v16_horizon_matched_cohort_sleeve(
        M03R_V16_SETTINGS[setting_index],
        **_inputs(),
    )
    trace.validate()
    decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
    steps = decisions + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
    assert trace.policy_gross_returns.shape == (steps,)
    assert int((trace.cohort_entry_one_way_mass > 0.0).sum()) == decisions
    if setting_index in {1, 2}:
        assert float(trace.policy_gross_returns[-1]) > float(
            trace.benchmark_gross_returns[-1]
        )
    assert trace.terminal_liquidation_one_way_turnover >= 0.0
    ten_bp = M03R_V16_PREDICTIVE_SPEC.evaluation_cost_basis_points.index(10.0)
    assert torch.allclose(
        trace.absolute_policy_cost_by_cost[ten_bp],
        trace.policy_one_way_turnover * 0.001,
    )
    assert torch.allclose(
        trace.net_active_return_by_cost[ten_bp],
        trace.gross_active_returns - trace.incremental_active_cost_by_cost[ten_bp],
    )


def test_v16_h30_final_decision_earns_exactly_thirty_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v16_cohort_runtime as runtime

    monkeypatch.setattr(runtime, "project_m03r_v9_active_book", _identity_projection)
    trace = run_m03r_v16_horizon_matched_cohort_sleeve(
        M03R_V16_SETTINGS[1],
        **_inputs(),
    )
    # Decision 62 earns on step 62 and then on the 29 no-new-decision steps.
    assert (
        trace.execution_origin_indices.numel()
        - M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
        == 29
    )
    assert float(trace.cohort_release_one_way_mass[-1]) > 0.0
    assert trace.final_decision_receives_full_horizon
