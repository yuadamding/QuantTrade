from __future__ import annotations

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PROTOCOL_GENERATION,
)
from rl_quant.training.hold30_alpha_m03r_v7_routes import (
    M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS,
    M03R_V7_SETTING_ROUTES,
    M03R_V7_SETTING_ROUTES_BY_ID,
    evaluate_m03r_v7_routes,
    require_m03r_v7_routes_ready,
)


def test_v7_routes_cover_exact_primary_panel_in_order() -> None:
    assert tuple(row.setting_id for row in M03R_V7_SETTING_ROUTES) == (
        M03R_V7_PRIMARY_SETTING_IDS
    )
    assert tuple(M03R_V7_SETTING_ROUTES_BY_ID) == M03R_V7_PRIMARY_SETTING_IDS
    assert len({row.route_contract_sha256 for row in M03R_V7_SETTING_ROUTES}) == 12


def test_v7_routes_bind_persistence_and_risk_budget_ablation_values() -> None:
    p00 = M03R_V7_SETTING_ROUTES_BY_ID["P00-no-soft-persistence-v7"]
    canonical = M03R_V7_SETTING_ROUTES_BY_ID[
        "M03R-soft-persistence-active-alpha-hold30-v7"
    ]
    p10 = M03R_V7_SETTING_ROUTES_BY_ID["P10-soft-persistence-10bp-v7"]
    fixed = M03R_V7_SETTING_ROUTES_BY_ID["A12-fixed-2pct-active-risk-budget-v7"]
    assert "persistence-0bp" in p00.objective_route_id
    assert "persistence-5bp" in canonical.objective_route_id
    assert "persistence-10bp" in p10.objective_route_id
    assert "fixed-2pct" in fixed.execution_route_id


def test_v7_all_routes_are_explicitly_missing_and_launch_blocked() -> None:
    status = evaluate_m03r_v7_routes()
    assert status.protocol_generation == M03R_V7_PROTOCOL_GENERATION
    assert status.launch_authorized is False
    assert len(status.blockers) == 12 * len(
        M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS
    )
    assert len(status.receipt_sha256) == 64
    assert all(
        row.implementation_status == "known-missing" for row in M03R_V7_SETTING_ROUTES
    )
    assert all(not row.launch_authorized for row in M03R_V7_SETTING_ROUTES)


def test_v7_ready_guard_always_fails_closed() -> None:
    with pytest.raises(ValueError, match="public production route is unavailable"):
        require_m03r_v7_routes_ready()
