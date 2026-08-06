from __future__ import annotations

import pytest

from rl_quant.models.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_MODEL_ROUTES,
    M03R_TOP2000_DEV_MODEL_ROUTES_BY_ID,
    M03RTop2000DevModelRouteError,
    require_m03r_top2000_dev_model_route_training_ready,
    resolve_m03r_top2000_dev_model_route,
    resolve_m03r_top2000_dev_model_switches,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import M03R_V7_PRIMARY_SETTING_IDS
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)


def test_model_routes_cover_exact_development_panel_without_relabeling() -> None:
    assert tuple(row.setting_id for row in M03R_TOP2000_DEV_MODEL_ROUTES) == (
        M03R_TOP2000_DEV_SETTING_IDS
    )
    assert tuple(M03R_TOP2000_DEV_MODEL_ROUTES_BY_ID) == (M03R_TOP2000_DEV_SETTING_IDS)
    assert len({row.route_sha256 for row in M03R_TOP2000_DEV_MODEL_ROUTES}) == 12
    assert all(row.development_only for row in M03R_TOP2000_DEV_MODEL_ROUTES)
    assert all(not row.training_authorized for row in M03R_TOP2000_DEV_MODEL_ROUTES)
    assert all(not row.promotion_eligible for row in M03R_TOP2000_DEV_MODEL_ROUTES)
    assert all(
        not row.scientific_reporting_eligible for row in M03R_TOP2000_DEV_MODEL_ROUTES
    )

    canonical = resolve_m03r_top2000_dev_model_route(M03R_TOP2000_DEV_SETTING_IDS[0])
    switches = resolve_m03r_top2000_dev_model_switches(canonical.setting_id)
    assert switches.setting_id == "M03R-soft-persistence-active-alpha-hold30"
    assert switches.setting_id != canonical.setting_id


@pytest.mark.parametrize(
    ("setting_index", "source_v6_id", "expected"),
    [
        (0, "M03R-soft-persistence-active-alpha-hold30", (True, True, False)),
        (1, "M03R-soft-persistence-active-alpha-hold30", (True, True, False)),
        (2, "M03R-soft-persistence-active-alpha-hold30", (True, True, False)),
        (3, "A08-fixed-exit-hazard-v6", (True, False, False)),
        (4, "A11-no-exact-hold-atom", (True, True, False)),
        (5, "A09-no-long-context-v6", (True, True, False)),
        (6, "M02-active-risk-no-alpha-heads-v6", (False, True, False)),
        (7, "A04-no-downside-score-adjustment-v6", (True, True, False)),
        (9, "M03R-soft-persistence-active-alpha-hold30", (True, True, False)),
        (10, "A06-sharpe-overlay-v6", (True, True, True)),
        (11, "A07-direct-sharpe-v6", (True, True, False)),
    ],
)
def test_supported_routes_map_to_truthful_existing_v6_capabilities(
    setting_index: int,
    source_v6_id: str,
    expected: tuple[bool, bool, bool],
) -> None:
    setting_id = M03R_TOP2000_DEV_SETTING_IDS[setting_index]
    route = resolve_m03r_top2000_dev_model_route(setting_id)
    switches = resolve_m03r_top2000_dev_model_switches(setting_id)
    assert route.model_capability_supported
    assert route.source_v6_model_setting_id == source_v6_id
    assert switches.setting_id == source_v6_id
    assert (
        switches.use_alpha_head,
        switches.use_three_way_exit_action,
        switches.use_total_risk_overlay,
    ) == expected
    if setting_index == 4:
        assert not switches.allow_exact_hold_atom
    if setting_index == 11:
        assert switches.use_direct_sharpe


def test_fixed_2pct_route_fails_closed_without_existing_model_capability() -> None:
    setting_id = M03R_TOP2000_DEV_SETTING_IDS[8]
    route = resolve_m03r_top2000_dev_model_route(setting_id)
    assert not route.model_capability_supported
    assert route.source_v6_model_setting_id is None
    assert route.unsupported_model_semantics == (
        "fixed-2pct-active-risk-budget-has-no-existing-m03r-v3-model-path",
    )
    with pytest.raises(M03RTop2000DevModelRouteError, match="fixed-2pct"):
        resolve_m03r_top2000_dev_model_switches(setting_id)


def test_non_model_semantics_and_cache_binding_remain_fail_closed() -> None:
    a10 = resolve_m03r_top2000_dev_model_route(
        "A10-no-factor-neutral-projection-top2000-dev-v1"
    )
    assert "no-factor-sector-projection-execution" in (a10.required_non_model_bindings)
    with pytest.raises(M03RTop2000DevModelRouteError, match="not training ready"):
        require_m03r_top2000_dev_model_route_training_ready(a10.setting_id)
    with pytest.raises(ValueError, match="unknown TOP2000"):
        resolve_m03r_top2000_dev_model_route(M03R_V7_PRIMARY_SETTING_IDS[0])
