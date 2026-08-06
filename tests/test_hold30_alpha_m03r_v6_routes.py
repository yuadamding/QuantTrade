"""Fail-closed route coverage for every immutable M03R v6 setting."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_PROTOCOL_GENERATION as M03R_V5_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v6_routes import (
    M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID,
    M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS,
    M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID,
    M03R_V6_SETTING_ROUTES,
    M03RV6RouteError,
    evaluate_m03r_v6_all_setting_routes,
    require_m03r_v6_all_setting_routes_ready,
)
from rl_quant.workflows.hold30_prelockbox import HOLD30_COMPONENT_TESTS


def test_route_inventory_covers_all_12_v6_settings_in_protocol_order() -> None:
    assert len(M03R_V6_SETTING_ROUTES) == 12
    assert tuple(route.setting_id for route in M03R_V6_SETTING_ROUTES) == (
        M03R_SETTING_IDS
    )
    assert tuple(route.setting_index for route in M03R_V6_SETTING_ROUTES) == tuple(
        range(12)
    )
    assert all(
        route.protocol_generation == M03R_PROTOCOL_GENERATION
        and route.design_id == M03R_DESIGN_ID
        for route in M03R_V6_SETTING_ROUTES
    )

    route_bindings = {
        (
            route.setting_id,
            route.objective_route_id,
            route.model_route_id,
            route.ensemble_route_id,
            route.execution_route_id,
            route.evaluator_route_id,
        )
        for route in M03R_V6_SETTING_ROUTES
    }
    assert len(route_bindings) == 12
    assert len({route.route_contract_sha256 for route in M03R_V6_SETTING_ROUTES}) == 12


def test_every_route_is_generation_qualified_and_never_aliases_v5() -> None:
    for route in M03R_V6_SETTING_ROUTES:
        payload = route.canonical_payload()
        assert payload["protocol_generation"] == M03R_PROTOCOL_GENERATION
        for field in (
            "objective_route_id",
            "model_route_id",
            "ensemble_route_id",
            "execution_route_id",
            "evaluator_route_id",
        ):
            route_id = payload[field]
            assert isinstance(route_id, str)
            assert route_id.startswith("m03r-v6-")
            assert "m03r-v5" not in route_id

    with pytest.raises(M03RV6RouteError, match="v5 remains immutable"):
        replace(
            M03R_V6_SETTING_ROUTES[0],
            protocol_generation=M03R_V5_PROTOCOL_GENERATION,
        )


def test_only_v6_proportional_persistence_objective_can_apply() -> None:
    assert M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID.endswith("/v2")
    for route in M03R_V6_SETTING_ROUTES:
        assert route.persistence_objective_route_id == (
            M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID
        )
        assert route.legacy_generic_early_exit_term_route_id == (
            M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID
        )
        assert not route.legacy_generic_early_exit_term_applicable
        assert "proportional-nav-session-soft-persistence-5bp/v2" in (
            route.objective_route_id
        )

    with pytest.raises(M03RV6RouteError, match="legacy generic early-exit"):
        replace(
            M03R_V6_SETTING_ROUTES[0],
            legacy_generic_early_exit_term_applicable=True,
        )


def test_route_ids_bind_setting_specific_causal_switches() -> None:
    routes = {route.setting_id: route for route in M03R_V6_SETTING_ROUTES}
    m02_model = routes["M02-active-risk-no-alpha-heads-v6"].model_route_id
    assert "alpha-heads-off" in m02_model
    assert "confidence-risk-on" in m02_model
    assert "confidence-two-stage-on" in m02_model
    assert (
        "exit-fixed-hold30-prior" in routes["A08-fixed-exit-hazard-v6"].model_route_id
    )
    assert "context-63s" in routes["A09-no-long-context-v6"].model_route_id
    assert (
        "no-factor-sector-projection"
        in routes["A10-no-factor-neutral-projection-v6"].execution_route_id
    )
    assert "exact-hold-off" in routes["A11-no-exact-hold-atom"].model_route_id
    for setting_id, route in routes.items():
        exact_exit = (
            "exact-exit-off"
            if setting_id == "A08-fixed-exit-hazard-v6"
            else "exact-exit-on"
        )
        assert exact_exit in route.model_route_id
        assert exact_exit in route.execution_route_id
    assert (
        "selection-contract-v2"
        in routes["M03R-soft-persistence-active-alpha-hold30"].evaluator_route_id
    )


def test_every_public_production_route_is_explicitly_missing_and_fail_closed() -> None:
    for route in M03R_V6_SETTING_ROUTES:
        assert route.implementation_status == "known-missing"
        assert route.missing_public_production_components == (
            M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS
        )

    status = evaluate_m03r_v6_all_setting_routes()
    assert not status.launch_authorized
    assert len(status.qualification_blockers) == (
        12 * len(M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS)
    )
    for route in M03R_V6_SETTING_ROUTES:
        for component in M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS:
            assert (
                f"missing-public-production-component:{route.setting_id}:{component}"
                in status.qualification_blockers
            )
    assert status.receipt_sha256

    with pytest.raises(
        M03RV6RouteError, match="public production route is unavailable"
    ):
        require_m03r_v6_all_setting_routes_ready()


def test_route_and_aggregate_status_cannot_overstate_readiness() -> None:
    route = M03R_V6_SETTING_ROUTES[0]
    with pytest.raises(M03RV6RouteError, match="cannot overstate"):
        replace(route, missing_public_production_components=())
    with pytest.raises(M03RV6RouteError, match="identifiers do not match"):
        replace(route, evaluator_route_id="m03r-v6-evaluator/forged/v1")

    status = evaluate_m03r_v6_all_setting_routes()
    with pytest.raises(M03RV6RouteError, match="must fail closed"):
        replace(status, launch_authorized=True)
    with pytest.raises(M03RV6RouteError, match="canonically derived"):
        replace(status, qualification_blockers=())


def test_route_module_is_in_the_software_qualification_inventory() -> None:
    assert (
        "src/rl_quant/training/hold30_alpha_m03r_v6_routes.py",
        "tests/test_hold30_alpha_m03r_v6_routes.py",
    ) in HOLD30_COMPONENT_TESTS
    assert (
        "src/rl_quant/models/hold30_confidence_v6.py",
        "tests/test_hold30_confidence_v6.py",
    ) in HOLD30_COMPONENT_TESTS
