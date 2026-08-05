"""Governed, fail-closed route coverage for every M03R v5 setting."""

import hashlib
import json

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v5_routes import (
    M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS,
    M03R_V5_SETTING_ROUTES,
    M03RV5AllSettingRouteStatus,
    M03RV5RouteError,
    build_m03r_v5_route_qualification_receipt,
    evaluate_m03r_v5_all_setting_routes,
)


def test_route_inventory_covers_exactly_all_eleven_v5_settings() -> None:
    assert len(M03R_V5_SETTING_ROUTES) == 11
    assert tuple(route.setting_id for route in M03R_V5_SETTING_ROUTES) == (
        M03R_SETTING_IDS
    )
    assert tuple(route.setting_index for route in M03R_V5_SETTING_ROUTES) == tuple(
        range(11)
    )
    assert all(
        route.protocol_generation == M03R_PROTOCOL_GENERATION
        and route.design_id == M03R_DESIGN_ID
        for route in M03R_V5_SETTING_ROUTES
    )
    route_tuples = {
        (
            route.objective_route_id,
            route.model_route_id,
            route.ensemble_route_id,
            route.execution_route_id,
        )
        for route in M03R_V5_SETTING_ROUTES
    }
    assert len(route_tuples) == 11
    assert all(route.route_contract_sha256 for route in M03R_V5_SETTING_ROUTES)


def test_known_unsupported_routes_are_explicit_and_cannot_be_qualified() -> None:
    assert M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS == frozenset(M03R_SETTING_IDS)
    for setting_id in M03R_SETTING_IDS:
        with pytest.raises(M03RV5RouteError, match="known-missing"):
            build_m03r_v5_route_qualification_receipt(
                setting_id=setting_id,
                qualification_artifact_sha256="1" * 64,
                source_tree_sha256="2" * 64,
            )


def test_empty_qualification_inventory_is_fail_closed_for_every_setting() -> None:
    status = evaluate_m03r_v5_all_setting_routes(())
    assert status.launch_authorized is False
    assert len(status.qualification_blockers) == 11
    for route in M03R_V5_SETTING_ROUTES:
        expected_prefix = (
            "missing-implementation"
            if route.setting_id in M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS
            else "missing-qualification-receipt"
        )
        assert f"{expected_prefix}:{route.setting_id}" in status.qualification_blockers
    assert status.receipt_sha256


def test_even_all_currently_available_receipts_cannot_hide_missing_routes() -> None:
    status = evaluate_m03r_v5_all_setting_routes(())
    assert status.launch_authorized is False
    assert status.qualification_blockers == tuple(
        f"missing-implementation:{setting_id}"
        for setting_id in M03R_SETTING_IDS
    )
    assert status.qualification_receipts == ()


def test_receipt_inventory_rejects_untyped_rows() -> None:
    with pytest.raises(M03RV5RouteError, match="untyped"):
        evaluate_m03r_v5_all_setting_routes(("not-a-receipt",))  # type: ignore[arg-type]


def test_direct_status_construction_cannot_forge_launch_authority() -> None:
    fields = {
        "qualification_receipts": (),
        "qualification_blockers": (),
        "launch_authorized": True,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
    }
    unsigned = M03RV5AllSettingRouteStatus.__new__(M03RV5AllSettingRouteStatus)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    payload = unsigned.canonical_payload()
    forged_receipt = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(M03RV5RouteError, match="canonically derived"):
        M03RV5AllSettingRouteStatus(
            **fields,
            receipt_sha256=forged_receipt,
        )
