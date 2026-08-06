"""Fail-closed route inventory for the immutable M03R v7 primary panel.

The routes bind scientific intent to stable identifiers.  They deliberately
do not dispatch training, Kubernetes work, outer evaluation, or promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, NoReturn

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_DESIGN_ID,
    M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA,
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PRIMARY_SETTINGS,
    M03R_V7_PROTOCOL_GENERATION,
    M03RV7Setting,
    validate_m03r_v7_artifact_identity,
)

M03R_V7_ROUTE_SCHEMA = "rl-quant.m03r-v7-setting-route-v1"
M03R_V7_ROUTE_STATUS_SCHEMA = "rl-quant.m03r-v7-route-status-v1"
M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS = (
    "public-v7-twelve-setting-training-driver",
    "public-v7-isolated-confidence-calibration-step",
    "public-v7-five-seed-output-ensemble",
    "public-v7-cause-typed-chronological-execution-adapter",
    "public-v7-factor-risk-projection-adapter",
    "public-v7-chronological-evaluator",
    "public-v7-route-receipt-writer",
)


class M03RV7RouteError(ValueError):
    """A v7 route identity or launch-status claim is inconsistent."""


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _objective_route_id(setting: M03RV7Setting) -> str:
    return (
        "m03r-v7-objective/c1-active-net-log-return/"
        f"persistence-{setting.persistence_coefficient_basis_points:g}bp/"
        f"sharpe-{setting.sharpe_mode}/"
        f"{M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA}"
    )


def _model_route_id(setting: M03RV7Setting) -> str:
    return (
        "m03r-v7-model/"
        f"residual-heads-{setting.residual_alpha_head_mode}/"
        f"exit-{setting.exit_hazard_mode}/"
        f"exact-hold-{'on' if setting.exact_hold_action_supported else 'off'}/"
        f"context-{setting.learned_temporal_context_trading_sessions}s/"
        f"risk-budget-{setting.active_risk_budget_mode}/v1"
    )


def _execution_route_id(setting: M03RV7Setting) -> str:
    factor_projection = (
        "factor-sector-neutral"
        if setting.factor_sector_neutral_projection
        else "factor-sector-neutral-disabled"
    )
    return (
        "m03r-v7-execution/"
        f"{factor_projection}/"
        f"active-risk-{setting.active_risk_budget_mode}/"
        "te-ceiling-6pct/active-beta-equivalence-10pct/"
        "max-stock-weight-1pct/v1"
    )


def _ensemble_route_id(setting: M03RV7Setting) -> str:
    if setting.sharpe_mode == "separate-total-risk-overlay":
        mode = "five-seed-alpha-core-plus-separate-sharpe-overlay"
    elif setting.residual_alpha_heads:
        mode = "five-seed-alpha-output-space-ensemble"
    else:
        mode = "five-seed-common-policy-output-space-ensemble"
    return f"m03r-v7-ensemble/{mode}/v1"


def _evaluator_route_id(setting: M03RV7Setting) -> str:
    return (
        "m03r-v7-evaluator/active-return-10-20-40bp/"
        "active-multifactor-alpha/block-bootstrap-lcb/"
        "holding-telemetry-nongating/"
        f"sharpe-{setting.sharpe_mode}/v1"
    )


@dataclass(frozen=True, slots=True)
class M03RV7SettingRoute:
    """Exact but non-executable route for one primary-panel row."""

    setting_index: int
    setting_id: str
    objective_route_id: str
    model_route_id: str
    execution_route_id: str
    ensemble_route_id: str
    evaluator_route_id: str
    missing_public_production_components: tuple[str, ...]
    implementation_status: Literal["known-missing"]
    launch_authorized: bool
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID

    def __post_init__(self) -> None:
        try:
            setting = validate_m03r_v7_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV7RouteError(str(exc)) from exc
        if self.setting_index != setting.setting_index:
            raise M03RV7RouteError("v7 route index does not match protocol")
        expected = (
            _objective_route_id(setting),
            _model_route_id(setting),
            _execution_route_id(setting),
            _ensemble_route_id(setting),
            _evaluator_route_id(setting),
        )
        observed = (
            self.objective_route_id,
            self.model_route_id,
            self.execution_route_id,
            self.ensemble_route_id,
            self.evaluator_route_id,
        )
        if observed != expected:
            raise M03RV7RouteError("v7 route identifiers drifted from setting")
        if (
            self.missing_public_production_components
            != M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS
            or self.implementation_status != "known-missing"
            or self.launch_authorized
        ):
            raise M03RV7RouteError("v7 routes must remain explicitly launch-blocked")

    def canonical_payload(self) -> dict[str, Any]:
        return {"schema": M03R_V7_ROUTE_SCHEMA, **asdict(self)}

    @property
    def route_contract_sha256(self) -> str:
        return _sha256(self.canonical_payload())


def _build_route(setting: M03RV7Setting) -> M03RV7SettingRoute:
    return M03RV7SettingRoute(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        objective_route_id=_objective_route_id(setting),
        model_route_id=_model_route_id(setting),
        execution_route_id=_execution_route_id(setting),
        ensemble_route_id=_ensemble_route_id(setting),
        evaluator_route_id=_evaluator_route_id(setting),
        missing_public_production_components=(
            M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS
        ),
        implementation_status="known-missing",
        launch_authorized=False,
    )


M03R_V7_SETTING_ROUTES = tuple(
    _build_route(setting) for setting in M03R_V7_PRIMARY_SETTINGS
)
M03R_V7_SETTING_ROUTES_BY_ID = {
    route.setting_id: route for route in M03R_V7_SETTING_ROUTES
}
if tuple(route.setting_id for route in M03R_V7_SETTING_ROUTES) != (
    M03R_V7_PRIMARY_SETTING_IDS
):
    raise RuntimeError("v7 route order must match the primary panel")


@dataclass(frozen=True, slots=True)
class M03RV7RouteStatus:
    """Aggregate fail-closed status for all twelve primary routes."""

    blockers: tuple[str, ...]
    launch_authorized: bool
    receipt_sha256: str
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V7_ROUTE_STATUS_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "routes": [route.canonical_payload() for route in M03R_V7_SETTING_ROUTES],
            "blockers": list(self.blockers),
            "launch_authorized": self.launch_authorized,
        }

    def __post_init__(self) -> None:
        if (
            self.protocol_generation != M03R_V7_PROTOCOL_GENERATION
            or self.design_id != M03R_V7_DESIGN_ID
        ):
            raise M03RV7RouteError("v7 route-status identity drifted")
        if self.blockers != _route_blockers() or self.launch_authorized:
            raise M03RV7RouteError("v7 route status cannot authorize launch")
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7RouteError("v7 route-status receipt hash mismatch")


def _route_blockers() -> tuple[str, ...]:
    return tuple(
        f"missing:{route.setting_id}:{component}"
        for route in M03R_V7_SETTING_ROUTES
        for component in route.missing_public_production_components
    )


def evaluate_m03r_v7_routes() -> M03RV7RouteStatus:
    fields: dict[str, Any] = {
        "blockers": _route_blockers(),
        "launch_authorized": False,
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
    }
    unsigned = M03RV7RouteStatus.__new__(M03RV7RouteStatus)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7RouteStatus(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


def require_m03r_v7_routes_ready() -> NoReturn:
    status = evaluate_m03r_v7_routes()
    raise M03RV7RouteError(
        "M03R v7 public production route is unavailable: " + ", ".join(status.blockers)
    )


__all__ = [
    "M03R_V7_MISSING_PUBLIC_PRODUCTION_COMPONENTS",
    "M03R_V7_ROUTE_SCHEMA",
    "M03R_V7_ROUTE_STATUS_SCHEMA",
    "M03R_V7_SETTING_ROUTES",
    "M03R_V7_SETTING_ROUTES_BY_ID",
    "M03RV7RouteError",
    "M03RV7RouteStatus",
    "M03RV7SettingRoute",
    "evaluate_m03r_v7_routes",
    "require_m03r_v7_routes_ready",
]
