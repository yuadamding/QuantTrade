"""Generation-qualified, fail-closed route inventory for all M03R v6 settings.

This module is declarative.  It binds the route required by every registered
v6 setting but does not dispatch training or evaluation.  All public production
components remain explicitly missing, so no route can authorize a launch or
mint performance evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, NoReturn

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
    M03R_SETTINGS,
    M03RSetting,
    validate_m03r_v6_artifact_identity,
)

M03R_V6_ROUTE_SCHEMA = "rl-quant.m03r-v6-governed-setting-route-v1"
M03R_V6_ROUTE_STATUS_SCHEMA = "rl-quant.m03r-v6-all-setting-route-status-v1"
M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID = (
    "m03r-v6-persistence/proportional-nav-session-quadratic-one-sided-5bp/v2"
)
M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID = (
    "legacy-generic-early-exit-term/inapplicable-to-m03r-v6"
)
M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS = (
    "public-all-setting-training-driver",
    "public-isolated-confidence-head-training-step",
    "public-five-seed-ensemble-driver",
    "public-cause-typed-execution-adapter",
    "public-chronological-evaluator-adapter",
    "public-route-receipt-writer",
)


class M03RV6RouteError(ValueError):
    """A v6 route identity or launch-status claim is inconsistent."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV6RouteError("v6 route payload is not canonical-JSON safe") from exc
    return rendered.encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _boolean_token(value: bool) -> str:
    return "on" if value else "off"


def _tracking_error_floor_token(setting: M03RSetting) -> str:
    if setting.annual_tracking_error_floor is None:
        return "none"
    basis_points = round(10_000.0 * setting.annual_tracking_error_floor)
    return f"{basis_points}bp-ann"


def _objective_route_id(setting: M03RSetting) -> str:
    return (
        "m03r-v6-objective/"
        f"{setting.objective_mode}/"
        f"te-floor-{_tracking_error_floor_token(setting)}/"
        f"sharpe-{setting.sharpe_mode}/"
        "proportional-nav-session-soft-persistence-5bp/v2"
    )


def _model_route_id(setting: M03RSetting) -> str:
    return (
        "m03r-v6-model/"
        f"alpha-heads-{_boolean_token(setting.residual_alpha_heads)}/"
        "downside-score-"
        f"{_boolean_token(setting.use_downside_adjusted_stock_score)}/"
        "confidence-risk-"
        f"{_boolean_token(setting.use_confidence_scaled_active_risk_budget)}/"
        "confidence-two-stage-"
        f"{_boolean_token(setting.use_confidence_scaled_active_risk_budget)}/"
        f"exit-{setting.exit_hazard_mode}/"
        f"exact-hold-{_boolean_token(setting.exact_hold_action_supported)}/"
        "exact-exit-"
        f"{_boolean_token(setting.exit_hazard_mode == 'learned-age-aware')}/"
        f"context-{setting.slow_context_trading_sessions}s/v2"
    )


def _ensemble_route_id(setting: M03RSetting) -> str:
    if setting.sharpe_mode == "separate-total-risk-overlay":
        suffix = "five-seed-alpha-core-plus-separate-risk-overlay"
    elif setting.residual_alpha_heads:
        suffix = "five-seed-alpha-output-mean"
    else:
        suffix = "five-seed-no-alpha-head-output-mean"
    return f"m03r-v6-ensemble/{suffix}/v1"


def _execution_route_id(setting: M03RSetting) -> str:
    projection = (
        "factor-sector-active-beta-projection"
        if setting.factor_sector_projection
        else "no-factor-sector-projection"
    )
    confidence = (
        "confidence-budgets-new-active-risk"
        if setting.use_confidence_scaled_active_risk_budget
        else "no-confidence-active-risk-budget"
    )
    return (
        f"m03r-v6-execution/{projection}/{confidence}/"
        f"te-floor-{_tracking_error_floor_token(setting)}/"
        f"exact-hold-{_boolean_token(setting.exact_hold_action_supported)}/"
        "exact-exit-"
        f"{_boolean_token(setting.exit_hazard_mode == 'learned-age-aware')}/"
        f"sharpe-{setting.sharpe_mode}/v2"
    )


def _evaluator_route_id(setting: M03RSetting) -> str:
    return (
        "m03r-v6-evaluator/"
        "active-return-20-40bp/"
        "selection-contract-v2/"
        "holding-and-censoring-telemetry-nongating/"
        f"sharpe-{setting.sharpe_mode}/v1"
    )


@dataclass(frozen=True, slots=True)
class M03RV6SettingRoute:
    """Exact scientific route required by one registered v6 setting."""

    setting_index: int
    setting_id: str
    objective_route_id: str
    model_route_id: str
    ensemble_route_id: str
    execution_route_id: str
    evaluator_route_id: str
    persistence_objective_route_id: str
    legacy_generic_early_exit_term_route_id: str
    legacy_generic_early_exit_term_applicable: bool
    missing_public_production_components: tuple[str, ...]
    implementation_status: Literal["known-missing"]
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def __post_init__(self) -> None:
        try:
            setting = validate_m03r_v6_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV6RouteError(str(exc)) from exc
        if self.setting_index != setting.setting_index:
            raise M03RV6RouteError("v6 route setting_index does not match protocol")
        expected_route_ids = (
            _objective_route_id(setting),
            _model_route_id(setting),
            _ensemble_route_id(setting),
            _execution_route_id(setting),
            _evaluator_route_id(setting),
            M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID,
            M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID,
        )
        observed_route_ids = (
            self.objective_route_id,
            self.model_route_id,
            self.ensemble_route_id,
            self.execution_route_id,
            self.evaluator_route_id,
            self.persistence_objective_route_id,
            self.legacy_generic_early_exit_term_route_id,
        )
        if observed_route_ids != expected_route_ids:
            raise M03RV6RouteError(
                "v6 route identifiers do not match setting semantics"
            )
        if self.legacy_generic_early_exit_term_applicable:
            raise M03RV6RouteError(
                "legacy generic early-exit term is inapplicable to every v6 route"
            )
        if (
            self.missing_public_production_components
            != M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS
            or self.implementation_status != "known-missing"
        ):
            raise M03RV6RouteError(
                "v6 route cannot overstate its public production implementation"
            )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the full generation-qualified route binding."""

        return {"schema": M03R_V6_ROUTE_SCHEMA, **asdict(self)}

    @property
    def route_contract_sha256(self) -> str:
        return _sha256(self.canonical_payload())


def _build_route(setting: M03RSetting) -> M03RV6SettingRoute:
    return M03RV6SettingRoute(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        objective_route_id=_objective_route_id(setting),
        model_route_id=_model_route_id(setting),
        ensemble_route_id=_ensemble_route_id(setting),
        execution_route_id=_execution_route_id(setting),
        evaluator_route_id=_evaluator_route_id(setting),
        persistence_objective_route_id=(
            M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID
        ),
        legacy_generic_early_exit_term_route_id=(
            M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID
        ),
        legacy_generic_early_exit_term_applicable=False,
        missing_public_production_components=(
            M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS
        ),
        implementation_status="known-missing",
    )


M03R_V6_SETTING_ROUTES = tuple(_build_route(setting) for setting in M03R_SETTINGS)
M03R_V6_SETTING_ROUTES_BY_ID = {
    route.setting_id: route for route in M03R_V6_SETTING_ROUTES
}

if tuple(route.setting_id for route in M03R_V6_SETTING_ROUTES) != M03R_SETTING_IDS:
    raise RuntimeError("M03R v6 routes must cover every setting in protocol order")
if len(M03R_V6_SETTING_ROUTES_BY_ID) != 12:
    raise RuntimeError("M03R v6 route identities must be unique")


def _canonical_blockers() -> tuple[str, ...]:
    return tuple(
        f"missing-public-production-component:{route.setting_id}:{component}"
        for route in M03R_V6_SETTING_ROUTES
        for component in route.missing_public_production_components
    )


@dataclass(frozen=True, slots=True)
class M03RV6AllSettingRouteStatus:
    """Content-bound aggregate status for the 12-setting v6 experiment."""

    qualification_blockers: tuple[str, ...]
    launch_authorized: bool
    receipt_sha256: str
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V6_ROUTE_STATUS_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "route_contracts": [
                route.canonical_payload() for route in M03R_V6_SETTING_ROUTES
            ],
            "qualification_blockers": list(self.qualification_blockers),
            "launch_authorized": self.launch_authorized,
        }

    def __post_init__(self) -> None:
        if self.protocol_generation != M03R_PROTOCOL_GENERATION:
            raise M03RV6RouteError("v6 route status protocol generation drifted")
        if self.design_id != M03R_DESIGN_ID:
            raise M03RV6RouteError("v6 route status design identity drifted")
        if self.qualification_blockers != _canonical_blockers():
            raise M03RV6RouteError(
                "v6 route blockers were not canonically derived from inventory"
            )
        if self.launch_authorized:
            raise M03RV6RouteError(
                "v6 route status must fail closed while public components are missing"
            )
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV6RouteError("v6 all-setting route status hash mismatch")


def evaluate_m03r_v6_all_setting_routes() -> M03RV6AllSettingRouteStatus:
    """Return the canonical fail-closed status; no caller evidence is accepted."""

    fields: dict[str, Any] = {
        "qualification_blockers": _canonical_blockers(),
        "launch_authorized": False,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
    }
    unsigned = M03RV6AllSettingRouteStatus.__new__(M03RV6AllSettingRouteStatus)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV6AllSettingRouteStatus(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


def require_m03r_v6_all_setting_routes_ready() -> NoReturn:
    """Fail closed until the declared public production components exist."""

    status = evaluate_m03r_v6_all_setting_routes()
    raise M03RV6RouteError(
        "M03R v6 all-setting public production route is unavailable: "
        + ", ".join(status.qualification_blockers)
    )


__all__ = [
    "M03R_V6_LEGACY_GENERIC_EARLY_EXIT_TERM_ROUTE_ID",
    "M03R_V6_MISSING_PUBLIC_PRODUCTION_COMPONENTS",
    "M03R_V6_PROPORTIONAL_PERSISTENCE_OBJECTIVE_ROUTE_ID",
    "M03R_V6_ROUTE_SCHEMA",
    "M03R_V6_ROUTE_STATUS_SCHEMA",
    "M03R_V6_SETTING_ROUTES",
    "M03R_V6_SETTING_ROUTES_BY_ID",
    "M03RV6AllSettingRouteStatus",
    "M03RV6RouteError",
    "M03RV6SettingRoute",
    "evaluate_m03r_v6_all_setting_routes",
    "require_m03r_v6_all_setting_routes_ready",
]
