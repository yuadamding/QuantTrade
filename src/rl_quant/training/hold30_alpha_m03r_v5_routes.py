"""Fail-closed all-setting route inventory for the M03R v5 experiment.

This module records which scientific route every registered setting requires.
It is deliberately not a dispatcher: unsupported routes remain explicit and
cannot acquire a qualification receipt.  Launch authority is possible only
after all eleven exact routes are implemented and independently qualified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
    M03R_SETTINGS,
    M03RSetting,
    validate_m03r_artifact_identity,
)

M03R_V5_ROUTE_SCHEMA = "rl-quant.m03r-v5-governed-setting-route-v1"
M03R_V5_ROUTE_QUALIFICATION_SCHEMA = "rl-quant.m03r-v5-setting-route-qualification-v1"
M03R_V5_ROUTE_STATUS_SCHEMA = "rl-quant.m03r-v5-all-setting-route-status-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")

# Route qualification is executable evidence, not a caller-provided digest.
# Until the governed all-setting driver emits typed artifacts for a route, that
# route remains missing even when its lower-level primitives have unit tests.
M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS = frozenset(M03R_SETTING_IDS)


class M03RV5RouteError(ValueError):
    """A route identity, qualification receipt, or launch claim is invalid."""


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
        raise M03RV5RouteError("route evidence is not canonical-JSON safe") from exc
    return rendered.encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M03RV5RouteError(f"{name} must be a lowercase SHA-256 digest")


def _boolean_token(value: bool) -> str:
    return "on" if value else "off"


def _tracking_error_floor_token(setting: M03RSetting) -> str:
    if setting.annual_tracking_error_floor is None:
        return "none"
    basis_points = round(10_000.0 * setting.annual_tracking_error_floor)
    return f"{basis_points}bp-ann"


def _objective_route_id(setting: M03RSetting) -> str:
    return (
        "m03r-v5-objective/"
        f"{setting.objective_mode}/"
        f"te-floor-{_tracking_error_floor_token(setting)}/"
        f"sharpe-{setting.sharpe_mode}/v1"
    )


def _model_route_id(setting: M03RSetting) -> str:
    return (
        "m03r-v5-model/"
        f"alpha-heads-{_boolean_token(setting.residual_alpha_heads)}/"
        "downside-score-"
        f"{_boolean_token(setting.use_downside_adjusted_stock_score)}/"
        "confidence-risk-"
        f"{_boolean_token(setting.use_confidence_scaled_active_risk_budget)}/"
        f"exit-{setting.exit_hazard_mode}/"
        f"context-{setting.slow_context_trading_sessions}s/v1"
    )


def _ensemble_route_id(setting: M03RSetting) -> str:
    if setting.sharpe_mode == "separate-total-risk-overlay":
        suffix = "five-seed-alpha-core-plus-separate-risk-overlay"
    elif setting.residual_alpha_heads:
        suffix = "five-seed-alpha-output-mean"
    else:
        suffix = "five-seed-no-alpha-head-output-mean"
    return f"m03r-v5-ensemble/{suffix}/v1"


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
        f"m03r-v5-execution/{projection}/{confidence}/"
        f"te-floor-{_tracking_error_floor_token(setting)}/"
        f"sharpe-{setting.sharpe_mode}/v1"
    )


@dataclass(frozen=True, slots=True)
class M03RV5SettingRoute:
    """Exact route required to execute one registered v5 setting."""

    setting_index: int
    setting_id: str
    objective_route_id: str
    model_route_id: str
    ensemble_route_id: str
    execution_route_id: str
    implementation_status: Literal[
        "available-unqualified",
        "known-missing",
    ]
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def __post_init__(self) -> None:
        try:
            setting = validate_m03r_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV5RouteError(str(exc)) from exc
        if self.setting_index != setting.setting_index:
            raise M03RV5RouteError("route setting_index does not match protocol")
        expected = (
            _objective_route_id(setting),
            _model_route_id(setting),
            _ensemble_route_id(setting),
            _execution_route_id(setting),
        )
        observed = (
            self.objective_route_id,
            self.model_route_id,
            self.ensemble_route_id,
            self.execution_route_id,
        )
        if observed != expected:
            raise M03RV5RouteError("route identifiers do not match setting semantics")
        expected_status = (
            "known-missing"
            if self.setting_id in M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS
            else "available-unqualified"
        )
        if self.implementation_status != expected_status:
            raise M03RV5RouteError(
                "implementation_status cannot overstate the known route surface"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V5_ROUTE_SCHEMA,
            **asdict(self),
        }

    @property
    def route_contract_sha256(self) -> str:
        return _sha256(self.canonical_payload())


def _build_route(setting: M03RSetting) -> M03RV5SettingRoute:
    status: Literal["available-unqualified", "known-missing"] = (
        "known-missing"
        if setting.setting_id in M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS
        else "available-unqualified"
    )
    return M03RV5SettingRoute(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        objective_route_id=_objective_route_id(setting),
        model_route_id=_model_route_id(setting),
        ensemble_route_id=_ensemble_route_id(setting),
        execution_route_id=_execution_route_id(setting),
        implementation_status=status,
    )


M03R_V5_SETTING_ROUTES = tuple(_build_route(setting) for setting in M03R_SETTINGS)
M03R_V5_SETTING_ROUTES_BY_ID = {
    route.setting_id: route for route in M03R_V5_SETTING_ROUTES
}

if tuple(route.setting_id for route in M03R_V5_SETTING_ROUTES) != M03R_SETTING_IDS:
    raise RuntimeError("M03R v5 governed routes must cover all settings in order")
if len(M03R_V5_SETTING_ROUTES_BY_ID) != 11:
    raise RuntimeError("M03R v5 governed route identities must be unique")


@dataclass(frozen=True, slots=True)
class M03RV5RouteQualificationReceipt:
    """Content-bound evidence that one available route passed qualification."""

    setting_id: str
    route_contract_sha256: str
    qualification_artifact_sha256: str
    source_tree_sha256: str
    receipt_sha256: str
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def __post_init__(self) -> None:
        for name in (
            "route_contract_sha256",
            "qualification_artifact_sha256",
            "source_tree_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        route = M03R_V5_SETTING_ROUTES_BY_ID.get(self.setting_id)
        if route is None:
            raise M03RV5RouteError("qualification setting_id is not registered")
        if self.protocol_generation != route.protocol_generation:
            raise M03RV5RouteError("qualification protocol generation drifted")
        if self.design_id != route.design_id:
            raise M03RV5RouteError("qualification design identity drifted")
        if route.implementation_status != "available-unqualified":
            raise M03RV5RouteError(
                f"route {route.setting_id!r} is known-missing and cannot be qualified"
            )
        if self.route_contract_sha256 != route.route_contract_sha256:
            raise M03RV5RouteError("qualification route contract hash drifted")
        if self.receipt_sha256 != self.recompute_receipt_sha256():
            raise M03RV5RouteError("route qualification receipt hash mismatch")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V5_ROUTE_QUALIFICATION_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "route_contract_sha256": self.route_contract_sha256,
            "qualification_artifact_sha256": self.qualification_artifact_sha256,
            "source_tree_sha256": self.source_tree_sha256,
        }

    def recompute_receipt_sha256(self) -> str:
        return _sha256(self.canonical_payload())


def build_m03r_v5_route_qualification_receipt(
    *,
    setting_id: str,
    qualification_artifact_sha256: str,
    source_tree_sha256: str,
) -> M03RV5RouteQualificationReceipt:
    """Seal qualification evidence only for a route marked available."""

    route = M03R_V5_SETTING_ROUTES_BY_ID.get(setting_id)
    if route is None:
        raise M03RV5RouteError("qualification setting_id is not registered")
    fields = {
        "setting_id": setting_id,
        "route_contract_sha256": route.route_contract_sha256,
        "qualification_artifact_sha256": qualification_artifact_sha256,
        "source_tree_sha256": source_tree_sha256,
        "protocol_generation": route.protocol_generation,
        "design_id": route.design_id,
    }
    unsigned = M03RV5RouteQualificationReceipt.__new__(M03RV5RouteQualificationReceipt)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt_sha256 = _sha256(unsigned.canonical_payload())
    return M03RV5RouteQualificationReceipt(
        **fields,
        receipt_sha256=receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class M03RV5AllSettingRouteStatus:
    """Fail-closed aggregate status for the eleven-setting experiment."""

    qualification_receipts: tuple[M03RV5RouteQualificationReceipt, ...]
    qualification_blockers: tuple[str, ...]
    launch_authorized: bool
    receipt_sha256: str
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V5_ROUTE_STATUS_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "route_contracts": [
                route.canonical_payload() for route in M03R_V5_SETTING_ROUTES
            ],
            "qualification_receipts": [
                receipt.canonical_payload() | {"receipt_sha256": receipt.receipt_sha256}
                for receipt in self.qualification_receipts
            ],
            "qualification_blockers": list(self.qualification_blockers),
            "launch_authorized": self.launch_authorized,
        }

    def __post_init__(self) -> None:
        if self.protocol_generation != M03R_PROTOCOL_GENERATION:
            raise M03RV5RouteError("route status protocol generation drifted")
        if self.design_id != M03R_DESIGN_ID:
            raise M03RV5RouteError("route status design identity drifted")
        canonical_receipts, canonical_blockers = _canonical_route_status_semantics(
            self.qualification_receipts
        )
        if self.qualification_receipts != canonical_receipts:
            raise M03RV5RouteError(
                "route status receipts do not match the canonical validated inventory"
            )
        if self.qualification_blockers != canonical_blockers:
            raise M03RV5RouteError(
                "route status blockers were not canonically derived from route evidence"
            )
        if self.launch_authorized != (not canonical_blockers):
            raise M03RV5RouteError(
                "launch authority was not canonically derived from route evidence"
            )
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV5RouteError("all-setting route status receipt hash mismatch")


def _canonical_route_status_semantics(
    qualification_receipts: Sequence[M03RV5RouteQualificationReceipt],
) -> tuple[
    tuple[M03RV5RouteQualificationReceipt, ...],
    tuple[str, ...],
]:
    """Revalidate evidence and derive the only authorized blocker inventory."""

    rows = tuple(qualification_receipts)
    if not all(isinstance(row, M03RV5RouteQualificationReceipt) for row in rows):
        raise M03RV5RouteError("qualification inventory contains an untyped receipt")
    # Dataclass construction is not a trust boundary: callers can manufacture
    # instances through ``__new__`` or deserialization. Re-run every semantic
    # and content-hash check at the aggregate authorization boundary.
    for row in rows:
        row.__post_init__()
    observed_ids = tuple(row.setting_id for row in rows)
    if len(set(observed_ids)) != len(rows):
        raise M03RV5RouteError("qualification receipts cannot duplicate a setting")
    expected_ids = tuple(
        setting_id for setting_id in M03R_SETTING_IDS if setting_id in observed_ids
    )
    if observed_ids != expected_ids:
        raise M03RV5RouteError(
            "qualification receipts must use canonical setting order"
        )
    blockers: list[str] = []
    receipts_by_setting = {row.setting_id: row for row in rows}
    for route in M03R_V5_SETTING_ROUTES:
        if route.implementation_status == "known-missing":
            blockers.append(f"missing-implementation:{route.setting_id}")
        elif route.setting_id not in receipts_by_setting:
            blockers.append(f"missing-qualification-receipt:{route.setting_id}")
    return rows, tuple(blockers)


def evaluate_m03r_v5_all_setting_routes(
    qualification_receipts: Sequence[M03RV5RouteQualificationReceipt],
) -> M03RV5AllSettingRouteStatus:
    """Validate exact receipts and report every missing implementation/evidence."""

    rows, blockers = _canonical_route_status_semantics(qualification_receipts)
    fields: dict[str, Any] = {
        "qualification_receipts": rows,
        "qualification_blockers": blockers,
        "launch_authorized": not blockers,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
    }
    unsigned = M03RV5AllSettingRouteStatus.__new__(M03RV5AllSettingRouteStatus)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt_sha256 = _sha256(unsigned.canonical_payload())
    return M03RV5AllSettingRouteStatus(
        **fields,
        receipt_sha256=receipt_sha256,
    )


__all__ = [
    "M03R_V5_KNOWN_MISSING_ROUTE_SETTINGS",
    "M03R_V5_ROUTE_QUALIFICATION_SCHEMA",
    "M03R_V5_ROUTE_SCHEMA",
    "M03R_V5_ROUTE_STATUS_SCHEMA",
    "M03R_V5_SETTING_ROUTES",
    "M03R_V5_SETTING_ROUTES_BY_ID",
    "M03RV5AllSettingRouteStatus",
    "M03RV5RouteError",
    "M03RV5RouteQualificationReceipt",
    "M03RV5SettingRoute",
    "build_m03r_v5_route_qualification_receipt",
    "evaluate_m03r_v5_all_setting_routes",
]
