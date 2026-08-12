"""Fail-closed source readiness for TOP2000 M03R-v9 risk inputs.

The predictive panel needs the same decision-origin exposure families for
factor-residual labels and the deterministic simple sleeve.  The legacy
TOP2000 cache does not contain a point-in-time sector classification by
itself; the separately hash-bound Polygon/cache materializer can supply that
surface.  This module keeps any remaining source or projector mismatch as an
explicit prelaunch receipt instead of allowing a worker to synthesize inputs
or spend GPU time before discovering the mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
)

M03R_V9_RISK_SOURCE_READINESS_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-risk-source-readiness-v1"
)


class M03RV9RiskSourceError(ValueError):
    """The predictive risk source is incomplete or its identity drifted."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV9RiskSourceError("risk-source identity is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV9RiskSourceInventory:
    """Exact metadata inventory available before package or Job creation."""

    source_id: str
    source_schema_sha256: str
    asset_axis_sha256: str
    source_columns: tuple[str, ...]
    sector_exposure_names: tuple[str, ...]
    style_risk_exposure_names: tuple[str, ...]
    active_beta_exposure_name: str | None
    point_in_time_sector_receipt_sha256: str | None
    point_in_time_style_risk_receipt_sha256: str | None
    point_in_time_active_beta_receipt_sha256: str | None
    origin_availability_receipt_sha256: str | None
    projector_manifest_sha256: str | None
    target_projector_exposure_names_match: bool

    def validate(self) -> None:
        if (
            not self.source_id
            or not self.source_columns
            or len(set(self.source_columns)) != len(self.source_columns)
            or len(set(self.sector_exposure_names)) != len(self.sector_exposure_names)
            or len(set(self.style_risk_exposure_names))
            != len(self.style_risk_exposure_names)
        ):
            raise M03RV9RiskSourceError("risk-source inventory is malformed")
        _digest(self.source_schema_sha256)
        _digest(self.asset_axis_sha256)
        for value in (
            self.point_in_time_sector_receipt_sha256,
            self.point_in_time_style_risk_receipt_sha256,
            self.point_in_time_active_beta_receipt_sha256,
            self.origin_availability_receipt_sha256,
            self.projector_manifest_sha256,
        ):
            if value is not None:
                _digest(value)


def _blockers(inventory: M03RV9RiskSourceInventory) -> tuple[str, ...]:
    blockers: list[str] = []
    if not inventory.sector_exposure_names:
        blockers.append("missing-point-in-time-sector-classification")
    if not inventory.style_risk_exposure_names:
        blockers.append("missing-point-in-time-style-risk-exposures")
    if inventory.active_beta_exposure_name is None:
        blockers.append("missing-point-in-time-active-beta")
    if inventory.point_in_time_sector_receipt_sha256 is None:
        blockers.append("missing-point-in-time-sector-receipt")
    if inventory.point_in_time_style_risk_receipt_sha256 is None:
        blockers.append("missing-point-in-time-style-risk-receipt")
    if inventory.point_in_time_active_beta_receipt_sha256 is None:
        blockers.append("missing-point-in-time-active-beta-receipt")
    if inventory.origin_availability_receipt_sha256 is None:
        blockers.append("missing-origin-availability-receipt")
    if inventory.projector_manifest_sha256 is None:
        blockers.append("missing-projector-manifest")
    if not inventory.target_projector_exposure_names_match:
        blockers.append("target-projector-exposure-name-mismatch")
    return tuple(blockers)


@dataclass(frozen=True, slots=True)
class M03RV9RiskSourceReadiness:
    source_inventory_sha256: str
    blocker_codes: tuple[str, ...]
    required_exposure_families: tuple[str, ...]
    predictive_worker_authorized: bool
    economic_panel_authorized: bool = False
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_RISK_SOURCE_READINESS_SCHEMA

    def validate(self) -> None:
        _digest(self.source_inventory_sha256)
        if (
            len(set(self.blocker_codes)) != len(self.blocker_codes)
            or self.required_exposure_families
            != M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
            or self.predictive_worker_authorized != (not self.blocker_codes)
            or self.economic_panel_authorized
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.schema != M03R_V9_RISK_SOURCE_READINESS_SCHEMA
        ):
            raise M03RV9RiskSourceError("risk-source readiness receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))

    def require_predictive_worker_authorized(self) -> None:
        self.validate()
        if not self.predictive_worker_authorized:
            raise M03RV9RiskSourceError(
                "V9 predictive worker is blocked: " + ",".join(self.blocker_codes)
            )


def audit_m03r_v9_risk_source(
    inventory: M03RV9RiskSourceInventory,
) -> M03RV9RiskSourceReadiness:
    inventory.validate()
    blockers = _blockers(inventory)
    result = M03RV9RiskSourceReadiness(
        source_inventory_sha256=_canonical_sha256(asdict(inventory)),
        blocker_codes=blockers,
        required_exposure_families=M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
        predictive_worker_authorized=not blockers,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V9_RISK_SOURCE_READINESS_SCHEMA",
    "M03RV9RiskSourceError",
    "M03RV9RiskSourceInventory",
    "M03RV9RiskSourceReadiness",
    "audit_m03r_v9_risk_source",
]
