"""Causal post-fill targets and origin-time residual operators.

This module owns the semantics shared by every future alpha model.  It keeps
the fill convention, economic outcome path, target horizon, asset axis, and
origin-time factor operator explicit so a training adapter cannot silently
change any of them while materializing tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from rl_quant.alpha.accounting import (
    EconomicValuePoint,
    TotalReturnTarget,
    compute_post_fill_total_return,
)
from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.protocol.canonical_artifact import semantic_sha256


ALPHA_TARGET_SPEC_SCHEMA = "rl-quant.alpha-target-spec-v1"
ORIGIN_RESIDUAL_OPERATOR_SCHEMA = "rl-quant.origin-residual-operator-v1"
MULTI_HORIZON_TARGET_SCHEMA = "rl-quant.multi-horizon-target-v1"


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PITAlphaDataError(f"{name} must be a non-empty canonical string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PITAlphaDataError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise PITAlphaDataError(f"{name} is outside its finite domain")
    return result


@dataclass(frozen=True, slots=True)
class DecisionFillConvention:
    """One frozen relationship between observations, decisions, and fills."""

    convention_id: str
    decision_rule: Literal["after-close", "pre-close-cutoff"]
    fill_rule: Literal["next-open-auction", "next-open-vwap", "same-close-auction"]
    target_begins_after_fill: bool = True
    input_cutoff_precedes_decision: bool = True

    def validate(self) -> None:
        _canonical_text("fill convention ID", self.convention_id)
        if self.decision_rule not in {"after-close", "pre-close-cutoff"}:
            raise PITAlphaDataError("decision rule is unsupported")
        allowed = {
            "after-close": {"next-open-auction", "next-open-vwap"},
            "pre-close-cutoff": {"same-close-auction"},
        }
        if self.fill_rule not in allowed[self.decision_rule]:
            raise PITAlphaDataError("decision and fill rules form a lookahead-prone pair")
        if not self.target_begins_after_fill or not self.input_cutoff_precedes_decision:
            raise PITAlphaDataError("alpha targets must be post-fill and causally observed")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "convention_id": self.convention_id,
            "decision_rule": self.decision_rule,
            "fill_rule": self.fill_rule,
            "target_begins_after_fill": self.target_begins_after_fill,
            "input_cutoff_precedes_decision": self.input_cutoff_precedes_decision,
        }


@dataclass(frozen=True, slots=True)
class AlphaTargetSpec:
    """Frozen multi-horizon target semantics for one experiment generation."""

    primary_horizon_sessions: int
    auxiliary_horizons_sessions: tuple[int, ...]
    fill_convention: DecisionFillConvention
    return_kind: Literal["economic-total-simple", "economic-total-log"]
    target_mode: Literal["factor-residual"]
    terminal_outcomes_included: bool
    future_survival_required: bool
    schema: str = ALPHA_TARGET_SPEC_SCHEMA

    def validate(self) -> None:
        self.fill_convention.validate()
        if (
            isinstance(self.primary_horizon_sessions, bool)
            or not isinstance(self.primary_horizon_sessions, int)
            or self.primary_horizon_sessions <= 0
        ):
            raise PITAlphaDataError("primary target horizon must be positive")
        if (
            tuple(sorted(set(self.auxiliary_horizons_sessions)))
            != self.auxiliary_horizons_sessions
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.auxiliary_horizons_sessions
            )
            or self.primary_horizon_sessions in self.auxiliary_horizons_sessions
        ):
            raise PITAlphaDataError("auxiliary target horizons must be positive, sorted, and unique")
        horizons = self.horizons
        if tuple(sorted(set(horizons))) != horizons:
            raise PITAlphaDataError("target horizons must be sorted and unique")
        if self.primary_horizon_sessions not in horizons:
            raise PITAlphaDataError("primary horizon is absent from the target panel")
        if self.return_kind not in {"economic-total-simple", "economic-total-log"}:
            raise PITAlphaDataError("return kind is unsupported")
        if self.target_mode != "factor-residual":
            raise PITAlphaDataError("reportable alpha targets must be factor residual")
        if not self.terminal_outcomes_included or self.future_survival_required:
            raise PITAlphaDataError("target support must include economic terminal outcomes")
        if self.schema != ALPHA_TARGET_SPEC_SCHEMA:
            raise PITAlphaDataError("alpha target schema drifted")

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted({self.primary_horizon_sessions, *self.auxiliary_horizons_sessions}))

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(
            {
                "schema": self.schema,
                "primary_horizon_sessions": self.primary_horizon_sessions,
                "auxiliary_horizons_sessions": self.auxiliary_horizons_sessions,
                "fill_convention": self.fill_convention.payload(),
                "return_kind": self.return_kind,
                "target_mode": self.target_mode,
                "terminal_outcomes_included": self.terminal_outcomes_included,
                "future_survival_required": self.future_survival_required,
            }
        )


@dataclass(frozen=True, slots=True)
class OriginExposurePanel:
    """Factor inputs that were observable at one decision origin."""

    origin_at_ms: int
    available_at_ms: int
    asset_ids: tuple[str, ...]
    exposure_names: tuple[str, ...]
    exposures: tuple[tuple[float, ...], ...]
    regression_weights: tuple[float, ...]
    qualified_asset_mask: tuple[bool, ...]
    source_receipt_sha256: str

    def validate(self) -> None:
        if (
            isinstance(self.origin_at_ms, bool)
            or not isinstance(self.origin_at_ms, int)
            or self.origin_at_ms < 0
            or isinstance(self.available_at_ms, bool)
            or not isinstance(self.available_at_ms, int)
            or self.available_at_ms < 0
            or self.available_at_ms > self.origin_at_ms
        ):
            raise PITAlphaDataError("factor exposures were unavailable at the origin")
        _digest("exposure source receipt", self.source_receipt_sha256)
        if (
            not self.asset_ids
            or tuple(sorted(self.asset_ids)) != self.asset_ids
            or len(set(self.asset_ids)) != len(self.asset_ids)
            or any(not value or value != value.strip() for value in self.asset_ids)
        ):
            raise PITAlphaDataError("exposure asset axis must be sorted and unique")
        if (
            not self.exposure_names
            or self.exposure_names[0] != "intercept"
            or len(set(self.exposure_names)) != len(self.exposure_names)
        ):
            raise PITAlphaDataError("exposures must include one leading intercept")
        assets = len(self.asset_ids)
        factors = len(self.exposure_names)
        if (
            len(self.exposures) != assets
            or len(self.regression_weights) != assets
            or len(self.qualified_asset_mask) != assets
        ):
            raise PITAlphaDataError("exposure panel asset dimensions disagree")
        for row in self.exposures:
            if len(row) != factors:
                raise PITAlphaDataError("exposure panel factor dimensions disagree")
            for value in row:
                _finite("factor exposure", value)
            if not math.isclose(row[0], 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise PITAlphaDataError("the declared intercept column is not one")
        for weight in self.regression_weights:
            _finite("regression weight", weight, positive=True)
        if any(not isinstance(value, bool) for value in self.qualified_asset_mask):
            raise PITAlphaDataError("qualified exposure mask must be boolean")
        if sum(self.qualified_asset_mask) <= factors:
            raise PITAlphaDataError("factor residualization has no residual degrees of freedom")


@dataclass(frozen=True, slots=True)
class OriginResidualOperator:
    """Compact weighted least-squares map frozen at one decision origin."""

    origin_at_ms: int
    asset_ids: tuple[str, ...]
    qualified_asset_mask: tuple[bool, ...]
    exposure_names: tuple[str, ...]
    qualified_design: tuple[tuple[float, ...], ...]
    qualified_weights: tuple[float, ...]
    coefficient_map: tuple[tuple[float, ...], ...]
    source_receipt_sha256: str
    receipt_sha256: str
    schema: str = ORIGIN_RESIDUAL_OPERATOR_SCHEMA

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "origin_at_ms": self.origin_at_ms,
            "asset_ids": self.asset_ids,
            "qualified_asset_mask": self.qualified_asset_mask,
            "exposure_names": self.exposure_names,
            "qualified_design": self.qualified_design,
            "qualified_weights": self.qualified_weights,
            "coefficient_map": self.coefficient_map,
            "source_receipt_sha256": self.source_receipt_sha256,
        }

    def validate(self) -> None:
        _digest("operator source receipt", self.source_receipt_sha256)
        _digest("operator receipt", self.receipt_sha256)
        if self.schema != ORIGIN_RESIDUAL_OPERATOR_SCHEMA:
            raise PITAlphaDataError("origin residual operator schema drifted")
        if (
            isinstance(self.origin_at_ms, bool)
            or not isinstance(self.origin_at_ms, int)
            or self.origin_at_ms < 0
            or not self.asset_ids
            or tuple(sorted(set(self.asset_ids))) != self.asset_ids
            or len(self.qualified_asset_mask) != len(self.asset_ids)
            or any(not isinstance(value, bool) for value in self.qualified_asset_mask)
            or not self.exposure_names
            or self.exposure_names[0] != "intercept"
            or len(set(self.exposure_names)) != len(self.exposure_names)
        ):
            raise PITAlphaDataError("origin residual operator identity drifted")
        selected = sum(self.qualified_asset_mask)
        factors = len(self.exposure_names)
        if (
            selected <= factors
            or len(self.qualified_design) != selected
            or len(self.qualified_weights) != selected
            or len(self.coefficient_map) != factors
            or any(len(row) != factors for row in self.qualified_design)
            or any(len(row) != selected for row in self.coefficient_map)
        ):
            raise PITAlphaDataError("origin residual operator dimensions drifted")
        design = np.asarray(self.qualified_design, dtype=np.float64)
        coefficient = np.asarray(self.coefficient_map, dtype=np.float64)
        if not np.isfinite(design).all() or not np.isfinite(coefficient).all():
            raise PITAlphaDataError("origin residual operator is nonfinite")
        identity = coefficient @ design
        if not np.allclose(identity, np.eye(factors), rtol=0.0, atol=2e-10):
            raise PITAlphaDataError("origin residual operator does not reproduce identity")
        if self.receipt_sha256 != semantic_sha256(self._payload()):
            raise PITAlphaDataError("origin residual operator receipt drifted")


def build_origin_residual_operator(panel: OriginExposurePanel) -> OriginResidualOperator:
    """Build a stable weighted-QR residual map from origin-available exposures."""

    panel.validate()
    selected = np.flatnonzero(np.asarray(panel.qualified_asset_mask, dtype=np.bool_))
    design = np.asarray(panel.exposures, dtype=np.float64)[selected]
    weights = np.asarray(panel.regression_weights, dtype=np.float64)[selected]
    root_weight = np.sqrt(weights)
    weighted_design = root_weight[:, None] * design
    q_value, r_value = np.linalg.qr(weighted_design, mode="reduced")
    if np.linalg.matrix_rank(r_value) != r_value.shape[0]:
        raise PITAlphaDataError("origin factor design is rank deficient")
    coefficient = np.linalg.solve(r_value, q_value.T * root_weight[None, :])
    qualified_design = tuple(
        tuple(float(value) for value in row) for row in design
    )
    qualified_weights = tuple(float(value) for value in weights)
    coefficient_map = tuple(
        tuple(float(value) for value in row) for row in coefficient
    )
    payload = {
        "schema": ORIGIN_RESIDUAL_OPERATOR_SCHEMA,
        "origin_at_ms": panel.origin_at_ms,
        "asset_ids": panel.asset_ids,
        "qualified_asset_mask": panel.qualified_asset_mask,
        "exposure_names": panel.exposure_names,
        "qualified_design": qualified_design,
        "qualified_weights": qualified_weights,
        "coefficient_map": coefficient_map,
        "source_receipt_sha256": panel.source_receipt_sha256,
    }
    result = OriginResidualOperator(
        origin_at_ms=panel.origin_at_ms,
        asset_ids=panel.asset_ids,
        qualified_asset_mask=panel.qualified_asset_mask,
        exposure_names=panel.exposure_names,
        qualified_design=qualified_design,
        qualified_weights=qualified_weights,
        coefficient_map=coefficient_map,
        source_receipt_sha256=panel.source_receipt_sha256,
        receipt_sha256=semantic_sha256(payload),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class ResidualizedCrossSection:
    values: tuple[float, ...]
    qualified_asset_mask: tuple[bool, ...]
    operator_receipt_sha256: str
    maximum_weighted_exposure_error: float

    def validate(self, operator: OriginResidualOperator) -> None:
        operator.validate()
        if (
            len(self.values) != len(operator.asset_ids)
            or self.qualified_asset_mask != operator.qualified_asset_mask
            or self.operator_receipt_sha256 != operator.receipt_sha256
            or not all(math.isfinite(value) for value in self.values)
            or not math.isfinite(self.maximum_weighted_exposure_error)
            or self.maximum_weighted_exposure_error > 2e-10
        ):
            raise PITAlphaDataError("residualized cross-section drifted")


def apply_origin_residual_operator(
    values: Sequence[float],
    operator: OriginResidualOperator,
) -> ResidualizedCrossSection:
    """Apply the same executable factor-null map to a score or target."""

    operator.validate()
    if len(values) != len(operator.asset_ids):
        raise PITAlphaDataError("cross-section does not match the residual asset axis")
    normalized = np.asarray(values, dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise PITAlphaDataError("cross-section contains a nonfinite value")
    mask = np.asarray(operator.qualified_asset_mask, dtype=np.bool_)
    selected = normalized[mask]
    design = np.asarray(operator.qualified_design, dtype=np.float64)
    weights = np.asarray(operator.qualified_weights, dtype=np.float64)
    coefficient = np.asarray(operator.coefficient_map, dtype=np.float64)
    residual = selected - design @ (coefficient @ selected)
    result_values = np.zeros_like(normalized)
    result_values[mask] = residual
    exposure_error = float(np.max(np.abs(design.T @ (weights * residual))))
    result = ResidualizedCrossSection(
        values=tuple(float(value) for value in result_values),
        qualified_asset_mask=operator.qualified_asset_mask,
        operator_receipt_sha256=operator.receipt_sha256,
        maximum_weighted_exposure_error=exposure_error,
    )
    result.validate(operator)
    return result


@dataclass(frozen=True, slots=True)
class SecurityMultiHorizonTarget:
    security_id: str
    decision_at_ms: int
    fill_at_ms: int
    label_available_at_ms: int
    targets: tuple[TotalReturnTarget, ...]
    target_spec_receipt_sha256: str
    receipt_sha256: str
    schema: str = MULTI_HORIZON_TARGET_SCHEMA

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "security_id": self.security_id,
            "decision_at_ms": self.decision_at_ms,
            "fill_at_ms": self.fill_at_ms,
            "label_available_at_ms": self.label_available_at_ms,
            "targets": tuple(
                {
                    "fill_session_index": target.fill_session_index,
                    "horizon_sessions": target.horizon_sessions,
                    "start_value": target.start_value,
                    "end_value": target.end_value,
                    "simple_return": target.simple_return,
                    "log_return": target.log_return,
                    "terminal_zero_value": target.terminal_zero_value,
                }
                for target in self.targets
            ),
            "target_spec_receipt_sha256": self.target_spec_receipt_sha256,
        }

    def validate(self, spec: AlphaTargetSpec) -> None:
        spec.validate()
        _canonical_text("target security ID", self.security_id)
        _digest("target spec receipt", self.target_spec_receipt_sha256)
        _digest("multi-horizon target receipt", self.receipt_sha256)
        if (
            self.schema != MULTI_HORIZON_TARGET_SCHEMA
            or self.target_spec_receipt_sha256 != spec.receipt_sha256
            or self.fill_at_ms <= self.decision_at_ms
            or self.label_available_at_ms < self.fill_at_ms
            or tuple(target.horizon_sessions for target in self.targets) != spec.horizons
        ):
            raise PITAlphaDataError("multi-horizon target chronology or identity drifted")
        for target in self.targets:
            target.validate()
            if spec.return_kind == "economic-total-log" and target.log_return is None:
                raise PITAlphaDataError("log target cannot represent an exact total loss")
        if self.receipt_sha256 != semantic_sha256(self._payload()):
            raise PITAlphaDataError("multi-horizon target receipt drifted")


def build_security_multi_horizon_target(
    *,
    security_id: str,
    decision_at_ms: int,
    fill_at_ms: int,
    fill_session_index: int,
    points: Sequence[EconomicValuePoint],
    spec: AlphaTargetSpec,
    built_at_ms: int,
) -> SecurityMultiHorizonTarget:
    """Build every frozen horizon from one complete economic value path."""

    spec.validate()
    _canonical_text("target security ID", security_id)
    if fill_at_ms <= decision_at_ms:
        raise PITAlphaDataError("fill must occur after the target decision")
    targets = tuple(
        compute_post_fill_total_return(
            points,
            fill_session_index=fill_session_index,
            horizon_sessions=horizon,
        )
        for horizon in spec.horizons
    )
    end_index = fill_session_index + max(spec.horizons)
    end_points = [point for point in points if point.session_index == end_index]
    if len(end_points) != 1:
        raise PITAlphaDataError("target path lacks its maximum-horizon endpoint")
    label_available_at_ms = end_points[0].available_at_ms
    if built_at_ms < label_available_at_ms:
        raise PITAlphaDataError("target was built before its economic endpoint was available")
    payload = {
        "schema": MULTI_HORIZON_TARGET_SCHEMA,
        "security_id": security_id,
        "decision_at_ms": decision_at_ms,
        "fill_at_ms": fill_at_ms,
        "label_available_at_ms": label_available_at_ms,
        "targets": tuple(
            {
                "fill_session_index": target.fill_session_index,
                "horizon_sessions": target.horizon_sessions,
                "start_value": target.start_value,
                "end_value": target.end_value,
                "simple_return": target.simple_return,
                "log_return": target.log_return,
                "terminal_zero_value": target.terminal_zero_value,
            }
            for target in targets
        ),
        "target_spec_receipt_sha256": spec.receipt_sha256,
    }
    result = SecurityMultiHorizonTarget(
        security_id=security_id,
        decision_at_ms=decision_at_ms,
        fill_at_ms=fill_at_ms,
        label_available_at_ms=label_available_at_ms,
        targets=targets,
        target_spec_receipt_sha256=spec.receipt_sha256,
        receipt_sha256=semantic_sha256(payload),
    )
    result.validate(spec)
    return result


def target_cross_section(
    rows: Sequence[SecurityMultiHorizonTarget],
    *,
    asset_ids: Sequence[str],
    horizon_sessions: int,
    return_kind: Literal["economic-total-simple", "economic-total-log"],
) -> tuple[float, ...]:
    """Align one horizon to a permanent asset axis without filling missing labels."""

    by_security = {row.security_id: row for row in rows}
    if len(by_security) != len(rows) or tuple(asset_ids) != tuple(sorted(set(asset_ids))):
        raise PITAlphaDataError("target cross-section asset identities are ambiguous")
    result: list[float] = []
    for security_id in asset_ids:
        row = by_security.get(security_id)
        if row is None:
            raise PITAlphaDataError("target cross-section cannot fabricate a missing asset")
        matches = [
            target for target in row.targets if target.horizon_sessions == horizon_sessions
        ]
        if len(matches) != 1:
            raise PITAlphaDataError("requested target horizon is unavailable")
        target = matches[0]
        value = target.simple_return if return_kind == "economic-total-simple" else target.log_return
        if value is None:
            raise PITAlphaDataError("log target cannot silently replace a total economic loss")
        result.append(value)
    return tuple(result)


__all__ = [
    "ALPHA_TARGET_SPEC_SCHEMA",
    "MULTI_HORIZON_TARGET_SCHEMA",
    "ORIGIN_RESIDUAL_OPERATOR_SCHEMA",
    "AlphaTargetSpec",
    "DecisionFillConvention",
    "OriginExposurePanel",
    "OriginResidualOperator",
    "ResidualizedCrossSection",
    "SecurityMultiHorizonTarget",
    "apply_origin_residual_operator",
    "build_origin_residual_operator",
    "build_security_multi_horizon_target",
    "target_cross_section",
]
