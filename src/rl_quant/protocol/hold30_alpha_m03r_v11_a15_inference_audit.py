"""Frozen post-hoc inference audit for the completed M03R-v11 a15 panel.

The audit consumes exact update-64 checkpoint bytes and already-consumed v11
qualification data.  It cannot train, select, promote, authorize an economic
generation, or access 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
)

M03R_V11_A15_AUDIT_GENERATION = "top2000-dev-m03r-v11-a15-posthoc-inference-audit-v2"
M03R_V11_A15_AUDIT_DESIGN_ID = (
    "exact-checkpoint-controls-calibration-action-attribution-semantic-risk-v2"
)
M03R_V11_A15_AUDIT_SETTING_INDEXES = (0, 1)
M03R_V11_A15_AUDIT_HORIZONS = (21, 30)
M03R_V11_A15_AUDIT_COST_BASIS_POINTS = (0, 10, 20, 40)
M03R_V11_A15_AUDIT_QUANTILE_COUNTS = (10, 20)
M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS = (10, 21, 30)
M03R_V11_A15_AUDIT_STARTUP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-startup-v2"
)
M03R_V11_A15_AUDIT_CURSOR_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-cursor-artifact-v2"
)
M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-worker-terminal-v2"
)
M03R_V11_A15_AUDIT_WORKER_ERROR_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-worker-error-v2"
)
M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-capacity-terminal-v2"
)
M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-static-terminal-v2"
)
M03R_V11_A15_AUDIT_FOLD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-fold-v2"
)
M03R_V11_A15_AUDIT_PANEL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-panel-v2"
)


class M03RV11A15InferenceAuditProtocolError(ValueError):
    """The immutable a15 post-hoc audit contract drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditVariant:
    variant_id: str
    signal_transform: Literal["original", "zero", "sign-flipped", "shuffled"]
    maximum_incremental_one_way_turnover: float

    def validate(self) -> None:
        expected = {
            "original-cap-200bp": ("original", 0.020),
            "original-cap-150bp": ("original", 0.015),
            "original-cap-100bp": ("original", 0.010),
            "original-cap-050bp": ("original", 0.005),
            "zero-signal-cap-200bp": ("zero", 0.020),
            "sign-flipped-cap-200bp": ("sign-flipped", 0.020),
            "shuffled-cap-200bp": ("shuffled", 0.020),
        }
        if expected.get(self.variant_id) != (
            self.signal_transform,
            self.maximum_incremental_one_way_turnover,
        ):
            raise M03RV11A15InferenceAuditProtocolError("a15 audit variant drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


M03R_V11_A15_AUDIT_VARIANTS = (
    M03RV11A15AuditVariant("original-cap-200bp", "original", 0.020),
    M03RV11A15AuditVariant("original-cap-150bp", "original", 0.015),
    M03RV11A15AuditVariant("original-cap-100bp", "original", 0.010),
    M03RV11A15AuditVariant("original-cap-050bp", "original", 0.005),
    M03RV11A15AuditVariant("zero-signal-cap-200bp", "zero", 0.020),
    M03RV11A15AuditVariant("sign-flipped-cap-200bp", "sign-flipped", 0.020),
    M03RV11A15AuditVariant("shuffled-cap-200bp", "shuffled", 0.020),
)


@dataclass(frozen=True, slots=True)
class M03RV11A15InferenceAuditSpec:
    source_protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    audited_setting_indexes: tuple[int, ...] = M03R_V11_A15_AUDIT_SETTING_INDEXES
    audited_horizons: tuple[int, ...] = M03R_V11_A15_AUDIT_HORIZONS
    cost_basis_points: tuple[int, ...] = M03R_V11_A15_AUDIT_COST_BASIS_POINTS
    quantile_counts: tuple[int, ...] = M03R_V11_A15_AUDIT_QUANTILE_COUNTS
    bootstrap_blocks: tuple[int, ...] = M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 202_608_13
    shuffle_seed: int = 17_011
    calibration_bins: int = 10
    qualification_data_status: str = "already-consumed-posthoc-diagnostic"
    checkpoint_rule: str = "exact-update-64-write-reload-evaluate-by-file-sha256"
    control_rule: str = "predeclared-target-blind-signal-controls-v1"
    economic_optimizer_updates: int = 0
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False
    generation: str = M03R_V11_A15_AUDIT_GENERATION
    design_id: str = M03R_V11_A15_AUDIT_DESIGN_ID

    def validate(self) -> None:
        for variant in M03R_V11_A15_AUDIT_VARIANTS:
            variant.validate()
        if (
            self.source_protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.audited_setting_indexes != (0, 1)
            or self.audited_horizons != (21, 30)
            or self.cost_basis_points != (0, 10, 20, 40)
            or self.quantile_counts != (10, 20)
            or self.bootstrap_blocks != (10, 21, 30)
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 202_608_13
            or self.shuffle_seed != 17_011
            or self.calibration_bins != 10
            or self.qualification_data_status != "already-consumed-posthoc-diagnostic"
            or self.checkpoint_rule
            != "exact-update-64-write-reload-evaluate-by-file-sha256"
            or self.control_rule != "predeclared-target-blind-signal-controls-v1"
            or self.economic_optimizer_updates != 0
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotable
            or self.generation != M03R_V11_A15_AUDIT_GENERATION
            or self.design_id != M03R_V11_A15_AUDIT_DESIGN_ID
        ):
            raise M03RV11A15InferenceAuditProtocolError(
                "a15 inference-audit specification drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        payload = asdict(self)
        payload["variant_receipt_sha256"] = tuple(
            row.receipt_sha256 for row in M03R_V11_A15_AUDIT_VARIANTS
        )
        return _sha256(payload)


M03R_V11_A15_INFERENCE_AUDIT_SPEC = M03RV11A15InferenceAuditSpec()
M03R_V11_A15_INFERENCE_AUDIT_SPEC.validate()
M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256 = (
    M03R_V11_A15_INFERENCE_AUDIT_SPEC.receipt_sha256
)


def resolve_m03r_v11_a15_audit_variant(
    value: str,
) -> M03RV11A15AuditVariant:
    for variant in M03R_V11_A15_AUDIT_VARIANTS:
        if variant.variant_id == value:
            return variant
    raise M03RV11A15InferenceAuditProtocolError(
        f"unknown a15 inference-audit variant: {value}"
    )


__all__ = [
    "M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS",
    "M03R_V11_A15_AUDIT_COST_BASIS_POINTS",
    "M03R_V11_A15_AUDIT_DESIGN_ID",
    "M03R_V11_A15_AUDIT_GENERATION",
    "M03R_V11_A15_AUDIT_HORIZONS",
    "M03R_V11_A15_AUDIT_QUANTILE_COUNTS",
    "M03R_V11_A15_AUDIT_SETTING_INDEXES",
    "M03R_V11_A15_AUDIT_VARIANTS",
    "M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256",
    "M03R_V11_A15_INFERENCE_AUDIT_SPEC",
    "M03RV11A15AuditVariant",
    "M03RV11A15InferenceAuditProtocolError",
    "M03RV11A15InferenceAuditSpec",
    "resolve_m03r_v11_a15_audit_variant",
]
