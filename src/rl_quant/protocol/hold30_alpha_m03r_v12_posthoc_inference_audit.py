"""Frozen post-hoc audit contract for the completed M03R-v12 panel.

The audit replays exact update-64 checkpoint outputs on already-consumed,
pre-2026 qualification data.  It cannot train, select a checkpoint, mint an
economic generation, or authorize access to 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
)

M03R_V12_POSTHOC_AUDIT_GENERATION = (
    "top2000-dev-m03r-v12-posthoc-inference-audit-v1"
)
M03R_V12_POSTHOC_AUDIT_DESIGN_ID = (
    "exact-v12-checkpoints-rank-vs-mean-causal-mask-corrected-chronology-v1"
)
M03R_V12_POSTHOC_AUDIT_HORIZON = 3
M03R_V12_POSTHOC_AUDIT_SCORE_CHANNELS = ("economic-mean", "rank-score")
M03R_V12_POSTHOC_AUDIT_SIGNAL_TRANSFORMS = (
    "original",
    "zero",
    "sign-flipped",
    "shuffled",
)
M03R_V12_POSTHOC_AUDIT_ACTIVE_MASS_CAPS = (0.0025, 0.005, 0.010, 0.020)
M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS = (0, 10, 20, 40)
M03R_V12_POSTHOC_AUDIT_BOOTSTRAP_BLOCKS = (10, 21, 30)
M03R_V12_POSTHOC_AUDIT_INPUT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-input-v1"
)
M03R_V12_POSTHOC_AUDIT_FOLD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-fold-v1"
)
M03R_V12_POSTHOC_AUDIT_PANEL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-panel-v1"
)


class M03RV12PosthocAuditProtocolError(ValueError):
    """The immutable v12 post-hoc audit contract drifted."""


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
class M03RV12PosthocAuditVariant:
    score_channel: Literal["economic-mean", "rank-score"]
    signal_transform: Literal["original", "zero", "sign-flipped", "shuffled"]
    maximum_active_one_way_mass: float

    def validate(self) -> None:
        if (
            self.score_channel not in M03R_V12_POSTHOC_AUDIT_SCORE_CHANNELS
            or self.signal_transform not in M03R_V12_POSTHOC_AUDIT_SIGNAL_TRANSFORMS
            or self.maximum_active_one_way_mass
            not in M03R_V12_POSTHOC_AUDIT_ACTIVE_MASS_CAPS
            or (
                self.signal_transform != "original"
                and self.maximum_active_one_way_mass != 0.020
            )
        ):
            raise M03RV12PosthocAuditProtocolError(
                "v12 post-hoc audit variant drifted"
            )

    @property
    def variant_id(self) -> str:
        self.validate()
        basis_points = round(self.maximum_active_one_way_mass * 10_000)
        return f"{self.score_channel}-{self.signal_transform}-cap-{basis_points:03d}bp"

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


M03R_V12_POSTHOC_AUDIT_VARIANTS = tuple(
    M03RV12PosthocAuditVariant(channel, transform, cap)  # type: ignore[arg-type]
    for channel in M03R_V12_POSTHOC_AUDIT_SCORE_CHANNELS
    for transform in M03R_V12_POSTHOC_AUDIT_SIGNAL_TRANSFORMS
    for cap in M03R_V12_POSTHOC_AUDIT_ACTIVE_MASS_CAPS
    if transform == "original" or cap == 0.020
)


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditSpec:
    source_protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    audited_setting_indexes: tuple[int, ...] = (0, 1, 2)
    audited_horizon_sessions: int = M03R_V12_POSTHOC_AUDIT_HORIZON
    score_channels: tuple[str, ...] = M03R_V12_POSTHOC_AUDIT_SCORE_CHANNELS
    signal_transforms: tuple[str, ...] = M03R_V12_POSTHOC_AUDIT_SIGNAL_TRANSFORMS
    active_mass_caps: tuple[float, ...] = M03R_V12_POSTHOC_AUDIT_ACTIVE_MASS_CAPS
    cost_basis_points: tuple[int, ...] = M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS
    bootstrap_blocks: tuple[int, ...] = M03R_V12_POSTHOC_AUDIT_BOOTSTRAP_BLOCKS
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 202_608_13
    shuffle_seed: int = 12_017
    quantile_count: int = 10
    action_mask_rule: str = (
        "origin-decision-availability-and-origin-regression-weight-only-v1"
    )
    target_mask_rule: str = "future-label-validity-diagnostic-only-never-action-v1"
    chronology_rule: str = "decision-t-action-earns-return-t-plus-1-v1"
    allocation_rule: str = "benchmark-relative-long-only-rank-mass-ladder-v1"
    runtime_rule: str = "one-visible-nvidia-h100-80gb-hbm3-no-training-v1"
    qualification_data_status: str = "already-consumed-posthoc-diagnostic"
    checkpoint_rule: str = "exact-update-64-write-reload-evaluate-by-file-sha256"
    economic_optimizer_updates: int = 0
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False
    generation: str = M03R_V12_POSTHOC_AUDIT_GENERATION
    design_id: str = M03R_V12_POSTHOC_AUDIT_DESIGN_ID

    def validate(self) -> None:
        for variant in M03R_V12_POSTHOC_AUDIT_VARIANTS:
            variant.validate()
        if (
            self.source_protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.audited_setting_indexes != (0, 1, 2)
            or self.audited_horizon_sessions != 3
            or self.score_channels != ("economic-mean", "rank-score")
            or self.signal_transforms
            != ("original", "zero", "sign-flipped", "shuffled")
            or self.active_mass_caps != (0.0025, 0.005, 0.010, 0.020)
            or self.cost_basis_points != (0, 10, 20, 40)
            or self.bootstrap_blocks != (10, 21, 30)
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 202_608_13
            or self.shuffle_seed != 12_017
            or self.quantile_count != 10
            or self.action_mask_rule
            != "origin-decision-availability-and-origin-regression-weight-only-v1"
            or self.target_mask_rule
            != "future-label-validity-diagnostic-only-never-action-v1"
            or self.chronology_rule != "decision-t-action-earns-return-t-plus-1-v1"
            or self.allocation_rule
            != "benchmark-relative-long-only-rank-mass-ladder-v1"
            or self.runtime_rule
            != "one-visible-nvidia-h100-80gb-hbm3-no-training-v1"
            or self.qualification_data_status
            != "already-consumed-posthoc-diagnostic"
            or self.checkpoint_rule
            != "exact-update-64-write-reload-evaluate-by-file-sha256"
            or self.economic_optimizer_updates != 0
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotable
            or self.generation != M03R_V12_POSTHOC_AUDIT_GENERATION
            or self.design_id != M03R_V12_POSTHOC_AUDIT_DESIGN_ID
        ):
            raise M03RV12PosthocAuditProtocolError(
                "v12 post-hoc audit specification drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        payload = asdict(self)
        payload["variant_receipt_sha256"] = tuple(
            row.receipt_sha256 for row in M03R_V12_POSTHOC_AUDIT_VARIANTS
        )
        return _sha256(payload)


M03R_V12_POSTHOC_AUDIT_SPEC = M03RV12PosthocAuditSpec()
M03R_V12_POSTHOC_AUDIT_SPEC.validate()
M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256 = (
    M03R_V12_POSTHOC_AUDIT_SPEC.receipt_sha256
)


def resolve_m03r_v12_posthoc_audit_variant(
    variant_id: str,
) -> M03RV12PosthocAuditVariant:
    for variant in M03R_V12_POSTHOC_AUDIT_VARIANTS:
        if variant.variant_id == variant_id:
            return variant
    raise M03RV12PosthocAuditProtocolError(
        f"unknown v12 post-hoc audit variant: {variant_id}"
    )


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_ACTIVE_MASS_CAPS",
    "M03R_V12_POSTHOC_AUDIT_BOOTSTRAP_BLOCKS",
    "M03R_V12_POSTHOC_AUDIT_COST_BASIS_POINTS",
    "M03R_V12_POSTHOC_AUDIT_DESIGN_ID",
    "M03R_V12_POSTHOC_AUDIT_FOLD_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_GENERATION",
    "M03R_V12_POSTHOC_AUDIT_HORIZON",
    "M03R_V12_POSTHOC_AUDIT_INPUT_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_PANEL_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256",
    "M03R_V12_POSTHOC_AUDIT_SCORE_CHANNELS",
    "M03R_V12_POSTHOC_AUDIT_SIGNAL_TRANSFORMS",
    "M03R_V12_POSTHOC_AUDIT_SPEC",
    "M03R_V12_POSTHOC_AUDIT_VARIANTS",
    "M03RV12PosthocAuditProtocolError",
    "M03RV12PosthocAuditSpec",
    "M03RV12PosthocAuditVariant",
    "resolve_m03r_v12_posthoc_audit_variant",
]
